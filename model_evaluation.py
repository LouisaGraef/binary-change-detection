
from pathlib import Path
import pandas as pd
import dill
import os
import csv
from scipy.stats import spearmanr
import networkx as nx
from correlation_clustering import cluster_correlation_search
from sklearn.metrics import adjusted_rand_score 
from sklearn import metrics
import numpy as np
from collections import defaultdict



# https://github.com/FrancescoPeriti/CSSDetection 
# https://github.com/FrancescoPeriti/CSSDetection/blob/main/src/model_evaluation.py 
# https://github.com/Garrafao/correlation_clustering 


# scores[word].append(dict(identifier1=row1['identifier'].iloc[0], identifier2=row2['identifier'].iloc[0], judgment=1-cosine(E1[-1], E2[-1])))

"""
compute purity score of a predicted clustering
"""
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true,y_pred)
    # purity
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    return purity



"""
Read uses and rename grouping to 1 and 2 if dataset is nor_dia_change-main
"""
def load_word_usages(dataset, word):
    uses = pd.read_csv(f'{dataset}/data/{word}/uses.csv', sep='\t', quoting=csv.QUOTE_NONE)
    
    if 'nor_dia_change-main/subset1' in dataset:
        uses['grouping'] = [1 if '1970-2015' == i else 2 for i in uses['grouping']]
    elif 'nor_dia_change-main/subset2' in dataset:
        uses['grouping'] = [1 if '1980-1990' == i else 2 for i in uses['grouping']]

    return uses


"""
Read stats_groupings (graded and binary change gold scores, ...)
"""
def load_gold_lsc(dataset):
    if 'nor_dia_change-main' in dataset:
        stats_groupings = pd.read_csv(f'{dataset}/stats/stats_groupings.tsv', sep='\t')
    else:
        stats_groupings = pd.read_csv(f'{dataset}/stats/opt/stats_groupings.csv', sep='\t')
    
    return stats_groupings


"""
Read data (edge predictions, gold edge weights, gold clusters, gold graded change) 
"""
def read_data(dataset):
    # read targets
    words = sorted(os.listdir(f'{dataset}/data/'))
    
    # read judgments by LM
    ds = os.path.basename(dataset)
    edge_preds = dill.load(open(f'./scores/{ds}/scores.dill', mode='rb'))   # dict: for every word list of dicts (identifier1, identifier2, judgment)
    words = [word for word in words if word in edge_preds]
    edge_preds = [pd.DataFrame(edge_preds[word]) for word in words]     # list of dataframes (for every word: identifier1, identifier2, judgment)
    
    
    # pre-processing
    for df in edge_preds:
        df['identifier1'] = df['identifier1'].apply(lambda x: x.replace('Name: identifier1', ''))
        df['identifier2'] = df['identifier2'].apply(lambda x: x.replace('Name: identifier2', ''))   
    
    gold_wic_dfs = list()   # list of dfs ['index', 'identifier1', 'identifier2', 'judgment', 'grouping1', 'grouping2']
    gold_clusters = list()

    for i, word in enumerate(words):

        # read gold judgments
        if 'nor_dia_change-main' in dataset:
            gold_wic_df = pd.read_csv(f'{dataset}/data_joint/data_joint.tsv', sep='\t')
            gold_wic_df = gold_wic_df[gold_wic_df['lemma']==word][['identifier1', 'identifier2', 'judgment']]
        else:
            gold_wic_df = pd.read_csv(f'{dataset}/data/{word}/judgments.csv', sep='\t')[['identifier1', 'identifier2', 'judgment']]

        gold_wic_df['identifier1'] = gold_wic_df['identifier1'].astype(str)
        gold_wic_df['identifier2'] = gold_wic_df['identifier2'].astype(str)
        # get mean annotation for every edge 
        gold_wic_df = gold_wic_df.sort_values(['identifier1', 'identifier2']).groupby(['identifier1', 'identifier2']).mean().reset_index()
        
        
        # load gold clusters dataframe 
        if 'nor_dia_change-main' in dataset:
            gold_cluster = pd.read_csv(f'{dataset}/clusters/{word}.tsv', sep='\t')
        else:
            gold_cluster = pd.read_csv(f'{dataset}/clusters/opt/{word}.csv', sep='\t')

        # ignore noisy word usages (i.e. cluster = -1)
        valid_idx = gold_cluster[gold_cluster.cluster!=-1].identifier.values  # nodes that are not in gold cluster -1 
        # filter -1 nodes out of gold clusters and -1 edges out of gold edge weights and predicted edge weights 
        gold_cluster = gold_cluster[gold_cluster.identifier.isin(valid_idx)]  
        gold_wic_df = gold_wic_df[gold_wic_df.identifier1.isin(valid_idx) & gold_wic_df.identifier2.isin(valid_idx)].sort_values(['identifier1', 'identifier2']).reset_index()
        edge_preds[i] = edge_preds[i][edge_preds[i].identifier1.isin(valid_idx) & edge_preds[i].identifier2.isin(valid_idx)].sort_values(['identifier1', 'identifier2'])
        

        # ignore usage pairs for which it wasn't possible to make a prediction
        usages = set(edge_preds[i].identifier1.tolist()+edge_preds[i].identifier2.tolist()) # all nodes that have edge predictions 
        gold_wic_df = gold_wic_df[gold_wic_df.identifier1.isin(usages) & gold_wic_df.identifier2.isin(usages)]  # only nodes that are also in edge_preds[i] (of word i)
        gold_cluster = gold_cluster[gold_cluster.identifier.isin(usages)]   # only nodes that are also in edge_preds[i] (of word i)


        # add grouping info
        uses = load_word_usages(dataset, word)[['identifier', 'grouping']]
        uses = {row['identifier']: row['grouping'] for _, row in uses.iterrows()}   # mapping from identifiers to grouping 
        gold_wic_df['grouping1'] = [uses[row['identifier1']] for _, row in gold_wic_df.iterrows()]  # grouping of identifier 1 
        gold_wic_df['grouping2'] = [uses[row['identifier2']] for _, row in gold_wic_df.iterrows()]  # grouping of identifier 2 
        edge_preds[i]['grouping1'] = [uses[row['identifier1']] for _, row in edge_preds[i].iterrows()]  # grouping of identifier 1 
        edge_preds[i]['grouping2'] = [uses[row['identifier2']] for _, row in edge_preds[i].iterrows()]  # grouping of identifier 2 

        # print(gold_wic_df.columns.tolist())    # ['index', 'identifier1', 'identifier2', 'judgment', 'grouping1', 'grouping2']
        # print(edge_preds[i].columns.tolist())  # ['identifier1', 'identifier2', 'judgment', 'grouping1', 'grouping2']
        # print(gold_cluster.columns.tolist())   # ['identifier', 'cluster']

        gold_wic_dfs.append(gold_wic_df)
        gold_clusters.append(gold_cluster)

    
    gold_lsc_df = load_gold_lsc(dataset)[['lemma', 'change_graded']]    # read graded change gold scores 
    if 'nor_dia_change-main' not in dataset:
        gold_lsc_df['lemma'] = gold_lsc_df['lemma'].str.normalize('NFKD') 
    gold_lsc_df = gold_lsc_df[gold_lsc_df['lemma'].isin(words)].sort_values('lemma')        # sort 
    # print(gold_lsc_df.shape, len(words))          # (44, 2) 50 -> columns 'lemma' and 'change_graded'
    
    return edge_preds, gold_wic_dfs, gold_clusters, gold_lsc_df



""" 
Evaluate WIC (edge weight predictions) with Spearman correlation
"""
def WiC(edge_preds, gold_wic_dfs):
    # print(edge_preds[0].columns.tolist())         ['identifier1', 'identifier2', 'judgment', 'grouping1', 'grouping2']
    # print(gold_wic_dfs[0].columns.tolist())   ['index', 'identifier1', 'identifier2', 'judgment', 'grouping1', 'grouping2']
    # store gold judgments and predicted judgments (list of (mean) judgment for all edges)
    y, y_true = list(), list() 
    for edge_pred, gold_wic_df in zip(edge_preds, gold_wic_dfs):            # iterate dataframes for words 
        y.extend(edge_pred.judgment.values.tolist())
        y_true.extend(gold_wic_df.judgment.values.tolist())
    return spearmanr(y, y_true)


"""
Cluster Graph with Correlation Clustering, Evaluate Clustering with Adjusted Rand Index 
"""
def WSI(edge_preds, gold_clusters):
    # print(edge_preds[0].columns.tolist())  # ['identifier1', 'identifier2', 'judgment', 'grouping1', 'grouping2'] for first word
    # print(gold_clusters[0].columns.tolist())         # ['identifier', 'cluster'] 
    cluster_dists = list()      # cluster probability distributions 
    cluster_labels = list()     # cluster labels 
    clusters_metrics = defaultdict(list)    # cluster metrics (evaluation metrics (ARI, Purity)) 


    # concatenate edge_preds - standardize judgments - and then split again
    start_end = [(0, edge_preds[0].shape[0])] + [(pd.concat(edge_preds[:i]).shape[0], pd.concat(edge_preds[:i]).shape[0]+edge_preds[i].shape[0]) for i in range(1, len(edge_preds))]
        # shows how many edge predictions (which indices) belong to which word 
        # print(start_end)      # [(0, 1105), (1105, 1318), (1318, 1972), (1972, 2197), ...]
    # print(edge_preds[0].shape)        # (1105, 5) (number of edge weight predictions, number of columns)

    # concatenate edge_preds
    edge_preds = pd.concat(edge_preds).reset_index(drop=True) 
    # standardize judgments -> mean of judgments is 0, standard deviation is 1 
    edge_preds['judgment'] = (edge_preds['judgment'].values - edge_preds['judgment'].values.mean()) / edge_preds['judgment'].values.std() 
    # split edge_preds again 
    edge_preds = [edge_preds.iloc[idx[0]:idx[1]] for idx in start_end]


    for i, word_edge_preds in enumerate(edge_preds):        # edge preds for one word 
        # Generate Graph 
        graph = nx.Graph()
        for _, edge_pred in word_edge_preds.iterrows():                               # one edge pred
            graph.add_edge(edge_pred['identifier1'] + '###' + str(edge_pred['grouping1']), 
                           edge_pred['identifier2'] + '###' + str(edge_pred['grouping2']), 
                           weight = edge_pred['judgment'])
        
        # Cluster Graph 
        classes = []
        for init in range(1):
            classes = cluster_correlation_search(graph, s=20, max_attempts=2000, max_iters=50000, initial=classes)

        # print(classes)          # Tupel: 1st element list of sets of nodes, 2nd element dict with parameters

        # Store cluster labels (cluster label for each node)
        labels = list()
        for k, cl in enumerate(classes[0]):     # enumerate sets of nodes
            for idx in cl:                   # enumerate nodes of cluster k 
                labels.append(dict(identifier=idx.split('###')[0], cluster=k))
        labels = pd.DataFrame(labels).sort_values('identifier')                  # convert labels from list of dicts to dataframe
        cluster_labels.append(classes[0])


        # compute cluster metrics for word i
        gold_clusters[i] = gold_clusters[i].sort_values('identifier')

        clusters_metrics['adjusted_rand_score'].append(adjusted_rand_score(labels.cluster.values, gold_clusters[i].cluster.values))
        clusters_metrics['purity'].append(purity_score(labels.cluster.values, gold_clusters[i].cluster.values))


        # compute cluster distributions 
        # compute cluster frequency distribution 
        count = defaultdict(lambda: defaultdict(int))
        for j, cluster in enumerate(classes[0]):            # enumerate sets of nodes 
            for node in cluster:                            # enumerate nodes of cluster j 
                time_period = int(node.split('###')[-1])
                count[time_period][j]+=1                # increase counter for cluster j in time period of node by 1
        
        # compute cluster probability distribution 
        prob = [[],[]]
        for j in range(len(classes[0])):            # iterate clusters (indexes) 
            for t, time_period in enumerate(list(count.keys())):        # enumerate time periods ([1,2])
                if j in count[time_period]:                             # if cluster label is in cluster frequency distibution
                    prob[t].append(count[time_period][j] / sum(count[time_period].values()))    # append probability of cluster label in the given time period
                else:
                    prob[t].append(0.0)

        cluster_dists.append(prob)      # append prob distribution of word to the cluster_dists list 

                
    # take mean values across all words cluster metrics 
    # clusters_metrics: {'adjusted_rand_score': [...], 'purity': [...]}
    clusters_metrics = {m: np.mean(v) for m, v in clusters_metrics.items()}         


    return clusters_metrics, cluster_labels, cluster_dists





"""
Evaluates the model performance on WIC (edges evaluation with Sprearman Correlation),
WSI (Correlation Clustering evaluation with ARI and Purity) and Graded Change Detection (Spearman Correlation)
"""
def evaluate_model(dataset, output_file="./stats/model_evaluation.tsv"):
    # file header
    header = 'dataset\ttask\tscore'

    # file output
    if not Path(output_file).is_file():
        lines = [header+ '\n']
    else:
        lines = open(output_file, mode='r', encoding='utf-8').readlines()
    pass

    # Read data 
    edge_preds, gold_wic_dfs, gold_clusters, gold_lsc_df = read_data(dataset)
    # print(edge_preds[0].columns.tolist())         ['identifier1', 'identifier2', 'judgment', 'grouping1', 'grouping2']
    # print(gold_wic_dfs[0].columns.tolist())   ['index', 'identifier1', 'identifier2', 'judgment', 'grouping1', 'grouping2']

    # Evaluate WIC (edge weight predictions) with Spearman correlation 
    wic_spearman = WiC(edge_preds, gold_wic_dfs)

    # Add WIC Spearman Score to record 
    record = "\t".join([str(i) for i in [dataset, 'wic', wic_spearman[0].round(3)]]) + '\n'
    lines.append(record)
    print(record)
    

    # Get Clusters (Correlation Clustering) for all dataset words
    clusters_metrics, clusters_labels, clusters_dists = WSI(edge_preds, gold_clusters)
    print('ok')
    quit()

    # __________________________________________________________________________________________

    # Save clusters and cluster probability distributions 
    Path(f'probs/{dataset}').mkdir(exist_ok=True, parents=True)
    Path(f'classes/{dataset}').mkdir(exist_ok=True, parents=True)

    with open(f'probs/{dataset}.dill', mode='+wb') as f:
        dill.dump(clusters_dists, f)
        
    with open(f'classes/{dataset}.dill', mode='+wb') as f:
        dill.dump(clusters_labels, f)

    # Add WSI evaluation metrics to record 
    for metric in clusters_metrics:
        record = "\t".join([str(i) for i in [dataset, f'wsi-{metric}', clusters_metrics[metric].round(3), pd.concat(edge_preds).shape[0]]]) + '\n'
        lines.append(record)
        print(record)




if __name__=="__main__":
    evaluate_model("./data/dwug_de")