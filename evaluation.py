
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import dill
import os
import csv
from scipy.stats import spearmanr
import networkx as nx
from sklearn.metrics import adjusted_rand_score 
from sklearn import metrics
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
import itertools
from cluster_graph import cluster_graph




# https://github.com/FrancescoPeriti/CSSDetection 
# https://github.com/FrancescoPeriti/CSSDetection/blob/main/src/model_evaluation.py 
# https://github.com/Garrafao/correlation_clustering 


# edge_preds[word].append(dict(identifier1=row1['identifier'].iloc[0], identifier2=row2['identifier'].iloc[0], judgment=1-cosine(E1[-1], E2[-1])))

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
        stats_groupings = pd.read_csv(f'{dataset}/stats/stats_groupings.tsv', sep='\t', encoding='utf-8')
    else:
        stats_groupings = pd.read_csv(f'{dataset}/stats/opt/stats_groupings.csv', sep='\t', encoding='utf-8')
        # lemmas need to be normalized in dwug_es so that lemmas are sorted in the same order as 'words' in evaluate_model
        # -> graded change score predictions at index i in lemmas and 'words' are for the same word
        if "dwug_es" in dataset:
            stats_groupings['lemma'] = stats_groupings['lemma'].str.normalize('NFKD')
    
    return stats_groupings


"""
Read data (edge predictions, gold edge weights, gold clusters, gold graded change) 
"""
def read_data(dataset, paper_reproduction):
    # read targets
    words = sorted(os.listdir(f'{dataset}/data/'))
    
    # read edge weight gold judgments and predictions 
    if paper_reproduction:
        ds = dataset.replace("./paper_data/", "")
        edge_preds = dill.load(open(f'./paper_edge_preds/{ds}/paper_edge_preds.dill', mode='rb'))   # dict: for every word dataframe (identifier1, identifier2, judgment, edge_pred)
        edge_preds_full = None
    else:
        ds = dataset.replace("./data/", "")
        edge_preds = dill.load(open(f'./edge_preds/{ds}/edge_preds.dill', mode='rb'))   # dict: for every word dataframe (identifier1, identifier2, judgment, edge_pred)
        edge_preds_full = dill.load(open(f'./edge_preds/{ds}/edge_preds_full.dill', mode='rb'))   # dict: for every word dataframe (identifier1, identifier2, edge_pred)
        words = [word for word in words if word in edge_preds_full]
        edge_preds_full = [edge_preds_full[word] for word in words]        # list of dataframes (['identifier1', 'identifier2', 'edge_pred'] for every word)
        
    words = [word for word in words if word in edge_preds]
    edge_preds = [edge_preds[word] for word in words]        # list of dataframes (['identifier1', 'identifier2', 'judgment', 'edge_pred'] for every word)
    
    
    # pre-processing
    if not dataset=="./data/dwug_la":       # no gold edge weights for dwug_la 
        for df in edge_preds:
            df['identifier1'] = df['identifier1'].apply(lambda x: x.replace('Name: identifier1', ''))
            df['identifier2'] = df['identifier2'].apply(lambda x: x.replace('Name: identifier2', ''))  
    if not paper_reproduction:
        for df in edge_preds_full:
            df['identifier1'] = df['identifier1'].apply(lambda x: x.replace('Name: identifier1', ''))
            df['identifier2'] = df['identifier2'].apply(lambda x: x.replace('Name: identifier2', ''))  
    
    gold_clusters = list()  # list of dfs ['identifier', 'cluster']

    for i, word in enumerate(words):

        # print(edge_preds[i].columns.tolist())         # ['identifier1', 'identifier2', 'judgment', 'edge_pred']
        

        # read gold clusters dataframe 
        if 'nor_dia_change-main' in dataset:
            gold_cluster = pd.read_csv(f'{dataset}/clusters/{word}.tsv', sep='\t')
        else:
            gold_cluster = pd.read_csv(f'{dataset}/clusters/opt/{word}.csv', sep='\t')

        # ignore noisy word usages (i.e. cluster = -1)
        if paper_reproduction==True:
            valid_idx = gold_cluster[gold_cluster.cluster!=-1].identifier.values  # nodes that are not in gold cluster -1 
            # filter -1 nodes out of gold clusters and -1 edges out of gold edge weights and predicted edge weights 
            gold_cluster = gold_cluster[gold_cluster.identifier.isin(valid_idx)]  
            edge_preds[i] = edge_preds[i][edge_preds[i].identifier1.isin(valid_idx) & edge_preds[i].identifier2.isin(valid_idx)].sort_values(['identifier1', 'identifier2']).reset_index()
        
        if not dataset=="./data/dwug_la":
            
            # ignore usage pairs for which it wasn't possible to make a prediction
            usages = set(edge_preds[i].loc[edge_preds[i]['edge_pred'].notna(), 'identifier1'].tolist()
                        + edge_preds[i].loc[edge_preds[i]['edge_pred'].notna(), 'identifier2'].tolist())           # all nodes that have edge predictions 
            edge_preds[i] = edge_preds[i][edge_preds[i].identifier1.isin(usages) & edge_preds[i].identifier2.isin(usages)]  # only edges where both nodes have edge predictions
            gold_cluster = gold_cluster[gold_cluster.identifier.isin(usages)]   # only nodes that have edge predictions 


        # add grouping info
        uses = load_word_usages(dataset, word)
        uses = uses[['identifier', 'grouping']]
        uses = {row['identifier']: row['grouping'] for _, row in uses.iterrows()}   # mapping from identifiers to grouping 
        if not dataset=="./data/dwug_la":
            edge_preds[i]['grouping1'] = [uses[row['identifier1']] for _, row in edge_preds[i].iterrows()]  # grouping of identifier 1 
            edge_preds[i]['grouping2'] = [uses[row['identifier2']] for _, row in edge_preds[i].iterrows()]  # grouping of identifier 2 
        if not paper_reproduction:  
            edge_preds_full[i]['grouping1'] = [uses[row['identifier1']] for _, row in edge_preds_full[i].iterrows()]  # grouping of identifier 1 
            edge_preds_full[i]['grouping2'] = [uses[row['identifier2']] for _, row in edge_preds_full[i].iterrows()]  # grouping of identifier 2 

        # print(edge_preds[i].columns.tolist())    # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2']
        # print(gold_cluster.columns.tolist())   # ['identifier', 'cluster']

        gold_clusters.append(gold_cluster)
    


    gold_gc = load_gold_lsc(dataset)[['lemma', 'change_graded']].sort_values('lemma')    # read graded change gold scores 
    gold_bc = load_gold_lsc(dataset)[['lemma', 'change_binary']].sort_values('lemma')    # read graded change gold scores 

    
    # print(gold_gc.shape, len(words))          # (50, 2) 50 -> columns 'lemma' and 'change_graded'
    

    # edge_preds: list of dataframes (one df for every word)
    # print(edge_preds[0].columns.tolist())     # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2']

    # edge_preds_full: list of dataframes (one df for every word)
    # print(edge_preds_full[0].columns.tolist())     # ['identifier1', 'identifier2', 'edge_pred', 'grouping1', 'grouping2']

    # gold_clusters: list of dataframes (one df for every word)
    # print(gold_clusters[0].columns.tolist())      # ['identifier', 'cluster']

    # gold_gc: dataframe 
    # print(gold_gc.columns.tolist())       # ['lemma', 'change_graded']
    
    return edge_preds, edge_preds_full, gold_clusters, gold_gc, gold_bc




""" 
Evaluate WIC (edge weight predictions) with Spearman correlation
"""
def WiC(edge_preds, paper_reproduction):
    # edge_preds: list of dataframes (one df for every word)
    # print(edge_preds[0].columns.tolist())     # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2']

    # store predicted judgments and gold judgments (for all edges and all words)
    y_pred, y_true = list(), list() 
    for df in edge_preds:
        y_pred.extend(df.edge_pred.values.tolist())
        y_true.extend(df.judgment.values.tolist())

    if paper_reproduction==False:
        # Filter pd.na values out
        valid_indices = [i for i, (x,y) in enumerate(zip(y_true, y_pred)) if not pd.isna(x) and not pd.isna(y)]
        filtered_y_true = [y_true[i] for i in valid_indices]
        filtered_y_pred = [y_pred[i] for i in valid_indices]
        spearman_corr = spearmanr(filtered_y_pred, filtered_y_true)
    else:
        spearman_corr = spearmanr(y_pred, y_true)
    return spearman_corr



"""
Standardize edge weight predictions 
-> mean of predicted edge weights is 0, standard deviation is 1
(only relevant for paper reproduciton)
"""
def standardize_edge_preds(edge_preds):
    # concatenate edge_preds - standardize predicted edge weights - and then split again
    start_end = [(0, edge_preds[0].shape[0])] + [(pd.concat(edge_preds[:i]).shape[0], pd.concat(edge_preds[:i]).shape[0]+edge_preds[i].shape[0]) for i in range(1, len(edge_preds))]
        # shows how many edge predictions (which indices) belong to which word 
        # print(start_end)      # [(0, 1105), (1105, 1318), (1318, 1972), (1972, 2197), ...]
    # print(edge_preds[0].shape)        # (1105, 5) (number of edge weight predictions, number of columns)

    # concatenate edge_preds
    edge_preds = pd.concat(edge_preds).reset_index(drop=True) 
    # standardize predicted edge weights -> mean of predicted edge weights is 0, standard deviation is 1 
    edge_preds['edge_pred'] = (edge_preds['edge_pred'].values - edge_preds['edge_pred'].values.mean()) / edge_preds['edge_pred'].values.std() 
    # split edge_preds again 
    edge_preds = [edge_preds.iloc[idx[0]:idx[1]] for idx in start_end]

    return edge_preds


"""
Generate Graph with predicted edge weights
"""
def generate_graph(word_edge_preds):
    # Generate Graph 
    graph = nx.Graph()
    for _, row in word_edge_preds.iterrows():                               # one edge weight prediction
        graph.add_edge(row['identifier1'] + '###' + str(row['grouping1']), 
                        row['identifier2'] + '###' + str(row['grouping2']), 
                        weight = row['edge_pred'])
    return graph




"""
Compute cluster distributions (cluster frequency distribution and cluster probability distribution) for one Graph
"""
def get_cluster_distributions(classes_sets):
    
    # compute cluster distributions 
    # compute cluster frequency distribution 
    freq_dist = defaultdict(lambda: defaultdict(int))
    for j, cluster in enumerate(classes_sets):            # enumerate sets of nodes 
        for node in cluster:                            # enumerate nodes of cluster j 
            time_period = int(node.split('###')[-1])
            freq_dist[time_period][j]+=1                # increase counter for cluster j in time period of node by 1
    
    # compute cluster probability distribution 
    prob_dist = [[],[]]
    for j in range(len(classes_sets)):            # iterate clusters (indexes) 
        for t, time_period in enumerate(list(freq_dist.keys())):        # enumerate time periods ([1,2])
            if j in freq_dist[time_period]:                             # if cluster label is in cluster frequency distibution
                prob_dist[t].append(freq_dist[time_period][j] / sum(freq_dist[time_period].values()))  # append probability of cluster label in the given time period
            else:
                prob_dist[t].append(0.0)
        
    return freq_dist, prob_dist



"""
Get Graded Change Scores with Jensenshannon distance, 
Evaluate predicted Graded Change Scores with Spearman correlation coefficient 
"""
def GCD(clusters_dists, gold_gc, paper_reproduction):
    # cluster_dists: list of cluster probability distributions (for every word prob = [[],[]])
    if paper_reproduction:
        glsc_pred = np.array([jensenshannon(d[0], d[1]) for d in clusters_dists])        # graded lsc value for every word 
    else:
        glsc_pred = np.array([jensenshannon(d[0], d[1], base=2.0) for d in clusters_dists])        # graded lsc value for every word 
    # gold_gc columns: ['lemma', 'change_graded']
    glsc_true = gold_gc.change_graded.values                                     # gold graded lsc value for every word 
    return spearmanr(glsc_pred, glsc_true)



"""
Predict binary change score of one word
"""
def predict_binary(freq_dist):
    bc_pred = 0
    cluster_ids = set(freq_dist[1].keys()).union(set(freq_dist[2].keys()))      # cluster ids (labels)
    for cluster_id in cluster_ids:
        freq1 = freq_dist[1].get(cluster_id,0)      # frequency of cluster id in time period 1
        freq2 = freq_dist[2].get(cluster_id,0)      # frequency of cluster id in time period 2
        if freq1<=1 and freq2>=3 or freq2<=1 and freq1>=3:
            bc_pred = 1

    return bc_pred
    


"""
Evaluate the model performance on WIC (edges evaluation with Spearman Correlation)
"""
def evaluate_wic(datasets, paper_reproduction):
    header = 'dataset\tWIC_spearman_correlation'
    lines = [header+ '\n']
    
    print("\nWIC evaluation with Spearman Correlation:\n")
    for dataset in datasets:
        # Read data 
        edge_preds, edge_preds_full, gold_clusters, gold_gc, gold_bc = read_data(dataset, paper_reproduction)
        # print(edge_preds[0].columns.tolist())     # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2']

        # Evaluate WIC (edge weight predictions) with Spearman correlation 
        wic_spearman = WiC(edge_preds, paper_reproduction)
        
        # Add WIC Spearman Score to record 
        if paper_reproduction:
            ds = dataset.replace("./paper_data/", "")
        else:
            ds = dataset.replace("./data/", "")
        new_line = "\t".join([str(i) for i in [ds, wic_spearman[0].round(3)]]) + '\n'
        lines.append(new_line)
        print(new_line)
    
    if paper_reproduction:
        output_file=f"./stats/paper_wic_evaluation.tsv"
    else:
        output_file=f"./stats/wic_evaluation.tsv"
    
    # Save WIC evaluation results to output_file
    with open(output_file, mode='w', encoding='utf-8') as f:
        f.writelines(lines)



"""
Evaluates the model performance on WSI (Clustering evaluation with ARI (and Purity)) and Graded Change Detection (Spearman Correlation)
"""
def evaluate_model(dataset, paper_reproduction, clustering_method, parameter_list):
    
    print(f'\nDataset: {dataset}  \nClustering Method: {clustering_method}')

    if paper_reproduction:
        ds = dataset.replace("./paper_data/", "")
    else:
        ds = dataset.replace("./data/", "")

    if paper_reproduction==True:                                    # Evaluation results on whole dataset (WSI and GCD)
        output_file=f"./stats/paper_model_evaluation.tsv"
        header = 'dataset\tWSI_ARI\tWSI_Purity\tGCD_Spearman_Correlation'

        # evaluation file output
        if Path(output_file).stat().st_size == 0:       # empty file
            paper_eval_lines = [header+ '\n']
            print('\n\nWSI and GCD evaluation for Paper reproduction: \n')
            print(header + '\n')
        else:
            paper_eval_lines = open(output_file, mode='r', encoding='utf-8').readlines()
        pass


    # Read data 
    edge_preds, edge_preds_full, gold_clusters, gold_gc, gold_bc = read_data(dataset, paper_reproduction=paper_reproduction)
    # print(edge_preds[0].columns.tolist())     # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2']
    # print(gold_clusters[0].columns.tolist())      # ['identifier', 'cluster']
    # print(gold_gc.columns.tolist())       # ['lemma', 'change_graded']
    

    # Standardize edge weight predictions -> mean of predicted edge weights is 0, standard deviation is 1 (only relevant for paper reproduciton)
    if paper_reproduction==True:
        edge_preds = standardize_edge_preds(edge_preds)

    

    
    # print(edge_preds[0].columns.tolist())  # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2'] for first word
    # print(gold_clusters[0].columns.tolist())         # ['identifier', 'cluster'] 
    cluster_dists = list()      # cluster probability distributions 
    clusters_metrics = defaultdict(list)    # cluster metrics (evaluation metrics (ARI, Purity)) 

    words = sorted(os.listdir(f'{dataset}/data/'))
    
    combinations = list(itertools.product(*parameter_list))     # all parameter combinations
    
    print(combinations)

    # delete old parameter grid if one exists
    df_output_file=f"./parameter_grids/{ds}/{clustering_method}/parameter_grid.tsv"
    if os.path.exists(df_output_file):
        os.remove(df_output_file)

    for comb in combinations:
        if comb[0] == 'ward' and not comb[1] == 'euclidean':    # Agglomerative clustering: Ward can only work with euclidean distances.
            continue
        if comb[0] == 'rbf' and (comb[1] == 10 or comb[1] == 15):   # Spectral clustering: n_neighbors not relevant for rbf (only let rbf run once)
            continue
        print(f'\n\nParameters: {comb}')
        
        # Parameter grid for crossvalidation (Clustering: Mapping from node identifiers to Cluster ID (dictionary))
        parameter_grid = pd.DataFrame(
            columns=['parameter_combination', 'word', 'BC_pred', 'BC_gold', 'GC_pred', 'GC_gold', 'clustering_pred', 'clustering_gold'])
        
        word_counter = 1
        
        for i, word in enumerate(words):                    # iterate over words
            if paper_reproduction:
                word_edge_preds = edge_preds[i]
            else:
                word_edge_preds = edge_preds_full[i]    # use edge predictions of all edges

            graph = generate_graph(word_edge_preds)         # generate graph with predicted edge weights for word i
            # cluster graph (cluster_labels: dataframe with columns ['identifier', 'cluster'])
            #print(word)
            cluster_labels, classes_sets = cluster_graph(graph, clustering_method, paper_reproduction, comb, word=word)    
            
            # Evaluate clustering with ARI and Purity
            gold_clusters[i] = gold_clusters[i].sort_values('identifier')

            # Filter cluster_labels of clusters that have no gold cluster out
            if paper_reproduction==False:
                cluster_labels = cluster_labels[cluster_labels['identifier'].isin(gold_clusters[i]['identifier'])].sort_values(['identifier']).reset_index(drop=True)
                gold_clusters[i] = gold_clusters[i][gold_clusters[i]['identifier'].isin(cluster_labels['identifier'])].sort_values(['identifier']).reset_index(drop=True)
            

            clusters_metrics['adjusted_rand_score'].append(adjusted_rand_score(cluster_labels.cluster.values, gold_clusters[i].cluster.values))
            clusters_metrics['purity'].append(purity_score(cluster_labels.cluster.values, gold_clusters[i].cluster.values))
            print(f'{word_counter}/{len(words)}\t ARI: {clusters_metrics['adjusted_rand_score'][-1]}\tPurity: {clusters_metrics['purity'][-1]}')

            # Compute cluster distributions (cluster frequency distribution and cluster probability distribution) for one Graph
            freq_dist, prob_dist = get_cluster_distributions(classes_sets)
            cluster_dists.append(prob_dist)     # cluster distributions for LSC 

            # Add evaluation results to parameter_grid
            gc_pred = jensenshannon(prob_dist[0], prob_dist[1], base=2.0)
            bc_pred = predict_binary(freq_dist)

            gc_true = gold_gc.iloc[i]['change_graded']
            bc_true = gold_bc.iloc[i]['change_binary']
            clustering_pred = cluster_labels.set_index('identifier')['cluster'].to_dict()    # maps node_ids to cluster ids
            # columns=['parameter_combination', 'word', 'BC_pred', 'BC_gold', 'GC_pred', 'GC_gold', 'clustering_pred', 'clustering_gold']
            gold_clusters_word = gold_clusters[i].set_index('identifier')['cluster'].to_dict()    # maps node_ids to cluster ids
            new_grid_row = {'parameter_combination': comb, 'word': word, 'GC_pred': gc_pred, 'GC_gold': gc_true, 
                            'BC_pred': bc_pred, 'BC_gold': bc_true, 'clustering_pred': clustering_pred, 'clustering_gold': gold_clusters_word}
            parameter_grid.loc[len(parameter_grid)] = new_grid_row

            word_counter+=1
        
        
        # Save Parameter grid of dataset (and clustering method)
        df_output_file=f"./parameter_grids/{ds}/{clustering_method}/parameter_grid.tsv"
        if not os.path.exists(df_output_file):
            os.makedirs(os.path.dirname(df_output_file), exist_ok=True)
            parameter_grid.to_csv(df_output_file, sep='\t', mode='w', index=False)
        else:
            parameter_grid.to_csv(df_output_file, sep='\t', mode='a', header=False, index=False)



    # Paper reproduction: evaluate on whole dataset (mean ARI, mean Purity, GCD with Spearman Correlation)
    if paper_reproduction:
        # take mean values across all words cluster metrics 
        # clusters_metrics: {'adjusted_rand_score': [...], 'purity': [...]}
        clusters_metrics = {m: np.mean(v) for m, v in clusters_metrics.items()}         # WSI: ARI, Purity

        gcd_spearman = GCD(cluster_dists, gold_gc, paper_reproduction=True)                                 # GCD: Spearman correlation

        # header = 'dataset\tWSI_ARI\tWSI_Purity\tGCD_Spearman_Correlation'             # Add paper evaluation results to paper_eval_lines
        record = "\t".join([str(i) for i in [ds, clusters_metrics['adjusted_rand_score'].round(3), 
                                             clusters_metrics['purity'].round(3), gcd_spearman[0].round(3)]]) + '\n'
        paper_eval_lines.append(record)
        print(record)


        # Save WSI and GCD evaluation results to output_file="./stats/paper_model_evaluation.tsv"
        with open(output_file, mode='w', encoding='utf-8') as f:
            f.writelines(paper_eval_lines)






if __name__=="__main__":

    parameter_list = [['ward', 'average', 'complete', 'single'],['euclidean', 'cosine']]
    evaluate_model("./data/dwug_la", paper_reproduction=False, clustering_method="agglomerative", parameter_list=parameter_list)
    quit()
    
    # Evalation with WIC;WSI;GCD 
    datasets = ["dwug_de", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       # no dwug_la 
    datasets = ["./data/" + dataset for dataset in datasets]
    
    # Read data 
    edge_preds, edge_preds_full, gold_clusters, gold_gc, gold_bc = read_data("./data/chiwug", paper_reproduction=False)
    # print(edge_preds[0].columns.tolist())     # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2']

    # Evaluate WIC (edge weight predictions) with Spearman correlation 
    wic_spearman = WiC(edge_preds, paper_reproduction=False)
    
    # Add WIC Spearman Score to record 
    new_line = "\t".join([str(i) for i in ["chiwug", wic_spearman[0].round(3)]]) + '\n'
    print(new_line)

    """
    # WSI and LSC evaluation 

    parameter_list = [[1],[2],[3]]
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="k-means", parameter_list=parameter_list)
    """