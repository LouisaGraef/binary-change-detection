
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
from correlation_clustering import cluster_correlation_search
from sklearn.metrics import adjusted_rand_score 
from sklearn import metrics
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
import itertools
from utils import transform_edge_weights
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score




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
        stats_groupings = pd.read_csv(f'{dataset}/stats/stats_groupings.tsv', sep='\t')
    else:
        stats_groupings = pd.read_csv(f'{dataset}/stats/opt/stats_groupings.csv', sep='\t')
    
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
    else:
        ds = dataset.replace("./data/", "")
        edge_preds = dill.load(open(f'./edge_preds/{ds}/edge_preds.dill', mode='rb'))   # dict: for every word dataframe (identifier1, identifier2, judgment, edge_pred)
    words = [word for word in words if word in edge_preds]
    edge_preds = [edge_preds[word] for word in words]        # list of dataframes (['identifier1', 'identifier2', 'judgment', 'edge_pred'] for every word)
    
    
    # pre-processing
    for df in edge_preds:
        df['identifier1'] = df['identifier1'].apply(lambda x: x.replace('Name: identifier1', ''))
        df['identifier2'] = df['identifier2'].apply(lambda x: x.replace('Name: identifier2', ''))   
    
    gold_clusters = list()  # list of dfs ['identifier', 'cluster']

    for i, word in enumerate(words):

        # edge weight gold judgments and predictions for one word
        wic_judgments = edge_preds[i]      
        # print(wic_judgments.columns.tolist())         # ['identifier1', 'identifier2', 'judgment', 'edge_pred']
        

        # read gold clusters dataframe 
        if 'nor_dia_change-main' in dataset:
            gold_cluster = pd.read_csv(f'{dataset}/clusters/{word}.tsv', sep='\t')
        else:
            gold_cluster = pd.read_csv(f'{dataset}/clusters/opt/{word}.csv', sep='\t')

        # ignore noisy word usages (i.e. cluster = -1)
        valid_idx = gold_cluster[gold_cluster.cluster!=-1].identifier.values  # nodes that are not in gold cluster -1 
        # filter -1 nodes out of gold clusters and -1 edges out of gold edge weights and predicted edge weights 
        gold_cluster = gold_cluster[gold_cluster.identifier.isin(valid_idx)]  
        wic_judgments = wic_judgments[wic_judgments.identifier1.isin(valid_idx) & wic_judgments.identifier2.isin(valid_idx)].sort_values(['identifier1', 'identifier2']).reset_index()
        

        # ignore usage pairs for which it wasn't possible to make a prediction
        usages = set(wic_judgments.loc[wic_judgments['edge_pred'].notna(), 'identifier1'].tolist()
                     + wic_judgments.loc[wic_judgments['edge_pred'].notna(), 'identifier2'].tolist())           # all nodes that have edge predictions 
        wic_judgments = wic_judgments[wic_judgments.identifier1.isin(usages) & wic_judgments.identifier2.isin(usages)]  # only edges where both nodes have edge predictions
        gold_cluster = gold_cluster[gold_cluster.identifier.isin(usages)]   # only nodes that have edge predictions 


        # add grouping info
        uses = load_word_usages(dataset, word)
        uses = uses[['identifier', 'grouping']]
        uses = {row['identifier']: row['grouping'] for _, row in uses.iterrows()}   # mapping from identifiers to grouping 
        wic_judgments['grouping1'] = [uses[row['identifier1']] for _, row in wic_judgments.iterrows()]  # grouping of identifier 1 
        wic_judgments['grouping2'] = [uses[row['identifier2']] for _, row in wic_judgments.iterrows()]  # grouping of identifier 2 

        # print(wic_judgments.columns.tolist())    # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2']
        # print(gold_cluster.columns.tolist())   # ['identifier', 'cluster']

        gold_clusters.append(gold_cluster)
        
        """
        # df ['identifier1', 'identifier2', 'predicted_edge_weight', 'gold_edge_weight'] for one word 
        word_edge_annotations = pd.concat([edge_preds[i][['identifier1', 'identifier2', 'edge_pred']], gold_wic_df[['judgment']]], axis=1)
        word_edge_annotations['node_pair'] = list(zip(word_edge_annotations.identifier1, word_edge_annotations.identifier2))
        # rename columns
        word_edge_annotations.columns = ['identifier1', 'identifier2', 'predicted_edge_weight', 'gold_edge_weight', 'node_pair']
        # delete first two columns and reorder columns 
        word_edge_annotations = word_edge_annotations.iloc[:,[4,2,3]]

        edge_annotations.append(word_edge_annotations)
        """

        edge_preds[i] = wic_judgments
    


    gold_lsc_df = load_gold_lsc(dataset)[['lemma', 'change_graded']]    # read graded change gold scores 

    # normalize lemmas in graded change gold scores df 
    if 'nor_dia_change' not in dataset:     
        gold_lsc_df['lemma'] = gold_lsc_df['lemma'].str.normalize('NFKD') 
    if 'dwug_de' in dataset or 'dwug_sv' in dataset:
        gold_lsc_df['lemma'] = gold_lsc_df['lemma'].str.normalize('NFKC') 

    gold_lsc_df = gold_lsc_df[gold_lsc_df['lemma'].isin(words)].sort_values('lemma')        # sort by lemmas (alphabetically)
    # print(gold_lsc_df.shape, len(words))          # (50, 2) 50 -> columns 'lemma' and 'change_graded'
    

    # edge_preds: list of dataframes (one df for every word)
    # print(edge_preds[0].columns.tolist())     # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2']

    # gold_clusters: list of dataframes (one df for every word)
    # print(gold_clusters[0].columns.tolist())      # ['identifier', 'cluster']

    # gold_lsc_df: dataframe 
    # print(gold_lsc_df.columns.tolist())       # ['lemma', 'change_graded']
    
    return edge_preds, gold_clusters, gold_lsc_df




""" 
Evaluate WIC (edge weight predictions) with Spearman correlation
"""
def WiC(edge_preds):
    # edge_preds: list of dataframes (one df for every word)
    # print(edge_preds[0].columns.tolist())     # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2']

    # store predicted judgments and gold judgments (for all edges and all words)
    y_pred, y_true = list(), list() 
    for df in edge_preds:
        y_pred.extend(df.edge_pred.values.tolist())
        y_true.extend(df.judgment.values.tolist())
    # Filter nan values out
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    filtered_y_true = y_true[valid_indices]
    filtered_y_pred = y_pred[valid_indices]
    return spearmanr(filtered_y_pred, filtered_y_true)



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
Cluster graph
"""
def cluster_graph(graph, clustering_method, paper_reproduction, parameters):
    if "correlation" in clustering_method:
        labels, classes_sets = cluster_graph_cc(graph, paper_reproduction, parameters)
    elif clustering_method=="k-means":
        labels, classes_sets = cluster_graph_kmeans(graph,parameters)
    elif clustering_method=="agglomerative":
        labels, classes_sets = cluster_graph_agglomerative(graph,parameters)
    elif clustering_method=="spectral":
        labels, classes_sets = cluster_graph_spectral(graph,parameters)
    
    return labels, classes_sets


"""
Cluster Graph with Correlation Clustering, return cluster_labels (dataframe with columns ['identifier', 'cluster'])
parameters[0] = edge_shift_value, parameters[1] = max_attempts, parameters[2] = max_iters
"""
def cluster_graph_cc(graph, paper_reproduction, parameters):
    
    # Cluster Graph 
    classes = []
    max_attempts = parameters[1]
    max_iters = parameters[2]
    for init in range(1):
        if paper_reproduction:
            classes = cluster_correlation_search(graph, s=20, max_attempts=max_attempts, max_iters=max_iters, initial=classes)
        else:
            threshold = parameters[0]
            weight_transformation = lambda x: x-threshold
            graph = transform_edge_weights(graph, transformation = weight_transformation) # shift edge weights
            classes = cluster_correlation_search(graph, s=10, max_attempts=parameters[1], max_iters=parameters[2], initial=classes)

    # print(classes)          # Tupel: 1st element list of sets of nodes, 2nd element dict with parameters

    # Store cluster labels (cluster label for each node)
    labels = list()
    for k, cl in enumerate(classes[0]):     # enumerate sets of nodes
        for idx in cl:                   # enumerate nodes of cluster k 
            labels.append(dict(identifier=idx.split('###')[0], cluster=k))
    labels = pd.DataFrame(labels).sort_values('identifier').reset_index(drop=True)                  # convert labels from list of dicts to dataframe

    classes_sets = classes[0]           # list of sets of nodes that are in the same cluster

    return labels, classes_sets


"""
Cluster Graph with K-means (determines number of clusters with silhouette score)
"""
def cluster_graph_kmeans(graph, parameters):
    adj_matrix = nx.to_numpy_array(graph)        # Graph adjacency matrix as a numpy array
    sil_scores = []
    n_clusters = range(2, min(11, len(graph.nodes)))
    max_sil_score = -1
    best_k = None

    # Determine best number of clusters with Silhouette Score

    for k in n_clusters:                                            # 2 to 10 clusters 
        kmeans = KMeans(n_clusters=k).fit(adj_matrix)
        labels = kmeans.labels_                                     # list of cluster ids
        sil_score = silhouette_score(adj_matrix, labels)            # Silhouette Score
        sil_scores.append(sil_score)
        # print(f"Silhouette score for {k} clusters: {sil_score}")
        if sil_score > max_sil_score:
            max_sil_score = sil_score
            best_k = k

    # Cluster Adjacency Matrix with best number of clusters 
    kmeans = KMeans(n_clusters=best_k).fit(adj_matrix)
    labels = kmeans.labels_                                     
    labels = {node: label for node, label in zip(graph.nodes(), labels)}        # mapping node id to cluster id

    # Get classes_sets, convert labels to df
    classes_sets = [set() for i in range(best_k)]       # list with best_k sets 
    for node, label in labels.items():
        classes_sets[label].add(node)                   

    labels = {key.split('###')[0]: value for key, value in labels.items()}
    labels = pd.DataFrame(list(labels.items()), columns=['identifier', 'cluster']).sort_values('identifier').reset_index(drop=True) # labels as dataframe

    # print(f'Best number of clusters: {best_k} \nSilhouette score with {best_k} clusters: {max_sil_score}')
    

    return labels, classes_sets


"""
Cluster Graph with Agglomerative Clustering (determines number of clusters with silhouette score)
"""
def cluster_graph_agglomerative(graph, parameters):
    adj_matrix = nx.to_numpy_array(graph)        # Graph adjacency matrix as a numpy array
    sil_scores = []
    n_clusters = range(2, min(11, len(graph.nodes)))
    max_sil_score = -1
    best_k = None

    # Determine best number of clusters with Silhouette Score

    for k in n_clusters:                                            # 2 to 10 clusters 
        agglomerative = AgglomerativeClustering(n_clusters=k).fit(adj_matrix)
        labels = agglomerative.labels_                                     # list of cluster ids
        sil_score = silhouette_score(adj_matrix, labels)            # Silhouette Score
        sil_scores.append(sil_score)
        # print(f"Silhouette score for {k} clusters: {sil_score}")
        if sil_score > max_sil_score:
            max_sil_score = sil_score
            best_k = k

    # Cluster Adjacency Matrix with best number of clusters 
    agglomerative = AgglomerativeClustering(n_clusters=best_k).fit(adj_matrix)
    labels = agglomerative.labels_                                     
    labels = {node: label for node, label in zip(graph.nodes(), labels)}        # mapping node id to cluster id

    # Get classes_sets, convert labels to df
    classes_sets = [set() for i in range(best_k)]       # list with best_k sets 
    for node, label in labels.items():
        classes_sets[label].add(node)                   

    labels = {key.split('###')[0]: value for key, value in labels.items()}
    labels = pd.DataFrame(list(labels.items()), columns=['identifier', 'cluster']).sort_values('identifier').reset_index(drop=True) # labels as dataframe

    # print(f'Best number of clusters: {best_k} \nSilhouette score with {best_k} clusters: {max_sil_score}')
    

    return labels, classes_sets


"""
Cluster Graph with Spectral Clustering (determines number of clusters with silhouette score)
"""
def cluster_graph_spectral(graph, parameters):
    adj_matrix = nx.to_numpy_array(graph)        # Graph adjacency matrix as a numpy array
    sil_scores = []
    n_clusters = range(2, min(11, len(graph.nodes)))
    max_sil_score = -1
    best_k = None

    # Determine best number of clusters with Silhouette Score

    for k in n_clusters:                                            # 2 to 10 clusters 
        spectral = SpectralClustering(n_clusters=k).fit(adj_matrix)
        labels = spectral.labels_                                     # list of cluster ids
        sil_score = silhouette_score(adj_matrix, labels)            # Silhouette Score
        sil_scores.append(sil_score)
        # print(f"Silhouette score for {k} clusters: {sil_score}")
        if sil_score > max_sil_score:
            max_sil_score = sil_score
            best_k = k

    # Cluster Adjacency Matrix with best number of clusters 
    spectral = SpectralClustering(n_clusters=best_k).fit(adj_matrix)
    labels = spectral.labels_                                     
    labels = {node: label for node, label in zip(graph.nodes(), labels)}        # mapping node id to cluster id

    # Get classes_sets, convert labels to df
    classes_sets = [set() for i in range(best_k)]       # list with best_k sets 
    for node, label in labels.items():
        classes_sets[label].add(node)                   

    labels = {key.split('###')[0]: value for key, value in labels.items()}
    labels = pd.DataFrame(list(labels.items()), columns=['identifier', 'cluster']).sort_values('identifier').reset_index(drop=True) # labels as dataframe

    # print(f'Best number of clusters: {best_k} \nSilhouette score with {best_k} clusters: {max_sil_score}')
    

    return labels, classes_sets




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
def GCD(clusters_dists, gold_lsc_df):
    # cluster_dists: list of cluster probability distributions (for every word prob = [[],[]])
    glsc_pred = np.array([jensenshannon(d[0], d[1]) for d in clusters_dists])        # graded lsc value for every word 
    # gold_lsc_df columns: ['lemma', 'change_graded']
    glsc_true = gold_lsc_df.change_graded.values                                     # gold graded lsc value for every word 
    return spearmanr(glsc_pred, glsc_true)
    


"""
Evaluate the model performance on WIC (edges evaluation with Spearman Correlation)
"""
def evaluate_wic(datasets, paper_reproduction):
    header = 'dataset\tWIC_spearman_correlation'
    lines = [header+ '\n']
    
    print("\nWIC evaluation with Spearman Correlation:\n")
    for dataset in datasets:
        # Read data 
        edge_preds, gold_clusters, gold_lsc_df = read_data(dataset, paper_reproduction)
        # print(edge_preds[0].columns.tolist())     # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2']

        # Evaluate WIC (edge weight predictions) with Spearman correlation 
        wic_spearman = WiC(edge_preds)
        
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
Evaluates the model performance on WIC (edges evaluation with Sprearman Correlation),
WSI (Correlation Clustering evaluation with ARI and Purity) and Graded Change Detection (Spearman Correlation)
"""
def evaluate_model(dataset, paper_reproduction, clustering_method, parameter_list):

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
    edge_preds, gold_clusters, gold_lsc_df = read_data(dataset, paper_reproduction=paper_reproduction)
    # print(edge_preds[0].columns.tolist())     # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2']
    # print(gold_clusters[0].columns.tolist())      # ['identifier', 'cluster']
    # print(gold_lsc_df.columns.tolist())       # ['lemma', 'change_graded']
    

    # Standardize edge weight predictions -> mean of predicted edge weights is 0, standard deviation is 1 (only relevant for paper reproduciton)
    if paper_reproduction==True:
        edge_preds = standardize_edge_preds(edge_preds)

    
    # Parameter grid for crossvalidation (Clustering: Mapping from node identifiers to Cluster ID (dictionary))
    parameter_grid = pd.DataFrame(columns=['parameter_combination', 'word', 'ARI', 'GC', 'BC', 'clustering'])

    
    # print(edge_preds[0].columns.tolist())  # ['index', 'identifier1', 'identifier2', 'judgment', 'edge_pred', 'grouping1', 'grouping2'] for first word
    # print(gold_clusters[0].columns.tolist())         # ['identifier', 'cluster'] 
    cluster_dists = list()      # cluster probability distributions 
    clusters_metrics = defaultdict(list)    # cluster metrics (evaluation metrics (ARI, Purity)) 

    words = sorted(os.listdir(f'{dataset}/data/'))
    
    combinations = list(itertools.product(*parameter_list))     # all parameter combinations

    for comb in combinations:
        for i, word in enumerate(words):                    # iterate over words
            word_edge_preds = edge_preds[i]
            graph = generate_graph(word_edge_preds)         # generate graph with predicted edge weights for word i
            # cluster graph (cluster_labels: dataframe with columns ['identifier', 'cluster'])
            cluster_labels, classes_sets = cluster_graph(graph, clustering_method, paper_reproduction, comb)    
            
            # Evaluate clustering with ARI and Purity
            gold_clusters[i] = gold_clusters[i].sort_values('identifier')

            clusters_metrics['adjusted_rand_score'].append(adjusted_rand_score(cluster_labels.cluster.values, gold_clusters[i].cluster.values))
            clusters_metrics['purity'].append(purity_score(cluster_labels.cluster.values, gold_clusters[i].cluster.values))

            # Compute cluster distributions (cluster frequency distribution and cluster probability distribution) for one Graph
            freq_dist, prob_dist = get_cluster_distributions(classes_sets)
            cluster_dists.append(prob_dist)     # cluster distributions for LSC 

            # Add evaluation results to parameter_grid
            row_ARI = clusters_metrics['adjusted_rand_score'][-1]
            # glsc_pred = jensenshannon(prob_dist[0], prob_dist[1])
            clustering = cluster_labels.set_index('identifier')['cluster'].to_dict()    # maps node_ids to cluster ids
            new_grid_row = {'parameter_combination': comb, 'word': word, 'ARI': row_ARI, 'GC': "-", 'BC': "-", 'clustering': clustering}
            parameter_grid.loc[len(parameter_grid)] = new_grid_row
        print(comb)



    # Paper reproduction: evaluate on whole dataset (mean ARI, mean Purity, GCD with Spearman Correlation)
    if paper_reproduction:
        # take mean values across all words cluster metrics 
        # clusters_metrics: {'adjusted_rand_score': [...], 'purity': [...]}
        clusters_metrics = {m: np.mean(v) for m, v in clusters_metrics.items()}         # WSI: ARI, Purity

        gcd_spearman = GCD(cluster_dists, gold_lsc_df)                                 # GCD: Spearman correlation

        # header = 'dataset\tWSI_ARI\tWSI_Purity\tGCD_Spearman_Correlation'             # Add paper evaluation results to paper_eval_lines
        record = "\t".join([str(i) for i in [ds, clusters_metrics['adjusted_rand_score'].round(3), 
                                             clusters_metrics['purity'].round(3), gcd_spearman[0].round(3)]]) + '\n'
        paper_eval_lines.append(record)
        print(record)

        """
        # Save clusters and cluster probability distributions 
        ds = os.path.basename(dataset)
        Path(f'./prob_dists/{ds}').mkdir(exist_ok=True, parents=True)
        Path(f'./classes/{ds}').mkdir(exist_ok=True, parents=True)

        with open(f'./prob_dists/{ds}/prob_dists.dill', mode='+wb') as f:
            dill.dump(clusters_dists, f)
            
        with open(f'./classes/{ds}/classes.dill', mode='+wb') as f:
            dill.dump(clusters_labels, f)
        """

        # Save WSI and GCD evaluation results to output_file="./stats/paper_model_evaluation.tsv"
        with open(output_file, mode='w', encoding='utf-8') as f:
            f.writelines(paper_eval_lines)


    # Save Parameter grid of dataset (and clustering method)
    df_output_file=f"./parameter_grids/{ds}/{clustering_method}/parameter_grid.tsv"
    os.makedirs(os.path.dirname(df_output_file), exist_ok=True)
    parameter_grid.to_csv(df_output_file, sep='\t', index=False)


if __name__=="__main__":
    
    # Evalation with WIC;WSI;GCD 
    datasets = ["dwug_de", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       # no dwug_la 
    datasets = ["./paper_data/" + dataset for dataset in datasets]
    
    # WIC evaluation for all datasets 
    evaluate_wic(datasets)
    
    # WSI and LSC evaluation 

    output_file="./stats/paper_model_evaluation.tsv"
    # Delete content of output_file="./stats/paper_model_evaluation.tsv"
    with open(output_file, mode='w', encoding='utf-8') as f:
        pass

    # parameter=[s=20, max_attempts=2000, max_iters=50000]  # s=max_clusters
    parameter_list = [[20],[2000],[50000]]

    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=True, clustering_method="correlation_paper", parameter_list=parameter_list)