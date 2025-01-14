
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

import pandas as pd
import networkx as nx
from correlation_clustering import cluster_correlation_search
from utils import transform_edge_weights
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
import graph_tool
from clustering_interface_wsbm import wsbm_clustering


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
    elif clustering_method=="wsbm":
        labels, classes_sets = cluster_graph_wsbm(graph, parameters)
    
    return labels, classes_sets


"""
Cluster Graph with Correlation Clustering, return cluster_labels (dataframe with columns ['identifier', 'cluster'])
parameters[0] = edge_shift_value, parameters[1] = max_attempts, parameters[2] = max_iters
"""
def cluster_graph_cc(graph, paper_reproduction, parameters):
    
    # Cluster Graph 
    classes = []
    for init in range(1):
        if paper_reproduction:
            classes = cluster_correlation_search(graph, s=20, max_attempts=parameters[1], max_iters=parameters[2], initial=classes)
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
Cluster Graph with Weighted Stochastic Block Model (WSBM)
parameter: distribution either 'real-normal' or 'real-exponential'.
"""
def cluster_graph_wsbm(graph, parameters):
    graph = graph.copy()
    # shift edge weights +1
    for u, v, edge_data in graph.edges(data=True):
        edge_data['weight'] += 1

    classes = wsbm_clustering(graph, distribution=parameters[0], is_weighted=True, weight_attributes=['weight'], weight_data_type='float')

    # print(classes)          # Tupel: 1st element list of sets of nodes, 2nd element dict with parameters

    # Store cluster labels (cluster label for each node)
    labels = list()
    for k, cl in enumerate(classes):     # enumerate sets of nodes
        for idx in cl:                   # enumerate nodes of cluster k 
            labels.append(dict(identifier=idx.split('###')[0], cluster=k))
    labels = pd.DataFrame(labels).sort_values('identifier').reset_index(drop=True)                  # convert labels from list of dicts to dataframe

    classes_sets = classes           # list of sets of nodes that are in the same cluster


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
        kmeans = KMeans(n_clusters=k, n_init=parameters[0], max_iter=parameters[1]).fit(adj_matrix)
        labels = kmeans.labels_                                     # list of cluster ids
        sil_score = silhouette_score(adj_matrix, labels)            # Silhouette Score
        sil_scores.append(sil_score)
        # print(f"Silhouette score for {k} clusters: {sil_score}")
        if sil_score > max_sil_score:
            max_sil_score = sil_score
            best_k = k

    # Cluster Adjacency Matrix with best number of clusters 
    kmeans = KMeans(n_clusters=best_k, n_init=parameters[0], max_iter=parameters[1]).fit(adj_matrix)
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
        agglomerative = AgglomerativeClustering(n_clusters=k, linkage=parameters[0], metric=parameters[1]).fit(adj_matrix)
        labels = agglomerative.labels_                                     # list of cluster ids
        sil_score = silhouette_score(adj_matrix, labels)            # Silhouette Score
        sil_scores.append(sil_score)
        # print(f"Silhouette score for {k} clusters: {sil_score}")
        if sil_score > max_sil_score:
            max_sil_score = sil_score
            best_k = k

    # Cluster Adjacency Matrix with best number of clusters 
    agglomerative = AgglomerativeClustering(n_clusters=best_k, linkage=parameters[0], metric=parameters[1]).fit(adj_matrix)
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
        spectral = SpectralClustering(n_clusters=k, affinity=parameters[0], n_neighbors=parameters[1]).fit(adj_matrix)
        labels = spectral.labels_                                     # list of cluster ids
        sil_score = silhouette_score(adj_matrix, labels)            # Silhouette Score
        sil_scores.append(sil_score)
        # print(f"Silhouette score for {k} clusters: {sil_score}")
        if sil_score > max_sil_score:
            max_sil_score = sil_score
            best_k = k

    # Cluster Adjacency Matrix with best number of clusters 
    spectral = SpectralClustering(n_clusters=best_k, affinity=parameters[0], n_neighbors=parameters[1]).fit(adj_matrix)
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


