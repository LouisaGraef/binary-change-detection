

from generate_graph import *
import networkx as nx
from itertools import combinations
from correlation_clustering import cluster_correlation_search
from utils import get_clusters, transform_edge_weights
import numpy as np
from correlation_clustering import Loss 
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# https://github.com/Garrafao/correlation_clustering 
# https://github.com/Garrafao/correlation_clustering/blob/main/src/correlation.py 
# https://github.com/Garrafao/correlation_clustering/blob/main/src/test.py  


"""
Clusters one graph with Correlation Clustering (https://github.com/Garrafao/correlation_clustering).
Input: graph
return: clusters, cluster_stats 
"""
def cluster_graph_cc(graph):
    # Prepare graph for clustering
    threshold = 0.5                       # edge weights with cosine similarity are between 0 and 1 
    weight_transformation = lambda x: x-threshold
    graph = transform_edge_weights(graph, transformation = weight_transformation) # shift edge weights 

    # Cluster graph
    clusters, cluster_stats = cluster_correlation_search(graph, s = 5, max_attempts = 100, max_iters = 200) 
    
    # Display results
    node2cluster_inferred = {node:i for i, cluster in enumerate(clusters) for node in cluster}
    node2cluster_inferred = {node:node2cluster_inferred[node] for node in graph.nodes}
    print('clusters_inferred', node2cluster_inferred)
    print('clusters', node2cluster_inferred.values())
    print('loss', cluster_stats['loss'])
    
    # Clustering again and initializing with the previous solution can improve the solution in many cases (this can be done multiple times)
    clusters, cluster_stats = cluster_correlation_search(graph, s = 5, max_attempts = 100, max_iters = 200, initial = clusters)

    # Display results after second iteration
    node2cluster_inferred = {node:i for i, cluster in enumerate(clusters) for node in cluster}
    node2cluster_inferred = {node:node2cluster_inferred[node] for node in graph.nodes}
    print('clusters_inferred 2nd (dependent) iteration', node2cluster_inferred)
    print('clusters', node2cluster_inferred.values())
    print('loss', cluster_stats['loss'])

    return clusters, cluster_stats






if __name__=="__main__":
    uses = "./data/dwug_en/data/bar_nn/uses.csv"
    graph = generate_graph(uses)    
    #print(graph.nodes['fic_1964_16147.txt-1494-12'], '\n')
    clusters, cluster_stats = cluster_graph_cc(graph)
    