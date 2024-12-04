
from cluster_graph import cluster_graph_cc
from generate_graph import *
from sklearn.metrics import rand_score, adjusted_rand_score 
from collections import Counter

"""
Evaluate a clustering with Adjusted Rand Index
"""
def evaluate_clustering(clusters, gold_clusters):

    # Get predicted labels 
    node2cluster_inferred = {node:i for i, cluster in enumerate(clusters) for node in cluster}
    node2cluster_inferred = {node:node2cluster_inferred[node] for node in graph.nodes}      # dictionary, maps node ids to cluster ids 
    labels_pred = [node2cluster_inferred[node] for node in sorted(node2cluster_inferred.keys())]        # predicted labels (clusters) for all node ids (sorted by node ids)
    
    print('Predicted labels: \n', labels_pred)

    # Get Gold labels 
    with open(gold_clusters, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
        node2cluster_gold = {}      # gold dictionary to map node ids to cluster ids 
        for row in reader:
            node_id = row['identifier']
            cluster = row['cluster']
            node2cluster_gold[node_id] = cluster 
    
    labels_true = [node2cluster_gold[node] for node in sorted(node2cluster_gold.keys())]        # gold labels (clusters) for all node ids (sorted by node ids)
    
    print('Gold labels: \n', labels_true)

    # Get Adjusted Rand Index 
    ri = rand_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    print(f'Rand Index: {ri}')
    print(f'Adjusted Rand Index: {ari}')





if __name__=="__main__":
    uses = "./data/dwug_en/data/face_nn/uses.csv"
    graph = generate_graph(uses)    
    clusters, cluster_stats = cluster_graph_cc(graph)
    print('Cluster stats: \n', cluster_stats)
    gold_clusters = "./data/dwug_en/clusters/opt/face_nn.csv"
    evaluate_clustering(clusters, gold_clusters)