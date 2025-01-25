

import numpy as np
import pickle
from itertools import combinations, product
from modules import get_node_std, get_clusters, get_low_prob_clusters
import csv
import os



"""
Paper: https://openreview.net/pdf?id=BlbrJvKv6L 
Paper Code: https://github.com/Garrafao/wug_cluster_clean/blob/main/analyze_semeval_de1.ipynb 
WUG graph2clean2.py: https://github.com/Garrafao/WUGs/blob/main/scripts/graph2clean2.py 
"""



"""
Clean a graph with specified method. 
"""
def clean_graph(g, method, annotators):
    if method=="stdnode":
        g = clean_stdnode(g, annotators, std_nodes=0.07)
    elif method=="clustersize":
        g = clean_clustersize(g, cluster_size_min=3)
    elif method=="dgrnode":
        g = clean_dgrnode(g, degree_remove=5)
    elif method=="cntcluster":
        g = clean_cntcluster(g, cluster_connect_min=0.95)
    
    return g



"""
remove nodes with average standard deviation on its edges above the threshold.
"""
def clean_stdnode(g, annotators, std_nodes):
    # remove nodes with high standard deviation
    std_nodes=float(std_nodes)    
    node2stds = get_node_std(g, annotators, normalization=lambda x: ((x-1)/3.0))
    nodes_high_stds = [n for n in g.nodes() if np.nanmean(node2stds[n]) > std_nodes]
    print('Removing {0} nodes with standard deviation above {1}.'.format(len(nodes_high_stds),std_nodes))
    g.remove_nodes_from(nodes_high_stds)    
    g.graph['cleaning_stats'] = g.graph['cleaning_stats'] | {'std_nodes':std_nodes}

    return g


"""
remove nodes in low-frequency clusters
"""
def clean_clustersize(g, cluster_size_min):
    # remove nodes in low-frequency clusters
    cluster_size_min=int(cluster_size_min)    
    clusters, _, _ = get_clusters(g)
    cluster_ids_remove = get_low_prob_clusters(clusters, threshold=cluster_size_min)
    nodes_remove = [node for cluster_id in cluster_ids_remove for node in clusters[cluster_id]]
    g.remove_nodes_from(nodes_remove) 
    print('Removing {0} nodes from clusters with size less than {1}.'.format(len(nodes_remove),cluster_size_min))
    g.graph['cleaning_stats'] = g.graph['cleaning_stats'] | {'cluster_size_remove':cluster_size_min}

    return g


"""
remove nodes with degree (number of edges) below the threshold.
"""
def clean_dgrnode(g, degree_remove):
    # remove nodes with low degree
    degree_remove=int(degree_remove)    
    nodes_degrees = [node for (node, d) in g.degree() if d<degree_remove]
    g.remove_nodes_from(nodes_degrees) 
    print('Removing {0} nodes with degree less than {1}.'.format(len(nodes_degrees),degree_remove))
    g.graph['cleaning_stats'] = g.graph['cleaning_stats'] | {'degree_remove':degree_remove}

    return g


"""
remove poorly connected clusters. We calculate the percentage of annotated edges for each cluster pair 
and then average these percentages per cluster. We then remove clusters with an average connectedness below the threshold.
"""
def clean_cntcluster(g, cluster_connect_min):
    # remove nodes in poorly connected clusters
    cluster_connect_min=float(cluster_connect_min)    
    if cluster_connect_min>0.0:
        clusters, _, _ = get_clusters(g, is_include_noise = False, is_include_main = True)
        combo2edges = {}
        #combo2sizes = {}
        for (c1,c2) in combinations(range(len(clusters)), 2):
            cluster1, cluster2 = clusters[c1], clusters[c2]
            combo2edges[(c1,c2)] = []
            size = 0
            for (i,j) in product(cluster1,cluster2):
                size += 1
                if j in g.neighbors(i):
                    combo2edges[(c1,c2)].append((i,j))
            #combo2sizes[(c1,c2)] = size
        cluster2connectedness_values = {}    
        for c in range(len(clusters)):
            connectedness_values = []
            for (c1,c2), edges in combo2edges.items():
                if c1==c or c2==c:
                    value = 1 if len(edges)>0 else 0
                    connectedness_values.append(value)
            cluster2connectedness_values[c] = connectedness_values
        print(cluster2connectedness_values)
        assert len(cluster2connectedness_values.keys()) == len(clusters)
        clusters_remove = [c for c, values in cluster2connectedness_values.items() if np.mean(values)<cluster_connect_min]
        nodes_remove = [node for c in clusters_remove for node in clusters[c]]
        g.remove_nodes_from(nodes_remove) 
    g.graph['cleaning_stats'] = g.graph['cleaning_stats'] | {'cluster_size_remove':cluster_connect_min}
    print('Removing {0} nodes from clusters with average connectedness less than {1}.'.format(len(nodes_remove),cluster_connect_min))

    return g



"""
def plot_stats(avg_df):
    pass

def evaluate_clean_graph(dataset):
    methods = ["dgrnode", "random"]
    graph = None
    words = []
    dfs = []
    for method in methods:
        for word in words:
            for n_nodes in range(len(graph.nodes)):
                graph_cleaned = clean_graph(graph)
                # evaluate cleaning (n_nodes, n_clusters, ARI)
                cleaning_df = None

    avg_df = np.avg(dfs)
    plot_stats(avg_df)


def get_conflicts(dataset):
    methods = ["dgrnode", "random"]
    graph = None
    words = []
    dfs = []
    for method in methods:
        for word in words:
            for n_nodes in range(len(graph.nodes)):
                graph_cleaned = clean_graph(graph)
                # evaluate cleaning (n_nodes, n_clusters, n_conflicts) by counting conflicts
                cleaning_df = None

    avg_df = np.avg(dfs)
    plot_stats(avg_df)
"""


def clean_graphs(dataset):
    
    
    words = sorted(os.listdir(f'{dataset}/data/'))

    for word in words:

        # load graph
        with open(f'{dataset}/graphs/opt/{word}', 'rb') as f:
            graph = pickle.load(f)
        with open(f"{dataset}/data/{word}/judgments.csv", encoding='utf-8') as csvfile: 
            reader = csv.DictReader(csvfile, delimiter='\t',quoting=csv.QUOTE_NONE,strict=True)
            annotators = [row['annotator'] for row in reader]

        # clean graph 
        methods = ["stdnode", "dgrnode", "clustersize", "cntcluster"]
        for method in methods:
            g = graph.copy()
            print('Input graph: ', g)
            g.graph['cleaning_stats'] = {}
            g = clean_graph(g, method, annotators)
            print('Cleaned graph: ', g)





if __name__=="__main__":
    clean_graphs(dataset="./paper_data/dwug_de")

    

