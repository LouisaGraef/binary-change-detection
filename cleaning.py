

import numpy as np
import pickle
from itertools import combinations, product
from modules import get_node_std, get_clusters, get_low_prob_clusters, get_nan_edges, get_edge_std
import csv
import os
import pandas as pd
from pathlib import Path
import dill
from download_data import download_paper_datasets, download_new_datasets
import unicodedata
import networkx as nx



"""
Paper: https://openreview.net/pdf?id=BlbrJvKv6L 
Paper Code: https://github.com/Garrafao/wug_cluster_clean/blob/main/analyze_semeval_de1.ipynb 
WUG graph2clean2.py: https://github.com/Garrafao/WUGs/blob/main/scripts/graph2clean2.py 
"""


"""
Get a list of parameters for every cleaning method.
Returns a dictionary with cleaning parameters for every cleaning method ("stdnode", "dgrnode", "clustersize", "cntcluster")
"""
def get_parameters(dataset):
    parameters = {}
    words = sorted(os.listdir(f'{dataset}/data/'))

    
    dataset_node_avg_stds = []
    dataset_nodes_degrees = []
    dataset_clustersizes = []
    dataset_cntcluster_values = []
    word_counter = 0

    with open(f'{dataset}/annotators.csv', encoding='utf-8') as csvfile: 
        reader = csv.DictReader(csvfile, delimiter='\t',quoting=csv.QUOTE_NONE,strict=True)
        annotators = [row['annotator'] for row in reader]

    for word in words:      # words for which a graph exists

        # load graph
        with open(f'{dataset}/graphs/opt/{word}', 'rb') as f:
            g = pickle.load(f)

        # remove nan edges and noise
        nan_edges = get_nan_edges(g)    
        g.remove_edges_from(nan_edges)
        noise, _, _ = get_clusters(g, is_include_noise = True, is_include_main = False)
        nodes_noise = [node for cluster in noise for node in cluster]
        g.remove_nodes_from(nodes_noise) # Remove noise nodes before ambiguity measures  
        

        # stdnode (average standard deviation on the node's edges)
        node2stds = get_node_std(g, annotators, non_value=0.0, normalization=lambda x: ((x-1)/3.0))
        # list of average standard deviation of each node 
        #node_avg_stds = [np.nanmean(node2stds[n]) if n in node2stds and len(node2stds[n]) > 0 and not all(np.isnan(node2stds[n])) else 0 for n in g.nodes()]       
        nodes_mean_stds = [np.nanmean(node2stds[n]) for n in g.nodes()]
        dataset_node_avg_stds= dataset_node_avg_stds + nodes_mean_stds

        # dgrnode (degree of nodes (number of edges))
        nodes_degrees = [d for (node, d) in g.degree()]
        #nodes_degrees.append(0)
        #nodes_degrees = [g.degree(node) for node in g.nodes()]
        dataset_nodes_degrees = dataset_nodes_degrees + nodes_degrees

        # clustersize (size of clusters)
        clusters, _, _ = get_clusters(g)
        clustersizes = [len(cluster) for cluster in clusters]
        dataset_clustersizes = dataset_clustersizes + clustersizes

        # cntcluster ()
        clusters, _, _ = get_clusters(g, is_include_noise = False, is_include_main = True)
        combo2edges = {}
        for (c1,c2) in combinations(range(len(clusters)), 2):
            cluster1, cluster2 = clusters[c1], clusters[c2]
            combo2edges[(c1,c2)] = []
            size = 0
            for (i,j) in product(cluster1,cluster2):
                size += 1
                if j in g.neighbors(i):
                    combo2edges[(c1,c2)].append((i,j))
        cluster2connectedness_values = {}    
        for c in range(len(clusters)):
            connectedness_values = []
            for (c1,c2), edges in combo2edges.items():
                if c1==c or c2==c:
                    value = 1 if len(edges)>0 else 0
                    connectedness_values.append(value)
            cluster2connectedness_values[c] = connectedness_values
        assert len(cluster2connectedness_values.keys()) == len(clusters)
        cluster_connectedness_values = [np.mean(values) for c, values in cluster2connectedness_values.items()]
        dataset_cntcluster_values = dataset_cntcluster_values + cluster_connectedness_values
        #print(dataset_cntcluster_values)
        
        word_counter += 1
        print(word_counter)


    methods = "stdnode", "dgrnode", "clustersize", "cntcluster"
    percentiles = [2*i for i in range(51)]

    # sort, get percentiles, remove duplikate percentiles
    for k, method in zip([dataset_node_avg_stds, dataset_nodes_degrees, dataset_clustersizes, dataset_cntcluster_values], methods):
        #k.sort()
        #print(type(k), len(k), k[:5])
        #k = [x for x in k if not np.isnan(x)]
        #percentiles = np.percentile(k, np.linspace(0, 100, 51))    # get 50 percentiles
        variable_percentiles = np.nanpercentile(k, percentiles)
        variable_percentiles = np.unique(variable_percentiles)
        parameters[method] = variable_percentiles
    
    print(parameters)


    # save cleaning parameters
    if dataset.startswith("./paper_data/"):
        ds = dataset.replace("./paper_data/", "")
        os.makedirs(f"./cleaning_parameters/paper/{ds}", exist_ok=True)
        with open(f"./cleaning_parameters/paper/{ds}/cleaning_parameters.pkl", "wb") as file:
            dill.dump(parameters, file)
    else:
        ds = dataset.replace("./data/", "")
        os.makedirs(f"./cleaning_parameters/{ds}", exist_ok=True)
        with open(f"./cleaning_parameters/{ds}/cleaning_parameters.pkl", "wb") as file:
            dill.dump(parameters, file)


    



"""
Clean a graph with specified method. 
"""
def clean_graph(g, method, annotators, parameter):
    
    #print(g)

    if method=="stdnode":
        g = clean_stdnode(g, annotators, std_nodes=parameter)
    elif method=="clustersize":
        g = clean_clustersize(g, cluster_size_min=parameter)
    elif method=="dgrnode":
        g = clean_dgrnode(g, degree_remove=parameter)
    elif method=="cntcluster":
        g = clean_cntcluster(g, cluster_connect_min=parameter)
    
    return g



"""
remove nodes with average standard deviation on its edges above the threshold.
"""
def clean_stdnode(g, annotators, std_nodes):
    # remove nodes with high standard deviation
    std_nodes=float(std_nodes)    
    node2stds = get_node_std(g, annotators, normalization=lambda x: ((x-1)/3.0))
    nodes_high_stds = [n for n in g.nodes() if np.nanmean(node2stds[n]) > std_nodes]
    #print('Removing {0} nodes with standard deviation above {1}.'.format(len(nodes_high_stds),std_nodes))
    g.remove_nodes_from(nodes_high_stds)    
    g.graph['cleaning_stats'] = g.graph['cleaning_stats'] | {'std_nodes':std_nodes}
    
    isolates = list(nx.isolates(g))     
    g.remove_nodes_from(isolates)

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
    #print('Removing {0} nodes from clusters with size less than {1}.'.format(len(nodes_remove),cluster_size_min))
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
    #print('Removing {0} nodes with degree less than {1}.'.format(len(nodes_degrees),degree_remove))
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
        #print(cluster2connectedness_values)
        assert len(cluster2connectedness_values.keys()) == len(clusters)
        clusters_remove = [c for c, values in cluster2connectedness_values.items() if np.mean(values)<cluster_connect_min]
        nodes_remove = [node for c in clusters_remove for node in clusters[c]]
        g.remove_nodes_from(nodes_remove) 
    g.graph['cleaning_stats'] = g.graph['cleaning_stats'] | {'cluster_size_remove':cluster_connect_min}
    #print('Removing {0} nodes from clusters with average connectedness less than {1}.'.format(len(nodes_remove),cluster_connect_min))

    return g





"""
Clean graphs of one dataset with different methods ("stdnode", "dgrnode", "clustersize", "cntcluster")
"""
def clean_graphs(dataset):
    
    df_cleaning = pd.DataFrame(columns=['identifier', 'cluster', 'model', 'strategy', 'threshold', 'lemma'])
    words = sorted(os.listdir(f'{dataset}/data/'))
    counter = 0

    if dataset.startswith("./paper_data/"):
        ds = dataset.replace("./paper_data/", "")
        with open(f"./cleaning_parameters/paper/{ds}/cleaning_parameters.pkl", "rb") as file:
            parameters = dill.load(file)
    else:
        ds = dataset.replace("./data/", "")
        with open(f"./cleaning_parameters/{ds}/cleaning_parameters.pkl", "rb") as file:
            parameters = dill.load(file)


    for word in words:
        word = unicodedata.normalize('NFC', word)
        clusters = pd.read_csv(f'{dataset}/clusters/opt/{word}.csv', sep="\t")

        # load graph
        with open(f'{dataset}/graphs/opt/{word}', 'rb') as f:
            graph = pickle.load(f)
        with open(f'{dataset}/annotators.csv', encoding='utf-8') as csvfile: 
            reader = csv.DictReader(csvfile, delimiter='\t',quoting=csv.QUOTE_NONE,strict=True)
            annotators = [row['annotator'] for row in reader]
        
        # remove nan edges and noise
        nan_edges = get_nan_edges(graph)    
        graph.remove_edges_from(nan_edges)
        noise, _, _ = get_clusters(graph, is_include_noise = True, is_include_main = False)
        nodes_noise = [node for cluster in noise for node in cluster]
        graph.remove_nodes_from(nodes_noise) # Remove noise nodes 
        
        
        # clean graph 
        #methods = ["stdnode", "dgrnode", "clustersize", "cntcluster"]
        methods = ["stdnode", "dgrnode", "clustersize", "cntcluster"]
        for method in methods:
            for parameter in parameters[method]:
                model = str(method) + "_" + str(parameter)
                g = graph.copy()
                #print('Input graph: ', g)
                g.graph['cleaning_stats'] = {}

                g = clean_graph(g, method, annotators, parameter)
                
                #print('Cleaned graph: ', g)

                for node in g.nodes:
                    cluster = clusters.loc[clusters['identifier'] == node, 'cluster'].values[0]     # cluster of node 
                    # insert new row to df_cleaning 
                    new_row = pd.DataFrame({'identifier': [node], 'cluster': [cluster], 'model': [model], 'strategy': [method], 
                                            'threshold': [parameter], 'lemma': [word]})
                    df_cleaning = pd.concat([df_cleaning, new_row], ignore_index=True)
                    #df_cleaning.loc[len(df_cleaning)] = [node, cluster, model, method, parameter, word]
                #print(df_cleaning)
                #print(word, method, parameter)
                #quit()
            print(".")
        counter +=1
        print(counter)


    # save cleaning parameter grid
    if dataset.startswith("./paper_data/"):
        ds = dataset.replace("./paper_data/", "")
        os.makedirs(f"./cleaning_parameter_grids/paper/{ds}", exist_ok=True)
        with open(f"./cleaning_parameter_grids/paper/{ds}/cleaning_parameter_grid.pkl", "wb") as file:
            dill.dump(df_cleaning, file)
    else:
        ds = dataset.replace("./data/", "")
        os.makedirs(f"./cleaning_parameter_grids/{ds}", exist_ok=True)
        with open(f"./cleaning_parameter_grids/{ds}/cleaning_parameter_grid.pkl", "wb") as file:
            dill.dump(df_cleaning, file)

    return df_cleaning





if __name__=="__main__":
    #download_paper_datasets()
    #get_parameters(dataset="./paper_data/dwug_de")
    #clean_graphs(dataset="./paper_data/dwug_de")
    download_new_datasets()
    get_parameters(dataset="./data/dwug_de")
    clean_graphs(dataset="./data/dwug_de")

    

