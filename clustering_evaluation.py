
from cluster_graph import cluster_graph_cc
from generate_graph import *
from sklearn.metrics import rand_score, adjusted_rand_score 
from collections import Counter
import os
import glob
from pathlib import Path
from statistics import mean
from sklearn.model_selection import train_test_split
import unicodedata


"""
Relabels graph nodes from 'identifer_system' to 'identifier', so that they 
can be used as input for nor_dia_change clustering evaluation
"""
def map_identifier_system_to_identifier(graph, uses):
    with open(uses, encoding='utf-8') as csvfile:                                                   # read in uses
        reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
        uses = [row for row in reader]                                                              # uses as list of dictionaries

        # Get mapping for relabeling
        system2identifier = {}                    # maps identifier_system to identifier for every node identifier 
        for (k, row) in enumerate(uses):
            row = row.copy()
            system2identifier[str(row['identifier_system'])] = row['identifier']

        # Relabel nodes 
        graph = nx.relabel_nodes(graph, mapping=system2identifier)      

    return graph 


"""
Evaluate a clustering of one dataset with Adjusted Rand Index
"""
def evaluate_clustering(dataset, words):
    ari_stats = {}                      # data to be saved 
    ari_stats['dataset'] = dataset
    dataset = "./data/" + dataset       # full path to dataset

    ari_values = []       # list of ari values of all words in the dataset 

    for word in words:
        # Get Gold labels 
        if dataset == "./data/nor_dia_change-main/subset1" or dataset == "./data/nor_dia_change-main/subset2":
            gold_clusters = dataset + "/clusters/" + word + ".tsv"      # read in gold clusters 
        else:
            gold_clusters = dataset + "/clusters/opt/" + word + ".csv"      # read in gold clusters 
        with open(gold_clusters, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
            node2cluster_gold = {}      # gold dictionary to map node ids to cluster ids 
            for row in reader:
                node_id = row['identifier']
                cluster = row['cluster']
                node2cluster_gold[node_id] = cluster 
        
        labels_true = [node2cluster_gold[node] for node in sorted(node2cluster_gold.keys())]        # gold labels (clusters) (sorted by node ids)
        

        # Get predicted labels 
        uses = dataset + "/data/" + word + "/uses.csv" 
        if dataset == "./data/discowug":
            uses = unicodedata.normalize('NFC', uses)
        graph = generate_graph(uses)                                    # generate graph 
        if dataset == "./data/nor_dia_change-main/subset1" or dataset == "./data/nor_dia_change-main/subset2":
            graph = map_identifier_system_to_identifier(graph, uses)      # relabel graph nodes from 'identifer_system' to 'identifier'

        clusters, cluster_stats = cluster_graph_cc(graph)               # cluster graph 

        node2cluster_inferred = {node:i for i, cluster in enumerate(clusters) for node in cluster}
        node2cluster_inferred = {node:node2cluster_inferred[node] for node in node2cluster_gold.keys()}   # dictionary, maps node ids to cluster ids 
        labels_pred = [node2cluster_inferred[node] for node in sorted(node2cluster_inferred.keys())]      # predicted labels (clusters) (sorted by node ids)


        # Get Adjusted Rand Index 
        ri = rand_score(labels_true, labels_pred)
        ari = adjusted_rand_score(labels_true, labels_pred)
        ari_values.append(ari)
        #print(f'Rand Index: {ri}')
        #print(f'Adjusted Rand Index: {ari}')

    mean_ari = mean(ari_values)     # mean Adjusted Rand Index of dataset

    ari_stats['mean_adjusted_rand_index'] = mean_ari

    return ari_stats, mean_ari 





if __name__=="__main__":
    #uses = "./data/dwug_en/data/face_nn/uses.csv"
    #graph = generate_graph(uses)    
    #clusters, cluster_stats = cluster_graph_cc(graph)
    #print('Cluster stats: \n', cluster_stats)
    #gold_clusters = "./data/dwug_en/clusters/opt/face_nn.csv"
    #evaluate_clustering(clusters, gold_clusters)

    print("---")
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]           
    datasets_paper_versions = ["dwug_de", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]
    
    is_header = True        # create header first when exporting stats
    os.makedirs('./stats', exist_ok=True)     # create directory for stats (no exception is raised if directory aready exists)
    with open('./stats/clustering_evaluation_stats.csv', 'w', encoding='utf-8') as f_out:     # 'w' mode deletes contents of file 
        pass
    
    for dataset in datasets:
        if dataset == "nor_dia_change-main/subset1" or dataset == "nor_dia_change-main/subset2":
            words = sorted([f.stem for f in Path("./data/" + dataset + "/clusters").iterdir() if f.is_file], key=str.lower)
        else:
            words = sorted([f.stem for f in Path("./data/" + dataset + "/clusters/opt").iterdir() if f.is_file], key=str.lower) # list of all words in the clusters directory 
        
        ari_stats, mean_ari  = evaluate_clustering(dataset, words)            # get clustering evaluation stats of one dataset 
        print("\nDataset: ", dataset)
        print("Adjusted Rand Index: ", mean_ari)

        # export stats 
        with open('./stats/clustering_evaluation_stats.csv', 'a', encoding='utf-8') as f_out:
            if is_header:
                f_out.write('\t'.join([key for key in ari_stats]) + '\n')
                is_header = False 
            f_out.write('\t'.join([str(ari_stats[key]) for key in ari_stats]) + '\n')