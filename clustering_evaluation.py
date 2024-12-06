
from cluster_graph import cluster_graph_cc
from generate_graph import *
from sklearn.metrics import rand_score, adjusted_rand_score 
from collections import Counter
import os
import glob
from pathlib import Path
from statistics import mean

"""
Evaluate a clustering of one dataset with Adjusted Rand Index
"""
def evaluate_clustering(dataset):
    ari_stats = {}                      # data to be saved 
    ari_stats['dataset'] = dataset
    dataset = "./data/" + dataset       # full path to dataset

    words = sorted([d.name for d in Path(dataset + "/data").iterdir() if d.is_dir()])    # list of all words in the data directory 
    ri_values = []        # list of ri values of all words in the dataset 
    ari_values = []       # list of ari values of all words in the dataset 

    for word in words:
        uses = dataset + "/data/" + word + "/uses.csv" 
        graph = generate_graph(uses)                                    # generate graph 
        clusters, cluster_stats = cluster_graph_cc(graph)               # cluster graph 
        #print('Cluster stats: \n', cluster_stats)

        # Get predicted labels 
        node2cluster_inferred = {node:i for i, cluster in enumerate(clusters) for node in cluster}
        node2cluster_inferred = {node:node2cluster_inferred[node] for node in graph.nodes}      # dictionary, maps node ids to cluster ids 
        labels_pred = [node2cluster_inferred[node] for node in sorted(node2cluster_inferred.keys())]      # predicted labels (clusters) (sorted by node ids)
        
        #print('Predicted labels: \n', labels_pred)

        # Get Gold labels 
        gold_clusters = dataset + "/clusters/opt/" + word + ".csv"      # read in gold clusters 
        with open(gold_clusters, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
            node2cluster_gold = {}      # gold dictionary to map node ids to cluster ids 
            for row in reader:
                node_id = row['identifier']
                cluster = row['cluster']
                node2cluster_gold[node_id] = cluster 
        
        labels_true = [node2cluster_gold[node] for node in sorted(node2cluster_gold.keys())]        # gold labels (clusters) (sorted by node ids)
        
        #print('Gold labels: \n', labels_true)

        # Get Adjusted Rand Index 
        ri = rand_score(labels_true, labels_pred)
        ari = adjusted_rand_score(labels_true, labels_pred)
        ri_values.append(ri)
        ari_values.append(ari)
        #print(f'Rand Index: {ri}')
        #print(f'Adjusted Rand Index: {ari}')

    mean_ri = mean(ri_values)       # mean Rand Index of dataset
    mean_ari = mean(ari_values)     # mean Adjusted Rand Index of dataset

    ari_stats['mean_rand_index'] = mean_ri
    ari_stats['mean_adjusted_rand_index'] = mean_ari

    return ari_stats, mean_ri, mean_ari 





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
        ari_stats, mean_ri, mean_ari  = evaluate_clustering(dataset)            # get clustering evaluation stats of one dataset 
        print("\nDataset: ", dataset)
        print("Rand Index: ", mean_ri)
        print("Adjusted Rand Index: ", mean_ari)

        # export stats 
        with open('./stats/clustering_evaluation_stats.csv', 'a', encoding='utf-8') as f_out:
            if is_header:
                f_out.write('\t'.join([key for key in ari_stats]) + '\n')
                is_header = False 
            f_out.write('\t'.join([str(ari_stats[key]) for key in ari_stats]) + '\n')