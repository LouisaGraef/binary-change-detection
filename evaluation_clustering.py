import sys
import csv
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from itertools import combinations, chain
from collections import defaultdict, Counter
import seaborn as sb
from pandas import DataFrame
import unicodedata
import pandas as pd
from sklearn.metrics import cohen_kappa_score, hamming_loss, accuracy_score, f1_score
from sklearn.metrics.cluster import adjusted_rand_score, rand_score
from scipy.stats import spearmanr
import requests


"""
Paper: https://openreview.net/pdf?id=BlbrJvKv6L 
Paper Code: https://github.com/Garrafao/wug_cluster_clean/blob/main/analyze_semeval_de1.ipynb 
            https://github.com/Garrafao/wug_cluster_clean/blob/main/Nikolay_analyze_semeval_de1.ipynb 
"""



def evaluate_clustering(dataset, clustering_method="k-means"):

    
    print(f'\nDataset: {dataset}  \nClustering Method: {clustering_method}')

    if "paper_data" in dataset:
        ds = dataset.replace("./paper_data/", "")
    else:
        ds = dataset.replace("./data/", "")



    # load Gold Clustering 
    df_dwug_de = pd.DataFrame()                                                         # gold clustering 
    for p in Path(f'{dataset}/').glob('clusters/opt/*.csv'):    
        lemma = str(p).replace('\\', '/').split('/')[-1].replace('.csv','')
        lemma = unicodedata.normalize('NFC', lemma)
        df = pd.read_csv(p, delimiter='\t', quoting=3, na_filter=False)
        df['lemma'] = lemma
        df_dwug_de = pd.concat([df_dwug_de, df])    

    # Extract grouping (time) information
    df_dwug_de_uses = pd.DataFrame()
    for p in Path(f'{dataset}/data').glob('*/uses.csv'):
        df_dwug_de_uses = pd.concat([df_dwug_de_uses, pd.read_csv(p, delimiter='\t', quoting=3, na_filter=False)])
    #display(df_dwug_de_uses)


    df_dwug_de = df_dwug_de.merge(df_dwug_de_uses[['identifier', 'lemma', 'grouping']], how='left',      # columns: ['identifier', 'cluster', 'lemma', 'grouping']
                                                on = ['identifier', 'lemma'])
    
    print(df_dwug_de)




    # load Clustering Parameter Grid 
    clustering_parameter_grid = pd.read_csv(f"./parameter_grids/{ds}/{clustering_method}/parameter_grid.tsv", sep='\t')
    print(clustering_parameter_grid)
    print(clustering_parameter_grid.columns.tolist())



    # Add columns to clustering_parameter_grid: model, method, ARI
    model = f'{clustering_method}_{ds}_'
    if "paper_data" in dataset:
        ds = "paper_data/" + ds
    clustering_parameter_grid['model'] = (f'{clustering_method}_{ds}_' + clustering_parameter_grid['parameter_combination'])
    clustering_parameter_grid['method'] = (f'{clustering_method}')
    #clustering_parameter_grid['ARI'] = adjusted_rand_score()

    ari_values = []
    spearmanr_values = []
    for index, row in clustering_parameter_grid.iterrows():
        # evaluate clustering
        pred_clusters = eval(row['clustering_pred'])
        gold_clusters = eval(row['clustering_gold'])
        pred_labels = [pred_clusters[node] for node in sorted(pred_clusters)]
        gold_labels = [gold_clusters[node] for node in sorted(gold_clusters)]
        ari = adjusted_rand_score(gold_labels, pred_labels)
        ari_values.append(ari)
        
        # evaluate Graded Change
        spearman, p_value = spearmanr(row['GC_pred'], row['GC_gold'])
        spearmanr_values.append(spearman)

    # evaluate Binary Change
    f1 = f1_score(clustering_parameter_grid['BC_gold'], clustering_parameter_grid['BC_pred'])


    # Add columns and reorder 
    clustering_parameter_grid['ARI'] = ari_values
    clustering_parameter_grid['GC_Spearmanr'] = spearmanr_values
    clustering_parameter_grid['BC_F1'] = f1
    clustering_parameter_grid = clustering_parameter_grid[['model', 'word', 'ARI', 'GC_Spearmanr', 'BC_F1', 'BC_pred', 'BC_gold', 'GC_pred', 
                                                           'GC_gold', 'method','parameter_combination', 'clustering_pred', 'clustering_gold']]

    print(clustering_parameter_grid)
    print(clustering_parameter_grid.columns.tolist())













if __name__=="__main__":
    dataset = "./paper_data/dwug_de"
    evaluate_clustering(dataset, clustering_method="k-means")