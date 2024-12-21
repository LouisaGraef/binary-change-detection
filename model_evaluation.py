
from pathlib import Path
import pandas as pd
import dill
import os
import csv



# https://github.com/FrancescoPeriti/CSSDetection 
# https://github.com/FrancescoPeriti/CSSDetection/blob/main/src/model_evaluation.py 
# https://github.com/Garrafao/correlation_clustering 


# scores[word].append(dict(identifier1=row1['identifier'].iloc[0], identifier2=row2['identifier'].iloc[0], judgment=1-cosine(E1[-1], E2[-1])))

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
    if 'nor_dia_change-main/subset1' in dataset:
        stats_groupings = pd.read_csv(f'{dataset}/stats/stats_groupings.tsv', sep='\t')
    else:
        stats_groupings = pd.read_csv(f'{dataset}/stats/opt/stats_groupings.csv', sep='\t')
    
    return stats_groupings


"""
Read data (edge predictions, gold edge weights, gold clusters, gold graded change) 
"""
def read_data(dataset):
    # read targets
    words = sorted(os.listdir(f'{dataset}/data/'))
    
    # read judgments by LM
    ds = os.path.basename(dataset)
    edge_preds = dill.load(open(f'./scores/{ds}/scores.dill', mode='rb'))   # dict: for every word list of dicts (identifier1, identifier2, judgment)
    words = [word for word in words if word in edge_preds]
    edge_preds = [pd.DataFrame(edge_preds[word]) for word in words]     # list of dataframes (for every word: identifier1, identifier2, judgment)
    
    
    # pre-processing
    for df in edge_preds:
        df['identifier1'] = df['identifier1'].apply(lambda x: x.replace('Name: identifier1', ''))
        df['identifier2'] = df['identifier2'].apply(lambda x: x.replace('Name: identifier2', ''))   
    
    gold_wic_dfs = list()   # list of dfs ['index', 'identifier1', 'identifier2', 'judgment', 'grouping1', 'grouping2']
    gold_clusters = list()

    for i, word in enumerate(words):

        # read gold judgments
        if 'nor_dia_change-main' in dataset:
            gold_wic_df = pd.read_csv(f'{dataset}/data_joint/data_joint.tsv', sep='\t')
            gold_wic_df = gold_wic_df[gold_wic_df['lemma']==word][['identifier1', 'identifier2', 'judgment']]
        else:
            gold_wic_df = pd.read_csv(f'{dataset}/data/{word}/judgments.csv', sep='\t')[['identifier1', 'identifier2', 'judgment']]

        gold_wic_df['identifier1'] = gold_wic_df['identifier1'].astype(str)
        gold_wic_df['identifier2'] = gold_wic_df['identifier2'].astype(str)
        # get mean annotation for every edge 
        gold_wic_df = gold_wic_df.sort_values(['identifier1', 'identifier2']).groupby(['identifier1', 'identifier2']).mean().reset_index()
        
        
        # load gold clusters dataframe 
        if 'nor_dia_change-main' in dataset:
            gold_cluster = pd.read_csv(f'{dataset}/clusters/{word}.tsv', sep='\t')
        else:
            gold_cluster = pd.read_csv(f'{dataset}/clusters/opt/{word}.csv', sep='\t')

        # ignore noisy word usages (i.e. cluster = -1)
        valid_idx = gold_cluster[gold_cluster.cluster!=-1].identifier.values  # nodes that are not in gold cluster -1 
        # filter -1 nodes out of gold clusters and -1 edges out of gold edge weights and predicted edge weights 
        gold_cluster = gold_cluster[gold_cluster.identifier.isin(valid_idx)]  
        gold_wic_df = gold_wic_df[gold_wic_df.identifier1.isin(valid_idx) & gold_wic_df.identifier2.isin(valid_idx)].sort_values(['identifier1', 'identifier2']).reset_index()
        edge_preds[i] = edge_preds[i][edge_preds[i].identifier1.isin(valid_idx) & edge_preds[i].identifier2.isin(valid_idx)].sort_values(['identifier1', 'identifier2'])
        

        # ignore usage pairs for which it wasn't possible to make a prediction
        usages = set(edge_preds[i].identifier1.tolist()+edge_preds[i].identifier2.tolist()) # all nodes that have edge predictions 
        gold_wic_df = gold_wic_df[gold_wic_df.identifier1.isin(usages) & gold_wic_df.identifier2.isin(usages)]  # only nodes that are also in edge_preds[i] (of word i)
        gold_cluster = gold_cluster[gold_cluster.identifier.isin(usages)]   # only nodes that are also in edge_preds[i] (of word i)


        # add grouping info
        uses = load_word_usages(dataset, word)[['identifier', 'grouping']]
        uses = {row['identifier']: row['grouping'] for _, row in uses.iterrows()}   # mapping from identifiers to grouping 
        gold_wic_df['grouping1'] = [uses[row['identifier1']] for _, row in gold_wic_df.iterrows()]  # grouping of identifier 1 
        gold_wic_df['grouping2'] = [uses[row['identifier2']] for _, row in gold_wic_df.iterrows()]  # grouping of identifier 2 
        edge_preds[i]['grouping1'] = [uses[row['identifier1']] for _, row in edge_preds[i].iterrows()]  # grouping of identifier 1 
        edge_preds[i]['grouping2'] = [uses[row['identifier2']] for _, row in edge_preds[i].iterrows()]  # grouping of identifier 2 

        # print(gold_wic_df.columns.tolist())    # ['index', 'identifier1', 'identifier2', 'judgment', 'grouping1', 'grouping2']
        # print(edge_preds[i].columns.tolist())  # ['identifier1', 'identifier2', 'judgment', 'grouping1', 'grouping2']
        # print(gold_cluster.columns.tolist())   # ['identifier', 'cluster']

        gold_wic_dfs.append(gold_wic_df)
        gold_clusters.append(gold_cluster)

    
    gold_lsc_df = load_gold_lsc(dataset)[['lemma', 'change_graded']]    # read graded change gold scores 
    if 'nor_dia_change-main' not in dataset:
        gold_lsc_df['lemma'] = gold_lsc_df['lemma'].str.normalize('NFKD') 
    gold_lsc_df = gold_lsc_df[gold_lsc_df['lemma'].isin(words)].sort_values('lemma')        # sort 
    print(gold_lsc_df.shape, len(words))
    
    return edge_preds, gold_wic_dfs, gold_clusters, gold_lsc_df



"""
Evaluate WIC (edge weight predictions) with Spearman correlation
"""
def WiC(dfs, gold_wic_dfs):
    pass


"""
Cluster Graph with Correlation Clustering 
"""
def WSI(dfs, gold_wsi_dfs):
    pass




"""
Evaluates the model performance on WIC (edges evaluation with Sprearman Correlation),
WSI (Correlation Clustering evaluation with ARI and Purity) and Graded Change Detection (Spearman Correlation)
"""
def evaluate_model(dataset, output_file="./stats/model_evaluation.tsv"):
    # file header
    header = 'dataset\ttask\tscore\trows'

    # file output
    if not Path(output_file).is_file():
        lines = [header+ '\n']
    else:
        lines = open(output_file, mode='r', encoding='utf-8').readlines()
    pass

    # Read data 
    dfs, gold_wic_dfs, gold_wsi_dfs, gold_lsc_df = read_data(dataset)
    print("ok")
    quit()

    # Evaluate WIC (edge weight predictions) with Spearman correlation 
    wic_spearman = WiC(dfs, gold_wic_dfs)

    # Add WIC Spearman Score to record 
    record = "\t".join([str(i) for i in [dataset, 'wic', wic_spearman[0].round(3), pd.concat(dfs).shape[0]]]) + '\n'
    lines.append(record)
    print(record)

    # Get Clusters (Correlation Clustering) for all dataset words
    clusters_metrics, clusters_labels, clusters_dists = WSI(dfs, gold_wsi_dfs)

    # Save clusters and cluster probability distributions 
    Path(f'probs/{dataset}').mkdir(exist_ok=True, parents=True)
    Path(f'classes/{dataset}').mkdir(exist_ok=True, parents=True)

    with open(f'probs/{dataset}.dill', mode='+wb') as f:
        dill.dump(clusters_dists, f)
        
    with open(f'classes/{dataset}.dill', mode='+wb') as f:
        dill.dump(clusters_labels, f)

    # Add WSI evaluation metrics to record 
    for metric in clusters_metrics:
        record = "\t".join([str(i) for i in [dataset, f'wsi-{metric}', clusters_metrics[metric].round(3), pd.concat(dfs).shape[0]]]) + '\n'
        lines.append(record)
        print(record)




if __name__=="__main__":
    evaluate_model("./data/dwug_de")