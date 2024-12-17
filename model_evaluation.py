
from pathlib import Path
import pandas as pd
import dill



# https://github.com/FrancescoPeriti/CSSDetection 
# https://github.com/FrancescoPeriti/CSSDetection/blob/main/src/model_evaluation.py 
# https://github.com/Garrafao/correlation_clustering 



"""
Read data
"""
def read_data(dataset):
    pass


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