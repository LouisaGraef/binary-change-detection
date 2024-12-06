

from generate_graph import *
from scipy.stats import spearmanr
import glob 
import pandas as pd
from statistics import mean
import os



"""
Evaluate the predicted annotation (graph generation with XL-Lexeme vectors and cosine similarity) 
by calculating the Spearman correlation with human judgements.
Input: dataset
return: Mean Spearman correlation and p-value between predicted edge weights and human judgements
"""
def get_correlation(dataset): 
    corr_stats = {}                     # data to be saved 
    corr_stats['dataset'] = dataset
    dataset = "./data/" + dataset       # full path to dataset

    words = sorted(glob.glob(dataset + "/data/*"))      # list of directories of all words in the data directory 
    corr_values = []        # list of spearman correlation values of all words in the dataset 
    p_values = []           # list of p-values of all words in the dataset 
    for word in words:                     
        uses = word + "/uses.csv" 
        df = pd.read_csv(word + "/judgments.csv", sep='\t') 
        judgements = sorted(list(df['judgment']))                       # sorted list of human judgements of the given word 
        annotated = list(zip(df['identifier1'].astype(str), df['identifier2'].astype(str)))     # list of human annotated edges 

        graph = generate_graph(uses)         # generate graph 
        pred_ann = []                               # list of predicted annotations for all edges with human annotations given in data 
        for edge in annotated:      # iterate annotated edges
            pred = graph.get_edge_data(edge[0], edge[1])['weight']    # predicted weight
            pred_ann.append(pred)   
        pred_ann.sort()             # sort list of predicted annotations  

        if len(set(judgements)) == 1:       # Spearman correlation is undefined if 'judgements' is constant 
            pass
        else:
            corr, pval = spearmanr(judgements, pred_ann)
            corr_values.append(corr)
            p_values.append(pval)

    mean_corr = round(mean(corr_values), 3)        # mean spearman correlation of dataset 
    mean_p = mean(p_values)        # mean p-value of dataset 

    corr_stats['mean_spearman_correlation'] = mean_corr
    corr_stats['mean_p_value'] = mean_p

    return corr_stats, mean_corr, mean_p




if __name__=="__main__":
    #uses = "./data/dwug_en/data/face_nn/uses.csv"
    #graph = generate_graph(uses)    
    print("---")
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]
    datasets_paper_versions = ["dwug_de", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]
    
    is_header = True        # create header first when exporting stats
    os.makedirs('./stats', exist_ok=True)     # create directory for stats (no exception is raised if directory aready exists)
    with open('./stats/correlation_stats.csv', 'w', encoding='utf-8') as f_out:     # 'w' mode deletes contents of file 
        pass
    
    for dataset in datasets:
        corr_stats, corr, p_value = get_correlation(dataset)            # get correlation stats of one dataset 
        print("\nDataset: ", dataset)
        print("Spearman's correlation coefficient:", corr)
        print("p_value:", p_value)

        # export stats 
        with open('./stats/correlation_stats.csv', 'a', encoding='utf-8') as f_out:
            if is_header:
                f_out.write('\t'.join([key for key in corr_stats]) + '\n')
                is_header = False 
            f_out.write('\t'.join([str(corr_stats[key]) for key in corr_stats]) + '\n')