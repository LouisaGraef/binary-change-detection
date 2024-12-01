

from generate_graph import *
from scipy.stats import spearmanr
import glob 
import pandas as pd
from statistics import mean



"""
Evaluate the predicted annotation (graph generation with XL-Lexeme vectors and cosine similarity) 
by calculating the Spearman correlation with human judgements.
Input: dataset
return: Mean Spearman correlation and p-value between predicted edge weights and human judgements
"""
def get_correlation(dataset): 
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

    return mean_corr, mean_p




if __name__=="__main__":
    #uses = "./data/dwug_en/data/face_nn/uses.csv"
    #graph = generate_graph(uses)    
    print("---")
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]
    
    for ds in datasets:
        dataset = "./data/" + ds
        corr, p_value = get_correlation(dataset)
        print("\nDataset: ", ds)
        print("Spearman's correlation coefficient:", corr)
        print("p_value:", p_value)