

from generate_graph import *
from scipy.stats import spearmanr
import glob 
import pandas as pd



"""
Evaluate the predicted annotation (graph generation with XL-Lexeme vectors and cosine similarity) 
by calculating the Spearman correlation with human judgements.
Input: dataset
return: Spearman correlation and p-value between predicted edge weights and human judgements
"""
def get_correlation(dataset): 
    words = sorted(glob.glob(dataset + "/data/*"))      # list of directories of all words in the data directory 
    for word in words:                     
        uses = word + "/uses.csv" 
        df = pd.read_csv(word + "/judgments.csv", sep='\t') 
        judgements = sorted(list(df['judgment']))                       # sorted list of human judgements of the given word 
        annotated = list(zip(df['identifier1'], df['identifier2']))     # list of human annotated edges 

        graph = generate_graph(uses)         # generate graph 
        pred_ann = []                               # list of predicted annotations for all edges with human annotations given in data 
        for edge in annotated:      # iterate annotated edges
            pred = graph.get_edge_data(edge[0], edge[1])['weight']    # predicted weight
            pred_ann.append(pred)   
        pred_ann.sort()             # sort list of predicted annotations  

        corr, pval = spearmanr(judgements, pred_ann)
        print("\nSpearman's correlation coefficient:", corr)
        print("p-value:", pval)




if __name__=="__main__":
    #uses = "./data/dwug_en/data/face_nn/uses.csv"
    #graph = generate_graph(uses)    
    dataset = "./data/dwug_en"
    get_correlation(dataset)