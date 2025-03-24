
import pandas as pd
from sklearn.metrics import adjusted_rand_score 
from scipy.stats import spearmanr
import numpy as np
import ast
from evaluation import purity_score


def evaluate_paper_comparison(ds, method, parameter):

    # Load Parameter grid of dataset and clustering method
    df_output_file=f"./parameter_grids/{ds}/{method}/parameter_grid.tsv"
    df = pd.read_csv(df_output_file, sep='\t')
    #print(df.head)

    # only rows with best parameters 
    df_filtered = df[df['parameter_combination'] == parameter].reset_index(drop=True)
    #print(df_filtered.head)


    ari_scores = []
    purity_scores = []

    # iterate df_filtered
    for _,row in df_filtered.iterrows():
        pred_dict = ast.literal_eval(row['clustering_pred'])
        gold_dict = ast.literal_eval(row['clustering_gold'])
        identifier = sorted(set(pred_dict.keys()) & set(gold_dict.keys()))
        pred_labels = [pred_dict[i] for i in identifier]
        gold_labels = [gold_dict[i] for i in identifier]
        ari = adjusted_rand_score(gold_labels, pred_labels)
        ari_scores.append(ari)
        pur = purity_score(gold_labels, pred_labels)
        purity_scores.append(pur)

    ari_mean = np.mean(ari_scores).round(3)
    purity_mean = np.mean(purity_scores).round(3)


    gcd_spearmanr = spearmanr(df_filtered.GC_pred.values, df_filtered.GC_gold.values)[0].round(3)
    print(f"\nDataset: {ds}\nMean ARI/ Purity: {ari_mean} / {purity_mean}\nMean GCD Spearmanr: {gcd_spearmanr}")












if __name__=="__main__":

    evaluate_paper_comparison("dwug_es", "correlation", "(0.6, 5000, 20000)")
    evaluate_paper_comparison("nor_dia_change-main/subset1", "wsbm", "('discrete-poisson', False)")
    evaluate_paper_comparison("nor_dia_change-main/subset2", "correlation", "(0.65, 200, 20000)")
    evaluate_paper_comparison("chiwug", "correlation", "(0.65, 5000, 20000)")