
import numpy 
import pandas as pd



# https://github.com/Garrafao/WUGs/tree/main/scripts



"""
Prints stats of a dataset 
(Mean number of nodes, Mean number of nodes in grouping 1, Mean number of nodes in grouping 2, 
Mean normalized loss, Mean number of judgements, Mean number of judgements per edge) 
and returns a list of cluster frequency distributions in grouping 1, a list of cluster frequency distributions in grouping 2
and a list of binary change values. 
For NorDiaChange the two Subsets are concatenated and the mean stats of the whole dataset gets returned. 
"""

def get_dataset_stats(dataset):
    # stats from stats_groupings.csv 
    if dataset=="dwug_de_sense":
        stats_groupings = "./data/dwug_de_sense/stats/maj_2/stats_groupings.csv"
        df = pd.read_csv(stats_groupings, sep='\t')     # stats_groupings dataframe 
    elif dataset=="dwug_de_sense_maj3":
        stats_groupings = "./data/dwug_de_sense/stats/maj_3/stats_groupings.csv"
        df = pd.read_csv(stats_groupings, sep='\t')     # stats_groupings dataframe 
    elif dataset=="nor_dia_change-main":
        sg_subset1 = "./data/nor_dia_change-main/subset1/stats/stats_groupings.tsv"
        sg_subset2 = "./data/nor_dia_change-main/subset2/stats/stats_groupings.tsv"
        df_subset1 = pd.read_csv(sg_subset1, sep='\t')  # stats_groupings dataframe of subset 1  
        df_subset2 = pd.read_csv(sg_subset2, sep='\t')  # stats_groupings dataframe of subset 2  
        df = pd.concat([df_subset1, df_subset2], axis=0)    # concatenate both subsets to gets stats of the whole dataset 
    else: 
        stats_groupings = f"./data/{dataset}/stats/opt/stats_groupings.csv"
        df = pd.read_csv(stats_groupings, sep='\t')     # stats_groupings dataframe 

    nodes = list(df.iloc[:,2])     # number of nodes per graph (word) 
    nodes1 = list(df.iloc[:,3])     # number of nodes in grouping 1 per graph (word) 
    nodes2 = list(df.iloc[:,4])     # number of nodes in grouping 2 per graph (word) 
    mean_nodes = round(numpy.mean(nodes),2)     # mean number of nodes rounded to 2 decimal digits 
    nodes_std_deviation = round(numpy.std(nodes),2)     # standard deviation of number of nodes 
    mean_nodes1 = round(numpy.mean(nodes1),2)     # mean number of nodes in grouping 1 rounded to 2 decimal digits 
    mean_nodes2 = round(numpy.mean(nodes2),2)     # mean number of nodes in grouping 2 rounded to 2 decimal digits 
    frequ_dist1 = list(df['cluster_freq_dist1'])   # cluster frequ. distribution in grouping 1 
    frequ_dist2 = list(df['cluster_freq_dist2'])   # cluster frequ. distribution in grouping 2 
    binary = list(df['change_binary'])      # binary change scores 

    print(f"\nDataset: {dataset}")
    print(f"\nMean number of nodes: {mean_nodes} \nStandard deviation of mean number of nodes: {nodes_std_deviation}")
    print(f"Mean number of nodes in grouping 1: {mean_nodes1} \nMean number of nodes in grouping 2: {mean_nodes2} \n")
    

    # stats from stats.csv (no values for dwug_de_sense, because no judgements of semantic similarity between uses)
    if dataset=="dwug_de_sense" or dataset=="dwug_de_sense_maj3":
        pass
    else:
        if dataset=="nor_dia_change-main":
            sg_subset1 = "./data/nor_dia_change-main/subset1/stats/stats.tsv"
            sg_subset2 = "./data/nor_dia_change-main/subset2/stats/stats.tsv"
            df2_subset1 = pd.read_csv(sg_subset1, sep='\t')  # stats_groupings dataframe of subset 1  
            df2_subset2 = pd.read_csv(sg_subset2, sep='\t')  # stats_groupings dataframe of subset 2  
            df2 = pd.concat([df2_subset1, df2_subset2], axis=0)    # concatenate both subsets to gets stats of the whole dataset 
        else: 
            stats = f"./data/{dataset}/stats/opt/stats.csv"
            df2 = pd.read_csv(stats, sep='\t')     # stats dataframe 

        norm_loss = list(df2['loss_normalized'])     # normalized loss per graph (word) 
        judg = list(df2['judgments_total'])             # number of judgements per graph (word) 
        judg_edge = list(df2['judgments_per_edge'])     # number of judgements per edge per graph (word) 
        mean_norm_loss = round(numpy.mean(norm_loss),4)     # mean normalized loss 
        mean_judg = round(numpy.mean(judg),2)           # mean number of judgements 
        mean_judg_edge = round(numpy.mean(judg_edge),2)           # mean number of judgements per edge 

        print(f"Mean normalized loss: {mean_norm_loss} \nMean number of judgements: {mean_judg} \nMean number of judgements per edge: {mean_judg_edge}\n")

    return frequ_dist1, frequ_dist2, binary







if __name__=="__main__":
    # List of datasets: DWUG DE, DiscoWUG, RefWUG, DWUG EN, DWUG SV, DWUG LA, DWUG ES, ChiWUG, NorDiaChange, DWUG DE Sense 
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_la", "dwug_es", "chiwug", 
                "nor_dia_change-main", "dwug_de_sense", "dwug_de_sense_maj3"]
    frequ_dist1, frequ_dist2, binary = get_dataset_stats("dwug_en")                 # get stats of one dataset 
    print(binary)
    print("")






