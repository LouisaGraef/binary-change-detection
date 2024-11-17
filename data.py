import requests 
from zipfile import ZipFile
import os
import csv
import numpy 
import pandas as pd
import networkx as nx 
import pickle 
from collections import Counter 
from itertools import combinations
from constellation import Constellation



# https://github.com/Garrafao/WUGs/tree/main/scripts


"""
Downloads datasets into directory "./data", extracts downloaded zip files and deletes zip files. 
"""

def download_datasets(zipurls):
    # create data directory
    if not os.path.exists("./data"):
        os.makedirs("./data")
    # download data 
    for zipurl in zipurls:
        zipresponse = requests.get(zipurl)   # download zip file from URL
        zipfile_path = "./data/zipdata.zip"
        with open (zipfile_path, "wb") as f:   # create new file
            f.write(zipresponse.content)        # write URL zip content to new zip file 
        print("ZIP file downloaded.")
        with ZipFile("./data/zipdata.zip") as zf:      # open created zip file 
            zf.extractall(path="./data")                  # extract zip file to "./data" 
        print("ZIP file extracted.")
        os.remove(zipfile_path)                # delete zip file
        print("ZIP file deleted.")




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





# reads graph g, returns sense frequency distributions 1 and 2 of g
def get_graph_stats_old(g):
    print(graph.number_of_nodes())
    #print(list(graph.nodes(data=True))[:5])     # list of nodes 
    cluster_labels1 = [node['cluster'] for _, node in graph.nodes(data=True) if node['grouping']=='1']   # list of cluster labels of all nodes 
    cluster_labels2 = [node['cluster'] for _, node in graph.nodes(data=True) if node['grouping']=='2']   # list of cluster labels of all nodes 
    print(cluster_labels1)
    print(cluster_labels2)
    freq_dist1 = Counter(cluster_labels1)
    freq_dist2 = Counter(cluster_labels2)
    print(freq_dist1)
    print(freq_dist2)
    print("\n")

    return freq_dist1, freq_dist2



"""
returns mappings: dictionary which contains mapping from node ids to grouping of nodes (1 or 2).
"""
def get_data_maps_nodes(g, attributes={'type':'usage'}):
    mappings = {}       # dictionary which will contain node2period
    node2period = {}        # maps node ids to grouping (1 or 2) 
    for node in g.nodes():
        node_data = g.nodes()[node]     # data dictionary of node 
        #print(node_data)
        #print("")
        if all([node_data[k]==v  for (k,v) in attributes.items()]):     # if node_data['type']=='usage' 
            node2period[node] = node_data['grouping']

    mappings['node2period'] = node2period
    #print(mappings)

    return mappings


"""
returns time related stats (groupings stats) from a clustered graph as a dictionary
(frequency distributions in grouping 1 and grouping 2 of a graph G, binary change score of graph G).
"""
def get_time_stats(g, old, new, threshold=0.5, lower_range=(1,3), upper_range=(3,5), lower_prob=0.001, upper_prob=0.1, attributes={'type':'usage'}):
    g = g.copy()
    stats = {} 

    # get node stats 
    nodes = [node for node in g.nodes if all([g.nodes()[node][k]==v for (k,v) in attributes.items()])]      # list of node identifiers of nodes with 'type':'usage'
    oldnodes = [node for node in nodes if g.nodes()[node]['grouping']==old]     # list of all nodes with 'grouping':'1' (old is '1')
    newnodes = [node for node in nodes if g.nodes()[node]['grouping']==new]     # list of all nodes with 'grouping':'2' (new is '2')
    frequency = len(nodes) 
    frequency1 = len(oldnodes)
    frequency2 = len(newnodes)
    stats['nodes'] = frequency          # total number of nodes
    stats['nodes1'] = frequency1          # number of nodes from grouping 1
    stats['nodes2'] = frequency2          # number of nodes from grouping 2

    # Define thresholds
    lower_prob1 = round(lower_prob*frequency1)
    lower_prob2 = round(lower_prob*frequency2)
    upper_prob1 = round(upper_prob*frequency1)
    upper_prob2 = round(upper_prob*frequency2)
    lowerbound1 = min(max(lower_range[0],lower_prob1),lower_range[1])
    lowerbound2 = min(max(lower_range[0],lower_prob2),lower_range[1])
    upperbound1 = min(max(upper_range[0],upper_prob1),upper_range[1])
    upperbound2 = min(max(upper_range[0],upper_prob2),upper_range[1]) 

    try:    
        co = Constellation(graph=g, bound1=upperbound1, bound2=upperbound2, lowerbound1=lowerbound1, lowerbound2=lowerbound2, is_prob=False, old=old, new=new, threshold=threshold)        
        is_clusters = True
    except KeyError:
        print('No clusters found.')
        is_clusters = False

    if is_clusters:    
    
        stats['cluster_freq_dist'] = co.distribution
        stats['cluster_freq_dist1'] = co.distribution1
        stats['cluster_freq_dist2'] = co.distribution2
        stats['cluster_prob_dist'] = [round(pr, 3) for pr in co.prob]
        stats['cluster_prob_dist1'] = [round(pr, 3) for pr in co.prob1]
        stats['cluster_prob_dist2'] = [round(pr, 3) for pr in co.prob2]
        stats['cluster_number'] = len(co.prob)
        stats['cluster_number1'] = len([1 for pr in co.prob1 if pr > 0.0])
        stats['cluster_number2'] = len([1 for pr in co.prob2 if pr > 0.0])
    
        stats['change_binary'] = co.c_mb
        stats['change_binary_gain'] = co.i_mb
        stats['change_binary_loss'] = co.r_mb
        stats['change_graded'] = co.c_u
    
        stats['k1'] = lowerbound1
        stats['n1'] = upperbound1
        stats['k2'] = lowerbound2
        stats['n2'] = upperbound2

    return stats


"""
returns stats of graph G.
"""
def get_graph_stats(g):
    mappings_nodes = get_data_maps_nodes(g)     # dictionary which contains mapping from node ids to grouping of nodes (1 or 2)
    node2period = mappings_nodes['node2period']     # maps node ids to grouping (1 or 2) 
    periods = sorted(set(node2period.values()))     # list (['1', '2'])
    combos = combinations(periods, 2) if (len(periods)>1 or len(periods)==0) else [(periods[0],None)]       # list(combos) = [('1', '2')]
    for (old, new) in combos:
        time_stats = get_time_stats(g, old=old, new=new)

    return time_stats




if __name__=="__main__":
    # List of datasets: DWUG DE, DiscoWUG, RefWUG, DWUG EN, DWUG SV, DWUG LA, DWUG ES, ChiWUG, NorDiaChange, DWUG DE Sense 
    zipurls = ["https://zenodo.org/records/14028509/files/dwug_de.zip?download=1", "https://zenodo.org/records/14028592/files/discowug.zip?download=1", 
                "https://zenodo.org/records/5791269/files/refwug.zip?download=1", "https://zenodo.org/records/14028531/files/dwug_en.zip?download=1",
                "https://zenodo.org/records/14028906/files/dwug_sv.zip?download=1", "https://zenodo.org/records/5255228/files/dwug_la.zip?download=1",
                "https://zenodo.org/records/6433667/files/dwug_es.zip?download=1", "https://zenodo.org/records/10023263/files/chiwug.zip?download=1",
                "https://github.com/ltgoslo/nor_dia_change/archive/refs/heads/main.zip", "https://zenodo.org/records/14041715/files/dwug_de_sense.zip?download=1"]
    #download_datasets(zipurls)
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_la", "dwug_es", "chiwug", 
                "nor_dia_change-main", "dwug_de_sense", "dwug_de_sense_maj3"]
    frequ_dist1, frequ_dist2, binary = get_dataset_stats("dwug_en")                 # get stats of one dataset 
    print(binary)
    print("")

    input_file = "./data/dwug_en/graphs/opt/afternoon_nn"
    with open(input_file, "rb") as f:                   # open graph 
        graph = pickle.load(f)

    graph_stats = get_graph_stats(graph)         # get stats of one graph (freq dist 1 and 2, binary change value)
    print(graph_stats)






