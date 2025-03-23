
from download_data import download_paper_datasets, download_new_datasets
from cleaning import clean_graph, get_parameters, clean_graphs
from evaluation_cleaning import evaluate_cleaning
from evaluation_cleaning2 import evaluate_cleaning2
import subprocess


# Paper: https://openreview.net/pdf?id=BlbrJvKv6L 
# Paper Code: https://github.com/Garrafao/wug_cluster_clean/blob/main/analyze_semeval_de1.ipynb 
# WUG graph2clean2.py: https://github.com/Garrafao/WUGs/blob/main/scripts/graph2clean2.py 




if __name__=="__main__":
    
    # Download data
    download_paper_datasets()             # dwug_de 2.3.0
    download_new_datasets()               # dwug_de 3.0.0

    
    # Evaluate with ARI

    # Paper dataset versions
    dataset = "./paper_data/dwug_de"
    dwug_de_sense = "./paper_data/dwug_de_sense"
    get_parameters(dataset)                         # get parameters
    clean_graphs(dataset)                           # get parameter grid 
    evaluate_cleaning(dataset, dwug_de_sense)       # Evaluate cleaning

    # New dataset versions
    dataset = "./data/dwug_de"
    dwug_de_sense = "./data/dwug_de_sense"
    get_parameters(dataset)                         # get parameters
    clean_graphs(dataset)                           # get parameter grid 
    evaluate_cleaning(dataset, dwug_de_sense)       # Evaluate cleaning





    # Evaluate with normalized conflicts in clusterings (for different numbers of deleted nodes) (no gold for datasets -> no ARI)


    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_la", "dwug_es", "chiwug",      
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]                               
    datasets = ["./data/" + dataset for dataset in datasets]

    
    dataset = "./paper_data/dwug_de"
    evaluate_cleaning2(dataset)
    
    dataset = "./data/dwug_de"
    evaluate_cleaning2(dataset)

    dataset = "./data/discowug"
    get_parameters(dataset)
    clean_graphs(dataset)
    evaluate_cleaning2(dataset)

    dataset = "./data/refwug"
    get_parameters(dataset)
    clean_graphs(dataset)
    evaluate_cleaning2(dataset)
    
    dataset = "./data/dwug_en"
    get_parameters(dataset)
    clean_graphs(dataset)
    evaluate_cleaning2(dataset)

    dataset = "./data/dwug_sv"
    get_parameters(dataset)
    clean_graphs(dataset)
    evaluate_cleaning2(dataset)

    dataset = "./data/dwug_la"
    get_parameters(dataset)
    clean_graphs(dataset)
    evaluate_cleaning2(dataset)

    dataset = "./data/dwug_es"
    get_parameters(dataset)
    clean_graphs(dataset)
    evaluate_cleaning2(dataset)

    dataset = "./data/chiwug"
    get_parameters(dataset)
    clean_graphs(dataset)
    evaluate_cleaning2(dataset)
    
    dataset = "./data/nor_dia_change-main/subset1"
    subprocess.run(['bash', './wug_data2graph_pipeline.sh', dataset])       # get graph with uses and judgments 
    get_parameters(dataset)
    clean_graphs(dataset)
    evaluate_cleaning2(dataset)

    dataset = "./data/nor_dia_change-main/subset2"
    subprocess.run(['bash', './wug_data2graph_pipeline.sh', dataset])       # get graph with uses and judgments 
    get_parameters(dataset)
    clean_graphs(dataset)
    evaluate_cleaning2(dataset)

