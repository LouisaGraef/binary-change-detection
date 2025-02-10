
from download_data import download_paper_datasets, download_new_datasets
from cleaning import clean_graph, evaluate_clean_graph, get_conflicts, get_parameters, clean_graphs
from evaluation_cleaning import evaluate_cleaning
from evaluation_cleaning2 import evaluate_cleaning2


# Paper: https://openreview.net/pdf?id=BlbrJvKv6L 
# Paper Code: https://github.com/Garrafao/wug_cluster_clean/blob/main/analyze_semeval_de1.ipynb 
# WUG graph2clean2.py: https://github.com/Garrafao/WUGs/blob/main/scripts/graph2clean2.py 




if __name__=="__main__":
    
    # Download data
    download_paper_datasets()             # dwug_de 2.3.0
    download_new_datasets()               # dwug_de 3.0.0


    # Paper dataset versions
    # Get parameters for different cleaning methods 
    dataset = "./paper_data/dwug_de"
    dwug_de_sense = "./paper_data/dwug_de_sense"
    get_parameters(dataset)     # saves parameters
    clean_graphs(dataset)        # saves parameter grid 

    # Evaluate cleaning
    evaluate_cleaning(dataset, dwug_de_sense)        # dwug_de 2.3.0


    # New dataset versions
    # Get parameters for different cleaning methods 
    dataset = "./data/dwug_de"
    dwug_de_sense = "./data/dwug_de_sense"
    get_parameters(dataset)     # saves parameters
    clean_graphs(dataset)        # saves parameter grid 

    # Evaluate cleaning
    evaluate_cleaning(dataset, dwug_de_sense)        # dwug_de 3.0.0




    # evaluate all datasets by using conflicts in clusterings (for different numbers of deleted nodes) 
    # (to validate if conflicts method is good for evaluation of cleaning (bc for other datasets we don't have gold for cleaning -> no ARI)
    # -> should produce similar results (that dgrnode is better than the random baseline, similar graphics))


    # evaluate cleaning for all datasets with conflicts method 
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_la", "dwug_es", "chiwug",      # all datasets 
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

    dataset = "nor_dia_change-main/subset1"
    get_parameters(dataset)
    clean_graphs(dataset)
    evaluate_cleaning2(dataset)

    dataset = "nor_dia_change-main/subset2"
    get_parameters(dataset)
    clean_graphs(dataset)
    evaluate_cleaning2(dataset)

