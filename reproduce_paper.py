
from download_data import download_paper_datasets
from extract_embeddings import *
from comp_ann import *
from evaluation import *
import itertools

"""
Paper: https://arxiv.org/pdf/2402.12011 
Paper Code: https://github.com/FrancescoPeriti/CSSDetection/blob/main/run_comparison.sh 
"""





if __name__=="__main__":
    # Download datasets used in paper  
                                                     # TODO: add dwug_la to datasets 
    download_paper_datasets()        
    datasets = ["dwug_de", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       
    datasets = ["./paper_data/" + dataset for dataset in datasets]
    

    # Computational Annotation 
    for dataset in datasets:
        get_computational_annotation(dataset, paper_reproduction=True)
    

    # Evalation with WIC;WSI;GCD 
    datasets = ["dwug_de", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       # no dwug_la 
    datasets = ["./paper_data/" + dataset for dataset in datasets]
    

    # WIC evaluation for all datasets 
    evaluate_wic(datasets, paper_reproduction=True)
    


    # WSI and GCD evaluation 

    output_file="./stats/paper_model_evaluation.tsv"
    # Delete content of output_file="./stats/paper_model_evaluation.tsv"
    with open(output_file, mode='w', encoding='utf-8') as f:
        pass

    
    # parameter=[s=20, max_attempts=2000, max_iters=50000]  # s=max_clusters
    parameter_list = [[20],[2000],[50000]]


    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=True, clustering_method="correlation_paper", parameter_list=parameter_list)
