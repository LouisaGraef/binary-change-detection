
from download_data import download_new_datasets
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
                                                     # TODO: remove comment, add dwug_la to datasets 
    download_new_datasets()        
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       
    datasets = ["./data/" + dataset for dataset in datasets]


    # TODO: wug_data2graph_pipeline.sh hier ausführen, damit in get_computational_annotation 
    # die Goldkantengewichte aus den mit der DWUG Pipeline erstellten Graphen abgelesen werden können. 
    

    # Computational Annotation 
    for dataset in datasets:
        get_computational_annotation(dataset, paper_reproduction=False)
    

    # Evalation with WIC;WSI;GCD 
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       # no dwug_la 
    datasets = ["./data/" + dataset for dataset in datasets]
    

    # WIC evaluation for all datasets 
    evaluate_wic(datasets, paper_reproduction=False)
    


    # Create Parameter Grids for WSI, GCD and BCD evaluation 

    output_file="./stats/paper_model_evaluation.tsv"
    # Delete content of output_file="./stats/paper_model_evaluation.tsv"
    with open(output_file, mode='w', encoding='utf-8') as f:
        pass

    
    # parameter=[s=20, max_attempts=2000, max_iters=50000]  # s=max_clusters
    parameter_list = [[20],[2000],[50000]]


    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=True, clustering_method="correlation_paper", parameter_list=parameter_list)
