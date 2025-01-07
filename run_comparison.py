
from download_data import download_new_datasets
from extract_embeddings import *
from comp_ann import *
from evaluation import *
import itertools
import subprocess

"""
Paper: https://arxiv.org/pdf/2402.12011 
Paper Code: https://github.com/FrancescoPeriti/CSSDetection/blob/main/run_comparison.sh 
"""





if __name__=="__main__":
    
    # Download datasets used in paper  
                                                     # TODO: add dwug_la to datasets 
    download_new_datasets()        
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       
    datasets = ["./data/" + dataset for dataset in datasets]

    # Get gold edge weights from uses and judgments
    for dataset in datasets:
        subprocess.run(['bash', './wug_data2graph_pipeline.sh', dataset])           


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

    
    # Correlation Clustering

    # parameter=[edge_shift_value, max_attempts, max_iters]  # s=max_clusters=10
    # https://euralex.jezik.hr/wp-content/uploads/2021/09/Euralex-XXI-proceedings_1st.pdf p.163
    # https://elib.uni-stuttgart.de/bitstream/11682/11923/3/Bachelorarbeit_SWT_Tunc.pdf 
    #parameter_list = [[0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7],
    #                  [100, 200, 500, 1000, 5000],[5000, 10000, 20000]]
    parameter_list = [[0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
                      [100, 500, 1000, 5000],[10000, 20000]]
    
    # Correlation Clustering 
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list) # create parameter grid
    

    parameter_list = [[1],[2],[3]]
    
    # K-means Clustering
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="k-means", parameter_list=parameter_list)   # create parameter grid
    
    # Agglomerative Clustering
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="agglomerative", parameter_list=parameter_list)   # create parameter grid
    
    # Spectral Clustering
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="spectral", parameter_list=parameter_list)   # create parameter grid
    