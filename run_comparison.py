
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
    """
    # Download datasets used in paper  
    
    download_new_datasets()        
    
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_es", "chiwug",         # no edge judgments in dwug_la
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       
    datasets = ["./data/" + dataset for dataset in datasets]
    
    
    # Get gold edge weights from uses and judgments                             
    for dataset in datasets:
        subprocess.run(['bash', './wug_data2graph_pipeline.sh', dataset])           
    
    
    # Computational Annotation (predict edge weights)
    for dataset in datasets:
        get_computational_annotation(dataset, paper_reproduction=False)
    get_computational_annotation("./data/dwug_la", paper_reproduction=False)
    
    
    

    # Evalation with WIC;WSI;GCD 
    
    
    # WIC evaluation for all datasets except dwug_la
    evaluate_wic(datasets, paper_reproduction=False)
    
    """
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_la", "dwug_es", "chiwug",      # all datasets
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]                               
    datasets = ["./data/" + dataset for dataset in datasets]

    
    """
    # Create Parameter Grids for WSI, GCD and BCD evaluation 

    
    # Correlation Clustering

        # parameter=[edge_shift_value, max_attempts, max_iters]  # s = max_clusters = 10
        # https://euralex.jezik.hr/wp-content/uploads/2021/09/Euralex-XXI-proceedings_1st.pdf p.163
        # https://elib.uni-stuttgart.de/bitstream/11682/11923/3/Bachelorarbeit_SWT_Tunc.pdf + Standardwerte (10, 200, 5000) für (s, max_attempts, max_iters)
    #parameter_list = [[0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7],
    #                  [100, 200, 500, 1000, 5000], [5000, 10000, 20000]]
    # parameter_list = [[0.4, 0.45, 0.5, 0.55, 0.6, 0.65], [100, 500, 1000, 5000], [10000, 20000]]
    parameter_list = [[0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
                      [200, 500, 1000, 5000], [5000, 10000, 20000]]
    
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list) # create parameter grid
    
    """
    """
    # K-means Clustering
    # Parameter: n_init = [10, 30, 50], max_iter = [300, 400, 500]
    parameter_list = [[10, 50],[300, 500]]
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="k-means", parameter_list=parameter_list)   # create parameter grid
    
    # Agglomerative Clustering
    # parameters: linkage = ['ward', 'average', 'complete', 'single']   (which linkage criterion to use)
    # metric = ['euclidean', 'cosine', 'precomputed']  (metric used to compute the linkage)
    parameter_list = [['ward', 'average', 'complete', 'single'],['euclidean', 'cosine', 'precomputed']]
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="agglomerative", parameter_list=parameter_list)   # create parameter grid
    
    # Spectral Clustering
    # Parameters: We use the scikit-learn21 implementation with default hyperparameters. We apply the K-means
    # algorithm to find clusters in the reduced-dimensional space (Sense through time)
    # Parameters: which clustering algorithm to apply in the reduced dimensional space?
    # affinity: ['nearest_neighbors', 'rbf', 'precomputed'] (how to construct the affinity matrix)
    # n_neighbors: [5, 10, 15] (number of neighbors if nearest_neighbors is used)
    parameter_list = [['nearest_neighbors', 'rbf', 'precomputed'],[5, 10, 15]]
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="spectral", parameter_list=parameter_list)   # create parameter grid
    
    """
    # WSBM Clustering 
    # Parameters: Exponential and normal distribution (Sense through time)
    #parameter_list = [["real-normal"], ["real-exponential"]]
    parameter_list = [["real-exponential"]]
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list)   # create parameter grid
    