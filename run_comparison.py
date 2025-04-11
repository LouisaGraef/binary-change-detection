
from download_data import download_new_datasets
from extract_embeddings import *
from comp_ann import *
from evaluation import *
import itertools
import subprocess
from evaluation_clustering import evaluate_clustering
from evaluation_clustering_plot import evaluate_clustering_plot
from evaluation_paper_comparison import evaluate_paper_comparison
from cleaned_datasets_stats import get_dataset_stats

"""
Paper: https://arxiv.org/pdf/2402.12011 
Paper Code: https://github.com/FrancescoPeriti/CSSDetection/blob/main/run_comparison.sh 
"""





if __name__=="__main__":

    """
    
    # Download datasets
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
    
    
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_la", "dwug_es", "chiwug",      # all datasets 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]                               
    datasets = ["./data/" + dataset for dataset in datasets]

    





    
    
    # Create Parameter Grids for WSI, GCD and BCD evaluation 

    
    # Correlation Clustering

        # parameter=[edge_shift_value, max_attempts, max_iters]  # s = max_clusters = 7
        # https://euralex.jezik.hr/wp-content/uploads/2021/09/Euralex-XXI-proceedings_1st.pdf p.163
        # https://elib.uni-stuttgart.de/bitstream/11682/11923/3/Bachelorarbeit_SWT_Tunc.pdf  für (max_attempts, max_iters)

    parameter_list = [[0.45, 0.5, 0.55, 0.6, 0.65], [200, 5000], [5000, 20000]]

    evaluate_model(dataset="./data/dwug_de", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list)
    evaluate_model(dataset="./data/discowug", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list)
    evaluate_model(dataset="./data/refwug", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list)
    evaluate_model(dataset="./data/dwug_la", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list)
    evaluate_model(dataset="./data/dwug_es", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list)
    evaluate_model(dataset="./data/dwug_en", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list)
    evaluate_model(dataset="./data/dwug_sv", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list)
    evaluate_model(dataset="./data/chiwug", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list)
    evaluate_model(dataset="./data/nor_dia_change-main/subset1", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list)
    evaluate_model(dataset="./data/nor_dia_change-main/subset2", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list)


    
    # K-means Clustering
    # Parameter: n_init = [1, 10, 20], max_iter = [300, 400, 500]
    parameter_list = [[1, 10, 20], [300, 400, 500]]
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="k-means", parameter_list=parameter_list)   # create parameter grid
    
    # Agglomerative Clustering
    # parameters: linkage = ['ward', 'average', 'complete', 'single']   (which linkage criterion to use)
    # metric = ['euclidean', 'cosine']  (metric used to compute the linkage)
    parameter_list = [['ward', 'average', 'complete', 'single'],['euclidean', 'cosine']]
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="agglomerative", parameter_list=parameter_list)   # create parameter grid
    
    # Spectral Clustering
    # affinity: ['nearest_neighbors', 'rbf'] (how to construct the affinity matrix)
    # n_neighbors: [5, 10, 15] (number of neighbors if nearest_neighbors is used)
    parameter_list = [['nearest_neighbors', 'rbf'],[5, 10, 15]]
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="spectral", parameter_list=parameter_list)   # create parameter grid
    
    
    
    # WSBM Clustering 
    # Parameters: different distributions (Sense through time), adjacency False or True
    parameter_list = [["real-normal", "real-exponential", "discrete-geometric", "discrete-poisson", "discrete-binomial"], [False, True]]

    evaluate_model("./data/discowug", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list)   # create parameter grid
    evaluate_model("./data/refwug", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list)   # create parameter grid
    evaluate_model("./data/dwug_la", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list)   # create parameter grid
    evaluate_model("./data/dwug_es", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list)   # create parameter grid
    
    evaluate_model("./data/chiwug", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list)   # create parameter grid
    evaluate_model("./data/nor_dia_change-main/subset1", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list)   # create parameter grid
    evaluate_model("./data/nor_dia_change-main/subset2", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list)   # create parameter grid

    evaluate_model("./data/dwug_de", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list)   # create parameter grid
    evaluate_model("./data/dwug_en", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list)   # create parameter grid
    evaluate_model("./data/dwug_sv", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list)   # create parameter grid
    













    # Cluster cleaned Graphs (clustersize 20 or 10 depending on dataset)
    """
    datasets = ["discowug", "refwug", "dwug_la", "dwug_es", "chiwug",      # all datasets 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2", "dwug_de", "dwug_en", "dwug_sv"]                               
    datasets = ["./data/" + dataset for dataset in datasets]
    """
    # Correlation Clustering
    parameter_list = [[0.45, 0.5, 0.55, 0.6, 0.65], [200, 5000], [5000, 20000]]

    evaluate_model(dataset="./data/discowug", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list, cleaned=True)
    evaluate_model(dataset="./data/refwug", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list, cleaned=True)
    evaluate_model(dataset="./data/dwug_la", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list, cleaned=True)
    evaluate_model(dataset="./data/dwug_es", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list, cleaned=True)
    evaluate_model(dataset="./data/chiwug", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list, cleaned=True)
    evaluate_model(dataset="./data/nor_dia_change-main/subset1", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list, cleaned=True)
    evaluate_model(dataset="./data/nor_dia_change-main/subset2", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list, cleaned=True)
    evaluate_model(dataset="./data/dwug_de", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list, cleaned=True)
    evaluate_model(dataset="./data/dwug_en", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list, cleaned=True)
    evaluate_model(dataset="./data/dwug_sv", paper_reproduction=False, clustering_method="correlation", parameter_list=parameter_list, cleaned=True)

    
    # K-means Clustering
    parameter_list = [[1, 10, 20], [300, 400, 500]]
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="k-means", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    
    # Agglomerative Clustering
    parameter_list = [['ward', 'average', 'complete', 'single'],['euclidean', 'cosine']]
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="agglomerative", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    
    # Spectral Clustering
    parameter_list = [['nearest_neighbors', 'rbf'],[5, 10, 15]]
    for dataset in datasets:
        evaluate_model(dataset, paper_reproduction=False, clustering_method="spectral", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    
    
    
    
    # WSBM Clustering 
    parameter_list = [["real-normal", "real-exponential", "discrete-geometric", "discrete-poisson", "discrete-binomial"], [False, True]]

    evaluate_model("./data/discowug", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    evaluate_model("./data/refwug", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    evaluate_model("./data/dwug_la", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    evaluate_model("./data/dwug_es", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    
    evaluate_model("./data/chiwug", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    evaluate_model("./data/nor_dia_change-main/subset1", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    evaluate_model("./data/nor_dia_change-main/subset2", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list, cleaned=True)   # create parameter grid

    evaluate_model("./data/dwug_de", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    evaluate_model("./data/dwug_en", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    evaluate_model("./data/dwug_sv", paper_reproduction=False, clustering_method="wsbm", parameter_list=parameter_list, cleaned=True)   # create parameter grid
    """










    
    # Clustering evaluation
    
    datasets = ["dwug_de", "refwug", "discowug", "dwug_en", "dwug_sv", "dwug_la", "dwug_es", "chiwug",      # all datasets 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]                               
    datasets = ["./data/" + dataset for dataset in datasets]

    """
    for dataset in datasets:
        evaluate_clustering(dataset, cleaned_gold=False, filter_minus_one_nodes=False, bc_min_max=[1,3])
        evaluate_clustering(dataset, cleaned_gold=False, filter_minus_one_nodes=True, bc_min_max=[1,3])
        evaluate_clustering(dataset, cleaned_gold=True, filter_minus_one_nodes=False, bc_min_max=[1,3])
        evaluate_clustering(dataset, cleaned_gold=True, filter_minus_one_nodes=True, bc_min_max=[1,3])
        
        evaluate_clustering(dataset, cleaned_gold=False, filter_minus_one_nodes=False, bc_min_max=[0,1])
        evaluate_clustering(dataset, cleaned_gold=False, filter_minus_one_nodes=True, bc_min_max=[0,1])
        evaluate_clustering(dataset, cleaned_gold=True, filter_minus_one_nodes=False, bc_min_max=[0,1])
        evaluate_clustering(dataset, cleaned_gold=True, filter_minus_one_nodes=True, bc_min_max=[0,1])
    



    # Plot Clustering evaluation 

    evaluate_clustering_plot(datasets, cleaned_gold=False, filter_minus_one_nodes=False, bcmm="13")
    evaluate_clustering_plot(datasets, cleaned_gold=False, filter_minus_one_nodes=True, bcmm="13")
    evaluate_clustering_plot(datasets, cleaned_gold=True, filter_minus_one_nodes=False, bcmm="13")
    evaluate_clustering_plot(datasets, cleaned_gold=True, filter_minus_one_nodes=True, bcmm="13")

    evaluate_clustering_plot(datasets, cleaned_gold=False, filter_minus_one_nodes=False, bcmm="01")
    evaluate_clustering_plot(datasets, cleaned_gold=False, filter_minus_one_nodes=True, bcmm="01")
    evaluate_clustering_plot(datasets, cleaned_gold=True, filter_minus_one_nodes=False, bcmm="01")
    evaluate_clustering_plot(datasets, cleaned_gold=True, filter_minus_one_nodes=True, bcmm="01")

    quit()


    # Evaluate best models with best parameters on whole datasets (for comparison with Periti paper)

    evaluate_paper_comparison("dwug_es", "correlation", "(0.6, 5000, 20000)")
    evaluate_paper_comparison("nor_dia_change-main/subset1", "wsbm", "('discrete-poisson', False)")
    evaluate_paper_comparison("nor_dia_change-main/subset2", "correlation", "(0.65, 200, 20000)")
    evaluate_paper_comparison("chiwug", "correlation", "(0.65, 5000, 20000)") 
    """


    # Get stats of datasets and cleaned datasets 
    get_dataset_stats("./data/refwug")