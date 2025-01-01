
from download_data import download_paper_datasets
from extract_embeddings import *
from comp_ann import *
from model_evaluate import *

"""
Paper: https://arxiv.org/pdf/2402.12011 
Paper Code: https://github.com/FrancescoPeriti/CSSDetection/blob/main/run_comparison.sh 
"""





if __name__=="__main__":
    # Download datasets used in paper  
    """                                                 # TODO: remove comment, add dwug_la to datasets 
    download_paper_datasets()        
    """
    datasets = ["dwug_de", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       
    datasets = ["./data/" + dataset for dataset in datasets]
    

    # Computational Annotation 
    for dataset in datasets:
        get_computational_annotation(dataset)



    # Evalation with WIC;WSI;GCD 
    datasets = ["dwug_de", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       # no dwug_la 
    datasets = ["./data/" + dataset for dataset in datasets]
    
    output_file="./stats/model_evaluation.tsv"
    # Delete content of output_file="./stats/model_evaluation.tsv"
    with open(output_file, mode='w', encoding='utf-8') as f:
        pass

    for dataset in datasets:
        evaluate_model(dataset)