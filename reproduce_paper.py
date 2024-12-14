
from download_data import download_paper_datasets
from extract_embeddings import *
from computational_annotation import get_computational_annotation
import subprocess

"""
Paper: https://arxiv.org/pdf/2402.12011 
Paper Code: https://github.com/FrancescoPeriti/CSSDetection/blob/main/run_comparison.sh 
"""





if __name__=="__main__":
    # Download datasets used in paper  
    download_paper_datasets()
    datasets = ["dwug_de", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       # TODO: add dwug_la 
    datsets = ["./data/" + dataset for dataset in datasets]

    # Embedding Extraction 
    #for dataset in datasets:
        #extract_embeddings(dataset)


    # Embedding Evaluation 



    # Computational Annotation 
    batch_size=16
    for dataset in datasets:
        get_computational_annotation(dataset, batch_size=batch_size)


    # Evalation with WIC;WSI;GCD 