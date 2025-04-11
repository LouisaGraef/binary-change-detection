
from evaluation import load_cleaned_gold_clustering
from evaluation_clustering import load_gold_clustering


def get_dataset_stats(dataset):
    data = load_gold_clustering(dataset)
    cleaned_data = load_cleaned_gold_clustering(dataset)          # df (identifier, cluster, lemma, grouping) only with identifiers left after cleaning 
    print(data)
    print(cleaned_data)
    # TODO: add column "BC_gold" based on gold clustering (and maybe "GC_gold" based on gold clustering)
    # TODO: make plots 


if __name__=="__main__":
    get_dataset_stats("./data/refwug")
