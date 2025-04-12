
from evaluation import load_cleaned_gold_clustering, get_cluster_distributions, predict_binary
from evaluation_clustering import load_gold_clustering
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import pandas as pd



def add_bc_and_gc_gold(data, bc_min_max):
    # add clustering_gold for each lemma (maps identifier to gold cluster)
    lemma_clustering_gold = data.groupby('lemma').apply(
        lambda df: dict(zip(df["identifier"], df["cluster"]))
        ).to_dict()
    
    data["clustering_gold"] = data["lemma"].map(lemma_clustering_gold)
    #print(data)
    

    # Add BC_gold and GC_gold
    lemma_groups = data.groupby('lemma')
    for lemma_name, lemma_df in lemma_groups:
            c = 0
            for index, row in tqdm(lemma_df.iterrows(), total=len(lemma_df), desc=f"{lemma_name}"):
                if c > 0:                           # only one iteration per lemma
                    continue
                # Get list of sets of identifiers that belong to the same cluster
                cluster_sets = {}
                id_to_grouping = {row2['identifier']: f"{row2['identifier']}###{row2['grouping']}" for index, row2 in data.iterrows()}
                for identifier, cluster in row['clustering_gold'].items():
                    if cluster not in cluster_sets.keys():
                        cluster_sets[cluster] = set()
                    identifier = id_to_grouping[identifier]
                    cluster_sets[cluster].add(identifier)
                classes_sets = list(cluster_sets.values())

                # Compute cluster distributions (cluster frequency distribution and cluster probability distribution) for one Graph
                freq_dist, prob_dist = get_cluster_distributions(classes_sets)
                #print(lemma_name)
                #print(prob_dist)

                data.loc[data['lemma'] == lemma_name, 'GC_gold'] = jensenshannon(prob_dist[0], prob_dist[1], base=2.0)
                data.loc[data['lemma'] == lemma_name, 'BC_gold'] = predict_binary(freq_dist, minf=bc_min_max[0], maxf=bc_min_max[1], gold=True) # 0 and 1 or 1 and 3
                c+=1
    return data





def get_dataset_stats(dataset, bc_min_max):
    data = load_gold_clustering(dataset)
    cleaned_data = load_cleaned_gold_clustering(dataset)          # df (identifier, cluster, lemma, grouping) only with identifiers left after cleaning 
    #print(data)             # 421 rows
    #print(cleaned_data)     # 327 rows

    data = add_bc_and_gc_gold(data, bc_min_max)                         # df (identifier, cluster, lemma, grouping, clustering_gold, GC_gold, BC_gold)
    cleaned_data = add_bc_and_gc_gold(cleaned_data, bc_min_max)         # df (identifier, cluster, lemma, grouping, clustering_gold, GC_gold, BC_gold)

    print(data)
    print(cleaned_data)


    # per lemma: number of clusters, BC_gold, GC_gold
    cluster_n_total = data.groupby("lemma")["cluster"].nunique().rename("cluster_n_total")
    cluster_n_1 = data[data["grouping"]==1].groupby("lemma")["cluster"].nunique().rename("cluster_n_1")
    cluster_n_2 = data[data["grouping"]==2].groupby("lemma")["cluster"].nunique().rename("cluster_n_2")
    BC_gold = data.groupby("lemma")["BC_gold"].first()
    GC_gold = data.groupby("lemma")["GC_gold"].first()
    cluster_n = pd.concat([cluster_n_total, cluster_n_1, cluster_n_2, BC_gold, GC_gold], axis=1).sort_values(by="lemma", key=lambda x: x.str.lower())
    print(f"\n\n{cluster_n}")

    
    # per lemma: number of clusters, BC_gold, GC_gold
    cleaned_cluster_n_total = cleaned_data.groupby("lemma")["cluster"].nunique().rename("cluster_n_total")
    cleaned_cluster_n_1 = cleaned_data[cleaned_data["grouping"]==1].groupby("lemma")["cluster"].nunique().rename("cluster_n_1")
    cleaned_cluster_n_2 = cleaned_data[cleaned_data["grouping"]==2].groupby("lemma")["cluster"].nunique().rename("cluster_n_2")
    BC_gold = cleaned_data.groupby("lemma")["BC_gold"].first()
    GC_gold = cleaned_data.groupby("lemma")["GC_gold"].first()
    cleaned_cluster_n = pd.concat([cleaned_cluster_n_total, cleaned_cluster_n_1, cleaned_cluster_n_2, BC_gold, GC_gold], axis=1).sort_values(by="lemma", key=lambda x: x.str.lower())
    print(f"\n\n{cleaned_cluster_n}")


    # TODO: add cluster_freq_dist, cluster_freq_dist1, cluster_freq_dist2 für jedes Lemma 
    # TODO: make plots 



if __name__=="__main__":
    get_dataset_stats("./data/refwug", bc_min_max=[1,3])
