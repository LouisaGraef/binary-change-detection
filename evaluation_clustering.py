import sys
import csv
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
from itertools import combinations, chain
from collections import defaultdict, Counter
import seaborn as sb
from pandas import DataFrame
import unicodedata
import pandas as pd
from sklearn.metrics import cohen_kappa_score, hamming_loss, accuracy_score, f1_score
from sklearn.metrics.cluster import adjusted_rand_score, rand_score
from scipy.stats import spearmanr
import requests

# Cross validation
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.utils.estimator_checks import check_estimator

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from collections import Counter
import ast
import seaborn as sns
import dill
import pickle 
from modules import get_nan_edges
from cleaning import clean_graph
from evaluation import get_cluster_distributions, predict_binary
from scipy.spatial.distance import jensenshannon


"""
Paper: https://openreview.net/pdf?id=BlbrJvKv6L 
Paper Code: https://github.com/Garrafao/wug_cluster_clean/blob/main/analyze_semeval_de1.ipynb 
            https://github.com/Garrafao/wug_cluster_clean/blob/main/Nikolay_analyze_semeval_de1.ipynb 
"""


def load_gold_clustering(dataset):
    
    # load Gold Clustering 
    df_dwug_de = pd.DataFrame()                                                         # gold clustering 
    for p in Path(f'{dataset}/').glob('clusters/opt/*.csv'):    
        lemma = str(p).replace('\\', '/').split('/')[-1].replace('.csv','')
        lemma = unicodedata.normalize('NFC', lemma)
        df = pd.read_csv(p, delimiter='\t', quoting=3, na_filter=False)
        df['lemma'] = lemma
        df_dwug_de = pd.concat([df_dwug_de, df])    

    # Extract grouping (time) information
    df_dwug_de_uses = pd.DataFrame()
    for p in Path(f'{dataset}/data').glob('*/uses.csv'):
        df_dwug_de_uses = pd.concat([df_dwug_de_uses, pd.read_csv(p, delimiter='\t', quoting=3, na_filter=False)])


    df_dwug_de = df_dwug_de.merge(df_dwug_de_uses[['identifier', 'lemma', 'grouping']], how='left',      
                                                on = ['identifier', 'lemma'])
    # columns: ['identifier', 'cluster', 'lemma', 'grouping']
    
    print(f'df_dwug_de: \n{df_dwug_de}\n')

    return df_dwug_de






def evaluate_clustering(dataset, cleaned_gold):

    
    print(f'\nDataset: {dataset}\n')

    # load Gold Clustering 
    if cleaned_gold == True:
        df_dwug_de = load_cleaned_gold_clustering(dataset)
    else:
        df_dwug_de = load_gold_clustering(dataset)

    print(df_dwug_de)


    # load Clustering Parameter Grid, Add columns to clustering_parameter_grid: model, method, ARI, GC_Spearmanr, BC_F1
    ds = dataset.replace("./data/", "")


    clustering_parameter_grid = pd.DataFrame()
    for p in Path(f'./parameter_grids/{ds}/').glob('*/parameter_grid.tsv'):
        method = os.path.basename(os.path.dirname(p))
        if method == "correlation_paper":
            continue
        df = pd.read_csv(p, delimiter='\t')
        df['method'] = method
        df['model'] = (f'{method}_{ds}_' + df['parameter_combination'].astype(str))

        valid_identifiers = set(df_dwug_de['identifier'])       # only nodes that are not removed in cleaning
        df['clustering_pred'] = df['clustering_pred'].apply(ast.literal_eval)        # string to dictionary
        df['clustering_gold'] = df['clustering_gold'].apply(ast.literal_eval)        # string to dictionary
        df['clustering_pred'] = df['clustering_pred'].apply(lambda d: {k: v for k, v in d.items() if k in valid_identifiers})   # filter cleaned nodes out
        df['clustering_gold'] = df['clustering_gold'].apply(lambda d: {k: v for k, v in d.items() if k in valid_identifiers})

        print(df)


        # Add new GC_gold, BC_gold, GC_pred and BC_pred for cleaned dataset version
        if cleaned_gold==True:
            lemma_groups = df.groupby('word')
            for lemma_name, lemma_df in lemma_groups:
                counter = 0
                for index, row in lemma_df.iterrows():
                    if index > 0:
                        continue
                    # Get list of sets of identifiers that belong to the same cluster
                    cluster_sets = {}
                    id_to_grouping = {row2['identifier']: f"{row2['identifier']}###{row2['grouping']}" for index, row2 in df_dwug_de.iterrows()}
                    for identifier, cluster in row['clustering_gold'].items():
                        if cluster not in cluster_sets.keys():
                            cluster_sets[cluster] = set()
                        identifier = id_to_grouping[identifier]
                        cluster_sets[cluster].add(identifier)
                    classes_sets = list(cluster_sets.values())

                    # Compute cluster distributions (cluster frequency distribution and cluster probability distribution) for one Graph
                    freq_dist, prob_dist = get_cluster_distributions(classes_sets)

                    df.loc[df['word'] == lemma_name, 'GC_gold'] = jensenshannon(prob_dist[0], prob_dist[1], base=2.0)
                    df.loc[df['word'] == lemma_name, 'BC_gold'] = predict_binary(freq_dist)
                    counter +=1
                    print(counter)


            counter = 0
            for index, row in df.iterrows():

                # Get list of sets of identifiers that belong to the same cluster
                cluster_sets = {}
                id_to_grouping = {row2['identifier']: f"{row2['identifier']}###{row2['grouping']}" for index, row2 in df_dwug_de.iterrows()}
                for identifier, cluster in row['clustering_pred'].items():
                    if cluster not in cluster_sets.keys():
                        cluster_sets[cluster] = set()
                    identifier = id_to_grouping[identifier]
                    cluster_sets[cluster].add(identifier)
                classes_sets = list(cluster_sets.values())

                # Compute cluster distributions (cluster frequency distribution and cluster probability distribution) for one Graph
                freq_dist, prob_dist = get_cluster_distributions(classes_sets)

                df.loc[index, 'GC_pred'] = jensenshannon(prob_dist[0], prob_dist[1], base=2.0)
                df.loc[index, 'BC_pred'] = predict_binary(freq_dist)

                counter +=1
                print(counter)



        # GC_Spearmanr and BC_F1 for each model seperately 
        model_groups = df.groupby('model')
        for model_name, model_df in model_groups:

            ari_values = []

            for index, row in model_df.iterrows():

                # evaluate clustering
                pred_clusters = row['clustering_pred']
                gold_clusters = row['clustering_gold']
                pred_labels = [pred_clusters[node] for node in sorted(pred_clusters)]
                gold_labels = [gold_clusters[node] for node in sorted(gold_clusters)]
                ari = adjusted_rand_score(gold_labels, pred_labels)
                ari_values.append(ari)
                

            # evaluate Graded Change
            spearman, p_value = spearmanr(model_df['GC_pred'], model_df['GC_gold'])

            # evaluate Binary Change
            f1 = f1_score(model_df['BC_gold'], model_df['BC_pred'])

            # Add columns and reorder 
            model_df['ARI'] = ari_values
            model_df['GC_Spearmanr'] = spearman
            model_df['BC_F1'] = f1
            model_df['cluster_no'] = model_df['clustering_pred'].apply(lambda x: len(set(x.values())))             # number of clusters (predicted)
            model_df = model_df[['model', 'word', 'ARI', 'GC_Spearmanr', 'BC_F1', 'cluster_no', 'BC_pred', 'BC_gold', 'GC_pred', 'GC_gold', 
                    'method','parameter_combination', 'clustering_pred', 'clustering_gold']]
            clustering_parameter_grid = pd.concat([clustering_parameter_grid, model_df], ignore_index=True)


    
    print('\ndf_results: ')
    df_results = clustering_parameter_grid
    print(df_results)
    print(df_results.columns.tolist())
    
    # ['model', 'word', 'ARI', 'GC_Spearmanr', 'BC_F1', 'BC_pred', 'BC_gold', 'GC_pred', 'GC_gold', 
    # 'method', 'parameter_combination', 'clustering_pred', 'clustering_gold']
    print('\nSorted by ARI: ')
    print(df_results.groupby('model').agg({'ARI': 'mean','GC_Spearmanr': 'mean','BC_F1': 'mean'}).sort_values(by='ARI', ascending=False))
    print('\nSorted by GC_Spearmanr: ')
    print(df_results.groupby('model').agg({'ARI': 'mean','GC_Spearmanr': 'mean','BC_F1': 'mean'}).sort_values(by='GC_Spearmanr', ascending=False))
    print('\nSorted by BC_F1: ')
    print(df_results.groupby('model').agg({'ARI': 'mean','GC_Spearmanr': 'mean','BC_F1': 'mean'}).sort_values(by='BC_F1', ascending=False))
    print('\nMethod = correlation:')
    print(df_results[df_results['method'].eq('correlation')].groupby('model').agg({'ARI': 'mean','GC_Spearmanr': 'mean','BC_F1': 'mean'}).sort_values(by='ARI', ascending=False))













    class ARI_Precomputed(BaseEstimator):

        def __init__(self, df, column_X, column_y, column_p, index2target, is_strict):
            self.df = df
            self.column_X = column_X
            self.column_y = column_y
            self.column_p = column_p
            self.index2target = index2target
            self.is_strict = is_strict

        def fit(self, X, y):
            # Check that X and y have correct shape
            X, y = check_X_y(X, y)
            
            self.X_ = X
            self.y_ = y
            
            X = [index2target[i] for i in X.flatten()]
            df_clean = self.df[self.df[self.column_X].isin(X)]         
            models_top = df_clean.groupby(self.column_p).agg({self.column_y: 'mean'}).nlargest(1, self.column_y, keep='all')
            try:
                assert len(models_top) <= 1
            except AssertionError as e:
                print('Warning: Found multiple top models.')
                if self.is_strict:
                    raise e
            model_top = models_top.index.to_list()[0]       
            self.model_top_ = model_top

            return self
        
        def score(self, X, y):
            X = [index2target[i] for i in X.flatten()]
            #print(X)
            model_top = self.model_top_
            df_clean = self.df[self.df[self.column_X].isin(X)]
            df_clean_model_top = df_clean[df_clean[self.column_p].eq(model_top)]
            assert len(df_clean_model_top) == len(X)
            mean = df_clean_model_top[self.column_y].mean()
            return mean





    targets = df_dwug_de['lemma'].unique()
    target2index = {target:index for index, target in enumerate(targets)}
    index2target = {index:target for index, target in enumerate(targets)}
    indices = np.array(list(target2index.values())).reshape(-1, 1)
    cluster = df_dwug_de['cluster']


    def cvalidate(metric):
        #df_results_reduced = df_results[(df_results['ambiguity'].eq('None')) & (df_results['collapse'].eq('None'))]
        df_results_reduced = df_results
        #print(f'\n\n\n {df_results_reduced}')
        print('\n\n\n\n')

        # Iterate over different methods
        methods = [('all','method'), ('wsbm','method'), ('correlation','method'), ('k-means','method'), ('spectral','method'), ('agglomerative','method')]
        #base_models = [('wsbm_dwug_de_2.3.0-0.0-None-None-binomial-False-False','model'), ('wsbm_dwug_de_2.3.0-0.0-None-None-binomial-True-False','model'), ('wsbm_dwug_de_2.3.0-0.0-None-None-binomial-True-True','model'), ('correlation_dwug_de_2.3.0-2.7-None-None-None-None-None','model'), ('correlation_dwug_de_2.3.0-2.6-None-None-None-None-None','model'), ('correlation_dwug_de_2.3.0-2.5-None-None-None-None-None','model'), ('correlation_dwug_de_2.3.0-2.4-None-None-None-None-None','model'), ('correlation_dwug_de_2.3.0-2.3-None-None-None-None-None','model'), ('correlation_dwug_de_2.3.0-2.2-None-None-None-None-None','model'), ('correlation_dwug_de_2.3.0-2.1-None-None-None-None-None','model'), ('correlation_dwug_de_2.3.0-2.0-None-None-None-None-None','model'), ('chinese_dwug_de_2.3.0-2.5-None-top-None-None-None','model')]
        #methods_models = methods + base_models
        methods_models = methods 
        models_to_plot, method2cvaris = {}, {}
        for m, col in methods_models:
            
            print('*' +m +'*')
            
            if m=='all':
                df_results_method = df_results_reduced
            else:
                df_results_method = df_results_reduced[df_results_reduced[col].eq(m)]
                
            
            estimator = ARI_Precomputed(df_results_method, 'word', metric, 'model', index2target, False)    
            cv_results = cross_validate(estimator, indices, indices.ravel(), cv=len(indices), return_estimator=True, error_score='raise')

            # Get average result
            test_score = cv_results['test_score']
            method2cvaris[m] = {lemma: test_score[i] for i,lemma in index2target.items()}
            test_score_mean, test_score_std = np.mean(test_score), np.std(test_score)
            #display(test_score)
            print('test_score_mean', 'test_score_std', test_score_mean, test_score_std)

            # Get best models
            test_estimators = cv_results['estimator']
            top_models = [estimator.model_top_ for estimator in test_estimators]
            #print('top_models', top_models)
            print('top_models unique', np.unique(top_models))
            print('top_models counts', Counter(top_models).most_common())
            
            #top_models_variance = len(set(top_models))/len(top_models)
            print('number top_models', len(np.unique(top_models)))
            models_to_plot[m] = top_models
            print('mean cluster_no top_models unique (true mean number of clusters = 2.75)', [df_results_reduced[df_results_reduced['model'].eq(tm)]['cluster_no'].mean() for tm in np.unique(top_models)])
            print('')





        # Models selected by CV
        method2modelscnt = {method: Counter(models) for method,models in models_to_plot.items()}
        df_results_reduced['cnt'] = df_results_reduced.apply(lambda r: method2modelscnt[r.method][r.model], axis=1)

        q = df_results_reduced[df_results_reduced.cnt>0]
        q2 = q.drop(columns=['ARI','GC_Spearmanr', 'BC_F1', 'word','cluster_no', 'clustering_pred', 'clustering_gold', 'BC_pred', 'BC_gold', 'GC_pred', 'GC_gold']).groupby('model').first().sort_values(by=['method','cnt'], ascending=False)

        print("\nModels selected by CV:")
        print(q2)
        os.makedirs(f"./clustering_evaluation/{ds}/{metric}", exist_ok=True)
        os.makedirs(f"./clustering_evaluation/{ds}_cleaned/{metric}", exist_ok=True)
        if cleaned_gold==False:
            q2.to_csv(f"./clustering_evaluation/{ds}/{metric}/models_selected_by_cv.csv", index=False)   # save
        else:
            q2.to_csv(f"./clustering_evaluation/{ds}_cleaned/{metric}/models_selected_by_cv.csv", index=False)   # save


        print('\n\nWARNING! Biased estimates of ARI: taking the most frequently selected configuration for each method!')
        biased_estimates = df_results_reduced[df_results_reduced.model.isin([method2modelscnt[m].most_common(1)[0][0] 
                                                        for m in ['wsbm','correlation', 'k-means', 'spectral', 'agglomerative']])].groupby('model').ARI.mean()
        print(biased_estimates)




        # Comparison of clustering methods: for each word we take its cross-validated ARI 

        df_cv = pd.DataFrame.from_records( ((method, lemma, ari) for method, lemma2ari in method2cvaris.items() 
                            for lemma,ari in lemma2ari.items()), columns=['method','lemma','ARI'])
        print('\n\nComparison of Clustering Methods: cross-validated ARI for each word')
        print(df_cv)




        sns.set_context(context='poster')


        q = df_cv[df_cv.method.isin(['wsbm','correlation', 'k-means', 'spectral', 'agglomerative'])]   
        q.method = q.method.replace('wsbm','WSBM').replace('correlation', 'CC').replace('agglomerative','AGGL')
        method_order = ['WSBM','CC', 'k-means', 'spectral', 'AGGL']
        g=sns.catplot(data=q.sort_values(by='lemma'), 
                        y='ARI', x='lemma', hue='method', hue_order=method_order, kind='bar', orient='v', legend_out=True, aspect=5)
        g.set_ylabels(f'{metric}')
        g.set_xticklabels(rotation=90)

        os.makedirs(f"./clustering_evaluation/{ds}/{metric}", exist_ok=True)
        os.makedirs(f"./clustering_evaluation/{ds}_cleaned/{metric}", exist_ok=True)
        if cleaned_gold==False:
            g.savefig(f'./clustering_evaluation/{ds}/{metric}/barplot_bestmodels_perword.pdf')
        else:
            g.savefig(f'./clustering_evaluation/{ds}_cleaned/{metric}/barplot_bestmodels_perword.pdf')

        ari_mean = q.groupby('method').ARI.mean()
        print(f'\nMean crossvalidated {metric}:')
        print(ari_mean)
        ari_mean_df = ari_mean.reset_index().rename(columns={'ARI': f'{metric}'})
        if cleaned_gold==False:
            ari_mean_df.to_csv(f"./clustering_evaluation/{ds}/{metric}/mean_crossvalidated_results.csv", index=False)   # save
        else:
            ari_mean_df.to_csv(f"./clustering_evaluation/{ds}_cleaned/{metric}/mean_crossvalidated_results.csv", index=False)   # save






    cvalidate('ARI')
    cvalidate('GC_Spearmanr')
    cvalidate('BC_F1')









def load_cleaned_gold_clustering(dataset):

    # Clean gold data with dgrnode with threshold 5
    
    df_cleaning = pd.DataFrame(columns=['identifier', 'cluster', 'model', 'strategy', 'threshold', 'lemma'])
    words = sorted(os.listdir(f'{dataset}/data/'))
    counter = 0

    ds = dataset.replace("./data/", "")
    with open(f"./cleaning_parameters/{ds}/cleaning_parameters.pkl", "rb") as file:
        parameters = dill.load(file)


    for word in words:
        #print(repr(word))
        if "dwug_es" in dataset:
            word = unicodedata.normalize('NFD', word)
        else:
            word = unicodedata.normalize('NFC', word)

        if os.path.exists(f'{dataset}/clusters/opt/{word}.csv'):
            clusters = pd.read_csv(f'{dataset}/clusters/opt/{word}.csv', sep="\t")
        else:
            clusters = pd.read_csv(f'{dataset}/clusters/{word}.tsv', sep="\t")
        

        # load graph
        if os.path.exists(f'{dataset}/graphs/opt/{word}'):
            with open(f'{dataset}/graphs/opt/{word}.csv', 'rb') as f:
                graph = pickle.load(f)
        else:
            with open(f'{dataset}/graphs/{word}', 'rb') as f:
                graph = pickle.load(f)
        # add cluster information to graph
        clusters_df = pd.read_csv(f'{dataset}/clusters/opt/{word}.csv', sep='\t')
        #print(clusters_df)
        identifier2cluster = dict(zip(clusters_df['identifier'], clusters_df['cluster']))
        for node in graph.nodes():
            identifier = graph.nodes[node]["identifier"]
            if identifier in identifier2cluster:
                graph.nodes[node]["cluster"] = identifier2cluster[identifier]
            else:
                graph.nodes[node]["cluster"] = None


        with open(f'{dataset}/annotators.csv', encoding='utf-8') as csvfile: 
            reader = csv.DictReader(csvfile, delimiter='\t',quoting=csv.QUOTE_NONE,strict=True)
            annotators = [row['annotator'] for row in reader]
        
        # remove nan edges 
        nan_edges = get_nan_edges(graph)
        graph.remove_edges_from(nan_edges)
        
        
        # clean graph 
        methods = ["dgrnode"]
        for method in methods:
            parameter = 5
            model = str(method) + "_" + str(parameter)
            g = graph.copy()
            #print('Input graph: ', g)
            g.graph['cleaning_stats'] = {}

            g = clean_graph(g, method, annotators, parameter)
            
            #print('Cleaned graph: ', g)

            for node in g.nodes:
                try:
                    cluster = clusters.loc[clusters['identifier'] == node, 'cluster'].values[0]     # cluster of node 
                except (IndexError):    # node with no cluster assignment
                    continue
                # insert new row to df_cleaning 
                new_row = pd.DataFrame({'identifier': [node], 'cluster': [cluster], 'model': [model], 'strategy': [method], 
                                        'threshold': [parameter], 'lemma': [word]})
                df_cleaning = pd.concat([df_cleaning, new_row], ignore_index=True)
                #df_cleaning.loc[len(df_cleaning)] = [node, cluster, model, method, parameter, word]
            #print(df_cleaning)
            #print(word, method, parameter)
            #quit()
        counter +=1
        print(counter)






    # load Gold Clustering (uncleaned)
    df_dwug_de = pd.DataFrame()                                                         # gold clustering 
    for p in Path(f'{dataset}/').glob('clusters/opt/*.csv'):    
        lemma = str(p).replace('\\', '/').split('/')[-1].replace('.csv','')
        lemma = unicodedata.normalize('NFC', lemma)
        df = pd.read_csv(p, delimiter='\t', quoting=3, na_filter=False)
        df['lemma'] = lemma
        df_dwug_de = pd.concat([df_dwug_de, df])    

    # Extract grouping (time) information
    df_dwug_de_uses = pd.DataFrame()
    for p in Path(f'{dataset}/data').glob('*/uses.csv'):
        df_dwug_de_uses = pd.concat([df_dwug_de_uses, pd.read_csv(p, delimiter='\t', quoting=3, na_filter=False)])


    df_dwug_de = df_dwug_de.merge(df_dwug_de_uses[['identifier', 'lemma', 'grouping']], how='left',      
                                                on = ['identifier', 'lemma'])
    # columns: ['identifier', 'cluster', 'lemma', 'grouping']
    
    print(f'df_dwug_de: \n{df_dwug_de}\n')




    # only nodes left after cleaning

    df_dwug_de_cleaned = df_dwug_de.merge(df_cleaning[['identifier', 'lemma', 'cluster']], how='right', 
                                          on=['identifier', 'lemma', 'cluster'])
    print(f'df_dwug_de cleaned: \n{df_dwug_de_cleaned}\n')


    return df_dwug_de_cleaned







if __name__=="__main__":
    dataset = "./data/dwug_de"
    evaluate_clustering(dataset, cleaned_gold=False)