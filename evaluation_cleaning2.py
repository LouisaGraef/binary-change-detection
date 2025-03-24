import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from pathlib import Path
import unicodedata
import numpy as np
import matplotlib.pyplot as plt
import os
from download_data import download_paper_datasets, download_new_datasets
from cleaning import clean_graphs
from modules import get_cluster_stats
import pickle
from cluster_ import get_clusters
from correlation import Loss
from tqdm import tqdm


"""
Paper: https://openreview.net/pdf?id=BlbrJvKv6L 
Paper Code: https://github.com/Garrafao/wug_cluster_clean/blob/main/analyze_semeval_de1.ipynb 
            https://github.com/Garrafao/wug_cluster_clean/blob/main/Nikolay_analyze_semeval_de1.ipynb 
WUG graph2clean2.py: https://github.com/Garrafao/WUGs/blob/main/scripts/graph2clean2.py 
"""



"""
Evaluate cleaning 
"""
def evaluate_cleaning2(dataset):

    df_dwug_de = pd.DataFrame()
    
    if os.path.exists(f'{dataset}/clusters/opt'):
        clusters_path = Path(f'{dataset}/').glob('clusters/opt/*.csv')
    else:
        clusters_path = Path(f'{dataset}/').glob('clusters/*.tsv')

    for p in clusters_path:    
        lemma = str(p).replace('\\', '/').split('/')[-1].replace('.csv','').replace('.tsv','')
        if "dwug_es" in dataset:
            lemma = unicodedata.normalize('NFD', lemma)
        else:
            lemma = unicodedata.normalize('NFC', lemma)
        df = pd.read_csv(p, delimiter='\t', quoting=3, na_filter=False)
        df['lemma'] = lemma
        df_dwug_de = pd.concat([df_dwug_de, df])    

    # Extract grouping (time) information
    df_dwug_de_uses = pd.DataFrame()
    for p in Path(f'{dataset}/data').glob('*/uses.csv'):
        df_dwug_de_uses = pd.concat([df_dwug_de_uses, pd.read_csv(p, delimiter='\t', quoting=3, na_filter=False)])

    #print(df_dwug_de_uses[df_dwug_de_uses['lemma'] == "下海"]['identifier'].value_counts())
    #print(df_dwug_de[df_dwug_de['lemma'] == "下海"])
    #print(df_dwug_de)
    #print(df_dwug_de_uses)

    df_dwug_de = df_dwug_de.merge(df_dwug_de_uses[['identifier', 'lemma', 'grouping']], how='left',      # columns: ['identifier', 'cluster', 'lemma', 'grouping']
                                                on = ['identifier', 'lemma'])
    
    #print(df_dwug_de[df_dwug_de['lemma'] == "下海"])
    #print(len(df_dwug_de[df_dwug_de['lemma'] == "下海"]))



    # Get some data

    #df_cleaning = pd.read_pickle("analyze_semeval_de1_df_cleaning.pkl")
    if "paper_data" in dataset:
        ds = dataset.replace("./paper_data/", "")
        df_cleaning = pd.read_pickle(f"./cleaning_parameter_grids/paper/dwug_de/cleaning_parameter_grid.pkl")
    else: 
        ds = dataset.replace("./data/", "")
        df_cleaning = pd.read_pickle(f"./cleaning_parameter_grids/{ds}/cleaning_parameter_grid.pkl")
    #print(df_cleaning)




    """
    https://github.com/Garrafao/WUGs/blob/main/scripts/modules.py
    """
    def get_cluster_stats2(graph):
        """
        Get clusters with conflicting judgments.       
        :param G: graph
        :param stats: dictionary with conflicts
        :param threshold: threshold
        :return :
        """
        try:
            clusters, _, _ = get_clusters(graph, is_include_noise = False)
        except KeyError:
            print('No clusters found.')
            return {}
        
        noise, _, _ = get_clusters(graph, is_include_noise = True, is_include_main = False)
        G_clean = graph.copy()    
        G_clean.remove_nodes_from([node for cluster in noise for node in cluster])
        stats = {}
        max_error = max(2.5-1, 4-2.5)

        n2i = {node:i for i, node in enumerate(G_clean.nodes())}
        n2c = {n2i[node]:i for i, cluster in enumerate(clusters) for node in cluster}
        
        edges_positive = set([(n2i[i],n2i[j],w-2.5) for (i,j,w) in G_clean.edges.data("weight") if w >= 2.5])
        edges_negative = set([(n2i[i],n2i[j],w-2.5) for (i,j,w) in G_clean.edges.data("weight") if w < 2.5])
        valid_edges = len(edges_positive) + len(edges_negative)
        
        cluster_state = np.array([n2c[n] for n in sorted(n2c.keys())])
        loss = Loss('linear_loss', edges_positive=edges_positive, edges_negative=edges_negative).loss(cluster_state)

        stats['loss'] = loss
        stats['loss_normalized'] = loss/(valid_edges*max_error) if (valid_edges*max_error) != 0.0 else 0.0

        between_conflicts = Loss('binary_loss', edges_positive=edges_positive, edges_negative=edges_negative, signs=['pos']).loss(cluster_state)
        within_conflicts = Loss('binary_loss', edges_positive=edges_positive, edges_negative=edges_negative, signs=['neg']).loss(cluster_state)
        stats['conflicts'] = between_conflicts + within_conflicts
        stats['conflicts_normalized'] = stats['conflicts']/valid_edges if valid_edges != 0.0 else 0.0
        stats['conflicts_between_clusters'] = between_conflicts
        stats['conflicts_within_clusters'] = within_conflicts

        edges_min = set([(n2i[i],n2i[j],w) for (i,j,w) in G_clean.edges.data("weight") if w == 1])
        edges_max = set([(n2i[i],n2i[j],w) for (i,j,w) in G_clean.edges.data("weight") if w == 4])
        edges_min_no = len(edges_min)
        edges_max_no = len(edges_max)
        edges_min_max_no = edges_min_no + edges_max_no
        loss_min = Loss('binary_loss_poles', edges_min=edges_min, edges_max=edges_max, signs=['min']).loss(cluster_state)
        loss_max = Loss('binary_loss_poles', edges_min=edges_min, edges_max=edges_max, signs=['max']).loss(cluster_state)
        win_min = edges_min_no - loss_min
        win_max = edges_max_no - loss_max
        win_min_max = win_min + win_max
        stats['win_min_normalized'] = win_min / edges_min_no if edges_min_no != 0.0 else float('nan') 
        stats['win_max_normalized'] = win_max / edges_max_no if edges_max_no != 0.0 else float('nan') 
        stats['win_min_max_normalized'] = win_min_max / edges_min_max_no if edges_min_max_no != 0.0 else float('nan')   
        
        return stats



    """
    Get eval_method of one lemma.
    """
    def get_eval_method(lemma, identifiers, eval_method):

        # load graph
        if os.path.exists(f'{dataset}/graphs/opt/{lemma}'):
            with open(f'{dataset}/graphs/opt/{lemma}', 'rb') as f:
                graph = pickle.load(f)
        else:
            with open(f'{dataset}/graphs/{lemma}', 'rb') as f:
                graph = pickle.load(f)
            # add cluster information to graph
            clusters_df = pd.read_csv(f'{dataset}/clusters/{lemma}.tsv', sep='\t')
            #print(clusters_df)
            identifier2cluster = dict(zip(clusters_df['identifier'], clusters_df['cluster']))
            for node in graph.nodes():
                identifier = graph.nodes[node]["identifier"]
                if identifier in identifier2cluster:
                    graph.nodes[node]["cluster"] = identifier2cluster[identifier]
                else:
                    graph.nodes[node]["cluster"] = None

        graph = graph.copy()

        # remove nodes that are not in the cleaned graph
        #valid_identifiers = set(df['identifier'])     # all node identifiers of the cleaned graph
        valid_identifiers = set(identifiers)     # all node identifiers of the cleaned graph
        nodes_to_remove = [node for node in graph.nodes() if node not in valid_identifiers]
        graph.remove_nodes_from(nodes_to_remove)
        
        cleaned_graph_cluster_stats = get_cluster_stats2(graph)   
        eval_stat = cleaned_graph_cluster_stats[f'{eval_method}']

        return eval_stat



    def get_perlemma_stats(df, original_stats=None):
        """
        Given a dataframe with usages (before or after filtering) calculates per-lemma statistics: number of usages,
        sense, clusters, ri, ari and their complements (1-ri), (1-ari). 
        If original_stats is given, it also calculates the absolute and relative changes in these statistics.
        If there are no uses in df for a particular lemma that is present in original_stats, we the number
        of usages, senses and clusters are considered being equal to 0 and the quality metrics equal to NaN.
        This way when calculating the averages across lemmas (outside this function) we will take the removed
        usages, senses and clusters for these lemmas into account, but not the quality metrics which are not defined
        for such lemmas.
        """
        l2stat = df.groupby('lemma').apply(lambda r: 
                            pd.DataFrame({
                                'ntargets':1, 'nuses':len(r), 
                                'nclusters':r.cluster.nunique(),
                                #'ari':adjusted_rand_score(r.label, r.cluster), 
                                f'{eval_method}':get_eval_method(r.name, r.identifier, eval_method), 
                                #'ri':rand_score(r.label, r.cluster),
                                #'(1-ari)':1-adjusted_rand_score(r.label, r.cluster), 
                                f'(1-{eval_method})':1-get_eval_method(r.name, r.identifier, eval_method), 
                                #'(1-ri)':1-rand_score(r.label, r.cluster),
                            }, index=[0]), include_groups=False).droplevel(level=1)
        if original_stats is not None:
            # restore words from original_stats that are absent in l2stat (all uses removed)
            l2stat = l2stat + original_stats * 0 
            zero_cols = ['ntargets','nuses','nclusters']
            # replace NaNs with 0 in zero_cols, but metrics should be undefined not to affect the average
            l2stat.loc[l2stat['nuses'].isnull(), zero_cols] = 0
            # calculate the absolute and the relative differences, when the original value of some metric is 0
            # and the absolute difference is 0 we define the relative difference as 0 (this helps when the original
            # (1-ari) or (1-ri) is 0, i.e. the original clustering is already perfect, since it cannot get worse after
            # removing nodes).
            l2stat_diff = (l2stat - original_stats)
            l2stat_reldiff = l2stat_diff.where(l2stat_diff==0.0, l2stat_diff / original_stats)
            l2stat_diff.rename(columns={c:f'Δ{c}' for c in l2stat_diff.columns if c!='lemma'}, inplace=True)
            l2stat_reldiff.rename(columns={c:f'Δ{c}/{c}' for c in l2stat_reldiff.columns if c!='lemma'}, inplace=True)
            # return a wide dataframe with the metrics, their absolute and relative changes
            l2stat = pd.concat([l2stat, l2stat_diff, l2stat_reldiff], axis=1)
        return l2stat




    
    # df_dwug_de: ['identifier', 'cluster', 'lemma', 'grouping']

    # df: without cluster=-1
    df = df_dwug_de.query("cluster!=-1")
    print(f"Dataset: {dataset}")
    #print(df)
    #print(df[df['lemma'] == "下海"])       # columns: ['identifier', 'cluster', 'lemma', 'grouping']        # 39
    #print(len(df[df['lemma'] == "下海"]))       # columns: ['identifier', 'cluster', 'lemma', 'grouping']       # 39



    #eval_methods = ['win_min_max_normalized', 'conflicts', 'conflicts_normalized', 'conflicts_between_clusters', 'conflicts_within_clusters',
    #              'win_min_normalized', 'win_max_normalized']      
    eval_methods = ['win_min_max_normalized', 'conflicts_normalized']
    for eval_method in tqdm(eval_methods, desc="Evaluating"):
        tqdm.write(f"Evaluation: {eval_method}")

        #________________________________________
        # df: before cleaning
        original_stats = get_perlemma_stats(df)  # columns: ('lemma'), ['ntargets', 'nuses', 'nclusters', 'conflicts_normalized', '(1-conflicts_normalized)']
        #print(original_stats)
        #print(original_stats.columns.tolist())          # ['ntargets', 'nuses', 'nclusters', 'win_min_max_normalized', '(1-win_min_max_normalized)']
        # df_cleaning.strategy.value_counts()
        # print(df_cleaning.query('strategy=="dgrnode"').model.value_counts())
        if "chiwug" in dataset:
            dfm = df.merge(df_cleaning[['identifier','cluster','model','strategy']],                       
                                            on=['identifier', 'cluster'], how='inner').drop_duplicates(subset=['identifier', 'lemma', 'model'])
        else:
            dfm = df.drop(columns='cluster').merge(df_cleaning[['identifier','cluster','model','strategy']],    
                                            on=['identifier'], how='inner')
        #print(df)
        #print(dfm)                                  # columns: ['identifier', 'label', 'lemma', 'grouping', 'cluster', 'model', 'strategy']
        print(len(dfm))
        tdf = dfm.groupby(['model','strategy'])
        print(len(tdf))

        #print(dfm[(dfm["lemma"] == "下海") & (dfm["model"] == "stdnode_0.08226495726495725")].sort_values(by="identifier"))
        #print(len(dfm[(dfm["lemma"] == "下海") & (dfm["model"] == "stdnode_0.08226495726495725")].sort_values(by="identifier")))    # 25

        pdf = dfm.groupby(['model','strategy']).apply(lambda x: get_perlemma_stats(x, original_stats)).reset_index()
        print(len(pdf))     # 4700
        print(len(pdf)/50)     
        
        #print(pdf.columns.tolist())     
        # ['model', 'strategy', 'lemma', 'ntargets', 'nuses', 'nclusters', 'win_min_max_normalized', '(1-win_min_max_normalized)', 
        # 'Δntargets', 'Δnuses', 'Δnclusters', 'Δwin_min_max_normalized', 'Δ(1-win_min_max_normalized)', 'Δntargets/ntargets', 'Δnuses/nuses', 
        # 'Δnclusters/nclusters', 'Δwin_min_max_normalized/win_min_max_normalized', 'Δ(1-win_min_max_normalized)/(1-win_min_max_normalized)']
        
        #pd.set_option("display.max_columns", None)  
        #pd.set_option("display.expand_frame_repr", False)  
        #print(pdf[pdf["lemma"] == "下海"])      # nuses max 40 
        #print(len(pdf[pdf["lemma"] == "下海"]))
        # print(len(pdf))     # 3168
        # print(pdf.columns.tolist())     
        # pd.set_option('display.max_rows', None) 
        # print(pdf.loc[pdf['strategy'] == 'cntcluster', ['model', 'Δnuses/nuses', 'Δ(1-conflicts_normalized)/(1-conflicts_normalized)']].sort_values(by='Δnuses/nuses'))
        
        # ['model', 'strategy', 'lemma', 'ntargets', 'nuses', 'nsenses', 'nclusters', 'ari', 'ri', '(1-ari)', '(1-ri)', 
        # 'Δntargets', 'Δnuses', 'Δnsenses', 'Δnclusters', 'Δari', 'Δri', 'Δ(1-ari)', 'Δ(1-ri)', 'Δntargets/ntargets', 'Δnuses/nuses', 
        # 'Δnsenses/nsenses', 'Δnclusters/nclusters', 'Δari/ari', 'Δri/ri', 'Δ(1-ari)/(1-ari)', 'Δ(1-ri)/(1-ri)']

        # pdf[pdf.isnull().any(axis=1)]     # shows rows with nuses = 0.0 (193 rows)
        # if the number of uses for a lemma is 0 after filtering, the metrics are undefined (NaNs); 
        # pandas will not take NaNs into account when averaging across target words, 
        # so the averages below are across target words that have usages survived after filtering



        # Add results for the random baseline 

        def annotate(l2stat, strategy, args):
            l2stat['strategy'] = strategy
            l2stat['model'] = strategy + '_' + args
            return l2stat

        pdf = pd.concat([pdf]+[annotate(
            get_perlemma_stats(df.groupby('lemma').sample(frac=frac, replace=False), original_stats).reset_index(), 
            'random', str(frac)+"_"+str(j))
                for frac in np.linspace(0.1,1.0,num=10) for j in range(100)], ignore_index=True)
        # print(pdf.columns.tolist())
        # print(pdf[['model', 'conflicts_normalized', '(1-conflicts_normalized)']])
        # same columns
        print(len(pdf)/50)
        #quit()






        # Plot

        import seaborn as sns
        sns.set_context('talk')

        #strategy_order = ['stdnode','degreeremove','clustersizemin','clusterconnectmin','random']
        strategy_order = ['stdnode','dgrnode','clustersize','cntcluster','random']
        # plotdf = pdf[pdf.strategy.isin(strategy_order)]
        plotdf = pdf


        # Individual plots for each lemma
        method2papername = {
            'stdnode':'stdnode',
            'dgrnode':'dgrnode',
            'clustersize':'sizecluster',
            'cntcluster':'cntcluster',
            'random':'random',
        }

        with sns.plotting_context('poster'):
            g = sns.relplot(data=plotdf, 
                        x='Δnuses/nuses', y=f'{eval_method}', hue='strategy', style='strategy',
                            hue_order=strategy_order, style_order=strategy_order,
                            col='lemma',col_wrap=6, kind='line', errorbar=('ci',95), 
                        markers=True)
        #     g.map_dataframe(sns.scatterplot, 'Δnuses/nuses','ari','strategy', hue_order=strategy_order[:1])
            for t in g.legend.texts:
                t.set_text(method2papername[t.get_text()])
            if "paper_data" in dataset:
                os.makedirs(f'cleaning_results/paper_versions/{ds}/{eval_method}/', exist_ok=True)
                g.savefig(f'cleaning_results/paper_versions/{ds}/{eval_method}/cleaning_individual_{eval_method}.pdf')
            else:
                os.makedirs(f'cleaning_results/{ds}/{eval_method}/', exist_ok=True)
                g.savefig(f'cleaning_results/{ds}/{eval_method}/cleaning_individual_{eval_method}.pdf')




        # Average across lemmas

        # 'abgebrüht','zersetzen' are ideally clustered right from the start, removing nodes cannot change anything
        # 'artikulieren' has 1 sense only, hence, clustering metrics behave strangely
        # !!!TODO: remove words with 1 sense (ARI is a bad metric to evaluate clustering quality for them), 
        # but leave two other words defining relative error change as 0.0
        # filter_lemmas = ['abgebrüht','zersetzen','artikulieren'] 
        filter_lemmas = ['artikulieren'] 
        plotdf = plotdf[~plotdf.lemma.isin(filter_lemmas)]

        # plotdf.loc[plotdf.nuses==0,'lemma'] = None  # when calculating nunique for lemma, None will be ignored
        # rows with lemma=None do not affect lemma:nunique; rows with NaN metric values do not affect the averages
        print(len(plotdf.groupby(['model'])))   # 1094 (dwug_de)
        mpdf = plotdf.groupby(['model']).agg({'strategy':'first','ntargets':'sum','nuses':'sum', 'nclusters':'sum'}|
                                    {c:'mean' for c in plotdf.columns if f'{eval_method}' in c or 'Δ' in c})
        # mpdf.rename(columns={'lemma':'ntargets'}, inplace=True)
        mpdf['size'] = mpdf.strategy.apply(lambda x: 2 if x in {'collapse','stdedge'} else 1)  # dot sizes




        # clustering quality metrics averaged across survived target words 
        def plot_mpdf(mpdf, x='Δnuses/nuses', y='Δ(1-ri)/(1-ri)', legend=True,aspect=1.5):
            g = sns.relplot(data=mpdf, 
                        x=x, y=y, hue='strategy',style='strategy',
                        hue_order=strategy_order, style_order=strategy_order,
                        markers=True, kind='line', errorbar=('ci',95), legend=legend, aspect=aspect)
            if legend:
                for t in g.legend.texts:
                    t.set_text(method2papername[t.get_text()])

            return g
        
        # plot sizecluster values on x axis
        def plot_mpdf_sizecluster_values_x(mpdf, y='Δ(1-ri)/(1-ri)', legend=True,aspect=1.5):
            mpdf = mpdf[mpdf.index.str.startswith('clustersize')].copy()
            mpdf['param'] = mpdf.index.str.split('_').str[-1].astype(float)
            x = 'param'
            g = sns.relplot(data=mpdf, 
                        x=x, y=y, hue='strategy',style='strategy',
                        hue_order=strategy_order, style_order=strategy_order,
                        markers=True, kind='line', errorbar=('ci',95), legend=legend, aspect=aspect)
            if legend:
                for t in g.legend.texts:
                    t.set_text(method2papername[t.get_text()])

            return g
        
        # plot sizecluster values on x axis
        def plot_mpdf_sizecluster_values_y(mpdf, x='Δnuses/nuses', legend=True,aspect=1.5):
            mpdf = mpdf[mpdf.index.str.startswith('clustersize')].copy()
            mpdf['param'] = mpdf.index.str.split('_').str[-1].astype(float)
            y = 'param'
            g = sns.relplot(data=mpdf, 
                        x=x, y=y, hue='strategy',style='strategy',
                        hue_order=strategy_order, style_order=strategy_order,
                        markers=True, kind='line', errorbar=('ci',95), legend=legend, aspect=aspect)
            if legend:
                for t in g.legend.texts:
                    t.set_text(method2papername[t.get_text()])

            return g


        sns.set_context('talk')
        sns.set_style("whitegrid")
        # 1st figure in the main part
        g = plot_mpdf(mpdf, y=f'Δ(1-{eval_method})/(1-{eval_method})')
        sns.move_legend(g, 'lower center', bbox_to_anchor=(0.4, 1), frameon=True, ncol=2)
        if "paper_data" in dataset:
            os.makedirs(f'cleaning_results/paper_versions/{ds}/{eval_method}', exist_ok=True)
            g.savefig(f'cleaning_results/paper_versions/{ds}/{eval_method}/cleaning_main1.pdf')
        else:
            os.makedirs(f'cleaning_results/{ds}/{eval_method}', exist_ok=True)
            g.savefig(f'cleaning_results/{ds}/{eval_method}/cleaning_main1.pdf')


        # appendix: compare nsenses and nclusters
        g = plot_mpdf(mpdf, y='Δnclusters/nclusters', legend=True)
        sns.move_legend(g, 'lower center', bbox_to_anchor=(0.4, 1), frameon=True, ncol=2)
        
        if "paper_data" in dataset:
            os.makedirs(f'cleaning_results/paper_versions/{ds}/{eval_method}', exist_ok=True)
            g.savefig(f'cleaning_results/paper_versions/{ds}/{eval_method}/app_nclusters_avg.pdf')
        else:
            os.makedirs(f'cleaning_results/{ds}/{eval_method}', exist_ok=True)
            g.savefig(f'cleaning_results/{ds}/{eval_method}/app_nclusters_avg.pdf')
        

        # plot sizecluster values on x axis
        g = plot_mpdf_sizecluster_values_x(mpdf, y=f'Δ(1-{eval_method})/(1-{eval_method})')
        sns.move_legend(g, 'lower center', bbox_to_anchor=(0.4, 1), frameon=True, ncol=2)
        
        if "paper_data" in dataset:
            os.makedirs(f'cleaning_results/paper_versions/{ds}/{eval_method}', exist_ok=True)
            g.savefig(f'cleaning_results/paper_versions/{ds}/{eval_method}/sizecluster_values_x_avg.pdf')
        else:
            os.makedirs(f'cleaning_results/{ds}/{eval_method}', exist_ok=True)
            g.savefig(f'cleaning_results/{ds}/{eval_method}/sizecluster_values_x_avg.pdf')
        

        # plot sizecluster values on y axis (x='Δnuses/nuses')
        g = plot_mpdf_sizecluster_values_y(mpdf)
        sns.move_legend(g, 'lower center', bbox_to_anchor=(0.4, 1), frameon=True, ncol=2)
        
        if "paper_data" in dataset:
            os.makedirs(f'cleaning_results/paper_versions/{ds}/{eval_method}', exist_ok=True)
            g.savefig(f'cleaning_results/paper_versions/{ds}/{eval_method}/sizecluster_values_y_avg.pdf')
        else:
            os.makedirs(f'cleaning_results/{ds}/{eval_method}', exist_ok=True)
            g.savefig(f'cleaning_results/{ds}/{eval_method}/sizecluster_values_y_avg.pdf')





if __name__=="__main__":
    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_la", "dwug_es", "chiwug",      
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]                               
    datasets = ["./data/" + dataset for dataset in datasets]

    dataset = './paper_data/dwug_de'
    evaluate_cleaning2(dataset)

    for dataset in datasets:
        evaluate_cleaning2(dataset)