
import pandas as pd
import requests
from pathlib import Path
import unicodedata
import numpy as np
from sklearn.metrics import adjusted_rand_score, rand_score
import matplotlib.pyplot as plt
import os
from download_data import download_paper_datasets, download_new_datasets
from cleaning import clean_graphs



"""
Paper: https://openreview.net/pdf?id=BlbrJvKv6L 
Paper Code: https://github.com/Garrafao/wug_cluster_clean/blob/main/analyze_semeval_de1.ipynb 
WUG graph2clean2.py: https://github.com/Garrafao/WUGs/blob/main/scripts/graph2clean2.py 
"""


"""
Cleaning evaluation of dwug_de 2.3.0 (Paper version) and 3.0.0 with ARI and RI
"""

#download_new_datasets()
#dataset = './data/dwug_de'                     # new dwug_de and dwug_de_sense versions 
#dwug_de_sense = './data/dwug_de_sense'

#download_paper_datasets()
#dataset = './paper_data/dwug_de'                # paper versions of dwug_de and dwug_de_sense
#dwug_de_sense = './paper_data/dwug_de_sense'




"""
Evaluate cleaning with ARI against dwug_de_sense
"""
def evaluate_cleaning(dataset, dwug_de_sense):

    df_dwug_de = pd.DataFrame()
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
    #display(df_dwug_de_uses)

    df_dwug_de = df_dwug_de.merge(df_dwug_de_uses[['identifier', 'grouping']], how='left',      # columns: ['identifier', 'cluster', 'lemma', 'grouping']
                                                left_on='identifier', right_on="identifier")
    




    # Get some data

    df_dwug_de_sense = pd.DataFrame()
    for p in Path(f'{dwug_de_sense}/labels/').glob('*/maj_3/labels_senses.csv'): 
        lemma = str(p).replace('\\', '/').split('/')[-3] # for windows
        lemma = unicodedata.normalize('NFC', lemma)
        df = pd.read_csv(p, delimiter='\t', quoting=3, na_filter=False)
        df['lemma'] = lemma
        df_dwug_de_sense = pd.concat([df_dwug_de_sense, df])        # columns: ['identifier', 'label', 'lemma']


    df_dwug_de_sense_clean = df_dwug_de_sense[~df_dwug_de_sense['label'].eq('-1')]      # filter uses with label -1 out


    #df_cleaning = pd.read_pickle("analyze_semeval_de1_df_cleaning.pkl")
    if "paper_data" in dataset:
        ds = dataset.replace("./paper_data/", "")
        df_cleaning = pd.read_pickle(f"./cleaning_parameter_grids/paper/dwug_de/cleaning_parameter_grid.pkl")
    else: 
        ds = dataset.replace("./data/", "")
        df_cleaning = pd.read_pickle(f"./cleaning_parameter_grids/{ds}/cleaning_parameter_grid.pkl")
    #print(df_cleaning)

        
    # check per lemma

    # df_dwug_de: ['identifier', 'cluster', 'lemma', 'grouping'], df_dwug_de_sense_clean: ['identifier', 'label', 'lemma']

    # df: without label=-1 and without cluster=-1
    df = df_dwug_de_sense_clean.merge(df_dwug_de.query("cluster!=-1"), on=['lemma','identifier'], how='inner', validate='1:1')
    # columns: ['identifier', 'label', 'lemma', 'cluster', 'grouping']
    print(df)
            




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
                                'ntargets':1, 'nuses':len(r), 'nsenses':r.label.nunique(), 
                                'nclusters':r.cluster.nunique(),
                                'ari':adjusted_rand_score(r.label, r.cluster), 
                                'ri':rand_score(r.label, r.cluster),
                                '(1-ari)':1-adjusted_rand_score(r.label, r.cluster), 
                                '(1-ri)':1-rand_score(r.label, r.cluster),
                            }, index=[0]), include_groups=False).droplevel(level=1)
        if original_stats is not None:
            # restore words from original_stats that are absent in l2stat (all uses removed)
            l2stat = l2stat + original_stats * 0 
            zero_cols = ['ntargets','nuses','nsenses','nclusters']
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
            #print(l2stat)
            #print(df)
            #quit()
        return l2stat


    #________________________________________
    original_stats = get_perlemma_stats(df)     # columns: ('lemma'), ['ntargets', 'nuses', 'nsenses', 'nclusters', 'ari', 'ri', '(1-ari)', '(1-ri)']
    print(original_stats)
    print(original_stats.columns.tolist())
    # df_cleaning.strategy.value_counts()
    # df_cleaning.query('strategy=="degreeremove"').model.value_counts()
    dfm = df.drop(columns='cluster').merge(df_cleaning[['identifier','cluster','model','strategy']], 
                                    on=['identifier'], how='inner')
    print(df)
    print(dfm)                                  # columns: ['identifier', 'label', 'lemma', 'grouping', 'cluster', 'model', 'strategy']
    print(len(dfm))
    tdf = dfm.groupby(['model','strategy'])
    print(len(tdf))

    pdf = dfm.groupby(['model','strategy']).apply(lambda x: get_perlemma_stats(x, original_stats), include_groups=False).reset_index()
    print(len(pdf))     # 2304
    print(len(pdf)/24)     
    #print(pdf.columns.tolist())     
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
    # same columns






    # Plot

    import seaborn as sns
    sns.set_context('talk')

    strategy_order = ['stdnode','dgrnode','clustersize','cntcluster','random']
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
                    x='Δnuses/nuses', y='ari', hue='strategy', style='strategy',
                        hue_order=strategy_order, style_order=strategy_order,
                        col='lemma',col_wrap=6, kind='line', errorbar=('ci',95), 
                    markers=True)
    #     g.map_dataframe(sns.scatterplot, 'Δnuses/nuses','ari','strategy', hue_order=strategy_order[:1])
        for t in g.legend.texts:
            t.set_text(method2papername[t.get_text()])
        if "paper_data" in dataset:
            os.makedirs(f'cleaning_results/ari/paper_versions/{ds}/', exist_ok=True)
            g.savefig(f'cleaning_results/ari/paper_versions/{ds}/cleaning_individual_ari.pdf')
        else:
            os.makedirs(f'cleaning_results/ari/{ds}/', exist_ok=True)
            g.savefig(f'cleaning_results/ari/{ds}/cleaning_individual_ari.pdf')



    # Average across lemmas

    # 'abgebrüht','zersetzen' are ideally clustered right from the start, removing nodes cannot change anything
    # 'artikulieren' has 1 sense only, hence, clustering metrics behave strangely
    # !!!TODO: remove words with 1 sense (ARI is a bad metric to evaluate clustering quality for them), 
    # but leave two other words defining relative error change as 0.0
    # filter_lemmas = ['abgebrüht','zersetzen','artikulieren'] 
    filter_lemmas = ['artikulieren'] 
    plotdf = plotdf[~plotdf.lemma.isin(filter_lemmas)]
    #print(len(plotdf.groupby(['model'])))

    # plotdf.loc[plotdf.nuses==0,'lemma'] = None  # when calculating nunique for lemma, None will be ignored
    # rows with lemma=None do not affect lemma:nunique; rows with NaN metric values do not affect the averages
    mpdf = plotdf.groupby(['model']).agg({'strategy':'first','ntargets':'sum','nuses':'sum','nsenses':'sum', 'nclusters':'sum'}|
                                {c:'mean' for c in plotdf.columns if 'ri' in c or 'Δ' in c})
    # mpdf.rename(columns={'lemma':'ntargets'}, inplace=True)
    mpdf['size'] = mpdf.strategy.apply(lambda x: 2 if x in {'collapse','stdedge'} else 1)  # dot sizes
    mpdf






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


    # appendix: compare nsenses and nclusters
    g = plot_mpdf(mpdf, y='Δnclusters/nclusters', legend=True)
    sns.move_legend(g, 'lower center', bbox_to_anchor=(0.4, 1), frameon=True, ncol=2)
    if "paper_data" in dataset:
        os.makedirs(f'cleaning_results/ari/paper_versions/{ds}/', exist_ok=True)
        g.savefig(f'cleaning_results/ari/paper_versions/{ds}/app_nclusters_avg.pdf')
    else:
        os.makedirs(f'cleaning_results/ari/{ds}/', exist_ok=True)
        g.savefig(f'cleaning_results/ari/{ds}/app_nclusters_avg.pdf')

    
    eval_methods = ['ari', 'ri']
    for eval_method in eval_methods:

        # 1st figure in the main part
        g = plot_mpdf(mpdf, y=f'Δ(1-{eval_method})/(1-{eval_method})')
        sns.move_legend(g, 'lower center', bbox_to_anchor=(0.4, 1), frameon=True, ncol=2)
        if "paper_data" in dataset:
            os.makedirs(f'cleaning_results/ari/paper_versions/{ds}/', exist_ok=True)
            g.savefig(f'cleaning_results/ari/paper_versions/{ds}/cleaning_{eval_method}.pdf')
        else:
            os.makedirs(f'cleaning_results/ari/{ds}/', exist_ok=True)
            g.savefig(f'cleaning_results/ari/{ds}/cleaning_{eval_method}.pdf')

            
        # plot sizecluster values on x axis
        g = plot_mpdf_sizecluster_values_x(mpdf, y=f'Δ(1-{eval_method})/(1-{eval_method})')
        sns.move_legend(g, 'lower center', bbox_to_anchor=(0.4, 1), frameon=True, ncol=2)
        
        if "paper_data" in dataset:
            os.makedirs(f'cleaning_results/ari/paper_versions/{ds}/', exist_ok=True)
            g.savefig(f'cleaning_results/ari/paper_versions/{ds}/{eval_method}_sizecluster_values_x_avg.pdf')
        else:
            os.makedirs(f'cleaning_results/ari/{ds}/', exist_ok=True)
            g.savefig(f'cleaning_results/ari/{ds}/{eval_method}_sizecluster_values_x_avg.pdf')
        

        # plot sizecluster values on y axis (x='Δnuses/nuses')
        g = plot_mpdf_sizecluster_values_y(mpdf)
        sns.move_legend(g, 'lower center', bbox_to_anchor=(0.4, 1), frameon=True, ncol=2)
        
        if "paper_data" in dataset:
            os.makedirs(f'cleaning_results/ari/paper_versions/{ds}/', exist_ok=True)
            g.savefig(f'cleaning_results/ari/paper_versions/{ds}/{eval_method}_sizecluster_values_y_avg.pdf')
        else:
            os.makedirs(f'cleaning_results/ari/{ds}/', exist_ok=True)
            g.savefig(f'cleaning_results/ari/{ds}/{eval_method}_sizecluster_values_y_avg.pdf')








if __name__=="__main__":
    dataset = './paper_data/dwug_de'                # paper versions of dwug_de and dwug_de_sense
    dwug_de_sense = './paper_data/dwug_de_sense'
    evaluate_cleaning(dataset, dwug_de_sense)

    dataset = './data/dwug_de'                # new versions of dwug_de and dwug_de_sense
    dwug_de_sense = './data/dwug_de_sense'
    evaluate_cleaning(dataset, dwug_de_sense)