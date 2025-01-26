
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

#download_new_datasets()
#dataset = './data/dwug_de'                     # new dwug_de and dwug_de_sense versions 
#dwug_de_sense = './data/dwug_de_sense'

#download_paper_datasets()
dataset = './paper_data/dwug_de'                # paper versions of dwug_de and dwug_de_sense
dwug_de_sense = './paper_data/dwug_de_sense'



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
targets = df_dwug_de_sense_clean['lemma'].unique()
target2index = {target:index for index, target in enumerate(targets)}
index2target = {index:target for index, target in enumerate(targets)}
indices = np.array(list(target2index.values())).reshape(-1, 1)
#print(indices)
labels = df_dwug_de_sense_clean['label']  


df_cleaning = pd.read_pickle("analyze_semeval_de1_df_cleaning.pkl")

print(df_cleaning)
quit()
df_cleaning = clean_graphs(dataset)
print(df_cleaning)
quit()





# Iterate over cleaning models, calculate ARI, other interesting statistics and random baseline model

gb_model = df_cleaning.groupby('model')    
groups_model = gb_model.groups
results = []
results_random = []
#degreeremove_keys = [k for k in groups_model.keys() if "degreeremove" in k]
for model in groups_model.keys():                   
    df_model = gb_model.get_group(model)
    #if not 'clustersizemin' in model:
    #    continue
    #display(df_model).bb   
    df_merged = df_model.merge(df_dwug_de_sense, how='outer', left_on=['identifier','lemma'], right_on=['identifier','lemma'])
    df_merged = df_merged.merge(df_dwug_de, how='outer', left_on=['identifier','lemma'], right_on=['identifier','lemma'], suffixes=('_clean', '_original'))
    gb_merged_lemma = df_merged.groupby('lemma')    
    groups_merged_lemma = gb_merged_lemma.groups
    #print(groups_merged_lemma.keys())
    assert len(groups_merged_lemma.keys())==50

    for lemma in groups_merged_lemma.keys():
        
        df_merged_lemma = gb_merged_lemma.get_group(lemma)
        strategy = df_merged_lemma['strategy'].tolist()[0]
        threshold = float(df_merged_lemma['threshold'].tolist()[0])
        
        # Calculate various data subsets needed below
        df_merged_lemma_cluster_original = df_merged_lemma[(~df_merged_lemma['cluster_original'].isnull())]
        df_merged_lemma_cluster_original_labels = df_merged_lemma[(df_merged_lemma['cluster_original']!=-1) & (~df_merged_lemma['cluster_original'].eq('-1')) & (~df_merged_lemma['cluster_original'].isnull())]
        df_merged_lemma_cluster_clean_labels = df_merged_lemma[(df_merged_lemma['cluster_clean']!=-1) & (~df_merged_lemma['cluster_clean'].eq('-1')) & (~df_merged_lemma['cluster_clean'].isnull())]
        df_merged_lemma_senses = df_merged_lemma[(~df_merged_lemma['label'].isnull())]
        df_merged_lemma_senses_labels = df_merged_lemma[(df_merged_lemma['label']!=-1) & (~df_merged_lemma['label'].eq('-1')) & (~df_merged_lemma['label'].isnull())]
        df_merged_lemma_senses_labels_cluster_clean_labels = df_merged_lemma_senses_labels[(df_merged_lemma_senses_labels['cluster_clean']!=-1) & (~df_merged_lemma_senses_labels['cluster_clean'].eq('-1')) & (~df_merged_lemma_senses_labels['cluster_clean'].isnull())] 

        if lemma == 'Ohrwurm':
            print(model, len(df_merged_lemma_cluster_clean_labels))

        if len(df_merged_lemma_senses) == 0: # skip lemmas without sense annotations
            continue

        # Calculate various statistics, e.g. number of uses left, number of clusters left, number of senses left after cleaning
        uses_original_all = len(df_merged_lemma_cluster_original)
        uses_original = len(df_merged_lemma_cluster_original_labels)
        uses_cleaned = len(df_merged_lemma_cluster_clean_labels)
        uses_removed = uses_original_all - uses_cleaned
        uses_left_percentage = uses_cleaned*100/uses_original_all
        clusters_original_all = len(df_merged_lemma_cluster_original['cluster_original'].unique())
        clusters_original = len(df_merged_lemma_cluster_original_labels['cluster_original'].unique())
        clusters_cleaned = len(df_merged_lemma_cluster_original_labels['cluster_clean'].unique())
        clusters_removed = clusters_original_all - clusters_cleaned
        clusters_left_percentage = clusters_cleaned*100/clusters_original_all
        senses_original_all = len(df_merged_lemma_senses['label'].unique())
        senses_original = len(df_merged_lemma_senses_labels['label'].unique())
        senses_cleaned = len(df_merged_lemma_senses_labels_cluster_clean_labels['label'].unique())
        senses_removed = senses_original_all - senses_cleaned
        senses_left_percentage = senses_cleaned*100/senses_original_all if senses_original_all >0 else np.nan
        clusters1_original = df_merged_lemma_cluster_original_labels[(df_merged_lemma_cluster_original_labels['grouping']==1)]['cluster_original'].unique().flatten().tolist()
        clusters2_original = df_merged_lemma_cluster_original_labels[(df_merged_lemma_cluster_original_labels['grouping']==2)]['cluster_original'].unique().flatten().tolist()
        cluster2change_label_original = {cluster:2 if (cluster in clusters1_original and cluster in clusters2_original) 
                                         else 1 for cluster in set(clusters1_original+clusters2_original)} 
        clusters1_clean = df_merged_lemma_cluster_clean_labels[(df_merged_lemma_cluster_clean_labels['grouping']==1)]['cluster_clean'].unique().flatten().tolist()
        clusters2_clean = df_merged_lemma_cluster_clean_labels[(df_merged_lemma_cluster_clean_labels['grouping']==2)]['cluster_clean'].unique().flatten().tolist()
        cluster2change_label_clean = {cluster:2 if (cluster in clusters1_clean and cluster in clusters2_clean) 
                                         else 1 for cluster in set(clusters1_clean+clusters2_clean)} 
        cluster_changes_left_percentage = len([c for c, l in cluster2change_label_clean.items() 
                                               if cluster2change_label_original[c]==l])*100/len(cluster2change_label_original)     
        senses1_original = df_merged_lemma_senses_labels[(df_merged_lemma_senses_labels['grouping']==1)]['label'].unique().flatten().tolist()
        senses2_original = df_merged_lemma_senses_labels[(df_merged_lemma_senses_labels['grouping']==2)]['label'].unique().flatten().tolist()
        sense2change_label_original = {cluster:2 if (cluster in senses1_original and cluster in senses2_original) 
                                         else 1 for cluster in set(senses1_original+senses2_original)} 
        senses1_clean = df_merged_lemma_senses_labels_cluster_clean_labels[(df_merged_lemma_senses_labels_cluster_clean_labels['grouping']==1)]['label'].unique().flatten().tolist()
        senses2_clean = df_merged_lemma_senses_labels_cluster_clean_labels[(df_merged_lemma_senses_labels_cluster_clean_labels['grouping']==2)]['label'].unique().flatten().tolist()
        sense2change_label_clean = {cluster:2 if (cluster in senses1_clean and cluster in senses2_clean) 
                                         else 1 for cluster in set(senses1_clean+senses2_clean)} 
        sense_changes_left_percentage = len([c for c, l in sense2change_label_clean.items()
                                               if sense2change_label_original[c]==l])*100/len(sense2change_label_original) 
        
        # Check for wrongly filtered data subsets
        assert len(cluster2change_label_original) == len(set(clusters1_original+clusters2_original))
        assert len(sense2change_label_clean) == len(set(senses1_clean+senses2_clean))
        assert not '-1' in sense2change_label_original and not -1 in sense2change_label_original
        assert not '-1' in cluster2change_label_original and not -1 in cluster2change_label_original
        assert not '-1' in sense2change_label_clean and not -1 in sense2change_label_clean
        assert not '-1' in cluster2change_label_clean and not -1 in cluster2change_label_clean
        assert not '-1' in df_merged_lemma_cluster_original_labels['cluster_original'].unique().tolist() and not -1 in df_merged_lemma_cluster_original_labels['cluster_original'].unique()
        assert not '-1' in df_merged_lemma_cluster_clean_labels['cluster_clean'].unique() and not -1 in df_merged_lemma_cluster_clean_labels['cluster_clean'].unique()
        assert not '-1' in df_merged_lemma_senses_labels['label'].unique() and not -1 in df_merged_lemma_senses_labels['label'].unique()
        assert not '-1' in df_merged_lemma_senses_labels_cluster_clean_labels['label'].unique() and not -1 in df_merged_lemma_senses_labels_cluster_clean_labels['label'].unique()
        assert not '-1' in df_merged_lemma_senses_labels_cluster_clean_labels['cluster_clean'].unique() and not -1 in df_merged_lemma_senses_labels_cluster_clean_labels['cluster_clean'].unique()
           
        # Calculate ARI results    
        data1 = df_merged_lemma_senses_labels_cluster_clean_labels['cluster_clean']
        data2 = df_merged_lemma_senses_labels_cluster_clean_labels['label']
        ari = adjusted_rand_score(data1, data2) if len(data1)>0 else np.nan  # to do: make sure that np.nan does not improve ARI with low sample numbers because of averaging effects
        uses_compared = len(data1) if len(data1)>0 else np.nan # not sure about this
        #print(' ', lemma, ari, len(data1))
        results.append({'model':model, 'lemma':lemma, 'ARI':ari, 'senses_left_percentage':senses_left_percentage, 
                        'cluster_changes_left_percentage':cluster_changes_left_percentage,
                        'sense_changes_left_percentage':sense_changes_left_percentage, 'clusters_left_percentage':clusters_left_percentage, 'uses_left_percentage':uses_left_percentage, 
                        'clusters_cleaned':clusters_cleaned, 'senses_cleaned':senses_cleaned, 'uses_compared':uses_compared, 'uses_original':uses_original, 
                        'uses_original_all':uses_original_all, 'uses_cleaned':uses_cleaned, 'uses_removed':uses_removed, 
                        'strategy':strategy, 'threshold': threshold})
        



        # Sample random baseline cleaning models
        #display(df_merged_lemma_cluster_original_labels)
        df_lemma_random = df_merged_lemma_cluster_original_labels.copy().reset_index(drop=True)
        indices = (~df_lemma_random['cluster_original'].isnull()) ### has no effect
        #display(df_lemma_random[indices]['cluster_original'].unique())
        indices_index = df_lemma_random.index[indices]
        assert len(indices_index) == len(df_lemma_random)
        assert not '-1' in df_lemma_random[indices]['cluster_original'].unique().tolist() and not -1 in df_lemma_random[indices]['cluster_original'].unique() and not np.isnan(df_lemma_random[indices]['cluster_original'].unique()).any()
        for resample in range(15):
            indices_sampled = np.random.choice(indices_index, size=uses_cleaned, replace=False)     
            #indices_sampled = np.random.choice(indices_index, size=30, replace=False) # for testing
            assert len(indices_sampled) == uses_cleaned
            #print(indices_sampled).lll
            df_sample = df_lemma_random.copy().loc[indices_sampled][['identifier', 'cluster_original', 'label', 'lemma']]
            # check above
            df_sample = df_sample.rename(columns={"cluster_original": "cluster_clean"})
            assert len(df_sample) == uses_cleaned
            df_sample_senses_labels_cluster_clean_labels = df_sample[(df_sample['label']!=-1) & (~df_sample['label'].eq('-1')) & (~df_sample['label'].isnull())] 
            assert not '-1' in df_sample_senses_labels_cluster_clean_labels['label'].unique().tolist() and not -1 in df_sample_senses_labels_cluster_clean_labels['label'].unique()
            assert not '-1' in df_sample_senses_labels_cluster_clean_labels['cluster_clean'].unique().tolist() and not -1 in df_sample_senses_labels_cluster_clean_labels['cluster_clean'].unique()
            # Calculate ARI results    
            data1 = df_sample_senses_labels_cluster_clean_labels['cluster_clean']
            data2 = df_sample_senses_labels_cluster_clean_labels['label']
            ari = adjusted_rand_score(data1, data2) if len(data1)>0 else np.nan
            #ari = rand_score(data1, data2) if len(data1)>0 else np.nan # for testing
            uses_compared = len(data1) if len(data1)>0 else np.nan # not sure about this
            #print(' ', lemma, ari, len(data1))
            results_random.append({'model':model, 'lemma':lemma, 'ARI':ari, 'senses_left_percentage':np.nan, 'cluster_changes_left_percentage':np.nan,
                            'sense_changes_left_percentage':np.nan, 'clusters_left_percentage':np.nan, 'uses_left_percentage':np.nan, 
                            'clusters_cleaned':np.nan, 'senses_cleaned':np.nan, 'uses_compared':uses_compared, 'uses_original':np.nan, 
                            'uses_original_all':np.nan, 'uses_cleaned':uses_cleaned, 'uses_removed':np.nan, 
                            'strategy':strategy, 'threshold': threshold, 'resample': resample}) # to do: calculate missing statistics, may be reuse/generalize code from above
            



df_results_cleaning = pd.DataFrame(results)
df_results_cleaning_random = pd.DataFrame(results_random)
#df_results.to_pickle("analyze_semeval_de1_df_results.pkl")
#display(df_results_cleaning)
df_results_cleaning_mean = df_results_cleaning.groupby(['model']).agg('mean', numeric_only=True).reset_index().sort_values(by='ARI', ascending=False)
df_results_cleaning_random_mean = df_results_cleaning_random.groupby(['model']).agg('mean', numeric_only=True).reset_index().sort_values(by='ARI', ascending=False)
#display(df_results_cleaning_mean)     
df_results_cleaning_std = df_results_cleaning.groupby(['model']).agg('std', numeric_only=True).reset_index().sort_values(by='ARI', ascending=False)
df_results_cleaning_random_std = df_results_cleaning_random.groupby(['model']).agg('std', numeric_only=True).reset_index().sort_values(by='ARI', ascending=False)
df_results_cleaning_mean = df_results_cleaning_mean.merge(df_results_cleaning_std, how='inner', left_on=['model'], right_on=['model'], suffixes=('_mean', '_std'))
df_results_cleaning_random_mean = df_results_cleaning_random_mean.merge(df_results_cleaning_random_std, how='inner', left_on=['model'], right_on=['model'], suffixes=('_mean', '_std'))
#display(df_results_cleaning_mean)     
def extract_strategy(model):
    return str(model).split('_')[0]
df_results_cleaning_mean['strategy'] = df_results_cleaning_mean['model'].apply(lambda x: extract_strategy(x))
df_results_cleaning_random_mean['strategy'] = df_results_cleaning_random_mean['model'].apply(lambda x: extract_strategy(x))
#print(df_results_cleaning_mean['uses_original_all_mean'].unique())
assert len(df_results_cleaning_mean['uses_original_all_mean'].unique()) == 1
#assert len(df_results_cleaning_random_mean['uses_original_all_mean'].unique()) == 1
print(df_results_cleaning_mean) 
print(df_results_cleaning_random_mean)
print(df_results_cleaning_mean.columns.tolist()) 
print(df_results_cleaning_random_mean.columns.tolist())





gb_strategy = df_results_cleaning_mean.groupby('strategy')    
groups_strategy = gb_strategy.groups
gb_strategy_random = df_results_cleaning_random_mean.groupby('strategy')
groups_strategy_random = gb_strategy_random.groups
results = []
reference_result = df_results_cleaning_mean[(df_results_cleaning_mean['strategy'].eq('degreeremove')) & (df_results_cleaning_mean['threshold_mean'] == 1.0)]   
#reference_result = df_results_cleaning_mean.iloc[0:1] # for testing
#display(reference_result)
reference_result_ari = reference_result['ARI_mean'].to_list()[0]

for strategy in groups_strategy.keys():
    #print(word)
    #if strategy == 'collapse':
    #    continue
    if strategy == 'degreeremove':
        df_strategy = gb_strategy.get_group(strategy).sort_values(by='threshold_mean', ascending=False)
        df_strategy_random = gb_strategy_random.get_group(strategy).sort_values(by='threshold_mean', ascending=False)
    else:
        df_strategy = gb_strategy.get_group(strategy).sort_values(by='threshold_mean', ascending=True)
        df_strategy_random = gb_strategy_random.get_group(strategy).sort_values(by='threshold_mean', ascending=True)

    random_baseline_color = 'darkgreen'    
    print(df_strategy)
    #plt.figure(figsize=(20, 7))
    #plt.title(strategy)
    fig, ax1 = plt.subplots() 

    ax1.set_xlabel('threshold') 
    ax1.set_ylabel('ARI (mean)') 
    #ax1.errorbar(df_strategy['threshold_mean'], df_strategy['ARI_mean'], fmt='-', label=strategy, yerr=df_strategy['ARI_std'], color = 'black')
    ax1.errorbar(df_strategy['threshold_mean'], df_strategy['ARI_mean'], fmt='-', label=strategy, color = 'black')
    ax1.errorbar(df_strategy_random['threshold_mean'], df_strategy_random['ARI_mean'], fmt='-', label='random baseline', yerr=df_strategy_random['ARI_std'], color = random_baseline_color)
    ax1.tick_params(axis ='y') 
    ax1.set_ylim(0.0,1.02)
    
    ax2 = ax1.twinx()   
    ax2.set_ylabel('% nodes left', color = 'blue') 
    #ax2.errorbar(df_strategy['threshold_mean'], df_strategy['uses_removed_mean'], fmt='-', yerr=df_strategy['uses_removed_std'], color = 'blue')
    ax2.errorbar(df_strategy['threshold_mean'], df_strategy['uses_left_percentage_mean'], fmt='-', label='% nodes left', color = 'blue')
    ax2.tick_params(axis ='y', labelcolor = 'blue') 
    ax2.set_ylim(0,100)
    
    #ax1.axhline(y=reference_result_ari_random, color='darkgray', linestyle='-', label='random')    
    ax1.axhline(y=reference_result_ari, color='gray', linestyle='-', label='no cleaning')    
    ax1.legend()    
    #ax2.legend(loc=0)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/{0}-{1}x{2}.png'.format(strategy,'threshold','ARI_mean'))
    plt.show() 
    plt.close() 
    
    # Plot ARI versus number of nodes left
    fig, ax1 = plt.subplots() 

    ax1.set_xlabel('% nodes left') 
    ax1.set_ylabel('ARI (mean)') 
    ax1.errorbar(df_strategy['uses_left_percentage_mean'], df_strategy['ARI_mean'], fmt='-', label=strategy, color = 'black')
    ax1.errorbar(df_strategy_random['uses_left_percentage_mean'], df_strategy_random['ARI_mean'], fmt='-', label='random baseline', yerr=df_strategy_random['ARI_std'], color = random_baseline_color)
    ax1.tick_params(axis ='y') 
    ax1.set_ylim(0.0,1.02)
    ax1.legend()
    plt.savefig('results/{0}-{1}x{2}.png'.format(strategy,'uses_left_percentage_mean','ARI_mean'))
    #plt.show() 
    plt.close() 
    
    #print(df_strategy['uses_removed_mean'], df_strategy['ARI_mean'])
 
    # Plot ARI versus number of clusters left
    fig, ax1 = plt.subplots() 

    ax1.set_xlabel('# clusters left') 
    ax1.set_ylabel('ARI (mean)') 
    ax1.errorbar(df_strategy['clusters_cleaned_mean'], df_strategy['ARI_mean'], fmt='-', label=strategy, color = 'black')
    ax1.errorbar(df_strategy_random['clusters_cleaned_mean'], df_strategy_random['ARI_mean'], fmt='-', label='random baseline', yerr=df_strategy_random['ARI_std'], color = random_baseline_color)
    ax1.tick_params(axis ='y') 
    ax1.set_ylim(0.0,1.02)
    ax1.legend()
    plt.savefig('results/{0}-{1}x{2}.png'.format(strategy,'clusters_cleaned_mean','ARI_mean'))
    plt.show() 
    plt.close() 
    
    # Plot ARI versus number of senses left
    fig, ax1 = plt.subplots() 

    ax1.set_xlabel('# senses left') 
    ax1.set_ylabel('ARI (mean)') 
    ax1.errorbar(df_strategy['senses_cleaned_mean'], df_strategy['ARI_mean'], fmt='-', label=strategy, color = 'black')
    ax1.errorbar(df_strategy_random['senses_cleaned_mean'], df_strategy_random['ARI_mean'], fmt='-', label='random baseline', yerr=df_strategy_random['ARI_std'], color = random_baseline_color)
    ax1.tick_params(axis ='y') 
    ax1.set_ylim(0.0,1.02)
    ax1.legend()
    plt.savefig('results/{0}-{1}x{2}.png'.format(strategy,'senses_cleaned_mean','ARI_mean'))
    plt.show() 
    plt.close() 
    
    
    # Plot ARI versus number of sense changes left
    fig, ax1 = plt.subplots() 

    ax1.set_xlabel('% sense changes left') 
    ax1.set_ylabel('ARI (mean)') 
    ax1.errorbar(df_strategy['sense_changes_left_percentage_mean'], df_strategy['ARI_mean'], fmt='-', label=strategy, color = 'black')
    ax1.errorbar(df_strategy_random['sense_changes_left_percentage_mean'], df_strategy_random['ARI_mean'], fmt='-', label='random baseline', yerr=df_strategy_random['ARI_std'], color = random_baseline_color)
    ax1.tick_params(axis ='y') 
    ax1.set_ylim(0.0,1.02)
    ax1.legend()
    plt.savefig('results/{0}-{1}x{2}.png'.format(strategy,'sense_changes_left_percentage_mean','ARI_mean'))
    plt.show() 
    plt.close() 
     
# check per lemma

# df_dwug_de: ['identifier', 'cluster', 'lemma', 'grouping'], df_dwug_de_sense_clean: ['identifier', 'label', 'lemma']
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
                          }, index=[0])).droplevel(level=1)
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
    return l2stat


original_stats = get_perlemma_stats(df)     # columns: ('lemma'), ['ntargets', 'nuses', 'nsenses', 'nclusters', 'ari', 'ri', '(1-ari)', '(1-ri)']
print(original_stats)
print(original_stats.columns.tolist())
# df_cleaning.strategy.value_counts()
# df_cleaning.query('strategy=="degreeremove"').model.value_counts()
dfm = df.drop(columns='cluster').merge(df_cleaning[['identifier','cluster','model','strategy']], 
                                 on=['identifier'], how='inner')
print(dfm)                                  # columns: ['identifier', 'label', 'lemma', 'grouping', 'cluster', 'model', 'strategy']

pdf = dfm.groupby(['model','strategy']).apply(lambda x: get_perlemma_stats(x, original_stats)).reset_index()
print(len(pdf))     # 3168
print(pdf.columns.tolist())     
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

strategy_order = ['stdnode','degreeremove','clustersizemin','clusterconnectmin','random']
strategy_order = ['stdnode','degreeremove','clustersizemin','clusterconnectmin','random']
# plotdf = pdf[pdf.strategy.isin(strategy_order)]
plotdf = pdf



# Individual plots for each lemma

method2papername = {
    'stdnode':'stdnode',
    'degreeremove':'dgrnode',
    'clustersizemin':'sizecluster',
    'clusterconnectmin':'cntcluster',
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
    g.savefig('cleaning_individual_ari.pdf')



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
mpdf = plotdf.groupby(['model']).agg({'strategy':'first','ntargets':'sum','nuses':'sum','nsenses':'sum', 'nclusters':'sum'}|
                              {c:'mean' for c in plotdf.columns if 'ri' in c or 'Δ' in c})
# mpdf.rename(columns={'lemma':'ntargets'}, inplace=True)
mpdf['size'] = mpdf.strategy.apply(lambda x: 2 if x in {'collapse','stdedge'} else 1)  # dot sizes
mpdf



# Collapse plots 

pdf['thres'] = pdf.model.str.split('_').str[-1].astype(float)
mpdf['thres'] = mpdf.index.str.split('_').str[-1].astype(float)
argmax_thres = mpdf.loc[mpdf[mpdf.strategy=='collapse'].ari.idxmax()].thres
print(argmax_thres)
with sns.plotting_context('poster'):
    g = sns.relplot(data=pdf.query('strategy=="collapse"'), x='thres',y='ari',col='lemma',col_wrap=6,
                kind='line')
    for ax in g.axes:
        ax.axvline(argmax_thres)
    g.savefig('collapsing_individual_ari.pdf')

g = sns.relplot(data=mpdf.query('strategy=="collapse"'), x='thres',y='ari', kind='line')
g.set(ylim=(0.0,0.8))
g.axes[0,0].axvline(argmax_thres)
g.savefig('collapsing_avg_ari.pdf')



# clustering quality metrics averaged across survived target words 

def plot_mpdf(mpdf, x='Δnuses/nuses', y='Δ(1-ri)/(1-ri)', legend=True,aspect=1.5):
    g = sns.relplot(data=mpdf, 
                x=x, y=y, hue='strategy',style='strategy',
                hue_order=strategy_order, style_order=strategy_order,
                markers=True, kind='line', errorbar=('ci',95), legend=legend, aspect=aspect)
#     sns.scatterplot(data=mpdf[mpdf.strategy.isin(strategy_order[:-1])], 
#                     size='size',
#                 x=x, y=y, hue='strategy',style='strategy',
#                 hue_order=strategy_order, style_order=strategy_order,
#                 legend=False)  
    if legend:
        for t in g.legend.texts:
            t.set_text(method2papername[t.get_text()])

    return g


sns.set_context('talk')
sns.set_style("whitegrid")
# 1st figure in the main part
g = plot_mpdf(mpdf, y='Δ(1-ari)/(1-ari)')
sns.move_legend(g, 'lower center', bbox_to_anchor=(0.4, 1), frameon=True, ncol=2)
g.savefig('cleaning_main1.pdf')


# appendix: compare nsenses and nclusters
g = plot_mpdf(mpdf, y='Δnclusters/nclusters', legend=True)
sns.move_legend(g, 'lower center', bbox_to_anchor=(0.4, 1), frameon=True, ncol=2)
g.savefig('app_nclusters_avg.pdf')