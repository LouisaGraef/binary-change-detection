
import os
from WordTransformer import WordTransformer,InputExample
from collections import defaultdict
import pandas as pd
import csv
import numpy as np
from scipy.spatial.distance import cosine
from pathlib import Path
import dill 
from scipy.stats import spearmanr
from model_evaluate import read_data



# https://github.com/FrancescoPeriti/CSSDetection 
# https://github.com/FrancescoPeriti/CSSDetection/blob/main/src/computational_annotation.py 




"""
Process uses for one dataset (rename uses['grouping'] for Nordiachange subsets 1 and 2, remove uses with cluster '-1')
"""
def processing(dataset, uses, word):

    # rename uses['grouping'] for Nordiachange subsets 1 and 2
    if 'nor_dia_change/subset1' in dataset:
        uses['grouping'] = [1 if '1970-2015' == i else 2 for i in uses['grouping']]
    elif 'nor_dia_change/subset2' in dataset:
        uses['grouping'] = [1 if '1980-1990' == i else 2 for i in uses['grouping']]

    # remove uses with cluster '-1'.
    try:
        clusters = pd.read_csv(f'{dataset}/clusters/opt/{word}.csv', sep='\t')
        uses_to_remove = clusters[clusters['cluster']==-1].identifier.values        # identifier of uses to remove 
        uses = uses[~uses.identifier.isin(uses_to_remove)]                          # remove uses to remove 
    except: 
        pass
    
    return uses




"""
get a dataframe of mean human judgements and uses for one word
"""
def get_human_judgments(dataset,word,words):
    
    # read in uses 
    uses = pd.read_csv(f'{dataset}/data/{word}/uses.csv', sep='\t', quoting=csv.QUOTE_NONE, encoding="utf-8")

    # read in human judgements 
    if 'nor_dia_change-main/subset' in dataset:
        judgments = pd.read_csv(f'{dataset}/data_joint/data_joint.tsv', sep='\t', encoding="utf-8")
        judgments = judgments[judgments['lemma']==word]
    else:
        judgments = pd.read_csv(f'{dataset}/data/{word}/judgments.csv', sep='\t', encoding="utf-8")

    

    # remove useless columns 
    uses = processing(dataset, uses, words)
    
    uses = uses[['identifier', 'context', 'indexes_target_token', 'grouping']]
    judgments = judgments[['identifier1', 'identifier2', 'annotator', 'judgment']]

    uses['identifier'] = uses['identifier'].astype(str)
    judgments['identifier1'] = judgments['identifier1'].astype(str)
    judgments['identifier2'] = judgments['identifier2'].astype(str)


    # take the mean judgment of annotators (for every edge)
    # (judgments.columns.tolist() = ['identifier1', 'identifier2', 'judgment'])
    judgments = judgments.sort_values(['identifier1', 'identifier2']).groupby(
        ['identifier1', 'identifier2'])['judgment'].mean().reset_index()


    # uses of all identifier1 nodes, uses of all identifier2 nodes
    # (df1.columns.tolist() = ['identifier', 'context', 'indexes_target_token', 'grouping'])
    # (df2.columns.tolist() = ['identifier', 'context', 'indexes_target_token', 'grouping'])
    df1 = judgments.merge(uses, left_on='identifier1', right_on='identifier').drop(columns=['identifier1', 'identifier2', 'judgment'])
    df2 = judgments.merge(uses, left_on='identifier2', right_on='identifier').drop(columns=['identifier1', 'identifier2', 'judgment'])


    return judgments, df1, df2




"""
Extract embeddings for word usages and predicted edge weights 
"""
def get_computational_annotation(dataset):

    # target words 
    words = sorted(os.listdir(f'{dataset}/data/'))
    # load xl-lexeme
    model = WordTransformer('pierluigic/xl-lexeme', device='cuda')  

    # E[word] = (E_1, E_2), all identifer 1 embeddings, all identifier 2 embeddings
    E = dict()

    # dfs = list()                            # for every word dataframe of mean human judgements 
    edge_preds = dict()              # dictionary: for every word df of edge weight judgments and predictions 

    for word in words:

        judgments, df1, df2 = get_human_judgments(dataset,word,words)
        # (judgments.columns.tolist() = ['identifier1', 'identifier2', 'judgment'])
        # (df1.columns.tolist() = ['identifier', 'context', 'indexes_target_token', 'grouping'])
        # (df2.columns.tolist() = ['identifier', 'context', 'indexes_target_token', 'grouping'])
        # dfs.append(judgments)
        
        E1, E2 = list(), list()     # embeddings of identifier1 nodes, embeddings of identifier2 nodes

        # add new column for edge predictions to judgments 
        judgments['edge_pred'] = pd.NA

        for j, row in df1.iterrows():
            row1, row2 = pd.DataFrame([dict(row)]), pd.DataFrame([dict(df2.loc[j])])    # nodes 1 and 2 of one human annotated edge

            # input data for xl-lexeme model 
            context1, context2 = row1.loc[0, 'context'], row2.loc[0, 'context'] 
            indexes1, indexes2 = row1.loc[0, 'indexes_target_token'], row2.loc[0, 'indexes_target_token']
            indexes1, indexes2 = list(map(int, indexes1.split(':'))), list(map(int, indexes2.split(':')))

            examples_1= InputExample(texts=context1, positions=indexes1)
            examples_2= InputExample(texts=context2, positions=indexes2)


            # generate embeddings
            try:
                e_1, e_2 = model.encode(examples_1), model.encode(examples_2)
            except:
                # no edge weight prediction if one sentence is too long 
                continue                                    
            else:
                E1.append(e_1)      # embeddings of identifier1 nodes 
                E2.append(e_2)
                # add edge weight prediction for word and identifier 1 and 2 to dictionary 'edge_preds' 
                judgment=1-cosine(E1[-1], E2[-1])
                judgments.at[j, 'edge_pred'] = judgment
                #edge_preds[word].append(dict(identifier1=row1['identifier'].iloc[0], identifier2=row2['identifier'].iloc[0], judgment=1-cosine(E1[-1], E2[-1])))
                
        try:
            E[word] = (np.array(E1), np.array(E2))          # all identifer 1 embeddings, all identifier 2 embeddings
        except:
            continue

        edge_preds[word]= judgments
        # print(judgments.columns.tolist())         # ['identifier1', 'identifier2', 'judgment', 'edge_pred']

        print('.')

    # save edge_preds 
    ds = os.path.basename(dataset)
    Path(f'./edge_preds/{ds}').mkdir(exist_ok=True, parents=True)
    with open(f'./edge_preds/{ds}/edge_preds.dill', mode='+wb') as f:
        dill.dump(edge_preds, f)  

    print('-----')










if __name__=="__main__":
    
    # get_computational_annotation("./data/dwug_de")

    
    datasets = ["dwug_de", "dwug_en", "dwug_sv", "dwug_es", "chiwug", 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]       # no dwug_la 
    datasets = ["./data/" + dataset for dataset in datasets]

    for dataset in datasets:
        get_computational_annotation(dataset)
    