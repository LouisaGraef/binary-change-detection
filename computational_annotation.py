
import os
from WordTransformer import WordTransformer,InputExample
from collections import defaultdict
import pandas as pd
import csv
import numpy as np



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
Extract embeddings for word usages in C1 and C2.
"""
def get_computational_annotation(dataset, batch_size=16, max_length=512):

    # target words 
    words = sorted(os.listdir(f'{dataset}/data/'))

    # Embeddings for sentences <s_1, s_2>
    E = dict()

    # load embedding model
    model = WordTransformer('pierluigic/xl-lexeme', device='cuda')  
    layer='tuned'

    # Dataframe of judgements and uses for every target word 
    dfs = list() 
    scores = defaultdict(list)              # value is empty list if the key has not been set -> no Key Error 

    for word in words:
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


        # join the two dataframes 
        # (expand identifier 1 and 2 in judgements with context, indexes_target_token and grouping of the respective use)
        # (df.columns.tolist() = ['identifier1', 'identifier2', 'annotator', 'judgment', 
        # 'context1', 'indexes_target_token1', 'grouping1', 'context2', 'indexes_target_token2', 'grouping2'])
        df = judgments.merge(uses, left_on='identifier1', right_on='identifier')
        df = df.rename(
            columns={'context': 'context1', 'indexes_target_token': 'indexes_target_token1', 'grouping': 'grouping1'})
        del df['identifier']
        df = df.merge(uses, left_on='identifier2', right_on='identifier')
        df = df.rename(
            columns={'context': 'context2', 'indexes_target_token': 'indexes_target_token2', 'grouping': 'grouping2'})
        del df['identifier']



        # take the mean judgment of annotators (for every edge)
        # (df.columns.tolist() = ['identifier1', 'identifier2', 'context1', 'context2', 
        # 'indexes_target_token1', 'indexes_target_token2', 'grouping1', 'grouping2', 'judgment'])
        df = df.groupby(
            ['identifier1', 'identifier2', 'context1', 'context2', 'indexes_target_token1', 'indexes_target_token2', 
             'grouping1', 'grouping2'])['judgment'].mean().reset_index()
        dfs.append(df)
        examples = list()







if __name__=="__main__":
    get_computational_annotation("./data/dwug_de")