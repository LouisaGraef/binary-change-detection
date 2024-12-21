
import os
from WordTransformer import WordTransformer,InputExample
from collections import defaultdict
import pandas as pd
import csv
import numpy as np
from scipy.spatial.distance import cosine
from pathlib import Path
import dill 


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
Extract embeddings for word usages and predicted edge weights 
"""
def get_computational_annotation(dataset):

    # target words 
    words = sorted(os.listdir(f'{dataset}/data/'))

    # Embeddings for sentences <s_1, s_2> (E[word] = (E_1, E_2), all identifer 1 embeddings, all identifier 2 embeddings)
    E = dict()

    # load embedding model
    model = WordTransformer('pierluigic/xl-lexeme', device='cuda')  

    
    dfs = list()                            # for every word dataframe of mean human judgements and uses 
    scores = defaultdict(list)              # for every word list of edge weight predictions 

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
        # (join uses and judgements)
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

        # split df 
        # identifier1 nodes (in judgments) (columns ['identifier', 'context', 'grouping', 'indexes_target_token'])
        df1 = df[['identifier1', 'context1', 'grouping1', 'indexes_target_token1']].reset_index(drop=True) 
        df1 = df1.rename(columns={c: c[:-1] for c in df1.columns}) 
        # identifier2 nodes (in judgments) (columns ['identifier', 'context', 'grouping', 'indexes_target_token'])
        df2 = df[['identifier2', 'context2', 'grouping2', 'indexes_target_token2']].reset_index(drop=True)
        df2 = df2.rename(columns={c: c[:-1] for c in df2.columns})



        # embeddings extraction 
        #E_1, E_2 = model.encode(df1), model.encode(df2)
        #E[word] = (E_1, E_2)
        
        E1, E2 = list(), list()
        for j, row in df1.iterrows():
            row1, row2 = pd.DataFrame([dict(row)]), pd.DataFrame([dict(df2.loc[j])])    # nodes 1 and 2 of one human annotated edge


            context1, context2 = row1.loc[0, 'context'], row2.loc[0, 'context'] 
            indexes1, indexes2 = row1.loc[0, 'indexes_target_token'], row2.loc[0, 'indexes_target_token']
            indexes1, indexes2 = list(map(int, indexes1.split(':'))), list(map(int, indexes2.split(':')))

            examples_1= InputExample(texts=context1, positions=indexes1)
            examples_2= InputExample(texts=context2, positions=indexes2)


            #e_1, e_2 = model.encode(examples_1), model.encode(examples_2)
            try:
                e_1, e_2 = model.encode(examples_1), model.encode(examples_2)
            except:
                # one sentence is too long
                continue                                    # no edge weight prediction if the sentence of one use is too long 
            else:
                E1.append(e_1)      # list of all embeddings of identifier1 nodes 
                E2.append(e_2)
                # add edge weight prediction for word and identifier 1 and 2 to dictionary 'scores' 
                scores[word].append(dict(identifier1=row1['identifier'].iloc[0], identifier2=row2['identifier'].iloc[0], judgment=1-cosine(E1[-1], E2[-1])))

        try:
            E[word] = (np.array(E1), np.array(E2))          # all identifer 1 embeddings, all identifier 2 embeddings
        except:
            continue

        print('.')

    # save scores 
    ds = os.path.basename(dataset)
    Path(f'./scores/{ds}').mkdir(exist_ok=True, parents=True)
    with open(f'./scores/{ds}/scores.dill', mode='+wb') as f:
        dill.dump(scores, f)  

    print('-----')









if __name__=="__main__":

    get_computational_annotation("./data/dwug_de")