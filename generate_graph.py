
# https://github.com/Garrafao/WUGs/blob/main/scripts/use2graph.py
# https://arxiv.org/pdf/2402.12011
# https://github.com/pierluigic/xl-lexeme 
# https://huggingface.co/pierluigic/xl-lexeme 

# Vector -> cosine similarity between 2 vectors -> DWUG
# variation of Correlation Clustering -> clustered DWUG -> ARI to quantify cluster agreement 

import networkx as nx
import csv
from WordTransformer import WordTransformer,InputExample





model = WordTransformer('pierluigic/xl-lexeme')         # load model (from hugging face)

"""
Generate a contextualized embedding of a use case with XL-Lexeme 
"""
def generate_embedding():

    examples = InputExample(texts="the quick fox jumps over the lazy dog", positions=[10,13])
    fox_embedding = model.encode(examples) #The embedding of the target word "fox"

    return fox_embedding



"""
Generate a graph of a target word given its uses
"""
def get_graph(uses):
    # Initialize graph 
    graph = nx.Graph()
    with open(uses, encoding='utf-8') as csvfile:                                                   # read in uses
        reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
        uses = [row for row in reader]                                                              # uses as list of dictionaries

    # Add uses as nodes
    identifier2data = {}        # maps node identifiers to their data 
    for (k, row) in enumerate(uses):
        row = row.copy()
        identifier = row['identifier']
        identifier2data[identifier] = row
        graph.add_node(identifier)

    nx.set_node_attributes(graph, identifier2data)
    #print(graph.nodes()[identifier])

    print('number of nodes: ', len(graph.nodes))



if __name__=="__main__":
    uses = "./data/dwug_en/data/gas_nn/uses.csv"
    get_graph(uses)
    emb = generate_embedding()
    print(type(emb))