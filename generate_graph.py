
# https://github.com/Garrafao/WUGs/blob/main/scripts/use2graph.py
# https://arxiv.org/pdf/2402.12011
# https://github.com/pierluigic/xl-lexeme 
# https://huggingface.co/pierluigic/xl-lexeme 


import networkx as nx
import csv
from WordTransformer import WordTransformer,InputExample
from numpy import dot
from numpy.linalg import norm





model = WordTransformer('pierluigic/xl-lexeme')         # load model (from hugging face)


"""
Generate a contextualized embedding of a use case with XL-Lexeme 
return: embedding (type numpy.ndarray, shape (1024,))
"""
def generate_embedding(context, indexes_target_token):

    examples = InputExample(texts=context, positions=indexes_target_token)
    embedding = model.encode(examples) #The embedding of the target word in the given use case (context)

    return embedding


"""
Adds all edges to graph between all node pairs node1 and node2.
Edge weight = cosine similarity between the embedding of node1 and the embedding of node2."""
def add_edges(graph):
    for node1 in graph.nodes:
        for node2 in graph.nodes:
            if graph.has_edge(node1,node2):     # edge already exists
                pass
            else:
                emb1 = graph.nodes[node1]['embedding']      # embedding of node 1 
                emb2 = graph.nodes[node2]['embedding']      # embedding of node 2 
                cos_sim = dot(emb1, emb2)/(norm(emb1)*norm(emb2))       # cosine similarity between the embedding of node1 and the embedding of node2
                graph.add_edge(node1, node2, weight=cos_sim)        # add edge
    return graph



"""
Generate a graph of 1 target word given its use cases. Nodes are identifiers (with attributes), no edges, 
embedding of use case is contained in node attributes 
"""
def get_graph(uses):
    # Initialize graph 
    graph = nx.Graph()
    with open(uses, encoding='utf-8') as csvfile:                                                   # read in uses
        reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
        uses = [row for row in reader]                                                              # uses as list of dictionaries

    # Add uses as nodes
    identifier2data = {}                    # maps node identifiers to their data 
    for (k, row) in enumerate(uses):
        row = row.copy()
        identifier = row['identifier']
        identifier2data[identifier] = row
        graph.add_node(identifier)              # add node 

        context = identifier2data[identifier]["context"]
        indexes = identifier2data[identifier]["indexes_target_token"]       # string (e.g. '119:122')
        indexes = list(map(int, indexes.split(':')))                        # list of integers (e.g. [119,122])
        emb = generate_embedding(context, indexes)          # generate embedding of use case 
        identifier2data[identifier]["embedding"]= emb       # add embedding to node data 

    nx.set_node_attributes(graph, identifier2data)          # set attributes of all nodes 

    print('\nnumber of nodes: ', len(graph.nodes))
    print('number of edges: ', len(graph.edges))            # 0

    # Add edges (edge weight = cosine similarity beween two nodes)
    graph = add_edges(graph)        

    print('\nnumber of nodes: ', len(graph.nodes))
    print('number of edges: ', len(graph.edges))

    return graph



if __name__=="__main__":
    uses = "./data/dwug_en/data/face_nn/uses.csv"
    graph = get_graph(uses)    
    print(graph.nodes['fic_1964_16147.txt-1494-12'])