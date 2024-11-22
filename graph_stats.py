

import pickle 
from itertools import combinations
from constellation import Constellation


"""
returns mappings: dictionary which contains mapping from node ids to grouping of nodes (1 or 2).
"""
def get_data_maps_nodes(g, attributes={'type':'usage'}):
    mappings = {}       # dictionary which will contain node2period
    node2period = {}        # maps node ids to grouping (1 or 2) 
    for node in g.nodes():
        node_data = g.nodes()[node]     # data dictionary of node 
        #print(node_data)
        #print("")
        if all([node_data[k]==v  for (k,v) in attributes.items()]):     # if node_data['type']=='usage' 
            node2period[node] = node_data['grouping']

    mappings['node2period'] = node2period
    #print(mappings)

    return mappings


"""
returns time related stats (groupings stats) from a clustered graph as a dictionary
(frequency distributions in grouping 1 and grouping 2 of a graph G, binary change score of graph G).
"""
def get_time_stats(g, old, new, threshold=0.5, lower_range=(1,3), upper_range=(3,5), lower_prob=0.001, upper_prob=0.1, attributes={'type':'usage'}):
    g = g.copy()
    stats = {} 

    # get node stats 
    nodes = [node for node in g.nodes if all([g.nodes()[node][k]==v for (k,v) in attributes.items()])]      # list of node identifiers of nodes with 'type':'usage'
    oldnodes = [node for node in nodes if g.nodes()[node]['grouping']==old]     # list of all nodes with 'grouping':'1' (old is '1')
    newnodes = [node for node in nodes if g.nodes()[node]['grouping']==new]     # list of all nodes with 'grouping':'2' (new is '2')
    frequency = len(nodes) 
    frequency1 = len(oldnodes)
    frequency2 = len(newnodes)
    stats['nodes'] = frequency          # total number of nodes
    stats['nodes1'] = frequency1          # number of nodes from grouping 1
    stats['nodes2'] = frequency2          # number of nodes from grouping 2

    # Define thresholds
    lower_prob1 = round(lower_prob*frequency1)
    lower_prob2 = round(lower_prob*frequency2)
    upper_prob1 = round(upper_prob*frequency1)
    upper_prob2 = round(upper_prob*frequency2)
    lowerbound1 = min(max(lower_range[0],lower_prob1),lower_range[1])
    lowerbound2 = min(max(lower_range[0],lower_prob2),lower_range[1])
    upperbound1 = min(max(upper_range[0],upper_prob1),upper_range[1])
    upperbound2 = min(max(upper_range[0],upper_prob2),upper_range[1]) 

    try:    
        co = Constellation(graph=g, bound1=upperbound1, bound2=upperbound2, lowerbound1=lowerbound1, lowerbound2=lowerbound2, is_prob=False, old=old, new=new, threshold=threshold)        
        is_clusters = True
    except KeyError:
        print('No clusters found.')
        is_clusters = False

    if is_clusters:    
    
        stats['cluster_freq_dist'] = co.distribution
        stats['cluster_freq_dist1'] = co.distribution1
        stats['cluster_freq_dist2'] = co.distribution2
        stats['cluster_prob_dist'] = [round(pr, 3) for pr in co.prob]
        stats['cluster_prob_dist1'] = [round(pr, 3) for pr in co.prob1]
        stats['cluster_prob_dist2'] = [round(pr, 3) for pr in co.prob2]
        stats['cluster_number'] = len(co.prob)
        stats['cluster_number1'] = len([1 for pr in co.prob1 if pr > 0.0])
        stats['cluster_number2'] = len([1 for pr in co.prob2 if pr > 0.0])
    
        stats['change_binary'] = co.c_mb
        stats['change_binary_gain'] = co.i_mb
        stats['change_binary_loss'] = co.r_mb
        stats['change_graded'] = co.c_u
    
        stats['k1'] = lowerbound1
        stats['n1'] = upperbound1
        stats['k2'] = lowerbound2
        stats['n2'] = upperbound2

    return stats


"""
returns stats of graph G.
"""
def get_graph_stats(g):
    mappings_nodes = get_data_maps_nodes(g)     # dictionary which contains mapping from node ids to grouping of nodes (1 or 2)
    node2period = mappings_nodes['node2period']     # maps node ids to grouping (1 or 2) 
    periods = sorted(set(node2period.values()))     # list (['1', '2'])
    combos = combinations(periods, 2) if (len(periods)>1 or len(periods)==0) else [(periods[0],None)]       # list(combos) = [('1', '2')]
    for (old, new) in combos:
        time_stats = get_time_stats(g, old=old, new=new)

    return time_stats




if __name__=="__main__":

    input_file = "./data/dwug_en/graphs/opt/afternoon_nn"
    with open(input_file, "rb") as f:                   # open graph 
        graph = pickle.load(f)

    graph_stats = get_graph_stats(graph)         # get stats of one graph (freq dist 1 and 2, binary change value)
    print(f"\nInput file: {input_file} \n")
    print(graph_stats)
    print("")
