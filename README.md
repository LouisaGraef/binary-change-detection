# binary-change-detection

- 'setup.sh': Creates a conda environment and installs all required packages.  
- 'download_data.py': Downloads datasets  
- data_stats.py: Prints statistics of the datasets, saves statistics in './stats/dataset_stats.csv' 
- graph_stats.py: Prints statistics of one graph 
- generate_graph.py: Generates a graph with XL-Lexeme and Cosinus Distance given a dataframe of uses 
- edges_evaluation.py: Evaluates the predicted edge weights of graphs with mean Spearman Correlation and p-value against humanly annotated edge weigths
(returns mean Spearman Correlation and mean p-value for all datasets, prints results and saves results in './stats/correlation_stats.csv') 
- cluster_graph.py: Clusters one graph
- clustering_evaluation.py: Evaluates a clustering with Adjusted Rand Index 
