# binary-change-detection

setup.sh: creates a conda environment and installs all required packages.
download_data.py: downloads datasets 
data_stats.py: prints statistics of the datasets, saves statistics in './stats/dataset_stats.csv' 
graph_stats.py: prints statistics of one graph 
generate_graph.py: generates a graph with XL-Lexeme and Cosinus Distance given a dataframe of uses 
edges_evaluation.py: evaluates the predicted edge weights of graphs with mean Spearman Correlation and p-value against humanly annotated edge weigths
(returns mean Spearman Correlation and mean p-value for all datasets, prints results and saves results in './stats/correlation_stats.csv') 
