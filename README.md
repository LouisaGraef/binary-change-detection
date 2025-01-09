# binary-change-detection

- `setup.sh`: Creates a conda environment and installs all required packages.  
- `download_data.py`: Downloads datasets.
- `data_stats.py`: Prints statistics of the datasets, saves statistics in './stats/dataset_stats.csv' 
- `comp_ann.py`: Extracts embeddings of uses, predicts edge weights (for all words in 1 dataset).
- `evaluation.py`: Evaluates WIC with Spearman Correlation, clusters graphs and evaluates clusterings (WSI) with Adjusted Rand Index, 
                    saves parameter grids to "./parameter_grids/{ds}/{clustering_method}/parameter_grid.tsv".
- `cluster_graph.py`: clusters one graph with specified clustering method 
- `reproduce_paper`: run for paper reproduction
- `run_comparison`: run for WIC (edge weight prediction, evaluation with Spearman correlation), WSI (different Clustering methods, evaluation with ARI) and LSC (Graded and Binary Change prediction, evaluation with Spearman correlation and F1-Score)
