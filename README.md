# binary-change-detection


- `setup.sh`: Creates a conda environment and installs all required packages.
- `reproduce_cleaning_paper.py`: Reproduces results of Anonymous (2024) ([[1]](#1)) and evaluates cleaning methods with conflicts_normalized and win_min_max_normalized.
- `reproduce_paper.py`: run for clustering paper reproduction (Periti and Tahmasebi, 2024)([[2]](#2))
- `run_comparison.py`: run for WIC (edge weight prediction, evaluation with Spearman correlation), WSI (different Clustering methods, evaluation with ARI) and LSC (Graded and Binary Change prediction, evaluation with Spearman correlation and F1-Score)


## References
<a id="1">[1]</a>
Anonymous. Clustering and cleaning of word usage graphs. In Submitted to ACL
Rolling Review - August 2024, 2024. URL https://openreview.net/forum?id=BlbrJvKv6L. under review.

<a id="2">[2]</a>
Francesco Periti and Nina Tahmasebi. A systematic comparison of contextualized
word embeddings for lexical semantic change. arXiv preprint arXiv:2402.12011,
2024.
