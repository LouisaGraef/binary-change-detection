
import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt


def evaluate_clustering_plot(datasets, cleaned_gold=False, filter_minus_one_nodes=False):
    for metric in ["ARI", "BC_F1", "GC_Spearmanr"]:


        # Read clustering results 

        metric_results = pd.DataFrame()
        print(f"\n\ncleaned_gold: {cleaned_gold}, filter_minus_one_nodes: {filter_minus_one_nodes}")
        print(f"Metric: {metric}\n")
        for dataset in datasets:
            ds = dataset.replace("./data/", "")

    
            if cleaned_gold==False:                     # uncleaned gold
                if filter_minus_one_nodes == True:
                    mr = f"./clustering_evaluation/{ds}_without_minus_one/{metric}/mean_crossvalidated_results.csv"
                    df = pd.read_csv(mr)
                    df['dataset'] = ds
                else:
                    mr = f"./clustering_evaluation/{ds}/{metric}/mean_crossvalidated_results.csv"
                    df = pd.read_csv(mr)
                    df['dataset'] = ds
            else:                                       # cleaned gold
                if filter_minus_one_nodes == True:
                    mr = f"./clustering_evaluation/{ds}_cleaned_without_minus_one/{metric}/mean_crossvalidated_results.csv"
                    df = pd.read_csv(mr)
                    df['dataset'] = ds
                else:
                    mr = f"./clustering_evaluation/{ds}_cleaned/{metric}/mean_crossvalidated_results.csv"
                    df = pd.read_csv(mr)
                    df['dataset'] = ds
            metric_results = pd.concat([metric_results, df])


        gc_mean_std = metric_results.groupby('method')[[f'{metric} mean', f'{metric} std']].mean()
        gc_mean_std["dataset"] = "Avg."
        gc_mean_std = gc_mean_std.reset_index()
        metric_results = pd.concat([metric_results, gc_mean_std])
        metric_results = metric_results.reset_index(drop=True)
        metric_results = metric_results.rename(columns={f'{metric} std': 'std'})
        print(metric_results)



        # Plot

        sns.set_context(context='poster')

        method_order = ['WSBM','CC', 'k-means', 'spectral', 'AGGL']
        g=sns.catplot(data=metric_results, 
                        y=f'{metric} mean', x='dataset', hue='method', hue_order=method_order, kind='bar', orient='v', legend_out=True, aspect=5)


        # add standard deviation
        ax = g.ax 
        for i, bar in enumerate(ax.patches):
            height = bar.get_height() # metric value
            if height >= 0.01:
                dset = bar.get_x() + bar.get_width() / 2  # Position 
                dataset_name = metric_results['dataset'].unique()[(i % 11)]
                method = method_order[i // 11]

                
                std_value = metric_results[
                    (metric_results['method'] == method) & 
                    (metric_results['dataset'] == dataset_name)
                ][f'std'].values[0]
                
                ax.errorbar(dset, bar.get_height(), yerr=std_value, fmt='none', capsize=5, ecolor='0.75')


        # add text
        g.set(ylim=(0,1))
        g.set_ylabels(f'{metric}')
        g.set_xticklabels(rotation=90)

        ax = g.ax
        for p in ax.patches:
            height = p.get_height() # metric value
            if height >= 0.01:
                ax.text(p.get_x() + p.get_width() / 2,
                        height + 0.02,
                        f'{height:.2f}',    # round (2 decimal digits)
                        ha='center', va='bottom', fontsize=14, rotation=90)


        # save plot
        os.makedirs(f"./clustering_evaluation_plots", exist_ok=True)
        
        if cleaned_gold==False:                     # uncleaned gold
            if filter_minus_one_nodes == True:
                g.savefig(f'./clustering_evaluation_plots/{metric}_without_minus_one.pdf')
            else:
                g.savefig(f'./clustering_evaluation_plots/{metric}_normal.pdf')
        else:                                       # cleaned gold
            if filter_minus_one_nodes == True:
                g.savefig(f'./clustering_evaluation_plots/{metric}_cleaned_without_minus_one.pdf')
            else:
                g.savefig(f'./clustering_evaluation_plots/{metric}_cleaned.pdf')





if __name__=="__main__":

    datasets = ["dwug_de", "discowug", "refwug", "dwug_en", "dwug_sv", "dwug_la", "dwug_es", "chiwug",      # all datasets 
                "nor_dia_change-main/subset1", "nor_dia_change-main/subset2"]                               
    datasets = ["./data/" + dataset for dataset in datasets]

    evaluate_clustering_plot(datasets, cleaned_gold=False, filter_minus_one_nodes=False)
    evaluate_clustering_plot(datasets, cleaned_gold=False, filter_minus_one_nodes=True)
    evaluate_clustering_plot(datasets, cleaned_gold=True, filter_minus_one_nodes=False)
    evaluate_clustering_plot(datasets, cleaned_gold=True, filter_minus_one_nodes=True)