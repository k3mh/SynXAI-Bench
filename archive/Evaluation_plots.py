import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os

# results = pd.read_pickle("results/results10000.plk")
#
# results.head()
#
#
# ds_result = results.copy()
# # ds_result = ds_result.loc[ds_result.dataset == dataset_n]
# ds_result = ds_result.loc[ds_result.lib == "anchor"]
# ds_result = ds_result.loc[ds_result.metric == "sensitivity"]
# exploded = ds_result.set_index("dataset").explode("score_list").reset_index()
# sns.boxplot(data=exploded, x="dataset", y="score_list")
# plt.show()
def plot_results (parent_dir):
    #parent_dir   = 'Dataset\\30Mar2020_21h09m33s\\'

    results_df   = pd.read_pickle(parent_dir + "evaluation_" + "all" + ".pkl")
    #results_dir  = parent_dir + '\\results'
    results_dir = os.path.join(parent_dir , 'results')
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    context = ['recall_partial', 'recall', 'FPR', 'sensitivity']

    for con in context:

        sub_results_df = results_df.loc[results_df["metric"] == con ]
        sub_results_df = sub_results_df.explode('score_list')
        sub_results_df.score_list = sub_results_df.score_list.astype(np.float)

        plt.clf()
        fig_dims = (12, 8)
        fig, ax = plt.subplots(figsize=fig_dims)
        #sns.boxplot(data=sub_results_df, x='dataset', y='score_list', hue= 'lib')
        sns.violinplot(data=sub_results_df, x='dataset', y='score_list', hue='lib', split=True, scale_hue=True, cut=0, ax=ax)
        #sns.violinplot(data=sub_results_df, x='dataset', y='score_list', hue='lib', split=True, cut=0, ax=ax)
        #sns.swarmplot(data=sub_results_df, x='dataset', y='score_list', hue='lib', ax=ax)

        ax.set_title(con)
        ax.figure.savefig(os.path.join(results_dir , con + '.png'))

def plot_dataset_signals(parent_dir):
    for i in Path(parent_dir).iterdir():
        if i.name.find("datasets_seq_lst") > -1:
            dataset_list =  pd.read_csv(i)
            dataset_list = dataset_list.loc[:, ~dataset_list.columns.str.contains('^Unnamed')]

            size = i.name.rsplit(".")[0].rsplit("_", 1)[1]
            size = int(size)
            plot_df = pd.DataFrame(columns=["dataset_name", "x", "y"])

            for row in dataset_list.iterrows():
                y = []
                non_zero_len = np.count_nonzero(~np.isnan(row[1]))
                dataset_por =  size / non_zero_len
                for cell in  row[1]:
                    if not np.isnan(cell):
                        #plt.axhline(y= cell ,xmin=x , xmax=x + dataset_por )
                        #x = x + dataset_por
                        y = y + [cell] * int(dataset_por)
                dataset_name = [row[0]] * len(y)
                x = range(len(y))
                temp_df = pd.DataFrame({"dataset_name": dataset_name, "x":x, "y":y})
                plot_df = pd.concat([plot_df, temp_df])


            plot_df["dataset_name"] = plot_df["dataset_name"].astype(str)
            #plot_df = plot_df.drop_duplicates(subset=["y", "dataset_name"])

            plt.clf()
            mks = itertools.cycle(['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'V'])
            styls = itertools.cycle(['-', '--', '-.', ':'])
            markers = [next(mks) for i in plot_df["dataset_name"].unique()]
            sns.set_context("paper")
            plt_pallete = sns.color_palette("Paired", len(plot_df["dataset_name"].unique()))
            ax = sns.lineplot(data=plot_df, x="x", y="y", hue="dataset_name", legend="full",
                              palette=plt_pallete, drawstyle='steps', style=styls)

            ax.set_title("Dataset generator signals")
            plt.show()


def plot_all(parent_dir):
    plot_results(parent_dir)
    plot_dataset_signals(parent_dir)
