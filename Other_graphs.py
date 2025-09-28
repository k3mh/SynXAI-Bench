
from sklearn.datasets import make_gaussian_quantiles, make_hastie_10_2, make_classification, make_moons
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import importlib
import numpy as np

importlib.reload(px)


"""Madelon rule example"""
from sklearn.datasets import  make_classification

temp_arr  = make_classification( n_samples=10000, n_features=2,  n_informative=2, n_redundant=0, n_repeated=0, class_sep=3, n_clusters_per_class=2)
temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0],\
                        "x2": temp_arr[0].transpose()[1], \
                        "y": temp_arr[1]})
temp_pd.y = temp_pd.y.astype(str)
fig = px.scatter(data_frame = temp_pd , x="x1",  y="x2", color="y")
fig.update_layout(
    title={
        'text': "Simple example of 2 informative features with 2 clusters per class",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()

""" Madelon rule end"""

""" make blobs example """
from sklearn.datasets import make_blobs
temp_arr  = make_blobs( n_samples=10000, n_features=2, centers=6)
temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0],\
                        "x2": temp_arr[0].transpose()[1], \
                        "y": temp_arr[1]})

# temp_pd.y = temp_pd.y.map( lambda x: 0 if x in (1,2,3) else 1)
temp_pd.y = temp_pd.y.astype(str)
fig = px.scatter(data_frame = temp_pd , x="x1",  y="x2", color="y")
fig.update_layout(
    title={
        'text': "Simple example of 2 class problem based on 2 features",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()

""" end of make blobs"""


""" make moons example """
from sklearn.datasets import make_blobs
temp_arr  = make_moons( n_samples=10000, random_state=100, noise = 0.1)
temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0],\
                        "x2": temp_arr[0].transpose()[1], \
                        "y": temp_arr[1]})

# temp_pd.y = temp_pd.y.map( lambda x: 0 if x in (1,2,3) else 1)
temp_pd.y = temp_pd.y.astype(str)
fig = px.scatter(data_frame = temp_pd , x="x1",  y="x2", color="y")
fig.update_layout(
    title={
        # 'text': "Simple example of 2 class problem based on 2 features",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()

""" end of make blobs"""

""""dataset accuracy plot : start"""
fig = px.line(x=range(1, len(accuracy_lst)+1), y=accuracy_lst, markers=True, range_x= range(1, len(accuracy_lst)+2))
fig.update_layout(
    title={
        # 'text': "Datasets performance based on individual rules.",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
""""dataset accuracy plot : end """


""" Dataset combinations and selection"""
file_path="src/my_benchmark_results/model_performance_summary.csv"
minimum_threshold = 0.60
dataset_comb_results = pd.read_csv(file_path)
dataset_comb_results = dataset_comb_results.loc[dataset_comb_results.auc >= minimum_threshold]
dataset_comb_results.sort_values("auc", inplace=True, ascending=False)
# dataset_comb_results.drop(columns=["Unnamed: 0"], inplace=True)
dataset_comb_results = dataset_comb_results.reset_index(drop=True).reset_index()
selected_inds=[]
for i in [0, 10, 20 , 30 , 40 , 50 , 60 , 70 , 80 , 90 , 100]:
    selected_inds.append(dataset_comb_results.iloc[(dataset_comb_results.accuracy - np.percentile(
        dataset_comb_results.accuracy, i)).abs().argsort()[:1]].index.values[0])
dataset_comb_results = dataset_comb_results.rename(columns={"index":"Dataset Index", "accuracy":"Accuracy", "auc":"AUC"})
fig = px.scatter(data_frame=dataset_comb_results, x="Dataset Index", y="AUC" )
fig.data[0].update(selectedpoints=selected_inds, selected=dict(marker=dict(color='purple', size=11)))

fig.show()


####################### OPS surface function #############
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def z_func(cov, com):
    z=np.sqrt(np.power(100-cov,2) + np.power(com, 2)) -1
    return z

OPS=[]
coverage = []
complexity = []
cov = np.linspace(0, 100, 30)
comp = np.linspace(1, 100, 30)

coverage, complexity = np.meshgrid(cov, comp)
OPS = z_func(coverage, complexity)


# Read data from a csv
# z_data = pd.DataFrame({"OPS": OPS, "Complexity":complexity, "Coverage":coverage}, index=range(len(OPS)))

fig = go.Figure(data=[go.Surface(x=coverage, y=complexity, z=OPS)])

fig.update_layout(title='OPS Function Surface', autosize=False,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90),
                  scene=dict(
                      xaxis_title='Coverage',
                      yaxis_title='Complexity',
                      zaxis_title='OPS',
                  ),
                  )

fig.show()


######################################################################################################################


import pandas as pd
import numpy as np
import plotly.express as px
import os


def select_datasets_by_percentiles(file_path: str, auc_threshold: float = 0.60) -> pd.DataFrame:
    """
    Loads model performance data, sorts by AUC, and selects a representative
    sample of datasets based on the percentiles of their accuracy scores.

    Args:
        file_path (str): The path to the model performance CSV file.
        auc_threshold (float): The minimum AUC score to include in the analysis.

    Returns:
        pd.DataFrame: The full, sorted DataFrame with a new 'Dataset Index' column.
                      Plotting should be handled separately.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file was not found at {file_path}")
        return pd.DataFrame(), []

    # --- 1. Load and Filter Data ---
    dataset_comb_results = pd.read_csv(file_path)

    # Filter out runs that are below the minimum AUC threshold
    dataset_comb_results = dataset_comb_results.loc[dataset_comb_results.auc >= auc_threshold]
    if dataset_comb_results.empty:
        print(f"No datasets found with AUC >= {auc_threshold}")
        return pd.DataFrame(), []

    # --- 2. Sort by AUC (Full Precision) ---
    # This is the key step you asked about. Pandas sorts using the full
    # precision of the float64 numbers, not a rounded version.
    # The sorting is accurate.
    dataset_comb_results.sort_values("auc", inplace=True, ascending=False)

    # Create a new index that represents the rank after sorting by AUC
    dataset_comb_results = dataset_comb_results.reset_index(drop=True).reset_index()
    dataset_comb_results.rename(columns={'index': 'Dataset Index'}, inplace=True)

    # --- 3. Select Representative Indices based on Auc Percentiles ---
    # This logic selects 11 datasets whose Auc scores are closest
    # to the 0th, 10th, 20th, ..., 100th percentiles of the AUC distribution.
    print("\nSelecting dataset indices closest to AUC percentiles...")
    selected_indices = []
    percentiles_to_find = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for p in percentiles_to_find:
        # Calculate the percentile value for the 'auc' column
        percentile_value = np.percentile(dataset_comb_results.auc, p)

        # Find the absolute difference between each row's auc and the percentile value
        abs_difference = (dataset_comb_results.auc - percentile_value).abs()

        # Find the index of the row with the minimum difference
        closest_index = abs_difference.idxmin()

        selected_indices.append(closest_index)

        print(f"  - Closest to {p}th percentile (auc ≈ {percentile_value:.4f}) is at index {closest_index} "
              f"(Actual AUC: {dataset_comb_results.loc[closest_index, 'auc']:.4f})")

    # Remove duplicate indices if any were selected more than once
    selected_indices = sorted(list(set(selected_indices)))
    print(f"\nFinal unique selected indices: {selected_indices}")

    return dataset_comb_results, selected_indices




file_path = "src/my_benchmark_results/model_performance_summary.csv"
minimum_threshold = 0.60

# --- Run the Analysis and Plotting ---
# Define the path to your actual results file here
# file_path = "src/my_benchmark_results/model_performance_summary.csv"

results_df, selected_point_indices = select_datasets_by_percentiles(file_path, auc_threshold=minimum_threshold)

if not results_df.empty:
    # Rename columns for better plot labels
    results_df.rename(columns={"accuracy": "Accuracy", "auc": "AUC"}, inplace=True)

    # Create the scatter plot using Plotly Express
    print("\nGenerating plot...")
    fig = px.scatter(
        data_frame=results_df,
        x="Dataset Index",
        y="AUC",
        hover_data=['AUC', 'config_indices']
    )

    # Highlight the selected points
    fig.data[0].update(
        selectedpoints=selected_point_indices,
        selected=dict(marker=dict(color='purple', size=12))
    )

    fig.update_layout(
        title="AUC vs. Dataset Index (Sorted by AUC)",
        xaxis_title="Dataset Index (Ranked by AUC)",
        yaxis_title="Area Under Curve (AUC)"
    )

    # fig.background_color = "WHITE"
    fig.update_layout(
        plot_bgcolor='white'
    )


    # fig.write_image("./All_comp_datasets_auc.png")
    fig.write_html("./All_comp_datasets_auc.html")

    fig.show()



selected_df = results_df.iloc[selected_point_indices]

"[11, 12]"
"[5, 8, 9, 11, 12]"
"[2, 4, 8, 10]"
"[1, 3, 7, 9, 10, 12]"
"[1, 6, 7, 9, 10, 12]"
"[2, 3, 4, 8, 10, 12]"
"[2, 3, 6, 7, 8, 10]"
"[2, 3, 4, 6, 7, 8, 11]"
"[1, 2, 3, 6, 8, 10, 12]"
"[2, 4, 5, 6, 7, 8, 9, 10, 11, 12]"
"[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"











