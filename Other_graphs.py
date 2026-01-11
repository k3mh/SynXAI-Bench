#
# from sklearn.datasets import make_gaussian_quantiles, make_hastie_10_2, make_classification, make_moons
# import matplotlib.pyplot as plt
# import plotly.express as px
# import pandas as pd
# import importlib
# import numpy as np
#
# importlib.reload(px)
#
#
# """Madelon rule example"""
# from sklearn.datasets import  make_classification
#
# temp_arr  = make_classification( n_samples=10000, n_features=2,  n_informative=2, n_redundant=0, n_repeated=0, class_sep=3, n_clusters_per_class=2)
# temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0],\
#                         "x2": temp_arr[0].transpose()[1], \
#                         "y": temp_arr[1]})
# temp_pd.y = temp_pd.y.astype(str)
# fig = px.scatter(data_frame = temp_pd , x="x1",  y="x2", color="y")
# fig.update_layout(
#     title={
#         'text': "Simple example of 2 informative features with 2 clusters per class",
#         'y':0.95,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'})
# fig.show()
#
# """ Madelon rule end"""
#
# """ make blobs example """
# from sklearn.datasets import make_blobs
# temp_arr  = make_blobs( n_samples=10000, n_features=2, centers=6)
# temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0],\
#                         "x2": temp_arr[0].transpose()[1], \
#                         "y": temp_arr[1]})
#
# # temp_pd.y = temp_pd.y.map( lambda x: 0 if x in (1,2,3) else 1)
# temp_pd.y = temp_pd.y.astype(str)
# fig = px.scatter(data_frame = temp_pd , x="x1",  y="x2", color="y")
# fig.update_layout(
#     title={
#         'text': "Simple example of 2 class problem based on 2 features",
#         'y':0.95,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'})
# fig.show()
#
# """ end of make blobs"""
#
#
# """ make moons example """
# from sklearn.datasets import make_blobs
# temp_arr  = make_moons( n_samples=10000, random_state=100, noise = 0.1)
# temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0],\
#                         "x2": temp_arr[0].transpose()[1], \
#                         "y": temp_arr[1]})
#
# # temp_pd.y = temp_pd.y.map( lambda x: 0 if x in (1,2,3) else 1)
# temp_pd.y = temp_pd.y.astype(str)
# fig = px.scatter(data_frame = temp_pd , x="x1",  y="x2", color="y")
# fig.update_layout(
#     title={
#         # 'text': "Simple example of 2 class problem based on 2 features",
#         'y':0.95,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'})
# fig.show()
#
# """ end of make blobs"""
#
# """"dataset accuracy plot : start"""
# fig = px.line(x=range(1, len(accuracy_lst)+1), y=accuracy_lst, markers=True, range_x= range(1, len(accuracy_lst)+2))
# fig.update_layout(
#     title={
#         # 'text': "Datasets performance based on individual rules.",
#         'y':0.95,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'})
# fig.show()
# """"dataset accuracy plot : end """
#
#
# """ Dataset combinations and selection"""
# file_path="src/my_benchmark_results/model_performance_summary.csv"
# minimum_threshold = 0.60
# dataset_comb_results = pd.read_csv(file_path)
# dataset_comb_results = dataset_comb_results.loc[dataset_comb_results.auc >= minimum_threshold]
# dataset_comb_results.sort_values("auc", inplace=True, ascending=False)
# # dataset_comb_results.drop(columns=["Unnamed: 0"], inplace=True)
# dataset_comb_results = dataset_comb_results.reset_index(drop=True).reset_index()
# selected_inds=[]
# for i in [0, 10, 20 , 30 , 40 , 50 , 60 , 70 , 80 , 90 , 100]:
#     selected_inds.append(dataset_comb_results.iloc[(dataset_comb_results.accuracy - np.percentile(
#         dataset_comb_results.accuracy, i)).abs().argsort()[:1]].index.values[0])
# dataset_comb_results = dataset_comb_results.rename(columns={"index":"Dataset Index", "accuracy":"Accuracy", "auc":"AUC"})
# fig = px.scatter(data_frame=dataset_comb_results, x="Dataset Index", y="AUC" )
# fig.data[0].update(selectedpoints=selected_inds, selected=dict(marker=dict(color='purple', size=11)))
#
# fig.show()
#
#
# ####################### OPS surface function #############
# import numpy as np
# import plotly.graph_objects as go
# import pandas as pd
#
# def z_func(cov, com):
#     z=np.sqrt(np.power(100-cov,2) + np.power(com, 2)) -1
#     return z
#
# OPS=[]
# coverage = []
# complexity = []
# cov = np.linspace(0, 100, 30)
# comp = np.linspace(1, 100, 30)
#
# coverage, complexity = np.meshgrid(cov, comp)
# OPS = z_func(coverage, complexity)
#
#
# # Read data from a csv
# # z_data = pd.DataFrame({"OPS": OPS, "Complexity":complexity, "Coverage":coverage}, index=range(len(OPS)))
#
# fig = go.Figure(data=[go.Surface(x=coverage, y=complexity, z=OPS)])
#
# fig.update_layout(title='OPS Function Surface', autosize=False,
#                   width=800, height=800,
#                   margin=dict(l=65, r=50, b=65, t=90),
#                   scene=dict(
#                       xaxis_title='Coverage',
#                       yaxis_title='Complexity',
#                       zaxis_title='OPS',
#                   ),
#                   )
#
# fig.show()
#

######################################################################################################################

#
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import os
#
#
# def select_datasets_by_percentiles(file_path: str, auc_threshold: float = 0.60) -> pd.DataFrame:
#     """
#     Loads model performance data, sorts by AUC, and selects a representative
#     sample of datasets based on the percentiles of their accuracy scores.
#
#     Args:
#         file_path (str): The path to the model performance CSV file.
#         auc_threshold (float): The minimum AUC score to include in the analysis.
#
#     Returns:
#         pd.DataFrame: The full, sorted DataFrame with a new 'Dataset Index' column.
#                       Plotting should be handled separately.
#     """
#     if not os.path.exists(file_path):
#         print(f"Error: The file was not found at {file_path}")
#         return pd.DataFrame(), []
#
#     # --- 1. Load and Filter Data ---
#     dataset_comb_results = pd.read_csv(file_path)
#
#     # Filter out runs that are below the minimum AUC threshold
#     dataset_comb_results = dataset_comb_results.loc[dataset_comb_results.auc >= auc_threshold]
#     if dataset_comb_results.empty:
#         print(f"No datasets found with AUC >= {auc_threshold}")
#         return pd.DataFrame(), []
#
#     # --- 2. Sort by AUC (Full Precision) ---
#     # This is the key step you asked about. Pandas sorts using the full
#     # precision of the float64 numbers, not a rounded version.
#     # The sorting is accurate.
#     dataset_comb_results.sort_values("auc", inplace=True, ascending=False)
#
#     # Create a new index that represents the rank after sorting by AUC
#     dataset_comb_results = dataset_comb_results.reset_index(drop=True).reset_index()
#     dataset_comb_results.rename(columns={'index': 'Dataset Index'}, inplace=True)
#
#     # --- 3. Select Representative Indices based on Auc Percentiles ---
#     # This logic selects 11 datasets whose Auc scores are closest
#     # to the 0th, 10th, 20th, ..., 100th percentiles of the AUC distribution.
#     print("\nSelecting dataset indices closest to AUC percentiles...")
#     selected_indices = []
#     percentiles_to_find = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#
#     for p in percentiles_to_find:
#         # Calculate the percentile value for the 'auc' column
#         percentile_value = np.percentile(dataset_comb_results.auc, p)
#
#         # Find the absolute difference between each row's auc and the percentile value
#         abs_difference = (dataset_comb_results.auc - percentile_value).abs()
#
#         # Find the index of the row with the minimum difference
#         closest_index = abs_difference.idxmin()
#
#         selected_indices.append(closest_index)
#
#         print(f"  - Closest to {p}th percentile (auc ≈ {percentile_value:.4f}) is at index {closest_index} "
#               f"(Actual AUC: {dataset_comb_results.loc[closest_index, 'auc']:.4f})")
#
#     # Remove duplicate indices if any were selected more than once
#     selected_indices = sorted(list(set(selected_indices)))
#     print(f"\nFinal unique selected indices: {selected_indices}")
#
#     return dataset_comb_results, selected_indices
#
#
#
#
# file_path = "src/my_benchmark_results/model_performance_summary.csv"
# minimum_threshold = 0.60
#
# # --- Run the Analysis and Plotting ---
# # Define the path to your actual results file here
# # file_path = "src/my_benchmark_results/model_performance_summary.csv"
#
# results_df, selected_point_indices = select_datasets_by_percentiles(file_path, auc_threshold=minimum_threshold)
#
# if not results_df.empty:
#     # Rename columns for better plot labels
#     results_df.rename(columns={"accuracy": "Accuracy", "auc": "AUC"}, inplace=True)
#
#     # Create the scatter plot using Plotly Express
#     print("\nGenerating plot...")
#     fig = px.scatter(
#         data_frame=results_df,
#         x="Dataset Index",
#         y="AUC",
#         hover_data=['AUC', 'config_indices']
#     )
#
#     # Highlight the selected points
#     fig.data[0].update(
#         selectedpoints=selected_point_indices,
#         selected=dict(marker=dict(color='purple', size=12))
#     )
#
#     fig.update_layout(
#         title="AUC vs. Dataset Index (Sorted by AUC)",
#         xaxis_title="Dataset Index (Ranked by AUC)",
#         yaxis_title="Area Under Curve (AUC)"
#     )
#
#     # fig.background_color = "WHITE"
#     fig.update_layout(
#         plot_bgcolor='white'
#     )
#
#
#     # fig.write_image("./All_comp_datasets_auc.png")
#     fig.write_html("./All_comp_datasets_auc.html")
#
#     fig.show()
#



###########################################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path


def load_and_process_data(results_dir: Path, num_configs: int = 11) -> pd.DataFrame:
    """
    Loads all validation CSVs, combines them, and restructures the data
    to include an explicit baseline (Step 0) for plotting.
    """
    all_dfs = []
    print(f"\nLoading data from '{results_dir}'...")
    for i in range(1, num_configs + 1):
        file_path = results_dir / f"validation_impact_correct_samples_auc_config_{i}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df['dataset_config_id'] = f"Config {i}"
                all_dfs.append(df)
            except pd.errors.EmptyDataError:
                print(f"Warning: File is empty, skipping: {file_path}")
        else:
            print(f"Warning: File not found, skipping: {file_path}")

    if not all_dfs:
        print("Error: No data files found. Cannot generate plot.")
        return pd.DataFrame()

    full_df = pd.concat(all_dfs, ignore_index=True)

    # --- Restructure data to create an explicit baseline point for each group ---
    baseline_rows = []
    # Group by each unique validation run
    for group_keys, group_df in full_df.groupby(['dataset_config_id', 'rgs_group']):
        # The first step of the additive process contains the baseline score
        first_step = group_df.sort_values('score_before_adding').iloc[0]

        baseline_score = first_step['score_before_adding']

        # Create a new row for the baseline (Step 0)
        baseline_row = {
            'dataset_config_id': group_keys[0],
            'rgs_group': group_keys[1],
            'step': 0,
            'score_after_adding': baseline_score,
            'original_group_score': first_step['original_group_score'],
            'feature_added_back': 'Baseline'
        }
        baseline_rows.append(baseline_row)

    baseline_df = pd.DataFrame(baseline_rows)

    # Increment the step for the original data so it starts from 1
    full_df['step'] = full_df.groupby(['dataset_config_id', 'rgs_group']).cumcount() + 1

    # Combine the new baseline data with the original stepped data
    final_df = pd.concat([baseline_df, full_df], ignore_index=True).sort_values(
        by=['dataset_config_id', 'rgs_group', 'step']
    )

    print("Data loaded and processed successfully.")
    return final_df


def create_validation_plot(df: pd.DataFrame, output_path: Path):
    """
    Generates and saves a high-quality facet grid plot for the validation results.
    """
    if df.empty:
        return

    print(f"\nGenerating plot...")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # --- Sort dataset configurations numerically ---
    dataset_configs = sorted(df['dataset_config_id'].unique(), key=lambda x: int(x.split(' ')[-1]))
    num_plots = len(dataset_configs)

    # --- Create figure with manual subplot layout ---
    cols = 3
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows), sharey=True)
    axes = axes.flatten()

    # --- Custom plotting loop for robust line drawing ---
    for i, config_id in enumerate(dataset_configs):
        ax = axes[i]
        subplot_data = df[df['dataset_config_id'] == config_id]

        # --- NEW SORTING LOGIC for annotations and plotting order ---
        rgs_groups_in_subplot = sorted(subplot_data['rgs_group'].unique())
        sort_info = []
        for rgs_group in rgs_groups_in_subplot:
            group_data = subplot_data[subplot_data['rgs_group'] == rgs_group]
            feature_order = group_data[group_data['step'] > 0].sort_values('step')['feature_added_back'].tolist()

            first_feature = feature_order[0] if feature_order else ''

            # Create a numerical sort key from the feature name (e.g., 'x12' -> 12)
            try:
                sort_key = int(first_feature.replace('x', ''))
            except (ValueError, AttributeError):
                sort_key = float('inf')  # Put groups with no features/non-standard names last

            sort_info.append((sort_key, rgs_group))

        sort_info.sort()
        hue_order = [rgs for key, rgs in sort_info]
        # --- END NEW SORTING LOGIC ---

        palette = sns.color_palette("deep", n_colors=len(hue_order))

        # Plot each RGS group's line individually using the new sorted order
        for j, rgs_group in enumerate(hue_order):
            group_data = subplot_data[subplot_data['rgs_group'] == rgs_group].sort_values('step')
            color = palette[j]

            ax.plot(
                group_data["step"], group_data["score_after_adding"],
                marker='o', markersize=7, color=color, label=rgs_group, zorder=5 + j
            )

            # Add horizontal line for original score
            original_score = group_data['original_group_score'].iloc[0]
            ax.axhline(original_score, ls='--', color=color, lw=1.5, alpha=0.6)

        # Add a text block with feature orders, also using the new sorted order
        y_pos = 0.98
        for j, rgs_group in enumerate(hue_order):
            group_data = subplot_data[subplot_data['rgs_group'] == rgs_group].sort_values('step')
            color = palette[j]

            feature_order = group_data[group_data['step'] > 0]['feature_added_back'].tolist()

            order_str = " → ".join([f"+{f}" for f in feature_order])

            # --- FORMATTING CHANGE HERE ---
            try:
                rgs_num = int(rgs_group.replace('RGS', ''))
                formatted_rgs = f"RR{rgs_num:02d}"
            except ValueError:
                formatted_rgs = rgs_group.replace('RGS', 'RR')  # Fallback

            display_text = f"{formatted_rgs}: {order_str}"

            ax.text(0.02, y_pos, display_text,
                    transform=ax.transAxes,
                    color=color,
                    fontsize=9,
                    fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

            y_pos -= 0.07

        # --- Customize titles, labels, and ticks ---
        config_number = config_id.split(' ')[-1]
        ax.set_title(f"Dataset {config_number}", size=18, pad=10)

        max_steps = subplot_data['step'].max()
        tick_positions = range(max_steps + 1)
        tick_labels = ['Baseline'] + list(range(1, max_steps + 1))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=11)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)

        if i % cols == 0:
            ax.set_ylabel("Model AUC Score", size=14)
        if i >= (num_plots - cols):
            ax.set_xlabel("Feature Restoration Step", size=14)

    # --- Hide any unused subplots ---
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Additive Counterfactual Validation: Model AUC Recovery", fontsize=22, y=1.0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    # --- Configuration ---
    RESULTS_DIRECTORY = Path("src/my_benchmark_results/SynXAI-DB_run_001")

    if not RESULTS_DIRECTORY.exists():
        print(f"Error: Directory '{RESULTS_DIRECTORY}' not found.")
        print("Please ensure the path is correct and contains the necessary validation CSV files.")
    else:
        processed_df = load_and_process_data(RESULTS_DIRECTORY)

        if not processed_df.empty:
            output_plot_path = RESULTS_DIRECTORY / "validation_impact_summary.png"
            create_validation_plot(processed_df, output_plot_path)

############################################################################################################
#
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.utils import check_random_state, shuffle as util_shuffle
# from typing import List, Tuple
#
#
# def make_parabolic_ranked_for_viz(
#         n_samples=100,
#         *,
#         feature_weights: List[float] = None,
#         noise=None,
#         random_state=None
# ):
#     """
#     A modified version of the generator that also returns the intermediate
#     decision values for visualization purposes.
#     """
#     if feature_weights is None:
#         feature_weights = [1.0, 1.0]
#
#     if len(feature_weights) != 2:
#         raise ValueError("feature_weights must be a list of length 2.")
#
#     generator = check_random_state(random_state)
#
#     # Step 1: Generate a Base Point Cloud (X_base)
#     X_base = generator.uniform(low=[-1.5, -1.5], high=[1.5, 1.5], size=(n_samples, 2))
#
#     # Step 2: Create the Hidden Signal (Weighted Decision Function)
#     weights = np.array(feature_weights)
#     decision_values = (weights[1] * X_base[:, 1]) - (weights[0] * X_base[:, 0] ** 2)
#
#     # Step 3: Calculate y Labels Directly from the Hidden Signal
#     threshold = np.median(decision_values)
#     y = (decision_values > threshold).astype(int)
#
#     # The returned X is the base one, but we add noise for the final plot
#     X_final = X_base.copy()
#     if noise is not None:
#         X_final += generator.normal(scale=noise, size=X_final.shape)
#
#     # Return all components needed for visualization
#     return X_base, X_final, y, decision_values, threshold
#
#
# if __name__ == "__main__":
#     # --- Configuration ---
#     n_samples = 1500
#     noise = 0.1
#     random_state = 42
#
#     # Define different weight scenarios to visualize their effect
#     weight_scenarios = {
#         "Equal Weights [1.0, 1.0]": [1.0, 1.0],
#         "High X² Weight [4.0, 1.0] (Narrower Parabola)": [4.0, 1.0],
#         "High Y Weight [1.0, 4.0] (Steeper Vertical Influence)": [1.0, 4.0]
#     }
#
#     num_scenarios = len(weight_scenarios)
#
#     # --- Visualization ---
#     sns.set_theme(style="whitegrid", context="talk")
#     fig, axes = plt.subplots(num_scenarios, 3, figsize=(21, 6.0 * num_scenarios), squeeze=False)
#     palette = "viridis"
#
#     # --- Loop through each weight scenario and plot ---
#     for i, (scenario_title, feature_weights) in enumerate(weight_scenarios.items()):
#         # --- Generate Data for the current scenario ---
#         X_base, X_final, y, decision_values, threshold = make_parabolic_ranked_for_viz(
#             n_samples=n_samples,
#             feature_weights=feature_weights,
#             noise=noise,
#             random_state=random_state
#         )
#
#         # Set a title for the entire row
#         axes[i, 0].set_ylabel(scenario_title, fontsize=16, labelpad=20)
#
#         # --- Plot 1: The "Hidden Signal" (The Rule) ---
#         scatter1 = axes[i, 0].scatter(
#             X_base[:, 0], X_base[:, 1], c=decision_values, cmap='coolwarm', s=20
#         )
#         axes[i, 0].tricontour(
#             X_base[:, 0], X_base[:, 1], decision_values, levels=[threshold],
#             colors='black', linewidths=3
#         )
#         axes[i, 0].set_title("1. The 'Hidden Signal' (Decision Values)", fontsize=16, pad=15)
#         axes[i, 0].set_xlabel("Feature 1")
#         fig.colorbar(scatter1, ax=axes[i, 0], label="Decision Value (d)")
#
#         # --- Plot 2: The "Puzzle" (Data without noise) ---
#         axes[i, 1].scatter(
#             X_base[:, 0], X_base[:, 1], c=y, cmap=palette, s=20, alpha=0.9
#         )
#         axes[i, 1].set_title("2. The 'Puzzle' (Clean Data & Final Labels)", fontsize=16, pad=15)
#         axes[i, 1].set_xlabel("Feature 1")
#
#         # --- Plot 3: The Final Dataset (with Noise) ---
#         axes[i, 2].scatter(
#             X_final[:, 0], X_final[:, 1], c=y, cmap=palette, s=20, alpha=0.9
#         )
#         axes[i, 2].set_title("3. The Final Dataset (with Noise)", fontsize=16, pad=15)
#         axes[i, 2].set_xlabel("Feature 1")
#
#     # Overall figure adjustments
#     fig.suptitle("Visualizing the Effect of feature_weights in 'make_parabolic_ranked'", fontsize=22, y=1.0)
#     plt.tight_layout(h_pad=4.0, rect=[0, 0, 1, 0.96])  # Adjust rect to make space for suptitle
#     plt.show()
#
