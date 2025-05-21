# File: evaluation_plots_v2.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import itertools
import numpy as np
import os  # Retained for os.path.join if preferred by user, but pathlib is generally better
from typing import List, Dict, Any, Union, Optional
import logging
# Consistent styling for plots (optional, can be configured further)
sns.set_theme(style="whitegrid")


def plot_metric_distributions(
        results_df: pd.DataFrame,
        metrics_to_plot: List[str],
        output_dir: Path,
        plot_type: str = "violin"
) -> None:
    """
    Generates and saves plots (violin or box) for specified metrics from the results DataFrame.

    Args:
        results_df (pd.DataFrame): DataFrame containing evaluation results.
            Expected columns: 'metric', 'dataset', 'score_list' (list of scores), 'lib' (explainer library).
        metrics_to_plot (List[str]): A list of metric names to plot (e.g., ['recall', 'fpr']).
        output_dir (Path): The directory where plots will be saved.
        plot_type (str): Type of plot to generate, 'violin' or 'box'.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)  # Get logger for this module

    for metric_name in metrics_to_plot:
        logger.info(f"Processing plots for metric: {metric_name}")

        # Filter for the current metric and explode the list of scores
        metric_df = results_df[
            results_df["metric"] == metric_name].copy()  # Use .copy() to avoid SettingWithCopyWarning
        if metric_df.empty:
            logger.warning(f"No data found for metric '{metric_name}'. Skipping plot.")
            continue

        if 'score_list' not in metric_df.columns:
            logger.error(f"'score_list' column not found for metric '{metric_name}'. Skipping plot.")
            continue

        # Explode converts list-like entries in 'score_list' to separate rows
        try:
            # Ensure score_list contains lists/iterables before exploding
            is_list_like = metric_df['score_list'].apply(lambda x: isinstance(x, (list, tuple, np.ndarray)))
            if not is_list_like.all():
                logger.warning(
                    f"Not all 'score_list' entries are list-like for metric '{metric_name}'. Attempting to proceed.")

            exploded_df = metric_df.explode('score_list')
            # Convert scores to numeric, coercing errors to NaN
            exploded_df['score_value'] = pd.to_numeric(exploded_df['score_list'], errors='coerce')
            # Drop rows where conversion to numeric failed, or handle as needed
            exploded_df.dropna(subset=['score_value'], inplace=True)

            if exploded_df.empty:
                logger.warning(
                    f"No valid numeric scores to plot for metric '{metric_name}' after exploding and conversion.")
                continue
        except Exception as e:
            logger.error(f"Error exploding or converting 'score_list' for metric '{metric_name}': {e}. Skipping plot.")
            continue

        plt.figure(figsize=(14, 8))  # Create a new figure for each plot

        if plot_type == "violin":
            sns.violinplot(data=exploded_df, x='dataset', y='score_value', hue='lib',
                           split=True, scale_hue=True, cut=0, inner="quartile")
        elif plot_type == "box":
            sns.boxplot(data=exploded_df, x='dataset', y='score_value', hue='lib')
        else:
            logger.error(f"Unsupported plot_type: '{plot_type}'. Choose 'violin' or 'box'.")
            plt.close()  # Close the figure if plot type is invalid
            continue

        plt.title(f"Distribution of {metric_name.replace('_', ' ').title()} Scores", fontsize=16)
        plt.xlabel("Dataset", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Explainer Library", loc="best")
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        try:
            plot_filename = output_dir / f"{metric_name}_distribution_{plot_type}.png"
            plt.savefig(plot_filename)
            logger.info(f"Saved plot: {plot_filename}")
        except Exception as e:
            logger.error(f"Failed to save plot for metric '{metric_name}': {e}")
        finally:
            plt.close()  # Close the figure to free memory


def plot_dataset_signals_from_csv(
        parent_dir: Path,
        output_dir: Path,
        filename_pattern: str = "datasets_seq_lst*.csv"  # More flexible pattern
) -> None:
    """
    Plots dataset generator signals from CSV files found in the parent directory.

    Args:
        parent_dir (Path): Directory containing the dataset signal CSV files.
        output_dir (Path): Directory where the combined signal plot will be saved.
        filename_pattern (str): Glob pattern to find the relevant CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    all_signals_plot_df = pd.DataFrame(
        columns=["dataset_name", "x_time", "y_signal_value"])  # Renamed columns for clarity

    csv_files_found = list(parent_dir.glob(filename_pattern))
    if not csv_files_found:
        logger.warning(
            f"No CSV files matching pattern '{filename_pattern}' found in '{parent_dir}'. Skipping signal plots.")
        return

    logger.info(f"Found {len(csv_files_found)} CSV files for dataset signals.")

    for csv_file_path in csv_files_found:
        logger.debug(f"Processing signal CSV file: {csv_file_path.name}")
        try:
            # Extract 'size' from filename, e.g., datasets_seq_lst_SOMEID_10000.csv -> 10000
            parts = csv_file_path.stem.split('_')  # .stem removes .csv
            if len(parts) < 2 or not parts[-1].isdigit():
                logger.warning(f"Could not extract 'size' from filename {csv_file_path.name}. Using default size 1.")
                dataset_total_size = 1  # Default or skip
            else:
                dataset_total_size = int(parts[-1])

            raw_signals_df = pd.read_csv(csv_file_path)
            # Remove potential "Unnamed" columns that pandas might add if CSV has leading commas
            raw_signals_df = raw_signals_df.loc[:, ~raw_signals_df.columns.str.contains('^Unnamed')]

            if raw_signals_df.empty:
                logger.warning(f"CSV file {csv_file_path.name} is empty. Skipping.")
                continue

            # Assuming first column is dataset name/identifier, rest are signal values
            if raw_signals_df.shape[1] < 2:
                logger.warning(f"CSV file {csv_file_path.name} has fewer than 2 columns. Skipping.")
                continue

            current_file_plot_data = []
            for _, row in raw_signals_df.iterrows():
                dataset_name = str(row.iloc[0])
                signal_values = row.iloc[1:].dropna().tolist()  # Get signal values, drop NaNs

                if not signal_values:
                    logger.debug(f"No signal values for dataset '{dataset_name}' in {csv_file_path.name}.")
                    continue

                # The 'dataset_por' logic upsamples each signal value
                # dataset_por = portion of the total 'size' that each signal value represents
                if len(signal_values) == 0: continue  # Should be caught by previous check
                points_per_signal_value = int(dataset_total_size / len(signal_values))
                if points_per_signal_value == 0: points_per_signal_value = 1  # Ensure at least one point

                y_upsampled = []
                for val in signal_values:
                    y_upsampled.extend([val] * points_per_signal_value)

                x_time_steps = list(range(len(y_upsampled)))

                for x_val, y_val in zip(x_time_steps, y_upsampled):
                    current_file_plot_data.append({
                        "dataset_name": dataset_name,
                        "x_time": x_val,
                        "y_signal_value": y_val
                    })

            if current_file_plot_data:
                all_signals_plot_df = pd.concat([all_signals_plot_df, pd.DataFrame(current_file_plot_data)],
                                                ignore_index=True)

        except pd.errors.EmptyDataError:
            logger.warning(f"CSV file {csv_file_path.name} is empty or invalid. Skipping.")
        except Exception as e:
            logger.error(f"Error processing dataset signal file {csv_file_path.name}: {e}")

    if all_signals_plot_df.empty:
        logger.info("No data to plot for dataset signals.")
        return

    # Ensure dataset_name is treated as categorical for consistent coloring/styling
    all_signals_plot_df["dataset_name"] = all_signals_plot_df["dataset_name"].astype("category")

    plt.figure(figsize=(15, 8))  # Create a new figure

    # Using a diverse palette and markers
    unique_datasets = all_signals_plot_df["dataset_name"].unique()
    palette = sns.color_palette("husl", n_colors=len(unique_datasets))

    # Markers and linestyles can be cycled if many datasets
    # For simplicity, relying on seaborn's default handling or a simpler cycle if needed.

    sns.lineplot(data=all_signals_plot_df, x="x_time", y="y_signal_value", hue="dataset_name",
                 legend="full", palette=palette, drawstyle='steps-post')  # steps-post is common for this type

    plt.title("Dataset Generator Signals Over Time", fontsize=16)
    plt.xlabel("Time Step / Sample Index (Upsampled)", fontsize=14)
    plt.ylabel("Signal Value", fontsize=14)
    plt.legend(title="Dataset Name", bbox_to_anchor=(1.02, 1), loc='upper left')  # Move legend outside
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend

    try:
        plot_filename = output_dir / "dataset_generator_signals.png"
        plt.savefig(plot_filename)
        logger.info(f"Saved dataset signals plot: {plot_filename}")
    except Exception as e:
        logger.error(f"Failed to save dataset signals plot: {e}")
    finally:
        plt.close()  # Close the figure


def run_all_plots(
        base_results_dir: Union[str, Path],
        evaluation_filename: str = "evaluation_all.pkl",
        metrics_to_plot: Optional[List[str]] = None,
        plot_signal_csv_pattern: str = "datasets_seq_lst*.csv"
) -> None:
    """
    Orchestrates the generation of all standard evaluation plots.

    Args:
        base_results_dir (Union[str, Path]): The parent directory where evaluation results
                                            (like 'evaluation_all.pkl') and dataset signal CSVs are stored.
        evaluation_filename (str): Name of the pickled DataFrame with evaluation results.
        metrics_to_plot (Optional[List[str]]): Specific metrics to plot distributions for.
            If None, defaults to ['recall', 'recall_partial_true', 'recall_partial_false', 'fpr', 'sensitivity'].
            Ensure these names match the 'metric' column in your evaluation_filename.
        plot_signal_csv_pattern (str): Pattern for signal CSV files.
    """
    logger = logging.getLogger(__name__)
    parent_path = Path(base_results_dir)
    plots_output_dir = parent_path / "plots"  # Centralized plots directory
    plots_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot Metric Distributions ---
    evaluation_file_path = parent_path / evaluation_filename
    if evaluation_file_path.exists():
        logger.info(f"Loading evaluation results from: {evaluation_file_path}")
        try:
            results_df = pd.read_pickle(evaluation_file_path)

            if metrics_to_plot is None:
                # Default metrics, assuming how 'recall_partial' might be stored.
                # Adjust these names to exactly match your 'metric' column in the pickle file.
                # For example, if your RecallMetric config is stored, you might filter on metric=='recall' & config.partial==True
                metrics_to_plot = ['recall', 'fpr', 'sensitivity']  # Base metrics
                # Add logic here if 'recall_partial' is a distinct metric name or derived from config
                if 'metric_config' in results_df.columns:  # Hypothetical column
                    if results_df[(results_df['metric'] == 'recall') & (
                    results_df['metric_config'].apply(lambda x: x.get('partial', False) == True))].shape[0] > 0:
                        metrics_to_plot.append(
                            'recall_partial_true_placeholder')  # Placeholder: you'd filter and rename or plot based on config
                elif 'recall_partial' in results_df['metric'].unique():  # If it's a distinct name
                    metrics_to_plot.append('recall_partial')

            logger.info(f"Plotting metric distributions for: {metrics_to_plot}")
            plot_metric_distributions(results_df, metrics_to_plot, plots_output_dir, plot_type="violin")
            plot_metric_distributions(results_df, metrics_to_plot, plots_output_dir,
                                      plot_type="box")  # Also generate box plots
        except Exception as e:
            logger.error(f"Could not load or plot from {evaluation_file_path}: {e}", exc_info=True)
    else:
        logger.warning(
            f"Evaluation results file not found: {evaluation_file_path}. Skipping metric distribution plots.")

    # --- Plot Dataset Signals ---
    # parent_path itself is where CSVs like 'datasets_seq_lst...' are expected.
    logger.info(f"Plotting dataset signals from CSVs in: {parent_path}")
    plot_dataset_signals_from_csv(parent_path, plots_output_dir, filename_pattern=plot_signal_csv_pattern)

    logger.info(f"All plots generated in: {plots_output_dir}")