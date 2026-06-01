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

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.8)

    # --- Sort dataset configurations numerically ---
    dataset_configs = sorted(df['dataset_config_id'].unique(), key=lambda x: int(x.split(' ')[-1]))
    num_plots = len(dataset_configs)

    # --- Create figure with manual subplot layout ---
    cols = 3
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8.5 * cols, 7.5 * rows), sharey=True)
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
        line_height = 0.06
        
        for j, rgs_group in enumerate(hue_order):
            group_data = subplot_data[subplot_data['rgs_group'] == rgs_group].sort_values('step')
            color = palette[j]

            feature_order = group_data[group_data['step'] > 0]['feature_added_back'].tolist()

            # --- FORMATTING CHANGE HERE ---
            try:
                rgs_num = int(rgs_group.replace('RGS', ''))
                formatted_rgs = f"RR{rgs_num:02d}"
            except ValueError:
                formatted_rgs = rgs_group.replace('RGS', 'RR')

            # Manual text wrapping: keep RR number on first line with features
            max_features_per_line = 8
            lines = []
            for k in range(0, len(feature_order), max_features_per_line):
                chunk = feature_order[k:k+max_features_per_line]
                line = " → ".join([f"+{f}" for f in chunk])
                if k == 0:
                    # First line: include RR number
                    lines.append(f"{formatted_rgs}: {line}")
                else:
                    # Subsequent lines: indent to align with features
                    lines.append(f"       {line}")
            
            display_text = "\n".join(lines)
            
            # Calculate how much vertical space this text will need
            num_lines = len(lines)
            text_height = num_lines * line_height

            ax.text(0.02, y_pos, display_text,
                    transform=ax.transAxes,
                    color=color,
                    fontsize=14,
                    fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

            y_pos -= text_height

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
