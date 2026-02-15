import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats
import pickle
import os
from pathlib import Path


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for the given data.

    Args:
        data: Array-like of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        tuple: (lower_bound, upper_bound, margin_of_error)
    """
    n = len(data)
    if n < 2:
        return (np.nan, np.nan, np.nan)

    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)  # t-distribution

    return (mean - margin, mean + margin, margin)


def evaluate_xai_faithfulness(dataset_id, X, y, gt_metadata, model):
    """
    Evaluates SHAP and LIME using Ground Truth features (GT_k).
    Ensures instances are evenly distributed among the rules (RGS) used in the dataset.
    Ref: [cite: 115, 514, 1131]

    Args:
        dataset_id: Dataset configuration ID
        X: Feature DataFrame
        y: Target labels
        gt_metadata: Ground truth metadata DataFrame with 'imp_vars' and 'RGS' columns
        model: Pre-trained XGBoost model

    Returns:
        tuple: (statistics_dict, num_instances_evaluated, detailed_results_df)
    """
    print(f"  Evaluating XAI methods for dataset {dataset_id}...")

    # Get correctly predicted instances
    y_pred = model.predict(X)
    correct_mask = (y_pred == y)

    # Create a DataFrame to track correct predictions with their RGS
    correct_df = pd.DataFrame({
        'index': X.index,
        'correct': correct_mask
    })

    # Merge with metadata to get RGS information
    if isinstance(gt_metadata, pd.DataFrame):
        metadata_with_index = gt_metadata.reset_index(drop=True)
        metadata_with_index['original_index'] = range(len(metadata_with_index))
        correct_df = correct_df.merge(
            metadata_with_index[['original_index', 'RGS']],
            left_on='index',
            right_on='original_index',
            how='left'
        )
    else:
        raise ValueError("gt_metadata must be a DataFrame with 'RGS' column")

    # Filter to only correctly predicted instances
    correct_df = correct_df[correct_df['correct']]

    # Get unique rules (RGS)
    unique_rules = correct_df['RGS'].unique()
    num_rules = len(unique_rules)

    print(f"  Found {len(correct_df)} correctly predicted instances across {num_rules} rules")
    print(f"  Rules: {sorted(unique_rules)}")

    # Calculate how many instances to sample per rule
    instances_per_rule = 1000 // num_rules
    remainder = 1000 % num_rules

    # Select instances evenly distributed across rules
    selected_indices = []

    for i, rule in enumerate(sorted(unique_rules)):
        rule_instances = correct_df[correct_df['RGS'] == rule]

        # Allocate extra instances from remainder to first rules
        num_to_sample = instances_per_rule + (1 if i < remainder else 0)
        num_to_sample = min(num_to_sample, len(rule_instances))

        if num_to_sample > 0:
            sampled = rule_instances.sample(n=num_to_sample, random_state=42)
            selected_indices.extend(sampled['index'].tolist())
            print(f"  Rule {rule}: sampled {num_to_sample} instances (out of {len(rule_instances)} available)")

    num_instances_evaluated = len(selected_indices)
    print(f"  Total instances selected for evaluation: {num_instances_evaluated}")

    # Setup Explainers [cite: 17, 218]
    background = shap.sample(X, 50)
    shap_explainer = shap.KernelExplainer(model.predict_proba, background)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values, feature_names=X.columns, class_names=['0', '1'], mode='classification'
    )

    results = []

    # Evaluate selected instances
    for original_idx in selected_indices:
        # Find the position in the metadata
        position_in_metadata = gt_metadata.index.get_loc(
            original_idx) if original_idx in gt_metadata.index else original_idx

        instance = X.loc[original_idx]

        # Get ground truth for this instance
        if isinstance(gt_metadata, pd.DataFrame):
            if 'imp_vars' in gt_metadata.columns:
                true_gt = gt_metadata.loc[original_idx, 'imp_vars']
            else:
                raise ValueError("Metadata DataFrame must have 'imp_vars' column")
        else:
            true_gt = gt_metadata[position_in_metadata]

        # SHAP Attribution
        shap_values = shap_explainer.shap_values(instance)
        # Handle binary classification output (usually index 1)
        shap_attr = np.abs(shap_values[1] if isinstance(shap_values, list) else shap_values)

        # LIME Attribution
        exp = lime_explainer.explain_instance(instance.values, model.predict_proba, num_features=len(X.columns))
        lime_attr = np.zeros(len(X.columns))
        for feature_idx, val in exp.as_map()[1]:
            lime_attr[feature_idx] = np.abs(val)

        # Calculate Fidelity Metric (GTF) [cite: 1131, 1187]
        def calc_gtf(attr, gt_list, feature_names):
            total_attr = np.sum(attr)
            if total_attr == 0:
                return 0
            gt_indices = [list(feature_names).index(f) for f in gt_list if f in feature_names]
            if not gt_indices:
                return 0
            gt_attr = np.sum(attr[gt_indices])
            return gt_attr / total_attr

        results.append({
            'SHAP_GTF': calc_gtf(shap_attr, true_gt, X.columns),
            'LIME_GTF': calc_gtf(lime_attr, true_gt, X.columns)
        })

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Calculate statistics including confidence intervals
    shap_scores = results_df['SHAP_GTF'].values
    lime_scores = results_df['LIME_GTF'].values

    # Calculate 95% confidence intervals
    shap_ci_lower, shap_ci_upper, shap_margin = calculate_confidence_interval(shap_scores)
    lime_ci_lower, lime_ci_upper, lime_margin = calculate_confidence_interval(lime_scores)

    statistics = {
        'SHAP_GTF_mean': np.mean(shap_scores),
        'SHAP_GTF_std': np.std(shap_scores, ddof=1),  # Sample standard deviation
        'SHAP_GTF_ci_lower': shap_ci_lower,
        'SHAP_GTF_ci_upper': shap_ci_upper,
        'SHAP_GTF_ci_margin': shap_margin,
        'LIME_GTF_mean': np.mean(lime_scores),
        'LIME_GTF_std': np.std(lime_scores, ddof=1),
        'LIME_GTF_ci_lower': lime_ci_lower,
        'LIME_GTF_ci_upper': lime_ci_upper,
        'LIME_GTF_ci_margin': lime_margin,
    }

    print(
        f"  SHAP: {statistics['SHAP_GTF_mean']:.4f} ± {shap_margin:.4f} (95% CI: [{shap_ci_lower:.4f}, {shap_ci_upper:.4f}])")
    print(
        f"  LIME: {statistics['LIME_GTF_mean']:.4f} ± {lime_margin:.4f} (95% CI: [{lime_ci_lower:.4f}, {lime_ci_upper:.4f}])")

    return statistics, num_instances_evaluated, results_df


def load_model(model_path):
    """
    Load a pre-trained XGBoost model from JSON file.

    Args:
        model_path: Path to the model JSON file

    Returns:
        Loaded XGBClassifier model
    """
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def load_dataset_and_metadata(base_path, config_id):
    """
    Load dataset and metadata from pickle files.

    Args:
        base_path: Base directory path containing the dataset files
        config_id: Configuration ID (1-11)

    Returns:
        tuple: (X, y, gt_metadata_df)
    """
    dataset_path = os.path.join(base_path, f'dataset_config_{config_id}.pkl')
    metadata_path = os.path.join(base_path, f'metadata_config_{config_id}.pkl')

    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    # Extract X, y from dataset
    # Based on run_xai_benchmark_v2.py structure, dataset should be a DataFrame
    if isinstance(dataset, pd.DataFrame):
        # Assuming 'y' is the target column
        feature_cols = [col for col in dataset.columns if col != 'y']
        X = dataset[feature_cols]
        y = dataset['y']
    elif isinstance(dataset, dict):
        X = dataset.get('X', dataset.get('data'))
        y = dataset.get('y', dataset.get('target'))
    elif isinstance(dataset, tuple):
        X, y = dataset[0], dataset[1]
    else:
        raise ValueError(f"Unexpected dataset structure: {type(dataset)}")

    # Metadata should be a DataFrame with 'imp_vars' and 'RGS' columns
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError(f"Metadata must be a DataFrame, got {type(metadata)}")

    if 'imp_vars' not in metadata.columns:
        raise ValueError("Metadata DataFrame must have 'imp_vars' column")

    if 'RGS' not in metadata.columns:
        raise ValueError("Metadata DataFrame must have 'RGS' column for rule identification")

    return X, y, metadata


def save_individual_result(results_dir, config_id, result_dict, detailed_results_df):
    """
    Save individual dataset results to CSV files (append mode).

    Args:
        results_dir: Directory to save results
        config_id: Configuration ID
        result_dict: Dictionary containing summary results for this dataset
        detailed_results_df: DataFrame with individual instance scores
    """
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    # Individual summary result file path
    individual_result_path = results_dir / f'dataset_config_{config_id}_results.csv'

    # Convert single result to DataFrame
    result_df = pd.DataFrame([result_dict])

    # Save summary to CSV
    result_df.to_csv(individual_result_path, index=False)
    print(f"  Individual summary saved to: {individual_result_path}")

    # Save detailed instance-level results
    detailed_result_path = results_dir / f'dataset_config_{config_id}_detailed_scores.csv'
    detailed_results_df.to_csv(detailed_result_path, index=False)
    print(f"  Detailed instance scores saved to: {detailed_result_path}")

    # Also append to a combined results file
    combined_results_path = results_dir / 'all_datasets_results.csv'

    # Check if file exists to determine if we need headers
    file_exists = combined_results_path.exists()

    # Append to combined file
    result_df.to_csv(combined_results_path, mode='a', header=not file_exists, index=False)

    if not file_exists:
        print(f"  Created combined results file: {combined_results_path}")
    else:
        print(f"  Appended to combined results file: {combined_results_path}")


def run_experiment():
    """
    Main experiment runner to evaluate SHAP and LIME on all 11 datasets.
    """
    # Define base path
    base_path = Path(__file__).parent.parent / 'my_benchmark_results' / 'SynXAI-DB_run_001'

    # Create results directory based on source dataset folder name
    source_folder_name = base_path.name  # 'SynXAI-DB_run_001'
    results_folder_name = f'{source_folder_name}_initial_expr_results_3'
    results_dir = base_path.parent / results_folder_name

    print(f"Results will be saved to: {results_dir}")

    # Store results for all datasets
    all_results = []

    # Iterate through all 11 datasets
    for config_id in range(1, 12):
        print(f"\n{'=' * 60}")
        print(f"Processing Dataset Config {config_id}")
        print(f"{'=' * 60}")

        try:
            # Load dataset and metadata
            print(f"Loading dataset and metadata...")
            X, y, gt_metadata = load_dataset_and_metadata(base_path, config_id)

            print(f"Dataset shape: {X.shape}")
            print(f"Metadata shape: {gt_metadata.shape}")
            print(f"Unique rules in dataset: {gt_metadata['RGS'].nunique()}")

            # Load pre-trained model
            model_path = base_path / f'model_config_{config_id}.json'
            print(f"Loading pre-trained model from {model_path}...")
            model = load_model(model_path)
            print(f"Model loaded successfully")

            # Evaluate XAI methods - now returns statistics dict, num_instances, and detailed results
            statistics, num_instances_evaluated, detailed_results = evaluate_xai_faithfulness(
                config_id, X, y, gt_metadata, model
            )

            # Store results with confidence intervals
            result_dict = {
                'dataset_id': config_id,
                'num_instances_evaluated': num_instances_evaluated,
                'SHAP_GTF_mean': statistics['SHAP_GTF_mean'],
                'SHAP_GTF_std': statistics['SHAP_GTF_std'],
                'SHAP_GTF_ci_lower': statistics['SHAP_GTF_ci_lower'],
                'SHAP_GTF_ci_upper': statistics['SHAP_GTF_ci_upper'],
                'SHAP_GTF_ci_margin': statistics['SHAP_GTF_ci_margin'],
                'SHAP_time_mean': statistics['SHAP_time_mean'],
                'SHAP_time_total': statistics['SHAP_time_total'],
                'LIME_GTF_mean': statistics['LIME_GTF_mean'],
                'LIME_GTF_std': statistics['LIME_GTF_std'],
                'LIME_GTF_ci_lower': statistics['LIME_GTF_ci_lower'],
                'LIME_GTF_ci_upper': statistics['LIME_GTF_ci_upper'],
                'LIME_GTF_ci_margin': statistics['LIME_GTF_ci_margin'],
                'LIME_time_mean': statistics['LIME_time_mean'],
                'LIME_time_total': statistics['LIME_time_total'],
                'dataset_samples': X.shape[0],
                'dataset_features': X.shape[1],
                'num_rules': gt_metadata['RGS'].nunique()
            }

            all_results.append(result_dict)

            print(f"\nResults for Config {config_id}:")
            print(f"  Instances Evaluated: {num_instances_evaluated}")

            # Save individual result immediately after processing
            save_individual_result(results_dir, config_id, result_dict, detailed_results)

        except Exception as e:
            print(f"Error processing config {config_id}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)

    # Save final summary to both original location and results directory
    # Save to original location (base_path)
    output_path_original = base_path / 'xai_faithfulness_evaluation_results.csv'
    results_df.to_csv(output_path_original, index=False)
    print(f"\n{'=' * 60}")
    print(f"Summary results saved to: {output_path_original}")

    # Save to results directory
    output_path_results = results_dir / 'xai_faithfulness_evaluation_results_summary.csv'
    results_df.to_csv(output_path_results, index=False)
    print(f"Summary results also saved to: {output_path_results}")
    print(f"{'=' * 60}")

    # Display summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Format for better display
    display_df = results_df.copy()
    for col in display_df.columns:
        if 'GTF' in col and col != 'dataset_id':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

    print(display_df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("OVERALL STATISTICS")
    print(f"{'=' * 60}")
    print(
        f"Average SHAP GTF across all datasets: {results_df['SHAP_GTF_mean'].mean():.4f} ± {results_df['SHAP_GTF_std'].mean():.4f}")
    print(
        f"Average LIME GTF across all datasets: {results_df['LIME_GTF_mean'].mean():.4f} ± {results_df['LIME_GTF_std'].mean():.4f}")
    print(f"Total instances evaluated: {results_df['num_instances_evaluated'].sum()}")
    print(f"Average instances per dataset: {results_df['num_instances_evaluated'].mean():.1f}")

    return results_df


if __name__ == "__main__":
    results = run_experiment()