# File: run_xai_benchmark.py
import sys
import logging
import argparse
import gc
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np  # Added for np.floor, np.sum used in proportion calculation

# Assuming these are your refactored module names
import Evaluation_plots_v2
import dataset_generation.DataSetGen_v3 as sdg  # Updated import path
import Evaluation_v2
import Explaination_v2

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# --- Global Configuration & Logging ---
# BasicConfig should ideally be called only once.
# If other modules also call it, it might not behave as expected.
# It's better if the main application entry point is the sole configurator.
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s',
                    level=logging.DEBUG, force=True)  # force=True if other modules might configure

logger = logging.getLogger("xai_benchmark_runner")
# Ensure level is set if basicConfig was already called by an import
logger.setLevel(logging.DEBUG)

# Avoid duplicate handlers if script is re-run in some environments (e.g. notebooks)
if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
    if not logger.handlers:  # Or only add if no handlers exist at all
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

TARGET_NAME = "y"
DEFAULT_DATASETS_SEQUENCES = [
    [1, 3], [3, 9, 10], [1, 5, 6, 11, 12], [1, 3, 7, 9],
    [2, 3, 4, 5, 6, 8, 10, 12], [2, 3, 5, 6, 8, 9, 10, 11],
    [1, 2, 3, 5, 6, 8, 9, 10, 11, 12], [1, 4, 5, 6, 7, 8, 10, 11, 12],
    [2, 3, 4, 5, 7, 8, 10], [4, 5, 6, 7, 8, 10, 11, 12], [5, 6, 7, 9, 10]
]


# --- Helper Functions from original script (generate_composite_dataset, train_model) ---
def generate_composite_dataset(
        dataset_indices: List[int],
        proportions: List[float],
        total_size: int,
        base_feature_names: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    all_data_dfs = []
    all_meta_dfs = []
    current_offset = 0
    actual_total_size = 0

    for i, dataset_idx in enumerate(dataset_indices):
        dataset_name_str = f"ds{dataset_idx}"
        if i < len(proportions) - 1:
            prop_size = int(proportions[i] * total_size)
            current_offset += prop_size
        else:
            prop_size = total_size - current_offset
        if prop_size <= 0:
            logger.warning(f"Calculated proportion size for {dataset_name_str} is {prop_size}. Skipping.")
            continue
        actual_total_size += prop_size
        logger.debug(f"Generating {prop_size} samples for {dataset_name_str}")
        try:
            synthetic_ds_obj = sdg.generate_dataset_by_name(name=dataset_name_str, size=prop_size)
            all_data_dfs.append(synthetic_ds_obj.data)
            all_meta_dfs.append(synthetic_ds_obj.meta_data)
        except Exception as e:
            logger.error(f"Error generating {dataset_name_str}: {e}", exc_info=True)
            continue
    if not all_data_dfs:
        return pd.DataFrame(), pd.DataFrame(), base_feature_names
    final_data_df = pd.concat(all_data_dfs, ignore_index=True)
    final_meta_df = pd.concat(all_meta_dfs, ignore_index=True)
    final_data_df = final_data_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_meta_df = final_meta_df.loc[final_data_df.index].reset_index(drop=True)
    logger.info(f"Generated composite dataset with {len(final_data_df)} samples.")
    return final_data_df, final_meta_df, base_feature_names


def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, random_state: int):
    if model_name.lower() == 'randomforest':
        model = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    elif model_name.lower() == 'extratrees':
        model = ExtraTreesClassifier(n_jobs=-1, random_state=random_state)
    elif model_name.lower() == 'xgboost':
        model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    logger.info(f"Training {model_name} model...")
    model.fit(X_train.values, y_train)
    logger.info(f"{model_name} model trained.")
    return model


# --- New Structured Functions ---

def setup_run_directory_and_config(args: argparse.Namespace) -> Tuple[Path, List[List[int]], List[List[float]]]:
    """Sets up the output directory and prepares dataset configurations."""
    run_output_dir = Path(args.output_dir) / args.run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run output will be saved to: {run_output_dir}")

    dataset_configs_to_run = DEFAULT_DATASETS_SEQUENCES

    dataset_proportions_list = []
    for seq in dataset_configs_to_run:
        if not seq: continue
        prop_val = 1.0 / len(seq)
        props = [prop_val] * len(seq)
        current_sum = np.sum(props[:-1])
        if len(props) > 0:
            props[-1] = 1.0 - current_sum
        dataset_proportions_list.append(props)

    pd.DataFrame(dataset_configs_to_run).to_csv(
        run_output_dir / f"run_dataset_sequences_size_{args.dataset_size}.csv",
        index=False, header=False
    )
    return run_output_dir, dataset_configs_to_run, dataset_proportions_list


def generate_all_explanations_for_config(
        dataset_id_str: str,
        X_test_df_processed: pd.DataFrame,
        X_train_df: pd.DataFrame,
        ml_model: Any,
        current_feature_names: List[str],
        args: argparse.Namespace,
        run_output_dir: Path
) -> Dict[str, pd.DataFrame]:
    """Generates or loads explanations for all specified libraries."""
    all_explanations = {}
    explainer_libs_to_run = ["lime", "anchor"]

    for lib_name in explainer_libs_to_run:
        exp_pkl_path = run_output_dir / f"explanations_{lib_name}_{dataset_id_str}.pkl"
        if args.load_explanations and exp_pkl_path.exists():
            logger.info(f"Loading {lib_name} explanations for {dataset_id_str} from file.")
            try:
                all_explanations[lib_name] = pd.read_pickle(exp_pkl_path)
            except Exception as e:
                logger.error(
                    f"Error loading {lib_name} explanations for {dataset_id_str} from {exp_pkl_path}: {e}. Will attempt to generate.",
                    exc_info=True)
                # Fall through to generation if load fails and generation is allowed
                if args.skip_explanations:  # If skipping explanations, don't generate
                    all_explanations[lib_name] = pd.DataFrame()
                    continue
                    # If not skipping, the 'else' block for generation will be hit.

        # This 'else' covers:
        # 1. Not loading (--load_explanations=False)
        # 2. Attempted to load but file didn't exist
        # 3. Attempted to load but failed (and generation is not skipped)
        if not (args.load_explanations and exp_pkl_path.exists() and lib_name in all_explanations and not
        all_explanations[lib_name].empty):
            if args.skip_explanations:  # Double check skip flag before generating
                logger.info(
                    f"Skipping generation of {lib_name} explanations for {dataset_id_str} as per --skip_explanations.")
                all_explanations[lib_name] = pd.DataFrame()
                continue

            if args.load_explanations and not exp_pkl_path.exists():
                logger.warning(
                    f"Explanations file for {lib_name} on {dataset_id_str} not found at {exp_pkl_path}. Generating instead.")
            elif args.load_explanations and lib_name in all_explanations and all_explanations[
                lib_name].empty:  # Loaded an empty df
                logger.warning(
                    f"Loaded empty explanations for {lib_name} on {dataset_id_str}. Will attempt to regenerate.")

            logger.info(f"Generating {lib_name} explanations for {dataset_id_str}...")
            explanations_df = pd.DataFrame()
            try:
                if lib_name == "lime":
                    explainer_wrapper = Explaination_v2.get_explainer_wrapper(lib_name, ml_model)
                    explainer_wrapper.fit(
                        training_data=X_train_df, feature_names=current_feature_names,
                        class_names=[str(c) for c in ml_model.classes_], mode="classification"
                    )
                    explanations_df = explainer_wrapper.explain_dataset(
                        X_test_df_processed, parallel=args.parallel_explain, n_jobs=args.n_jobs,
                        num_features=args.lime_num_features
                    )
                elif lib_name == "anchor":
                    categorical_names_anchor = {}
                    explainer_wrapper = Explaination_v2.get_explainer_wrapper(lib_name, ml_model)
                    explainer_wrapper.fit(
                        training_data=X_train_df, feature_names=current_feature_names,
                        class_names=[str(c) for c in ml_model.classes_],
                        categorical_names=categorical_names_anchor
                    )
                    explanations_df = explainer_wrapper.explain_dataset(
                        X_test_df_processed, parallel=args.parallel_explain, n_jobs=args.n_jobs,
                        threshold=args.anchor_threshold
                    )
                else:
                    logger.warning(f"Explainer {lib_name} not configured for generation in this loop.")

                all_explanations[lib_name] = explanations_df
                if args.save_explanations and not explanations_df.empty:
                    logger.info(f"Saving {lib_name} explanations for {dataset_id_str}.")
                    explanations_df.to_pickle(exp_pkl_path)
            except Exception as e:
                logger.error(f"Error generating {lib_name} explanations for {dataset_id_str}: {e}", exc_info=True)
                all_explanations[lib_name] = pd.DataFrame()
    return all_explanations


def evaluate_all_explanations_for_config(
        dataset_id_str: str,
        ds_indices_config_str: str,
        dict_of_explanations: Dict[str, pd.DataFrame],
        meta_test_df_aligned: pd.DataFrame,
        current_feature_names: List[str],
        args: argparse.Namespace
) -> List[Dict[str, Any]]:
    """Calculates evaluation metrics for all explanations."""
    config_metric_results_list = []

    ground_truth_for_eval = meta_test_df_aligned[['imp_vars', 'RGS']].copy()
    # 'instance_idx' in meta_test_df_aligned is the original index.
    # Explanations have 0-based 'instance_idx'.
    # The _align_data method in Evaluation_v2.BaseEvaluationMetric handles merging
    # based on 'instance_idx' columns. We need to ensure these columns are consistently named
    # and represent the same conceptual instance.
    # If meta_test_df_aligned's 'instance_idx' is original, and explanations_df's 'instance_idx' is 0-based,
    # they won't align directly.
    # Solution: Pass meta_test_df_aligned as is. The _align_data will reset its index if 'instance_idx' is not a column,
    # or use it if it is. Explanations_df should also have a consistent 'instance_idx' (0-based).
    # The crucial part is that both DataFrames passed to calculator.calculate() can be aligned.
    # Let's assume meta_test_df_aligned's 'instance_idx' (original index) is what we want to match against
    # if explanations_df also used these original indices.
    # However, X_test_df_processed had its index reset. So explanations_df's instance_idx is 0-based.
    # Thus, ground_truth_for_eval must also have a 0-based instance_idx.
    ground_truth_for_eval['instance_idx'] = range(len(ground_truth_for_eval))

    for lib_name, explanations_df in dict_of_explanations.items():
        if explanations_df is None or explanations_df.empty:
            logger.warning(f"No explanations to evaluate for {lib_name} on {dataset_id_str}.")
            continue

        if 'instance_idx' not in explanations_df.columns:
            logger.error(
                f"Explanations for {lib_name} on {dataset_id_str} missing 'instance_idx'. Skipping evaluation.")
            continue

        logger.info(f"Evaluating {lib_name} explanations for {dataset_id_str}...")
        metrics_to_calculate = {
            "recall_partial": Evaluation_v2.get_metric_calculator("recall", partial=True),
            "recall_full": Evaluation_v2.get_metric_calculator("recall", partial=False),
            "fpr": Evaluation_v2.get_metric_calculator("fpr"),
            "sensitivity": Evaluation_v2.get_metric_calculator("sensitivity", top_k=args.sensitivity_top_k)
        }
        for metric_key, calculator in metrics_to_calculate.items():
            logger.debug(f"Calculating {metric_key} for {lib_name} on {dataset_id_str}")
            try:
                metric_result = calculator.calculate(
                    ground_truth_df=ground_truth_for_eval,
                    explanations_df=explanations_df,
                    all_feature_names=current_feature_names
                )
                config_metric_results_list.append({
                    "dataset_id": dataset_id_str, "config_indices_str": ds_indices_config_str,
                    "metric_name": metric_result.get('metric_name', metric_key),
                    "metric_config": str(metric_result.get('config', {})), "lib": lib_name,
                    "average_score": metric_result.get('average_score'),
                    "score_list": metric_result.get('instance_scores'),
                })
            except Exception as e:
                logger.error(f"Error calculating metric {metric_key} for {lib_name} on {dataset_id_str}: {e}",
                             exc_info=True)
    return config_metric_results_list


def perform_final_steps(
        all_run_results_df: pd.DataFrame,
        model_performance_records: List[Dict[str, Any]],
        dataset_configs_to_run_count: int,
        args: argparse.Namespace,
        run_output_dir: Path
):
    """Handles saving aggregated results, plotting, and final score calculations."""
    if args.save_evaluation and not all_run_results_df.empty:
        final_results_pkl_path = run_output_dir / "evaluation_all_metrics.pkl"
        logger.info(f"Saving all aggregated evaluation results to {final_results_pkl_path}")
        all_run_results_df.to_pickle(final_results_pkl_path)
    elif not all_run_results_df.empty:
        logger.info("Aggregated evaluation results were generated but --save_evaluation is False. Not saving.")
    else:
        logger.info("No aggregated evaluation results to save.")

    if model_performance_records:
        model_perf_df = pd.DataFrame(model_performance_records)
        if not model_perf_df.empty:
            model_perf_csv_path = run_output_dir / "model_performance_summary.csv"
            logger.info(f"Saving model performance summary to {model_perf_csv_path}")
            model_perf_df.to_csv(model_perf_csv_path, index=False)
        else:
            logger.info("No model performance records to save (DataFrame was empty).")
    else:
        logger.info("No model performance records to save (list was empty).")

    if args.generate_plots:
        if not all_run_results_df.empty:
            logger.info("Generating plots...")
            metrics_for_plotting = all_run_results_df['metric_name'].unique().tolist()
            if not metrics_for_plotting: metrics_for_plotting = ['recall_full', 'recall_partial', 'fpr', 'sensitivity']
            Evaluation_plots_v2.run_all_plots(
                base_results_dir=run_output_dir,
                evaluation_filename="evaluation_all_metrics.pkl",
                metrics_to_plot=metrics_for_plotting,
                plot_signal_csv_pattern=f"run_dataset_sequences_size_*.csv"
            )
        else:
            logger.warning("Skipping plot generation as there are no evaluation results.")

    if args.calculate_final_scores:
        if not all_run_results_df.empty and model_performance_records:
            logger.info("Calculating final aggregate scores (Overall and XFA)...")
            model_perf_df = pd.DataFrame(model_performance_records)
            if model_perf_df.empty:
                logger.warning("Cannot calculate final scores: model performance data is empty.")
                return

            num_configs = dataset_configs_to_run_count
            if num_configs == 0:  # Handle case where no dataset configs were processed
                logger.warning("Cannot calculate final scores: dataset_configs_to_run_count is 0.")
                return

            complexity_values = list(range(1, num_configs + 1))
            dataset_id_list_for_scores = [f"config_{i + 1}" for i in range(num_configs)]
            dataset_complexity_map = {f"config_{i + 1}": complexity_values[i] for i in range(num_configs)}

            if 'dataset_id' not in model_perf_df.columns:
                logger.error("Cannot calculate final scores: 'dataset_id' missing in model performance data.")
                return
            model_accuracy_map = model_perf_df.set_index('dataset_id')['accuracy'].to_dict()

            overall_score_total, overall_scores_per_dataset = Evaluation_v2.calculate_overall_score(
                all_run_results_df, dataset_complexity_map, model_accuracy_map
            )
            logger.info(f"Total Overall Score: {overall_score_total}")
            pd.Series(overall_scores_per_dataset, name="overall_score").to_csv(
                run_output_dir / "overall_scores_per_dataset.csv"
            )
            for lib_name in all_run_results_df['lib'].unique():
                xfa_score = Evaluation_v2.calculate_xfa_score(
                    explainer_lib_id=lib_name, evaluation_results_df=all_run_results_df,
                    dataset_complexity_map=dataset_complexity_map,
                    dataset_order_for_complexity=dataset_id_list_for_scores
                )
                logger.info(f"XFA Score for {lib_name}: {xfa_score}")
                with open(run_output_dir / f"xfa_score_{lib_name}.txt", "w") as f:
                    f.write(str(xfa_score))
        else:
            logger.warning(
                "Skipping final score calculation as evaluation results or model performance records are missing/empty.")


# --- Main Orchestration Function ---
def main(args: argparse.Namespace):
    logger.info("====================== XAI Benchmark Runner: Start ============================")
    logger.info(f"Run arguments: {args}")

    run_output_dir, dataset_configs_to_run, dataset_proportions_list = setup_run_directory_and_config(args)

    all_run_results_df = pd.DataFrame()
    model_performance_records = []
    current_feature_names = sdg.BACKGROUND_FEATURE_NAMES

    # Determine if the main processing loop for datasets should run
    # This loop covers generating/loading datasets, training, explaining, and evaluating per config.
    run_full_pipeline_per_config = args.generate_dataset or args.load_datasets

    if run_full_pipeline_per_config:
        logger.info("Starting full pipeline per dataset configuration.")
        for i, ds_indices_config in enumerate(dataset_configs_to_run):
            dataset_id_str = f"config_{i + 1}"
            proportions_config = dataset_proportions_list[i]

            # --- Stage 1: Get Dataset (Generate or Load) ---
            dataset_df: Optional[pd.DataFrame] = None
            metadata_df: Optional[pd.DataFrame] = None
            dataset_pkl_path = run_output_dir / f"dataset_{dataset_id_str}.pkl"
            metadata_pkl_path = run_output_dir / f"metadata_{dataset_id_str}.pkl"

            if args.load_datasets and dataset_pkl_path.exists() and metadata_pkl_path.exists():
                logger.info(f"Attempting to load dataset and metadata for {dataset_id_str} from files.")
                try:
                    dataset_df = pd.read_pickle(dataset_pkl_path)
                    metadata_df = pd.read_pickle(metadata_pkl_path)
                    logger.info(f"Successfully loaded dataset and metadata for {dataset_id_str}.")
                except Exception as e:
                    logger.error(f"Error loading dataset/metadata for {dataset_id_str}: {e}", exc_info=True)
                    dataset_df, metadata_df = None, None

            if (dataset_df is None or metadata_df is None) and args.generate_dataset:
                if args.load_datasets:
                    logger.info(
                        f"Dataset/metadata for {dataset_id_str} not found or failed to load. Generating as --generate_dataset is set.")
                else:
                    logger.info(f"Generating dataset and metadata for {dataset_id_str}...")

                dataset_df, metadata_df, _ = generate_composite_dataset(
                    ds_indices_config, proportions_config, int(args.dataset_size), current_feature_names
                )
                if args.save_datasets and (dataset_df is not None and not dataset_df.empty):
                    logger.info(f"Saving generated dataset and metadata for {dataset_id_str}.")
                    dataset_df.to_pickle(dataset_pkl_path)
                    metadata_df.to_pickle(metadata_pkl_path)

            elif dataset_df is None or metadata_df is None:
                logger.warning(
                    f"Dataset for {dataset_id_str} not loaded (and --generate_dataset is false or generation failed). Skipping this config.")
                continue

            if dataset_df.empty:
                logger.warning(f"Dataset for {dataset_id_str} is empty after load/generation. Skipping.")
                continue

            # --- Stage 2: Model Training and Data Prep ---
            logger.info(f"Processing (splitting, training) for {dataset_id_str}...")
            X_train_df, X_test_df, y_train, y_test = train_test_split(
                dataset_df[current_feature_names], dataset_df[TARGET_NAME],
                train_size=0.80, random_state=args.random_state, stratify=dataset_df[TARGET_NAME]
            )
            meta_test_df_aligned = metadata_df.loc[X_test_df.index].copy()
            meta_test_df_aligned['instance_idx'] = X_test_df.index
            X_test_df_processed = X_test_df.reset_index(drop=True)

            ml_model = train_model(args.model_type, X_train_df, y_train, args.random_state)

            y_pred_test = ml_model.predict(X_test_df_processed.values)
            y_proba_test = ml_model.predict_proba(X_test_df_processed.values)[:, 1]
            accuracy = accuracy_score(y_test.values, y_pred_test)
            auc = roc_auc_score(y_test.values, y_proba_test)
            model_performance_records.append({
                'dataset_id': dataset_id_str, 'config_indices': ds_indices_config,
                'accuracy': accuracy, 'auc': auc
            })
            logger.info(f"Model Performance for {dataset_id_str}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")

            # --- Stage 3: Explanation Generation ---
            all_explanations_for_current_config = {}
            if not args.skip_explanations:
                all_explanations_for_current_config = generate_all_explanations_for_config(
                    dataset_id_str, X_test_df_processed, X_train_df, ml_model,
                    current_feature_names, args, run_output_dir
                )

            # --- Stage 4: Evaluation ---
            if not args.skip_evaluation:
                if all_explanations_for_current_config:
                    config_metric_results = evaluate_all_explanations_for_config(
                        dataset_id_str, str(ds_indices_config), all_explanations_for_current_config,
                        meta_test_df_aligned, current_feature_names, args
                    )
                    if config_metric_results:
                        all_run_results_df = pd.concat([all_run_results_df, pd.DataFrame(config_metric_results)],
                                                       ignore_index=True)
                else:
                    logger.info(
                        f"Skipping evaluation for {dataset_id_str} as no explanations were generated or loaded.")
        # End of for loop over dataset_configs_to_run

    elif args.load_explanations:
        # This block runs if NOT (generate_dataset OR load_datasets) AND load_explanations is TRUE.
        # It means we are trying to load explanations and evaluate them, assuming datasets and metadata exist.
        logger.info(
            "Full pipeline per config skipped. Attempting to load explanations and their prerequisite data for evaluation.")
        for i, ds_indices_config in enumerate(dataset_configs_to_run):
            dataset_id_str = f"config_{i + 1}"
            dataset_pkl_path = run_output_dir / f"dataset_{dataset_id_str}.pkl"
            metadata_pkl_path = run_output_dir / f"metadata_{dataset_id_str}.pkl"

            dataset_df, metadata_df = None, None
            if dataset_pkl_path.exists() and metadata_pkl_path.exists():
                try:
                    dataset_df = pd.read_pickle(dataset_pkl_path)
                    metadata_df = pd.read_pickle(metadata_pkl_path)
                except Exception as e:
                    logger.error(
                        f"Failed to load dataset/metadata for {dataset_id_str} (needed for evaluating loaded explanations): {e}")
                    continue
            else:
                logger.warning(
                    f"Dataset/metadata for {dataset_id_str} not found. Cannot evaluate loaded explanations for this config.")
                continue

            if dataset_df.empty:
                logger.warning(f"Loaded dataset for {dataset_id_str} is empty. Cannot evaluate for this config.")
                continue

            # Reconstruct meta_test_df_aligned for evaluation context
            # Note: Model is not trained here, so model performance for this config must be loaded separately if needed for final scores.
            _, X_test_df, _, _ = train_test_split(
                dataset_df[current_feature_names], dataset_df[TARGET_NAME],
                train_size=0.80, random_state=args.random_state, stratify=dataset_df[TARGET_NAME]
            )
            meta_test_df_aligned = metadata_df.loc[X_test_df.index].copy()
            meta_test_df_aligned['instance_idx'] = X_test_df.index

            # Load explanations (generate_all_explanations_for_config will only load due to args.load_explanations=True)
            # We need a dummy or loaded model if generate_all_explanations_for_config requires one for `fit`.
            # For simplicity, let's assume explanations are just loaded directly here.
            loaded_explanations = {}
            explainer_libs_to_run = ["lime", "anchor"]
            found_any_explanation_to_load = False
            for lib_name in explainer_libs_to_run:
                exp_pkl_path = run_output_dir / f"explanations_{lib_name}_{dataset_id_str}.pkl"
                if exp_pkl_path.exists():
                    try:
                        loaded_explanations[lib_name] = pd.read_pickle(exp_pkl_path)
                        logger.info(f"Loaded {lib_name} explanations for {dataset_id_str}.")
                        if not loaded_explanations[lib_name].empty:
                            found_any_explanation_to_load = True
                    except Exception as e:
                        logger.error(f"Error loading {lib_name} explanations for {dataset_id_str}: {e}")
                        loaded_explanations[lib_name] = pd.DataFrame()
                else:
                    logger.warning(f"Explanations file for {lib_name} on {dataset_id_str} not found at {exp_pkl_path}.")
                    loaded_explanations[lib_name] = pd.DataFrame()

            if not args.skip_evaluation and found_any_explanation_to_load:
                config_metric_results = evaluate_all_explanations_for_config(
                    dataset_id_str, str(ds_indices_config), loaded_explanations,
                    meta_test_df_aligned, current_feature_names, args
                )
                if config_metric_results:
                    all_run_results_df = pd.concat([all_run_results_df, pd.DataFrame(config_metric_results)],
                                                   ignore_index=True)
            elif args.skip_evaluation:
                logger.info(
                    f"Skipping evaluation for {dataset_id_str} as --skip_evaluation is set (in --load_explanations path).")
            else:
                logger.info(
                    f"Skipping evaluation for {dataset_id_str} as no loadable explanations found (in --load_explanations path).")

        # Load model performance records if they exist, as they are independent of this loop
        model_perf_csv_path = run_output_dir / "model_performance_summary.csv"
        if model_perf_csv_path.exists():
            try:
                model_perf_df_loaded = pd.read_csv(model_perf_csv_path)
                model_performance_records = model_perf_df_loaded.to_dict('records')
                logger.info(
                    f"Loaded model performance records from {model_perf_csv_path} (after --load_explanations path).")
            except Exception as e:
                logger.error(f"Error loading model performance records: {e}")
        else:
            logger.warning(
                f"Model performance summary file not found at {model_perf_csv_path} (in --load_explanations path). Needed for final scores.")


    elif args.load_evaluation:
        logger.info(
            "Full pipeline and explanation loading per config skipped. Attempting to load pre-aggregated evaluation results as --load_evaluation is set.")
        final_results_pkl_path = run_output_dir / "evaluation_all_metrics.pkl"
        model_perf_csv_path = run_output_dir / "model_performance_summary.csv"

        if final_results_pkl_path.exists():
            try:
                all_run_results_df = pd.read_pickle(final_results_pkl_path)
                logger.info(f"Loaded aggregated evaluation results from {final_results_pkl_path}")
            except Exception as e:
                logger.error(f"Error loading aggregated evaluation results from {final_results_pkl_path}: {e}",
                             exc_info=True)
        else:
            logger.warning(f"Aggregated evaluation results file not found: {final_results_pkl_path}.")

        if model_perf_csv_path.exists():
            try:
                model_perf_df_loaded = pd.read_csv(model_perf_csv_path)
                model_performance_records = model_perf_df_loaded.to_dict('records')
                logger.info(f"Loaded model performance records from {model_perf_csv_path}")
            except Exception as e:
                logger.error(f"Error loading model performance records from {model_perf_csv_path}: {e}", exc_info=True)
        else:
            logger.warning(f"Model performance summary file not found: {model_perf_csv_path}.")
    else:
        logger.info(
            "All processing loops skipped (due to flags). Proceeding to final steps with potentially empty data.")

    # --- Final Steps (Save aggregated if generated in loop, Plot, Score) ---
    perform_final_steps(all_run_results_df, model_performance_records, len(dataset_configs_to_run), args,
                        run_output_dir)

    logger.info("======================= XAI Benchmark Runner: End ============================")


sys.argv = [
        'run_xai_benchmark.py',  # sys.argv[0]
        '--run_id',  # sys.argv[1]
        'comprehensive_run_002',  # sys.argv[2]
        '--output_dir',  # sys.argv[3]
        'my_benchmark_results',  # sys.argv[4]
        '--random_state',  # sys.argv[5]
        '123',  # sys.argv[6] (note: numbers are passed as strings)
        '--dataset_size',  # sys.argv[7]
        '500',  # sys.argv[8] (string)
        '--model_type',  # sys.argv[11]
        'xgboost',  # sys.argv[12]
        '--lime_num_features',  # sys.argv[13]
        '7',  # sys.argv[14] (string)
        '--anchor_threshold',  # sys.argv[15]
        '0.85',  # sys.argv[16] (string)
        '--sensitivity_top_k',  # sys.argv[17]
        '4',  # sys.argv[18] (string)
        # '--save_datasets',  # sys.argv[19] (boolean flag)
        '--load_explanations',  # sys.argv[21] (boolean flag)
        '--save_evaluation',  # sys.argv[22] (boolean flag)
        '--generate_plots',  # sys.argv[23] (boolean flag)
        '--calculate_final_scores',  # sys.argv[24] (boolean flag)
        '--parallel_explain',  # sys.argv[25] (boolean flag)
        '--n_jobs',  # sys.argv[26]
        '3'  # sys.argv[27] (string)
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run XAI Benchmark Pipeline.")
    parser.add_argument("--run_id", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="Unique ID for the run.")
    parser.add_argument("--output_dir", type=str, default="benchmark_runs_output",
                        help="Parent directory for run outputs.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state.")
    parser.add_argument("--dataset_size", type=str, default="1000",
                        help="Total samples per composite dataset.")
    parser.add_argument("--model_type", type=str, default="RandomForest",
                        choices=["RandomForest", "ExtraTrees", "xgboost"], help="Model type.")
    parser.add_argument("--lime_num_features", type=int, default=5, help="Num features for LIME.")
    parser.add_argument("--anchor_threshold", type=float, default=0.90, help="Threshold for Anchor.")
    parser.add_argument("--sensitivity_top_k", type=int, default=3, help="Top K for sensitivity.")

    # Flags to control pipeline stages
    parser.add_argument("--generate_dataset", action="store_true",
                        help="Generate datasets from scratch. Triggers model training, explanations, and evaluations for these datasets.")
    parser.add_argument("--load_datasets", action="store_true",
                        help="Load datasets from disk if available. Triggers model training, explanations, and evaluations for these datasets.")
    parser.add_argument("--save_datasets", action="store_true", help="Save generated/loaded datasets.")

    parser.add_argument("--skip_explanations", action="store_true",
                        help="Skip explanation generation entirely (even if datasets are processed).")
    parser.add_argument("--load_explanations", action="store_true",
                        help="Load explanations from disk. If main dataset loop is skipped, this also tries to load corresponding datasets/metadata for evaluation context.")
    parser.add_argument("--save_explanations", action="store_true", help="Save generated explanations.")

    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip evaluation metric calculation (even if explanations are available).")
    parser.add_argument("--load_evaluation", action="store_true",
                        help="Load final aggregated evaluation results (evaluation_all_metrics.pkl) if skipping dataset processing and explanation loading loops.")
    parser.add_argument("--save_evaluation", action="store_true",
                        help="Save final aggregated evaluation results (from current run's evaluations).")

    parser.add_argument("--generate_plots", action="store_true", help="Generate plots.")
    parser.add_argument("--calculate_final_scores", action="store_true", help="Calculate overall and XFA scores.")

    parser.add_argument("--parallel_explain", action="store_true", help="Run explanations in parallel.")
    parser.add_argument("--n_jobs", type=int, default=-2, help="Num jobs for parallel processing.")

    args = parser.parse_args()

    # Default save behavior:
    # If a generation step happens (datasets, explanations, evaluations), save the output by default.
    # This can be overridden by explicitly NOT setting the save flag if a future arg like --no-save-datasets is added.
    # For now, if the action is performed, we assume saving is desired unless loading that specific artifact.
    if args.generate_dataset and not args.load_datasets: args.save_datasets = True
    if not args.skip_explanations and not args.load_explanations: args.save_explanations = True
    if not args.skip_evaluation: args.save_evaluation = True  # Saves the aggregated results if evaluation happened in this run.

    main(args)
