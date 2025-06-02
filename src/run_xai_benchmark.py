# File: run_xai_benchmark.py (renamed from start_v3.py for clarity)
import sys
import logging
# --- Global Configuration & Logging ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s', level=logging.DEBUG)
# Use a more specific logger for this application
logger = logging.getLogger("xai_benchmark_runner")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

import argparse
import gc

import warnings
from datetime import datetime
from pathlib import Path

import Evaluation_plots_v2
import pandas as pd
# --- Import refactored modules ---
# Assuming they are in the PYTHONPATH or same directory
import dataset_generation.DataSetGen_v3 as sdg
import Evaluation_v2
import Explaination_v2
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier # Uncomment if XGBoost is to be supported
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)  # Suppress some common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)  # For mean of empty slice etc.



TARGET_NAME = "y"

# --- Dataset Configuration (kept similar to original for now) ---
# This could be loaded from a config file in a more advanced setup
DEFAULT_DATASETS_SEQUENCES = [[1, 3]
    , [3, 9, 10]
    , [1, 5, 6, 11, 12]
    , [1, 3, 7, 9]
    , [2, 3, 4, 5, 6, 8, 10, 12]
    , [2, 3, 5, 6, 8, 9, 10, 11]
    , [1, 2, 3, 5, 6, 8, 9, 10, 11, 12]
    , [1, 4, 5, 6, 7, 8, 10, 11, 12]
    , [2, 3, 4, 5, 7, 8, 10]
    , [4, 5, 6, 7, 8, 10, 11, 12]
    , [5, 6, 7, 9, 10]
                              ]


# --- Helper Functions ---
def generate_composite_dataset(
        dataset_indices: List[int],
        proportions: List[float],
        total_size: int,
        base_feature_names: List[str]  # Pass this to ensure consistency if needed
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Generates a composite dataset by combining portions of different base synthetic datasets.
    """
    all_data_dfs = []
    all_meta_dfs = []

    # Ensure sum of proportions is close to 1.0 if they are meant to be exact fractions
    # For now, take them as specified.
    current_offset = 0
    actual_total_size = 0

    for i, dataset_idx in enumerate(dataset_indices):
        # Map original index (e.g., 1 for ds1) to dataset name string "dsX"
        # The original code used funs[dataset_index], where ds0 was index 0.
        # So, if dataset_indices are 1-based (like in datasets_seq_lst), adjust to 0-based for ds_name.
        # Or, assume dataset_indices are already 0-based for direct use.
        # Let's assume input dataset_indices are 0-based for `sdg.generate_dataset_by_name(f"ds{dataset_idx}")`
        dataset_name_str = f"ds{dataset_idx}"

        # Ensure proportions sum to roughly the total_size requested
        # This logic for prop_size ensures we try to get total_size
        if i < len(proportions) - 1:
            prop_size = int(proportions[i] * total_size)
            current_offset += prop_size
        else:  # Last dataset takes the remainder to match total_size
            prop_size = total_size - current_offset

        if prop_size <= 0:
            logger.warning(f"Calculated proportion size for {dataset_name_str} is {prop_size}. Skipping.")
            continue
        actual_total_size += prop_size

        logger.debug(f"Generating {prop_size} samples for {dataset_name_str} (Original index: {dataset_idx})")
        try:
            synthetic_ds_obj = sdg.generate_dataset_by_name(name=dataset_name_str, size=prop_size)
            all_data_dfs.append(synthetic_ds_obj.data)
            all_meta_dfs.append(synthetic_ds_obj.meta_data)
            # Feature names should be consistent from sdg.BACKGROUND_FEATURE_NAMES
            # but good to confirm from the first generated dataset.
            # No, sdg.BACKGROUND_FEATURE_NAMES is what's needed, not synthetic_ds_obj.feature_names
            # as the latter might be specific if a dataset only uses a subset.
            # The main loop should use sdg.BACKGROUND_FEATURE_NAMES.
        except ValueError as e:
            logger.error(f"Could not generate {dataset_name_str}: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error generating {dataset_name_str}: {e}", exc_info=True)
            continue

    if not all_data_dfs:
        logger.error("No dataframes were generated for the composite dataset.")
        return pd.DataFrame(), pd.DataFrame(), base_feature_names  # Return empty DFs

    final_data_df = pd.concat(all_data_dfs, ignore_index=True)
    final_meta_df = pd.concat(all_meta_dfs, ignore_index=True)

    # Shuffle the combined dataset to mix the different parts
    final_data_df = final_data_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_meta_df = final_meta_df.loc[final_data_df.index].reset_index(drop=True)

    logger.info(
        f"Generated composite dataset with {len(final_data_df)} actual samples (requested {total_size}, sum of parts {actual_total_size}).")
    return final_data_df, final_meta_df, base_feature_names


def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, random_state: int):
    """Trains a specified model."""
    if model_name.lower() == 'randomforest':
        model = RandomForestClassifier(n_jobs=-1, random_state=random_state)  # Use all available cores
    elif model_name.lower() == 'extratrees':
        model = ExtraTreesClassifier(n_jobs=-1, random_state=random_state)
    elif model_name.lower() == 'xgboost':
        model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    logger.info(f"Training {model_name} model...")
    model.fit(X_train.values, y_train)  # Scikit-learn models often prefer numpy arrays
    logger.info(f"{model_name} model trained.")
    return model


# --- Main Orchestration Function ---
def main(args):
    logger.info("=================================================================================")
    logger.info("====================== XAI Benchmark Runner: Start ============================")
    logger.info(f"Run arguments: {args}")

    run_output_dir = Path(args.output_dir) / args.run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run output will be saved to: {run_output_dir}")

    # Prepare dataset sequences and proportions
    # The original code uses 1-based indexing for datasets_seq_lst.
    # sdg.generate_dataset_by_name uses "RRs0", "RRs1", etc. So we adjust.
    # dataset_configs_to_run = DEFAULT_DATASETS_SEQUENCES

    if args.generate_dataset:
        if args.load_explanation:
            logger.warning(
                "`--generate-dataset` is specified; therefore, `--load-explanation` will be ignored and treated as False."
            )
            args.load_explanation = False

        if args.load_evaluation:
            logger.warning(
                "`--generate-dataset` is specified; therefore, `--load-evaluation` will be ignored and treated as False."
            )
            args.load_evaluation = False



        dataset_configs_to_run = DEFAULT_DATASETS_SEQUENCES


        dataset_proportions_list = []
        for seq in dataset_configs_to_run:

            if not seq: continue
            # Original code: np.floor((1/len(i)) * 1000) / 1000 - this ensures sum of proportions is slightly less than 1
            # if len(i) makes 1/len(i) have many decimal places.
            # A simpler way for equal proportions that sum to 1:
            prop_val = 1.0 / len(seq)
            props = [prop_val] * len(seq)
            # Adjust last proportion to ensure sum is exactly 1.0 if needed due to float precision
            current_sum = sum(props[:-1])
            if len(props) > 0:
                props[-1] = 1.0 - current_sum

            dataset_proportions_list.append(props)

        # Save the dataset sequences being used for this run
        print("=================================")
        print(dataset_configs_to_run)
        pd.DataFrame(dataset_configs_to_run).to_csv(run_output_dir / f"run_dataset_sequences_size_{args.dataset_size}.csv",
                                                    index=False,header=False)

        all_run_results_df = pd.DataFrame()
        model_performance_records = []

        # Consistent feature names from the synthetic generator config
        # Assumes sdg.BACKGROUND_FEATURE_NAMES are the features *before* target 'y' is added
        # And that all generated datasets will use these plus 'y'
        current_feature_names = sdg.BACKGROUND_FEATURE_NAMES

        for i, ds_indices_config in enumerate(dataset_configs_to_run):
            dataset_id = f"config_{i + 1}"  # Using a more descriptive ID
            logger.info(f"===== Processing Dataset Configuration: {dataset_id} (Indices: {ds_indices_config}) =====")
            gc.collect()

            proportions_config = dataset_proportions_list[i]

            # --- Dataset Generation ---
            dataset_pkl_path = run_output_dir / f"dataset_{dataset_id}.pkl"
            metadata_pkl_path = run_output_dir / f"metadata_{dataset_id}.pkl"

            if args.load_datasets and dataset_pkl_path.exists() and metadata_pkl_path.exists():
                logger.info(f"Loading dataset and metadata for {dataset_id} from files.")
                dataset_df = pd.read_pickle(dataset_pkl_path)
                metadata_df = pd.read_pickle(metadata_pkl_path)
                # features_names should be loaded or be consistent
            else:
                logger.info(f"Generating dataset and metadata for {dataset_id}...")
                # Adjust 1-based indices from config to 0-based for ds_name
                zero_based_ds_indices = [idx - 1 for idx in ds_indices_config if idx > 0]  # ds0 is index 0 # not needed as ds0 is not used
                zero_based_ds_indices = ds_indices_config # to disable the index 0
                if any(idx < 0 for idx in zero_based_ds_indices):
                    logger.error(f"Invalid dataset index in {ds_indices_config}. Must be > 0. Skipping.")
                    continue

                dataset_df, metadata_df, _ = generate_composite_dataset(
                    zero_based_ds_indices, proportions_config, int(args.dataset_size), current_feature_names
                )
                if args.save_datasets and not dataset_df.empty:
                    logger.info(f"Saving dataset and metadata for {dataset_id}.")
                    dataset_df.to_pickle(dataset_pkl_path)
                    metadata_df.to_pickle(metadata_pkl_path)

            if dataset_df.empty:
                logger.warning(f"Dataset for {dataset_id} is empty. Skipping further processing for this config.")
                continue

            # --- Data Splitting & Model Training ---
            X_train_df, X_test_df, y_train, y_test = train_test_split(
                dataset_df[current_feature_names], dataset_df[TARGET_NAME],
                train_size=0.80, random_state=args.random_state, stratify=dataset_df[TARGET_NAME]
            )
            # Align metadata with the test set using instance_idx if available, or index
            # The metadata_df from generate_composite_dataset should align with dataset_df by index
            meta_test_df = metadata_df.loc[X_test_df.index].reset_index().rename(columns={'index': 'original_dataset_idx'})
            # Add 'instance_idx' for metric calculation alignment
            meta_test_df['instance_idx'] = X_test_df.index
            X_test_df = X_test_df.reset_index(drop=True)  # Ensure X_test_df has simple 0-based index
            y_test = y_test.reset_index(drop=True)
            meta_test_df = meta_test_df.reset_index(drop=True)

            ml_model = train_model(args.model_type, X_train_df, y_train, args.random_state)

            y_pred_test = ml_model.predict(X_test_df.values)
            y_proba_test = ml_model.predict_proba(X_test_df.values)[:, 1]

            accuracy = accuracy_score(y_test, y_pred_test)
            auc = roc_auc_score(y_test, y_proba_test)
            model_performance_records.append({
                'dataset_id': dataset_id, 'config_indices': ds_indices_config,
                'accuracy': accuracy, 'auc': auc
            })
            logger.info(f"Model Performance for {dataset_id}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")

        # --- Explanation Generation ---
        current_config_explanations = {}  # Store explanations {'lime': df, 'anchor': df}

        if not args.skip_explanations:
            explainer_libs_to_run = ["lime", "anchor"]  # Can be parameterized
            for lib_name in explainer_libs_to_run:
                exp_pkl_path = run_output_dir / f"explanations_{lib_name}_{dataset_id}.pkl"
                if args.load_explanations and exp_pkl_path.exists():
                    logger.info(f"Loading {lib_name} explanations for {dataset_id} from file.")
                    current_config_explanations[lib_name] = pd.read_pickle(exp_pkl_path)
                else:
                    logger.info(f"Generating {lib_name} explanations for {dataset_id}...")
                    explainer_wrapper_args = {}
                    if lib_name == "lime":
                        explainer_wrapper = Explaination_v2.get_explainer_wrapper(lib_name, ml_model)
                        explainer_wrapper.fit(
                            training_data=X_train_df,  # Pass DataFrame
                            feature_names=current_feature_names,
                            class_names=[str(c) for c in ml_model.classes_],  # Get class names from model
                            mode="classification"  # Assuming classification
                        )
                        explanations_df = explainer_wrapper.explain_dataset(
                            X_test_df, parallel=args.parallel_explain, n_jobs=args.n_jobs,
                            num_features=args.lime_num_features
                        )
                    elif lib_name == "anchor":
                        # Define categorical_names for Anchor if applicable. For now, empty.
                        # This might require inspecting X_train_df dtypes or a config.
                        categorical_names_anchor = {}
                        explainer_wrapper = Explaination_v2.get_explainer_wrapper(lib_name, ml_model)
                        explainer_wrapper.fit(
                            training_data=X_train_df,
                            feature_names=current_feature_names,
                            class_names=[str(c) for c in ml_model.classes_],
                            categorical_names=categorical_names_anchor
                        )
                        explanations_df = explainer_wrapper.explain_dataset(
                            X_test_df, parallel=args.parallel_explain, n_jobs=args.n_jobs,
                            threshold=args.anchor_threshold, classifier_fn=ml_model.predict
                        )
                    else:
                        logger.warning(f"Explainer {lib_name} not implemented in this script's main loop.")
                        continue

                    current_config_explanations[lib_name] = explanations_df
                    if args.save_explanations and not explanations_df.empty:
                        logger.info(f"Saving {lib_name} explanations for {dataset_id}.")
                        # logger.info("========= debug ================")
                        # logger.info(explanations_df.shape)
                        # logger.info(explanations_df.columns)
                        # logger.info(explanations_df.head(5).values)
                        explanations_df.to_pickle(exp_pkl_path)

        # --- Evaluation ---
        # This DataFrame will store results for the current dataset_id (config)
        current_config_results_list = []

        if not args.skip_evaluation:
            # Iter_evaluation_pkl_path = run_output_dir / f"evaluation_metrics_{dataset_id}.pkl" # For all libs on this ds
            # if args.load_evaluation and Iter_evaluation_pkl_path.exists():
            #    logger.info(f"Loading evaluation metrics for {dataset_id} from {Iter_evaluation_pkl_path}")
            #    current_config_results_df = pd.read_pickle(Iter_evaluation_pkl_path)
            #    all_run_results_df = pd.concat([all_run_results_df, current_config_results_df], ignore_index=True)
            #    continue # Skip recalculation if loaded

            for lib_name, explanations_df in current_config_explanations.items():
                if explanations_df is None or explanations_df.empty:
                    logger.warning(f"No explanations found for {lib_name} on {dataset_id} to evaluate.")
                    continue

                # Ensure explanations_df has 'instance_idx' for alignment if not already present
                # The explain_dataset method should return 'instance_idx' mapped to original X_test_df index
                # If X_test_df was reset, its original index values need to be related to meta_test_df.
                # Let's assume explain_dataset returns 'instance_idx' that matches X_test_df's original index before reset_index.
                # And meta_test_df['instance_idx'] is also set to that.

                # The explanations_df from Explaination_v2 should have 'instance_idx'
                # corresponding to the index of X_test_df *before* it was reset for model prediction
                # This needs careful handling. Let's ensure meta_test_df's `instance_idx` refers to the
                # original index of X_test_df (which is also the index in `metadata_df`)
                # And that explanations_df['instance_idx'] also refers to this same original index.
                # The current `explain_dataset` returns instance_idx from `dataset.index`.
                # If `X_test_df` passed to it has been reset, then `dataset.index` will be 0-based.
                # `meta_test_df` has `instance_idx` corresponding to the original full dataset index.
                # This is a critical alignment point.

                # Simplification: Assume `meta_test_df` has `instance_idx` (0 to len(X_test_df)-1)
                # And `explanations_df` also has `instance_idx` (0 to len(X_test_df)-1)
                # This implies X_test_df was passed to explain_dataset *after* its index was reset.
                # And meta_test_df needs to be built on this reset index.

                # Let's adjust: meta_test_df should be prepared such that its `instance_idx` matches the
                # 0-based index of the `X_test_df` that is passed to the explainers.
                # The X_test_df passed to explainer should be the one with 0-based index.
                # `meta_test_df` needs to map its `imp_vars` (and `RGS`) to this 0-based index.

                # Re-aligning meta_test_df `instance_idx` to be 0-based to match a reset X_test_df
                meta_test_for_eval = meta_test_df[
                    ['instance_idx', 'imp_vars', 'RGS']].copy()  # Assuming these columns exist after _align_data
                # The _align_data in metrics will use 'instance_idx'
                # So, explanations_df needs 'instance_idx' 0..N-1, and meta_test_for_eval also needs 'instance_idx' 0..N-1

                logger.info(f"Evaluating {lib_name} explanations for {dataset_id}...")

                metrics_to_calculate = {
                    "recall_partial": Evaluation_v2.get_metric_calculator("recall", partial=True),
                    "recall_full": Evaluation_v2.get_metric_calculator("recall", partial=False),
                    "fpr": Evaluation_v2.get_metric_calculator("fpr"),
                    "sensitivity": Evaluation_v2.get_metric_calculator("sensitivity",
                                                                                top_k=args.sensitivity_top_k)
                }

                for metric_key, calculator in metrics_to_calculate.items():
                    logger.debug(f"Calculating {metric_key} for {lib_name} on {dataset_id}")
                    # `meta_test_df` (ground truth) needs `instance_idx` and `imp_vars`
                    # `explanations_df` needs `instance_idx`, `features`, `importance`
                    metric_result = calculator.calculate(
                        ground_truth_df=meta_test_df.rename(columns={'imp_vars_gt': 'imp_vars'}),  # Ensure 'imp_vars'
                        explanations_df=explanations_df,  # Assumes 'instance_idx', 'features', 'importance'
                        all_feature_names=current_feature_names
                    )
                    current_config_results_list.append({
                        "dataset_id": dataset_id,
                        "config_indices_str": str(ds_indices_config),  # Store the config for reference
                        "metric_name": metric_result.get('metric_name', metric_key),
                        "metric_config": str(metric_result.get('config', {})),
                        "lib": lib_name,
                        "average_score": metric_result.get('average_score'),
                        "score_list": metric_result.get('instance_scores'),
                        # "RGS_list": meta_test_df['RGS_gt'].tolist() # If RGS is per instance and needed
                    })

            if current_config_results_list:
                current_config_results_df = pd.DataFrame(current_config_results_list)
                all_run_results_df = pd.concat([all_run_results_df, current_config_results_df], ignore_index=True)
                # if args.save_evaluation:
                #    current_config_results_df.to_pickle(Iter_evaluation_pkl_path)





    # --- Save Final Aggregated Results ---
    if args.save_evaluation and not all_run_results_df.empty:
        final_results_pkl_path = run_output_dir / "evaluation_all_metrics.pkl"
        logger.info(f"Saving all aggregated evaluation results to {final_results_pkl_path}")
        all_run_results_df.to_pickle(final_results_pkl_path)

    if model_performance_records:
        model_perf_df = pd.DataFrame(model_performance_records)
        model_perf_csv_path = run_output_dir / "model_performance_summary.csv"
        logger.info(f"Saving model performance summary to {model_perf_csv_path}")
        model_perf_df.to_csv(model_perf_csv_path, index=False)

    # --- Plotting ---
    if args.generate_plots and not all_run_results_df.empty:
        logger.info("Generating plots...")
        # The plotting function expects 'metric' and 'score_list'
        # Ensure 'recall_partial' vs 'recall_full' are distinct metric names in all_run_results_df
        # or adapt plotting function/input df.
        # The metric_key from metrics_to_calculate provides distinct names like 'recall_partial'.

        # Metrics for plotting: use the keys from metrics_to_calculate
        metrics_for_plotting = list(metrics_to_calculate.keys())

        Evaluation_plots_v2.run_all_plots(
            base_results_dir=run_output_dir,  # Plots will be saved in run_output_dir/plots
            evaluation_filename="evaluation_all_metrics.pkl",  # Name of the file we just saved
            metrics_to_plot=metrics_for_plotting,
            plot_signal_csv_pattern=f"run_dataset_sequences_size_*.csv"
            # Use the saved sequence file for signals if needed, or original pattern
        )

    # --- Final Scoring (Overall and XFA) ---
    if args.calculate_final_scores and not all_run_results_df.empty and model_performance_records:
        logger.info("Calculating final aggregate scores (Overall and XFA)...")
        model_perf_df = pd.DataFrame(model_performance_records)

        # Create maps for complexity and accuracy for scoring functions
        # Using simple 1-based complexity for dataset configs for now.
        # Original `scaled_complexity = [x for x  in range(1,12)]` used for 11 datasets.
        # Here, we have len(dataset_configs_to_run) dataset configurations.
        num_configs = len(dataset_configs_to_run)
        complexity_values = list(range(1, num_configs + 1))  # Simple linear complexity

        dataset_id_list_for_scores = [f"config_{i + 1}" for i in range(num_configs)]

        dataset_complexity_map = {f"config_{i + 1}": complexity_values[i] for i in range(num_configs)}
        model_accuracy_map = model_perf_df.set_index('dataset_id')['accuracy'].to_dict()

        overall_score_total, overall_scores_per_dataset = Evaluation_v2.calculate_overall_score(
            all_run_results_df, dataset_complexity_map, model_accuracy_map
        )
        logger.info(f"Total Overall Score: {overall_score_total}")
        logger.info(f"Per-Dataset Overall Scores: {overall_scores_per_dataset}")
        pd.Series(overall_scores_per_dataset, name="overall_score").to_csv(
            run_output_dir / "overall_scores_per_dataset.csv")

        for lib_name in all_run_results_df['lib'].unique():
            xfa_score = Evaluation_v2.calculate_xfa_score(
                explainer_lib_id=lib_name,
                evaluation_results_df=all_run_results_df,
                dataset_complexity_map=dataset_complexity_map,
                dataset_order_for_complexity=dataset_id_list_for_scores  # Order for normalization sum
            )
            logger.info(f"XFA Score for {lib_name}: {xfa_score}")
            # Save XFA score
            with open(run_output_dir / f"xfa_score_{lib_name}.txt", "w") as f:
                f.write(str(xfa_score))

    logger.info("======================= XAI Benchmark Runner: End ============================")
    logger.info("=================================================================================")

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
        '--save_datasets',  # sys.argv[19] (boolean flag)
        '--save_explanations',  # sys.argv[21] (boolean flag)
        '--save_evaluation',  # sys.argv[22] (boolean flag)
        '--generate_plots',  # sys.argv[23] (boolean flag)
        '--calculate_final_scores',  # sys.argv[24] (boolean flag)
        '--parallel_explain',  # sys.argv[25] (boolean flag)
        '--n_jobs',  # sys.argv[26]
        '3'  # sys.argv[27] (string)
    ]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run XAI Benchmark Pipeline.")

    # Run control & Directory
    parser.add_argument("--run_id", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="Unique ID for the run, used for naming output directory.")
    parser.add_argument("--output_dir", type=str, default="benchmark_runs",
                        help="Parent directory to save run outputs.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")

    # Dataset parameters
    parser.add_argument("--dataset_size", type=int, default=10000,
                        help="Total number of samples per generated composite dataset configuration.")

    # Model parameters
    parser.add_argument("--model_type", type=str, default="RandomForest", choices=["RandomForest", "ExtraTrees", "xgboost"],
                        # Add more as needed
                        help="Type of model to train.")

    # Explainer parameters
    parser.add_argument("--lime_num_features", type=int, default=5, help="Number of features for LIME explanations.")
    parser.add_argument("--anchor_threshold", type=float, default=0.90,
                        help="Precision threshold for Anchor explanations.")

    # Evaluation parameters
    parser.add_argument("--sensitivity_top_k", type=int, default=3, help="Top K features for sensitivity metric.")

    # Execution flow control flags
    parser.add_argument("--generate_dataset", action="store_true", help="generate datasets.")
    parser.add_argument("--load_datasets", action="store_true", help="Load datasets from disk if available.")
    parser.add_argument("--save_datasets", action="store_true", help="Save generated datasets to disk.")
    parser.add_argument("--skip_explanations", action="store_true", help="Skip explanation generation.")
    parser.add_argument("--load_explanations", action="store_true", help="Load explanations from disk if available.")
    parser.add_argument("--save_explanations", action="store_true", help="Save generated explanations to disk.")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation metric calculation.")
    parser.add_argument("--load_evaluation", action="store_true", help="Load intermediate evaluation results from disk.") # More complex to manage per-config
    parser.add_argument("--save_evaluation", action="store_true", help="Save final aggregated evaluation results.")
    parser.add_argument("--generate_plots", action="store_true", help="Generate plots after evaluation.")
    parser.add_argument("--calculate_final_scores", action="store_true", help="Calculate overall and XFA scores.")
    parser.add_argument("--parallel_explain", action="store_true", help="Run explanations in parallel.")
    parser.add_argument("--n_jobs", type=int, default=-2,
                        help="Number of jobs for parallel processing (-1 for all, -2 for all but one).")

    args = parser.parse_args()

    # Inside your main function, after parsing arguments, e.g