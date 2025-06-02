# File: xai_evaluation_metrics.py

import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


# --- Base Evaluation Metric Class ---
class BaseEvaluationMetric(ABC):
    """
    Abstract base class for XAI evaluation metrics.
    """

    def __init__(self, name: str):
        self.name = name.lower()  # Store metric name in lowercase for consistency

    def _align_data(self,
                    ground_truth_df: pd.DataFrame,
                    explanations_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Aligns ground truth and explanation dataframes on 'instance_idx'.
        Logs warnings for mismatches.
        """
        if 'instance_idx' not in ground_truth_df.columns:
            ground_truth_df = ground_truth_df.reset_index().rename(columns={'index': 'instance_idx'})
            logger.debug("Added 'instance_idx' from ground_truth_df index.")

        if 'instance_idx' not in explanations_df.columns:
            # Try to use 'instance' if 'instance_idx' is missing, common from older format
            if 'instance' in explanations_df.columns:
                explanations_df = explanations_df.rename(columns={'instance': 'instance_idx'})
                logger.debug("Renamed 'instance' to 'instance_idx' in explanations_df.")
            else:
                explanations_df = explanations_df.reset_index().rename(columns={'index': 'instance_idx'})
                logger.debug("Added 'instance_idx' from explanations_df index.")

        # Ensure 'imp_vars' and 'features' columns exist (common names from original code)
        if 'imp_vars' not in ground_truth_df.columns:
            logger.error("'imp_vars' column missing in ground_truth_df.")
            return None
        if 'features' not in explanations_df.columns:  # This is 'features_exp' in the metric calculation
            logger.error("'features' column missing in explanations_df.")
            return None

        merged_df = pd.merge(ground_truth_df.add_suffix('_gt'),
                             explanations_df.add_suffix('_exp'),
                             left_on="instance_idx_gt",
                             right_on="instance_idx_exp",
                             how="inner")

        if len(merged_df) != len(ground_truth_df) and len(merged_df) != len(explanations_df):
            logger.warning(
                f"Potential mismatch in instances after merging for metric '{self.name}'. "
                f"GT rows: {len(ground_truth_df)}, Exp rows: {len(explanations_df)}, Merged rows: {len(merged_df)}. "
                "Ensure 'instance_idx' values align or only common instances are intended for evaluation."
            )

        if merged_df.empty:
            logger.warning(f"No common instances found for metric '{self.name}'. Evaluation will yield no results.")
            return None

        # Use one of the instance_idx columns as the definitive one
        merged_df['instance_idx'] = merged_df['instance_idx_gt']
        return merged_df

    @abstractmethod
    def calculate(self,
                  ground_truth_df: pd.DataFrame,
                  explanations_df: pd.DataFrame,
                  all_feature_names: Optional[List[str]] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Calculates the evaluation metric.

        Args:
            ground_truth_df (pd.DataFrame): DataFrame with ground truth.
                Expected to have 'instance_idx' (or be indexable) and 'imp_vars' (list of true feature names).
            explanations_df (pd.DataFrame): DataFrame with XAI explanations.
                Expected to have 'instance_idx' (or 'instance', or be indexable),
                'features' (list of explained feature names), and 'importance' (list of importances).
            all_feature_names (Optional[List[str]]): List of all possible features in the dataset.
                                                Required for metrics like FPR.
            **kwargs: Additional metric-specific parameters.

        Returns:
            Dict[str, Any]: A dictionary containing metric scores, typically:
                            {'metric_name': self.name,
                             'average_score': float,
                             'instance_scores': List[float],
                             'config': Dict[str, Any]} # To store metric specific config like 'partial' or 'top_k'
        """
        pass


# --- Specific Metric Implementations ---

class RecallMetric(BaseEvaluationMetric):
    def __init__(self, partial: bool = False):
        super().__init__("recall")
        self.partial = partial
        logger.info(f"RecallMetric initialized with partial={self.partial}.")

    def calculate(self, ground_truth_df: pd.DataFrame, explanations_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        merged_df = self._align_data(ground_truth_df, explanations_df)
        if merged_df is None or merged_df.empty:
            return {'metric_name': self.name, 'average_score': np.nan, 'instance_scores': [],
                    'config': {'partial': self.partial}}

        instance_scores = []
        for _, row in merged_df.iterrows():
            gt_imp_vars = set(row['imp_vars_gt'])
            exp_features = set(row['features_exp'])

            if not gt_imp_vars:  # No ground truth important variables
                # If also no explained features, perfect recall (found all 0 of 0). Otherwise, 0.
                score = 1.0 if not exp_features else 0.0
                instance_scores.append(score)
                continue

            intersection_count = len(gt_imp_vars.intersection(exp_features))

            if self.partial:
                instance_scores.append(1.0 if intersection_count > 0 else 0.0)
            else:
                score = intersection_count / len(gt_imp_vars) if len(gt_imp_vars) > 0 else 1.0  # Avoid div by zero
                instance_scores.append(score)

        avg_score = np.nanmean(instance_scores) if instance_scores else np.nan
        logger.info(f"Calculated {self.name} (partial={self.partial}): Average={avg_score:.4f}")
        return {'metric_name': self.name, 'average_score': avg_score, 'instance_scores': instance_scores,
                'config': {'partial': self.partial}}


class FPRMetric(BaseEvaluationMetric):
    def __init__(self):
        super().__init__("fpr")  # False Positive Rate
        logger.info("FPRMetric initialized.")

    def calculate(self, ground_truth_df: pd.DataFrame, explanations_df: pd.DataFrame,
                  all_feature_names: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        if all_feature_names is None:
            logger.error("all_feature_names must be provided for FPR calculation.")
            return {'metric_name': self.name, 'average_score': np.nan, 'instance_scores': []}

        merged_df = self._align_data(ground_truth_df, explanations_df)
        if merged_df is None or merged_df.empty:
            return {'metric_name': self.name, 'average_score': np.nan, 'instance_scores': []}

        instance_scores = []
        all_features_set = set(all_feature_names)

        for _, row in merged_df.iterrows():
            gt_imp_vars = set(row['imp_vars_gt'])
            exp_features = set(row['features_exp'])

            false_positives = len(exp_features.difference(gt_imp_vars))
            ground_truth_negatives_count = len(all_features_set.difference(gt_imp_vars))

            if ground_truth_negatives_count == 0:
                # All features are truly important, or no non-important features exist.
                # FPR is 0 if no false positives, otherwise undefined (or 1 if we consider any FP an issue).
                score = 0.0 if false_positives == 0 else np.nan
            else:
                score = false_positives / ground_truth_negatives_count
            instance_scores.append(score)

        avg_score = np.nanmean(instance_scores) if instance_scores else np.nan
        logger.info(f"Calculated {self.name}: Average={avg_score:.4f}")
        return {'metric_name': self.name, 'average_score': avg_score, 'instance_scores': instance_scores}


class SensitivityMetric(BaseEvaluationMetric):
    """
    Calculates sensitivity based on the overlap between top_k explained features
    (ranked by importance) and the ground truth important features.
    Sensitivity = |(Top K Explained) INTERSECTION (Ground Truth Imp)| / |Ground Truth Imp|
    """

    def __init__(self, top_k: int = 2):
        super().__init__("sensitivity")  # Original name was 'sensetivity'
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        self.top_k = top_k
        logger.info(f"SensitivityMetric initialized with top_k={self.top_k}.")

    def calculate(self, ground_truth_df: pd.DataFrame, explanations_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        merged_df = self._align_data(ground_truth_df, explanations_df)
        if merged_df is None or merged_df.empty:
            return {'metric_name': self.name, 'average_score': np.nan, 'instance_scores': [],
                    'config': {'top_k': self.top_k}}

        instance_scores = []
        for _, row in merged_df.iterrows():
            gt_imp_vars = set(row['imp_vars_gt'])
            exp_features_list = row['features_exp']
            exp_importances_list = row.get('importance_exp', [])  # Use .get for safety

            if not isinstance(exp_features_list, (list, np.ndarray)) or \
                    not isinstance(exp_importances_list, (list, np.ndarray)) or \
                    len(exp_features_list) != len(exp_importances_list):
                logger.warning(
                    f"Instance {row.get('instance_idx', 'Unknown')}: Mismatched features/importances or invalid type. Skipping sensitivity calculation.")
                instance_scores.append(np.nan)
                continue

            if not exp_features_list:  # No explained features
                instance_scores.append(0.0 if gt_imp_vars else 1.0)  # 0 if GT exists, 1 if GT also empty
                continue

            try:
                # Create pairs and sort by absolute importance, then take top_k features
                # Ensure importances are numeric
                numeric_importances = pd.to_numeric(exp_importances_list, errors='coerce')
                valid_indices = ~np.isnan(numeric_importances)

                sorted_exp_features = [
                    feat for _, feat in sorted(
                        zip(np.abs(numeric_importances[valid_indices]), np.array(exp_features_list)[valid_indices]),
                        key=lambda x: x[0], reverse=True
                    )
                ]
                top_k_explained_set = set(sorted_exp_features[:self.top_k])

            except Exception as e:
                logger.warning(
                    f"Instance {row.get('instance_idx', 'Unknown')}: Error processing importances for sensitivity: {e}. Skipping.")
                instance_scores.append(np.nan)
                continue

            if not gt_imp_vars:
                # No ground truth important features. Sensitivity is 1 if no top_k features explained, else 0.
                score = 1.0 if not top_k_explained_set else 0.0
            else:
                common_features_count = len(top_k_explained_set.intersection(gt_imp_vars))
                score = common_features_count / len(gt_imp_vars)
            instance_scores.append(score)

        avg_score = np.nanmean(instance_scores) if instance_scores else np.nan
        logger.info(f"Calculated {self.name} (top_k={self.top_k}): Average={avg_score:.4f}")
        return {'metric_name': self.name, 'average_score': avg_score, 'instance_scores': instance_scores,
                'config': {'top_k': self.top_k}}


# --- Metric Factory ---
_METRIC_CLASSES = {
    "recall": RecallMetric,
    "fpr": FPRMetric,
    "sensitivity": SensitivityMetric,  # Consistent name
}


def get_metric_calculator(metric_name: str, **kwargs) -> BaseEvaluationMetric:
    """
    Factory function to get an instance of a metric calculator.
    Args:
        metric_name (str): Name of the metric.
        **kwargs: Arguments to pass to the metric class constructor (e.g., partial=True).
    Returns:
        BaseEvaluationMetric: An instance of the metric calculator.
    """
    metric_class = _METRIC_CLASSES.get(metric_name.lower())
    if metric_class is None:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(_METRIC_CLASSES.keys())}")
    return metric_class(**kwargs)


# --- Composite Score Functions ---
# These functions now expect a DataFrame of pre-calculated individual metric results.
def calculate_overall_score(
        evaluation_results_df: pd.DataFrame,
        dataset_complexity_map: Dict[str, float],  # Key: dataset_id
        model_accuracy_map: Dict[str, float]  # Key: dataset_id
) -> Tuple[float, Dict[str, float]]:
    """
    Calculates an overall score per dataset and a total score based on aggregated metric results.

    Args:
        evaluation_results_df (pd.DataFrame): DataFrame with columns like 'dataset_id',
                                             'lib', 'metric_name', 'average_score'.
        dataset_complexity_map (Dict[str, float]): Mapping from dataset_id to its complexity.
        model_accuracy_map (Dict[str, float]): Mapping from dataset_id to model accuracy on it.

    Returns:
        Tuple[float, Dict[str, float]]: (total_overall_score, per_dataset_scores_map)
    """
    logger.info("Calculating Overall Score...")
    per_dataset_final_scores: Dict[str, float] = {}

    # Define which metrics contribute and how (lower is better for FPR)
    # Assuming 'average_score' for FPR is direct FPR, so we use 1 - FPR.
    # Ensure metric names match those produced by the calculator classes (all lowercase).
    metric_contribution = {
        "recall": lambda x: x,
        "fpr": lambda x: 1 - x if not np.isnan(x) else np.nan,  # Lower FPR is better
        "sensitivity": lambda x: x,
    }
    contributing_metric_names = list(metric_contribution.keys())

    # Iterate over unique combinations of dataset and explainer for overall scoring
    # This function seems to calculate a score for an implicit "current explainer" scenario
    # or an average over explainers if not filtered before.
    # Original code: `eval_all.dataset.unique()`. If `eval_all` contains multiple explainers, this would average over them.
    # Let's assume `evaluation_results_df` is for a single explainer or the aggregation is desired.

    # Group by dataset to calculate average base score per dataset
    for dataset_id, group_df in evaluation_results_df.groupby('dataset_id'):
        dataset_id_str = str(dataset_id)  # Ensure string key for maps

        valid_scores = []
        for metric_name in contributing_metric_names:
            metric_rows = group_df[group_df['metric_name'] == metric_name]
            if not metric_rows.empty:
                # Take the mean if multiple entries exist (e.g. recall with partial=True and partial=False)
                # Or ensure evaluation_results_df is pre-filtered to one version of each metric type.
                score_val = metric_rows['average_score'].mean()
                if not np.isnan(score_val):
                    valid_scores.append(metric_contribution[metric_name](score_val))

        if not valid_scores:
            avg_base_score = np.nan
        else:
            avg_base_score = np.mean(valid_scores)

        if np.isnan(avg_base_score):
            logger.warning(f"Could not compute base score for dataset '{dataset_id_str}'. Skipping.")
            per_dataset_final_scores[dataset_id_str] = np.nan
            continue

        complexity = dataset_complexity_map.get(dataset_id_str, 1.0)
        accuracy = model_accuracy_map.get(dataset_id_str, 1.0)

        if accuracy == 0:
            logger.warning(f"Model accuracy for dataset '{dataset_id_str}' is 0. Resulting score will be 0 or NaN.")
            dataset_final_score = 0.0 if avg_base_score == 0 else np.nan  # Avoid inf
        else:
            dataset_final_score = (avg_base_score * complexity) / accuracy

        per_dataset_final_scores[dataset_id_str] = dataset_final_score
        logger.debug(
            f"Dataset: {dataset_id_str}, AvgBaseScore: {avg_base_score:.3f}, Complexity: {complexity:.2f}, Accuracy: {accuracy:.3f}, FinalScore: {dataset_final_score:.3f}")

    total_overall_score = np.nansum(list(per_dataset_final_scores.values()))  # nansum treats NaNs as 0 for sum
    logger.info(f"Overall Score calculation finished. Total Score: {total_overall_score:.4f}")
    return total_overall_score, per_dataset_final_scores


def calculate_xfa_score(
        explainer_lib_id: str,
        evaluation_results_df: pd.DataFrame,  # Should contain 'dataset_id', 'metric_name', 'average_score'
        dataset_complexity_map: Dict[str, float],  # Key: dataset_id
        dataset_order_for_complexity: Optional[List[str]] = None  # If complexities are a vector
) -> float:
    """
    Calculates the XFA score for a specific explainer library.
    The original formula was: XFA_score_temp = (1 - FPR_ + recall_ + sensitivity_) * dataset_complexity * (100 / sum_total_complexity) / 3

    Args:
        explainer_lib_id (str): Identifier for the explainer library.
        evaluation_results_df (pd.DataFrame): DataFrame with individual metric results.
        dataset_complexity_map (Dict[str, float]): Maps dataset_id to its complexity value.
        dataset_order_for_complexity (Optional[List[str]]): If provided, defines the order and set of datasets
                                                            to consider for sum_total_complexity. If None, sum over
                                                            complexities of datasets present for the explainer.

    Returns:
        float: The total XFA score for the explainer.
    """
    logger.info(f"Calculating XFA Score for explainer '{explainer_lib_id}'...")
    logger.info(evaluation_results_df.columns)

    explainer_results = evaluation_results_df[evaluation_results_df['lib'] == explainer_lib_id]
    if explainer_results.empty:
        logger.warning(f"No results found for explainer '{explainer_lib_id}'. XFA score will be 0.")
        return 0.0

    sum_total_complexity = 0
    if dataset_order_for_complexity:
        for ds_id in dataset_order_for_complexity:
            sum_total_complexity += dataset_complexity_map.get(str(ds_id), 0)
    else:  # Sum complexities of datasets for which the explainer has results
        for ds_id in explainer_results['dataset_id'].unique():
            sum_total_complexity += dataset_complexity_map.get(str(ds_id), 0)

    if sum_total_complexity == 0:
        logger.warning(
            "Total dataset complexity is 0. XFA score cannot be normalized as per original formula, will be sum of unnormalized scores or NaN.")
        # The original normalization factor was (100 / sum_total_complexity)
        # If sum_total_complexity is 0, this factor is problematic.
        # Let's return NaN if sum_total_complexity is 0 and any score contribution exists.
        # If no score contributions, then 0.

    xfa_score_contributions = []
    for dataset_id, group_df in explainer_results.groupby('dataset_id'):
        dataset_id_str = str(dataset_id)

        recall_series = group_df[group_df['metric_name'] == 'recall']['average_score']
        fpr_series = group_df[group_df['metric_name'] == 'fpr']['average_score']
        sensitivity_series = group_df[group_df['metric_name'] == 'sensitivity']['average_score']

        if recall_series.empty or fpr_series.empty or sensitivity_series.empty:
            logger.warning(
                f"Missing one or more base metrics for dataset '{dataset_id_str}', explainer '{explainer_lib_id}'. Skipping for XFA.")
            continue

        # Take mean if multiple entries for a metric (e.g. different configs of recall)
        # Ideally, evaluation_results_df is filtered to specific configs before this.
        recall_val = recall_series.mean()
        fpr_val = fpr_series.mean()
        sensitivity_val = sensitivity_series.mean()

        if any(np.isnan([recall_val, fpr_val, sensitivity_val])):
            logger.warning(
                f"NaN metric value for dataset '{dataset_id_str}', explainer '{explainer_lib_id}'. Skipping for XFA.")
            continue

        current_dataset_complexity = dataset_complexity_map.get(dataset_id_str, 0)
        if current_dataset_complexity == 0:
            logger.debug(f"Complexity for dataset '{dataset_id_str}' is 0. It won't contribute to XFA score.")

        # Per-dataset term before normalization: (1 - FPR + Recall + Sensitivity) / 3 * Complexity
        term_value = (1 - fpr_val + recall_val + sensitivity_val) / 3.0
        xfa_score_contributions.append(term_value * current_dataset_complexity)
        logger.debug(
            f"DS: {dataset_id_str}, R:{recall_val:.2f}, F:{fpr_val:.2f}, S:{sensitivity_val:.2f}, BaseXFATerm:{term_value:.3f}, Compl:{current_dataset_complexity}, WeightedTerm:{term_value * current_dataset_complexity:.3f}")

    if not xfa_score_contributions:
        logger.info(f"No XFA score contributions for '{explainer_lib_id}'. Final XFA Score: 0.0")
        return 0.0

    sum_weighted_terms = np.nansum(xfa_score_contributions)

    if sum_total_complexity == 0:
        final_xfa_score = np.nan  # Or 0 if sum_weighted_terms is also 0
    else:
        final_xfa_score = sum_weighted_terms * (100 / sum_total_complexity)

    logger.info(
        f"XFA Score for '{explainer_lib_id}': SumWeightedTerms={sum_weighted_terms:.4f}, TotalComplexity={sum_total_complexity:.2f}, Final XFA Score={final_xfa_score:.4f}")
    return final_xfa_score


# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Mock data
    gt_data = pd.DataFrame({
        'instance_idx': [1, 2, 3, 4],
        'imp_vars': [['f1', 'f2'], ['f3'], ['f1', 'f4', 'f5'], ['f2']]
    })
    # Ensure 'features' and 'importance' are lists of the same length for each row
    exp_data = pd.DataFrame({
        'instance_idx': [1, 2, 3, 4, 5],  # Instance 5 won't be matched
        'lib': ['lime'] * 5,
        'dataset_id': ['ds1'] * 5,
        'features': [['f1', 'f3', 'f5'], ['f3', 'f4'], ['f1', 'f2'], ['f6'], ['f7']],
        'importance': [[0.5, 0.3, 0.1], [0.8, 0.1], [0.9, -0.2], [0.5], [0.6]]
    })
    all_feats = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']

    # --- Individual Metric Calculation ---
    print("\n--- Calculating Individual Metrics ---")
    recall_calc_partial = get_metric_calculator("recall", partial=True)
    recall_res_partial = recall_calc_partial.calculate(gt_data.copy(), exp_data.copy())
    print(f"Partial Recall: {recall_res_partial}")

    recall_calc_full = get_metric_calculator("recall", partial=False)
    recall_res_full = recall_calc_full.calculate(gt_data.copy(), exp_data.copy())
    print(f"Full Recall: {recall_res_full}")

    fpr_calc = get_metric_calculator("fpr")
    fpr_res = fpr_calc.calculate(gt_data.copy(), exp_data.copy(), all_feature_names=all_feats)
    print(f"FPR: {fpr_res}")

    sensitivity_calc = get_metric_calculator("sensitivity", top_k=2)
    sensitivity_res = sensitivity_calc.calculate(gt_data.copy(), exp_data.copy())
    print(f"Sensitivity (top_k=2): {sensitivity_res}")

    # --- Composite Score Calculation Example ---
    print("\n--- Calculating Composite Scores ---")
    # Create a mock evaluation_results_df
    # This df would typically be built by running all metric calculators over all datasets/explainers
    results_list_for_composite = [
        {'dataset_id': 'ds1', 'lib': 'lime', 'metric_name': 'recall',
         'average_score': recall_res_full['average_score']},
        {'dataset_id': 'ds1', 'lib': 'lime', 'metric_name': 'fpr', 'average_score': fpr_res['average_score']},
        {'dataset_id': 'ds1', 'lib': 'lime', 'metric_name': 'sensitivity',
         'average_score': sensitivity_res['average_score']},
        {'dataset_id': 'ds2', 'lib': 'lime', 'metric_name': 'recall', 'average_score': 0.7},
        {'dataset_id': 'ds2', 'lib': 'lime', 'metric_name': 'fpr', 'average_score': 0.15},
        {'dataset_id': 'ds2', 'lib': 'lime', 'metric_name': 'sensitivity', 'average_score': 0.6},
    ]
    mock_eval_results_df = pd.DataFrame(results_list_for_composite)

    ds_complexity = {'ds1': 1.5, 'ds2': 2.0}
    model_acc = {'ds1': 0.9, 'ds2': 0.85}

    total_overall, per_ds_overall = calculate_overall_score(mock_eval_results_df, ds_complexity, model_acc)
    print(f"Total Overall Score: {total_overall}, Per-Dataset Overall Scores: {per_ds_overall}")

    ds_complexity_ordered_ids = ['ds1', 'ds2']  # Order for complexity vector
    ds_complexity_map_for_xfa = {'ds1': 1.5, 'ds2': 2.0}  # Or use dataset_complexity_vector directly if ids are indices

    xfa_score_lime = calculate_xfa_score('lime', mock_eval_results_df, ds_complexity_map_for_xfa,
                                         ds_complexity_ordered_ids)
    print(f"XFA Score for LIME: {xfa_score_lime}")