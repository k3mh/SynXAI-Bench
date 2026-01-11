import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, accuracy_score

# Configure logger for this module
logger = logging.getLogger(__name__)


def validate_feature_correlation(
        X_test: pd.DataFrame,
        y_test: pd.Series,
        meta_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Performs correlation validation for each RGS group in the test set.

    For each rule group (RGS), this function calculates the Pearson correlation
    of every feature with the target variable, using only the instances
    belonging to that group. It then sorts the features based on the absolute
    correlation value to identify the most correlated features.

    Args:
        X_test (pd.DataFrame): The test data features.
        y_test (pd.Series): The corresponding true labels for the test data.
        meta_test (pd.DataFrame): The metadata for the test data, with 'imp_vars' and 'RGS'.

    Returns:
        pd.DataFrame: A DataFrame summarizing the correlation analysis for each RGS group,
                      with columns for the RGS name, its important features from metadata,
                      and the observed features and their correlations sorted by impact.
    """
    if not (X_test.index.equals(meta_test.index) and X_test.index.equals(y_test.index)):
        logger.error("X_test, y_test, and meta_test must have identical indices for alignment.")
        raise ValueError("Index mismatch between input DataFrames/Series.")

    correlation_results = []
    logger.info(f"Starting correlation validation for {len(meta_test['RGS'].unique())} RGS groups.")

    for rgs_group in meta_test['RGS'].unique():
        group_indices = meta_test[meta_test['RGS'] == rgs_group].index
        X_test_group = X_test.loc[group_indices]
        y_test_group = y_test.loc[group_indices]

        if X_test_group.empty or len(y_test_group.unique()) < 2:
            logger.warning(f"Skipping correlation analysis for RGS group '{rgs_group}' due to insufficient data.")
            continue

        # Get the important features for this rule from metadata
        rgs_important_features = meta_test.loc[group_indices[0]].get('imp_vars', [])

        try:
            # Calculate correlation of all features with the target for this subset
            all_correlations = X_test_group.corrwith(y_test_group)

            # Sort by absolute value to find the strongest relationships
            sorted_correlations = all_correlations.abs().sort_values(ascending=False)

            # Get the sorted feature names and their corresponding original correlation values
            sorted_feature_names = sorted_correlations.index.tolist()
            sorted_correlation_values = all_correlations.loc[sorted_feature_names].tolist()

            correlation_results.append({
                'rgs_group': rgs_group,
                'rgs_important_features': rgs_important_features,
                'correlated_features_sorted': sorted_feature_names,
                'correlation_values_sorted': sorted_correlation_values,
                'num_instances_in_group': len(X_test_group)
            })
        except Exception as e:
            logger.error(f"Could not compute correlations for RGS group '{rgs_group}': {e}")
            continue

    logger.info("Correlation validation complete.")
    if not correlation_results:
        logger.warning("Correlation validation finished, but no results were generated.")
        return pd.DataFrame()

    return pd.DataFrame(correlation_results)


def validate_model_additive_impact(
        ml_model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        meta_test: pd.DataFrame,
        metric: str = 'auc'
) -> pd.DataFrame:
    """
    Validates a model by measuring the additive impact on performance (AUC or Accuracy)
    as important features for a rule group (RGS) are added back one by one.

    This function operates on ALL provided test samples.

    The process starts by neutralizing all important features from all RGS groups
    globally across the test set by shuffling them. Then, for each RGS group, it
    evaluates performance on its subset of this neutralized data. It then iteratively
    adds back the features specific to that RGS group, measuring the performance gain
    on that group's subset at each step.

    Args:
        ml_model (Any): The trained model with `predict` and `predict_proba` methods.
        X_test (pd.DataFrame): The test data features.
        y_test (pd.Series): The corresponding true labels for the test data.
        meta_test (pd.DataFrame): The metadata for the test data, with 'imp_vars' and 'RGS'.
        metric (str): The performance metric to use for validation ('auc' or 'accuracy').

    Returns:
        pd.DataFrame: A DataFrame where each row represents the impact of adding back one
                      feature for one RGS group, measured by the gain in performance.
    """
    # This is an internal helper that contains the core logic.
    return _perform_additive_impact_validation(
        ml_model=ml_model,
        X_test=X_test,
        y_test=y_test,
        meta_test=meta_test,
        metric=metric,
        filter_correct_predictions=False
    )


def validate_model_additive_impact_on_correct_samples(
        ml_model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        meta_test: pd.DataFrame,
        metric: str = 'auc'
) -> pd.DataFrame:
    """
    Validates a model by measuring the additive impact on performance, but ONLY
    on the subset of test samples that the model initially predicted correctly.

    The process is identical to `validate_model_additive_impact`, but all operations
    (neutralization, subsetting, and performance calculation) are performed on
    the filtered set of correctly predicted instances.

    Args:
        ml_model (Any): The trained model with `predict` and `predict_proba` methods.
        X_test (pd.DataFrame): The test data features.
        y_test (pd.Series): The corresponding true labels for the test data.
        meta_test (pd.DataFrame): The metadata for the test data.
        metric (str): The performance metric to use for validation ('auc' or 'accuracy').

    Returns:
        pd.DataFrame: A DataFrame with the additive impact results for the
                      correctly predicted samples.
    """
    # This function acts as a wrapper that sets the flag to filter predictions.
    return _perform_additive_impact_validation(
        ml_model=ml_model,
        X_test=X_test,
        y_test=y_test,
        meta_test=meta_test,
        metric=metric,
        filter_correct_predictions=True
    )


def _perform_additive_impact_validation(
        ml_model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        meta_test: pd.DataFrame,
        metric: str,
        filter_correct_predictions: bool
) -> pd.DataFrame:
    """
    Internal core function to perform additive impact validation.
    It can operate on either all test samples or only correctly predicted ones.
    This version uses a full local neutralization to ensure a true baseline.
    """
    if not (X_test.index.equals(meta_test.index) and X_test.index.equals(y_test.index)):
        logger.error("X_test, y_test, and meta_test must have identical indices for alignment.")
        raise ValueError("Index mismatch between input DataFrames/Series.")
    if metric not in ['auc', 'accuracy']:
        raise ValueError("Metric must be either 'auc' or 'accuracy'.")

    validation_results = []

    # --- Optional Initial Filtering Step ---
    if filter_correct_predictions:
        logger.info("Filtering test set to include only correctly predicted samples.")
        y_pred = ml_model.predict(X_test.values)
        correct_indices = y_test.index[y_test == y_pred]

        X_test = X_test.loc[correct_indices]
        y_test = y_test.loc[correct_indices]
        meta_test = meta_test.loc[correct_indices]

        logger.info(f"Proceeding with {len(X_test)} correctly predicted samples out of {len(y_pred)} total.")
        if X_test.empty:
            logger.warning("No correctly predicted samples found. Cannot perform validation.")
            return pd.DataFrame()

    # --- Per-RGS Additive Analysis with Full Local Neutralization ---
    logger.info(f"Starting per-RGS additive validation for {len(meta_test['RGS'].unique())} groups using '{metric}'.")
    for rgs_group in meta_test['RGS'].unique():
        group_indices = meta_test[meta_test['RGS'] == rgs_group].index
        X_test_group = X_test.loc[group_indices]
        y_test_group = y_test.loc[group_indices]

        if X_test_group.empty or len(y_test_group.unique()) < 2:
            logger.warning(f"Skipping RGS group '{rgs_group}' due to insufficient data or only one class present.")
            continue

        rgs_important_features = meta_test.loc[group_indices[0]].get('imp_vars', [])

        if not rgs_important_features or (len(rgs_important_features) == 1 and rgs_important_features[0] == ''):
            logger.debug(f"Skipping RGS group '{rgs_group}' as it has no important features defined.")
            continue

        # --- 1. Calculate Original Score for this group ---
        try:
            if metric == 'auc':
                original_probas = ml_model.predict_proba(X_test_group.values)[:, 1]
                original_group_score = roc_auc_score(y_test_group, original_probas)
            else:  # accuracy
                original_preds = ml_model.predict(X_test_group.values)
                original_group_score = accuracy_score(y_test_group, original_preds)
        except Exception as e:
            logger.error(f"Could not calculate original score for RGS group '{rgs_group}': {e}")
            original_group_score = np.nan

        # --- 2. Full Local Neutralization: Shuffle ALL features within the group ---
        fully_neutralized_X_group = X_test_group.copy()
        for feature in fully_neutralized_X_group.columns:
            fully_neutralized_X_group[feature] = np.random.permutation(X_test_group[feature].values)

        # --- 3. Calculate True Baseline Score on fully shuffled data ---
        try:
            if metric == 'auc':
                neutralized_probas = ml_model.predict_proba(fully_neutralized_X_group.values)[:, 1]
                previous_score = roc_auc_score(y_test_group, neutralized_probas)
            else:  # accuracy
                neutralized_preds = ml_model.predict(fully_neutralized_X_group.values)
                previous_score = accuracy_score(y_test_group, neutralized_preds)
        except Exception as e:
            logger.error(f"Could not calculate initial score for RGS '{rgs_group}' on locally neutralized data: {e}")
            continue

        current_counterfactual_X_group = fully_neutralized_X_group.copy()
        active_features_at_step = []
        # Iterate from least to most important feature
        for feature_to_add_back in reversed(rgs_important_features):
            current_counterfactual_X_group[feature_to_add_back] = X_test_group[feature_to_add_back]
            active_features_at_step.append(feature_to_add_back)

            try:
                if metric == 'auc':
                    new_probas = ml_model.predict_proba(current_counterfactual_X_group.values)[:, 1]
                    new_score = roc_auc_score(y_test_group, new_probas)
                else:  # accuracy
                    new_preds = ml_model.predict(current_counterfactual_X_group.values)
                    new_score = accuracy_score(y_test_group, new_preds)
            except Exception as e:
                logger.error(
                    f"Model prediction/scoring failed for additive step of RGS '{rgs_group}' (feature {feature_to_add_back}): {e}")
                continue

            performance_gain = new_score - previous_score

            validation_results.append({
                'rgs_group': rgs_group,
                'num_instances_in_group': len(X_test_group),
                'validation_metric': metric,
                'original_group_score': original_group_score,
                'feature_added_back': feature_to_add_back,
                'active_important_features': sorted(active_features_at_step),
                'score_before_adding': previous_score,
                'score_after_adding': new_score,
                'performance_gain': performance_gain
            })

            previous_score = new_score

    logger.info("Additive counterfactual validation complete.")
    if not validation_results:
        logger.warning("Validation finished, but no results were generated.")
        return pd.DataFrame()

    return pd.DataFrame(validation_results)


def analyze_validation_results(
        validation_df: pd.DataFrame,
        meta_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyzes the results from additive counterfactual validation to rank features by impact
    and compare this ranking with the metadata.

    Args:
        validation_df (pd.DataFrame): The DataFrame produced by one of the `validate_model_additive_impact` functions.
        meta_test (pd.DataFrame): The test metadata, used to get the ground truth feature order.

    Returns:
        pd.DataFrame: A summary DataFrame with one row per RGS group, showing the
                      ground truth feature order, the observed impact-based order,
                      and a rank correlation score (Kendall's Tau).
    """
    if validation_df.empty:
        return pd.DataFrame()

    analysis_results = []

    for rgs_group in validation_df['rgs_group'].unique():
        meta_group = meta_test[meta_test['RGS'] == rgs_group]
        if meta_group.empty:
            continue
        ground_truth_order = meta_group.iloc[0]['imp_vars']

        validation_group = validation_df[validation_df['rgs_group'] == rgs_group]
        observed_order = validation_group.sort_values(by='performance_gain', ascending=False)[
            'feature_added_back'].tolist()

        tau, p_value = -1, -1
        if set(observed_order) == set(ground_truth_order) and len(ground_truth_order) > 1:
            gt_rank_map = {feature: rank for rank, feature in enumerate(ground_truth_order)}
            observed_ranks_for_gt_features = [gt_rank_map[feature] for feature in observed_order]
            tau, p_value = kendalltau(list(range(len(ground_truth_order))), observed_ranks_for_gt_features)

        analysis_results.append({
            'rgs_group': rgs_group,
            'ground_truth_feature_order': ground_truth_order,
            'observed_impact_feature_order': observed_order,
            'kendalls_tau_correlation': tau,
            'p_value': p_value,
            'num_instances_in_group': validation_group['num_instances_in_group'].iloc[0],
            'validation_metric': validation_group['validation_metric'].iloc[0]
        })

    return pd.DataFrame(analysis_results).sort_values(by='kendalls_tau_correlation', ascending=False)

