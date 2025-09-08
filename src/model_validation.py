# File: model_validation.py

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
    if not (X_test.index.equals(meta_test.index) and X_test.index.equals(y_test.index)):
        logger.error("X_test, y_test, and meta_test must have identical indices for alignment.")
        raise ValueError("Index mismatch between input DataFrames/Series.")
    if metric not in ['auc', 'accuracy']:
        raise ValueError("Metric must be either 'auc' or 'accuracy'.")

    validation_results = []

    # 1. Global Neutralization
    all_important_features = set()
    for imp_vars_list in meta_test['imp_vars']:
        if isinstance(imp_vars_list, list):
            all_important_features.update(imp_vars_list)
    all_important_features.discard('')  # Remove empty string if present
    all_important_features = sorted(list(all_important_features))

    logger.info(f"Identified {len(all_important_features)} unique important features for global neutralization.")

    globally_neutralized_X_test = X_test.copy()
    for feature in all_important_features:
        if feature in globally_neutralized_X_test.columns:
            globally_neutralized_X_test[feature] = np.random.permutation(X_test[feature].values)
        else:
            logger.warning(f"Global important feature '{feature}' not in X_test columns.")

    # 2. Per-RGS Additive Analysis
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

        try:
            neutralized_X_group_slice = globally_neutralized_X_test.loc[group_indices]
            if metric == 'auc':
                neutralized_probas = ml_model.predict_proba(neutralized_X_group_slice.values)[:, 1]
                previous_score = roc_auc_score(y_test_group, neutralized_probas)
            else:  # accuracy
                neutralized_preds = ml_model.predict(neutralized_X_group_slice.values)
                previous_score = accuracy_score(y_test_group, neutralized_preds)
        except Exception as e:
            logger.error(f"Could not calculate initial score for RGS '{rgs_group}' on globally neutralized data: {e}")
            continue

        current_counterfactual_X_group = neutralized_X_group_slice.copy()
        active_features_at_step = []
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
        validation_df (pd.DataFrame): The DataFrame produced by `validate_model_additive_impact`.
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


if __name__ == '__main__':
    # --- Example Usage ---
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    feature_names_example = [f'f{i}' for i in range(10)]
    X = pd.DataFrame(np.random.rand(200, 10), columns=feature_names_example)
    y = pd.Series(np.random.randint(0, 2, 200))

    meta_list = []
    for i in range(200):
        if i % 4 == 0:
            meta_list.append({'imp_vars': ['f1', 'f3', 'f5'], 'RGS': 'group_A'})
        elif i % 4 == 1:
            meta_list.append({'imp_vars': ['f2', 'f4'], 'RGS': 'group_B'})
        else:
            meta_list.append({'imp_vars': ['f7'], 'RGS': 'group_D'})
    meta = pd.DataFrame(meta_list)

    X_train_ex, X_test_ex, y_train_ex, y_test_ex, _, meta_test_ex = train_test_split(
        X, y, meta, test_size=0.5, random_state=42, stratify=y
    )

    model_ex = RandomForestClassifier(random_state=42)
    model_ex.fit(X_train_ex, y_train_ex)

    print("-" * 50)

    # --- Run Correlation Validation Step ---
    print("\n--- Running Correlation Validation ---")
    correlation_results_df = validate_feature_correlation(
        X_test=X_test_ex, y_test=y_test_ex, meta_test=meta_test_ex
    )
    if not correlation_results_df.empty:
        print("\n--- Correlation Validation Summary ---")
        pd.set_option('display.max_colwidth', None)
        print(correlation_results_df)
        print("-" * 50)

    # --- Run Additive Impact Validation using AUC ---
    print("\n--- Running Additive Validation with AUC Metric ---")
    validation_results_auc = validate_model_additive_impact(
        ml_model=model_ex, X_test=X_test_ex, y_test=y_test_ex,
        meta_test=meta_test_ex, metric='auc'
    )
    if not validation_results_auc.empty:
        analysis_summary_auc = analyze_validation_results(validation_results_auc, meta_test_ex)
        print("\n--- Additive Impact Results (AUC) ---")
        print(validation_results_auc)
        print("\n--- Analysis Summary (AUC) ---")
        print(analysis_summary_auc)
        print("-" * 50)
