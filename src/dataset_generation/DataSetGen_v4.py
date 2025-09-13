# File: dataset_generation/DataSetGen_v4.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.datasets import (
    make_gaussian_quantiles, make_hastie_10_2, make_classification,
    make_friedman1, make_friedman2, make_friedman3, make_blobs, make_moons
)
import logging
from typing import List, Dict, Any, Callable, NamedTuple, Set
from sklearn.utils import check_random_state, shuffle as util_shuffle
from itertools import product
from collections.abc import Iterable
import numbers




logger = logging.getLogger(__name__)

# --- Configuration ---
NUM_TOTAL_FEATURES = 48
ALL_FEATURE_NAMES = [f'x{i}' for i in range(1, NUM_TOTAL_FEATURES + 1)]


# --- Data Structure for Output ---
class SyntheticDataset(NamedTuple):
    data: pd.DataFrame
    meta_data: pd.DataFrame
    feature_names: List[str]
    name: str
    description: str


# --- Helper Functions ---

def _generate_uncorrelated_features(
        samples: int,
        feature_names: List[str],
        correlation_threshold: float = 0.03,
        max_retries: int = 20
) -> pd.DataFrame:
    """
    Generates a DataFrame of features with inter-feature correlation below a threshold.

    This function repeatedly generates features from a standard normal distribution
    until the maximum absolute correlation between any two features is less than
    the specified threshold.

    Args:
        samples (int): The number of samples (rows) to generate.
        feature_names (List[str]): The list of feature names for the columns.
        correlation_threshold (float): The maximum allowed absolute correlation
                                       between any two features.
        max_retries (int): The maximum number of attempts to meet the threshold
                           before proceeding with the last generated set.

    Returns:
        pd.DataFrame: A DataFrame of uncorrelated features, scaled to [0, 1].
    """
    if not feature_names or len(feature_names) < 2:
        if not feature_names:
            return pd.DataFrame(index=range(samples))
        # If only one feature, correlation is not applicable
        data_array = np.random.randn(samples, len(feature_names))
        df = pd.DataFrame(data_array, columns=feature_names)
        return pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)),
                            columns=feature_names, dtype=float)

    num_features = len(feature_names)
    for attempt in range(max_retries):
        data_array = np.random.randn(samples, num_features)
        df = pd.DataFrame(data_array, columns=feature_names)

        # Check correlation
        corr_matrix = df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)  # Ignore self-correlation
        max_corr = corr_matrix.max().max()

        if max_corr < correlation_threshold:
            logger.debug(
                f"Uncorrelated features generated successfully on attempt {attempt + 1} (max_corr: {max_corr:.4f})")
            # Scale the final accepted DataFrame
            df_scaled = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)),
                                     columns=feature_names, dtype=float)
            return df_scaled

    logger.warning(
        f"Could not generate features with correlation below {correlation_threshold} "
        f"within {max_retries} retries. Proceeding with the last set (max_corr: {max_corr:.4f}). "
        f"Consider increasing sample size or threshold."
    )
    # Scale and return the last attempt if threshold not met
    df_scaled = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)),
                             columns=feature_names, dtype=float)
    return df_scaled


def _combine_and_order_features(
        important_features_df: pd.DataFrame,
        noise_features_df: pd.DataFrame,
        target_series: pd.Series
) -> pd.DataFrame:
    """
    Combines important and noise features with the target, ensuring a consistent
    final column order.
    """
    # Concatenate all parts
    final_df = pd.concat([important_features_df, noise_features_df, target_series], axis=1)

    # Ensure final column order is x1, x2, ..., xN, y
    final_ordered_columns = ALL_FEATURE_NAMES + [target_series.name]

    # Reindex to ensure all columns are present, filling missing ones with noise if necessary
    # (This is a safeguard, but shouldn't be needed with the new logic)
    final_df = final_df.reindex(columns=final_ordered_columns)

    return final_df


# --- New Validation Function ---
def validate_dataset_correlations(
        dataset: SyntheticDataset,
        threshold: float = 0.1
) -> bool:
    """
    Validates that only important features are correlated with the target.

    This function checks three conditions:
    1. Noise vs. Noise: No noise feature has a high correlation with any other noise feature.
    2. Noise vs. Target: No noise feature has a high correlation with the target variable.
    3. Important vs. Target: At least one important feature has a significant correlation with the target.

    Args:
        dataset (SyntheticDataset): The generated dataset object to validate.
        threshold (float): The maximum absolute correlation allowed for noise features.

    Returns:
        bool: True if the dataset passes all validation checks, False otherwise.
    """
    logger.info(f"--- Running Correlation Validation for: {dataset.name} ---")
    data = dataset.data
    meta = dataset.meta_data

    # Get important features from metadata (use first row as representative)
    important_features = meta['imp_vars'].iloc[0]
    if not isinstance(important_features, list) or not important_features or important_features == ['']:
        important_features = []

    noise_features = [f for f in ALL_FEATURE_NAMES if f not in important_features]

    # Calculate correlations
    feature_corr_matrix = data[ALL_FEATURE_NAMES].corr().abs()
    target_corr = data[ALL_FEATURE_NAMES].corrwith(data['y']).abs()

    # --- Validation Checks ---
    validation_passed = True

    # 1. Check Noise vs. Noise correlation
    if noise_features:
        noise_corr = feature_corr_matrix.loc[noise_features, noise_features]
        np.fill_diagonal(noise_corr.values, 0)  # Ignore self-correlation
        max_noise_noise_corr = noise_corr.max().max()
        if max_noise_noise_corr > threshold:
            validation_passed = False
            logger.error(
                f"VALIDATION FAILED: Max noise-noise correlation is {max_noise_noise_corr:.4f} (exceeds threshold {threshold}).")
        else:
            logger.info(
                f"VALIDATION PASSED: Max noise-noise correlation is {max_noise_noise_corr:.4f} (within threshold).")

    # 2. Check Noise vs. Target correlation
    if noise_features:
        max_noise_target_corr = target_corr[noise_features].max()
        if max_noise_target_corr > threshold:
            validation_passed = False
            logger.error(
                f"VALIDATION FAILED: Max noise-target correlation is {max_noise_target_corr:.4f} (exceeds threshold {threshold}).")
        else:
            logger.info(
                f"VALIDATION PASSED: Max noise-target correlation is {max_noise_target_corr:.4f} (within threshold).")

    # 3. Check Important vs. Target correlation
    if important_features:
        max_important_target_corr = target_corr[important_features].max()
        if max_important_target_corr < threshold:
            validation_passed = False
            # This is a warning because non-linear relationships might not show high Pearson correlation
            logger.warning(
                f"VALIDATION WARNING: Max important-target correlation is {max_important_target_corr:.4f} (below threshold {threshold}). This may be acceptable for non-linear datasets.")
        else:
            logger.info(
                f"VALIDATION PASSED: Max important-target correlation is {max_important_target_corr:.4f} (above threshold).")

    if validation_passed:
        logger.info(f"--- Correlation Validation for {dataset.name}: SUCCESS ---")
    else:
        logger.error(f"--- Correlation Validation for {dataset.name}: FAILED ---")

    return validation_passed

def _generate_from_sklearn(
        size: int,
        sklearn_fn: Callable,
        important_feature_names: List[str],
        rgs_name: str,
        description: str
) -> SyntheticDataset:
    """
    Generic helper to create datasets using sklearn.datasets functions.
    """
    num_important = len(important_feature_names)
    noise_feature_names = [f for f in ALL_FEATURE_NAMES if f not in important_feature_names]

    # Generate important features and target from sklearn
    informative_features_np, y_target_raw = sklearn_fn(n_samples=size)

    # Ensure y is binary 0/1
    y_series = pd.Series(np.where(y_target_raw == y_target_raw.max(), 1, 0), name='y')

    # Scale important features and create DataFrame
    important_features_scaled = minmax_scale(informative_features_np, axis=0, feature_range=(0, 1))
    important_features_df = pd.DataFrame(important_features_scaled, columns=important_feature_names)

    # Generate noise features
    noise_features_df = _generate_uncorrelated_features(size, noise_feature_names)

    final_df = _combine_and_order_features(important_features_df, noise_features_df, y_series)

    meta_data = pd.DataFrame({"imp_vars": [important_feature_names] * size, "RGS": [rgs_name] * size})
    return SyntheticDataset(
        data=final_df, meta_data=meta_data, feature_names=ALL_FEATURE_NAMES,
        name=f"{rgs_name}_Sklearn", description=description
    )



def make_gaussian_quantiles_ranked(
        *,
        mean=None,
        cov=1.0,
        n_samples=100,
        n_features=2,
        n_classes=2,
        feature_weights: List[float] = None,
        shuffle=True,
        random_state=None,
):
    """
    Generate isotropic Gaussian features and label samples by a weighted distance.

    This is a modified version of sklearn's make_gaussian_quantiles. It generates
    features from a multi-dimensional standard normal distribution but creates the
    target variable based on a weighted squared Euclidean distance from the origin.
    This allows for a clear, non-linear feature ranking.

    Args:
        mean (array-like, optional): Mean of the distribution. Defaults to origin.
        cov (float, optional): Covariance factor. Defaults to 1.0.
        n_samples (int, optional): Total number of points. Defaults to 100.
        n_features (int, optional): Number of features for each sample. Defaults to 2.
        n_classes (int, optional): Number of classes. Defaults to 2.
        feature_weights (List[float], optional): List of weights to control the
            importance of each feature. Must match n_features. Defaults to equal weights.
        shuffle (bool, optional): Whether to shuffle samples. Defaults to True.
        random_state (int, optional): Random state for reproducibility. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (samples), y (labels).
    """
    if n_samples < n_classes:
        raise ValueError("n_samples must be at least n_classes")

    generator = check_random_state(random_state)

    if mean is None:
        mean = np.zeros(n_features)
    else:
        mean = np.array(mean)

    if feature_weights is None:
        feature_weights = np.ones(n_features)

    if len(feature_weights) != n_features:
        raise ValueError("Length of feature_weights must match n_features.")

    # Build multivariate normal distribution for features
    X = generator.multivariate_normal(mean, cov * np.identity(n_features), (n_samples,))

    # Create target based on a weighted squared distance from the mean.
    # This is equivalent to sum(w_i * (x_i - mean_i)^2) for each sample.
    weighted_distance_sq = np.dot((X - mean) ** 2, feature_weights)

    # Use quantiles of this weighted distance to define class boundaries
    quantiles = np.linspace(0, 1, n_classes + 1)
    thresholds = np.quantile(weighted_distance_sq, quantiles)

    # Ensure the last threshold is inclusive
    thresholds[-1] = np.inf

    y = np.zeros(n_samples, dtype=int)
    for i in range(n_classes):
        condition = (weighted_distance_sq >= thresholds[i]) & (weighted_distance_sq < thresholds[i + 1])
        y[condition] = i

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    return X, y


def make_hastie_10_2_ranked(
        n_samples=12000,
        *,
        feature_weights: List[float] = None,
        random_state=None
):
    """
    Generate data for binary classification with ranked features, based on
    Hastie et al. 2009, Example 10.2.

    The ten features are standard independent Gaussian and the target ``y`` is
    defined by a weighted sum of squares:
    y[i] = 1 if np.sum(weights * (X[i] ** 2)) > 9.34 else -1

    Args:
        n_samples (int, optional): The number of samples. Defaults to 12000.
        feature_weights (List[float], optional): A list of 10 weights to control the
            importance of each feature. Defaults to equal weights of 1.0.
        random_state (int, optional): Random state for reproducibility. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (samples), y (labels).
    """
    rs = check_random_state(random_state)

    if feature_weights is None:
        feature_weights = np.ones(10)

    if len(feature_weights) != 10:
        raise ValueError("feature_weights must be a list or array of length 10.")

    shape = (n_samples, 10)
    X = rs.normal(size=shape).reshape(shape)

    # Calculate the weighted sum of squares
    weighted_sum_of_squares = ((X ** 2.0) * feature_weights).sum(axis=1)

    threshold = np.median(weighted_sum_of_squares)

    y = (weighted_sum_of_squares > threshold).astype(np.float64, copy=False)

    return X, y


def make_friedman1_ranked(
        n_samples=100,
        n_features=10,
        *,
        feature_weights: List[float] = None,
        noise=0.0,
        random_state=None
):
    """
    Generate a modified "Friedman #1" regression problem with 5 ranked features.

    The output `y` is created according to a weighted version of the original formula,
    where the interactive term is split to allow individual ranking:
    y(X) = w[0]*sin(pi*X₀) + w[1]*sin(pi*X₁) + w[2]*(X₂-0.5)² + w[3]*X₃ + w[4]*X₄ + noise

    Args:
        n_samples (int, optional): The number of samples. Defaults to 100.
        n_features (int, optional): The number of features (>= 5). Defaults to 10.
        feature_weights (List[float], optional): A list of 5 weights to control the
            importance of each of the 5 terms. Defaults to weights that rank features
            x1 > x2 > x3 > x4 > x5.
        noise (float, optional): Std of Gaussian noise applied to output. Defaults to 0.0.
        random_state (int, optional): Random state for reproducibility. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (samples), y (labels).
    """
    if n_features < 5:
        raise ValueError("n_features must be at least 5.")

    generator = check_random_state(random_state)

    if feature_weights is None:
        # Default weights to rank features as: X0 > X1 > X2 > X3 > X4
        feature_weights = [20.0, 16.0, 12.0, 8.0, 4.0]

    if len(feature_weights) != 5:
        raise ValueError("feature_weights must be a list or array of length 5.")

    X = generator.uniform(size=(n_samples, n_features))

    # Apply weights to each of the 5 terms in the modified Friedman formula
    y = (
              feature_weights[0] * np.sin(np.pi * X[:, 0])
            + feature_weights[1] * np.sin(np.pi * X[:, 1])
            + feature_weights[2] * (X[:, 2] ) ** 2
            + feature_weights[3] * X[:, 3]
            + feature_weights[4] * X[:, 4]
            + noise * generator.standard_normal(size=(n_samples))
    )

    return X, y


# def make_friedman2_ranked_(
#         n_samples=100,
#         *,
#         feature_weights: List[float] = None,
#         noise=0.0,
#         random_state=None
# ):
#     """
#     Generate the "Friedman #2" regression problem with ranked features.
#
#     The output `y` is created by first scaling the features by the provided
#     weights, and then applying the original formula:
#     y(X) = ( (X'₀)² + (X'₁*X'₂ - 1/(X'₁*X'₃))² )⁰.⁵ + noise
#     where X'ᵢ = wᵢ * Xᵢ
#
#     Args:
#         n_samples (int, optional): The number of samples. Defaults to 100.
#         feature_weights (List[float], optional): A list of 4 weights to control the
#             importance of each feature. Defaults to weights that rank features
#             x1 > x2 > x3 > x4.
#         noise (float, optional): Std of Gaussian noise applied to output. Defaults to 0.0.
#         random_state (int, optional): Random state for reproducibility. Defaults to None.
#
#     Returns:
#         Tuple[np.ndarray, np.ndarray]: X (samples), y (labels).
#     """
#     generator = check_random_state(random_state)
#
#     if feature_weights is None:
#         # Default weights to rank features as: X0 > X1 > X2 > X3
#         feature_weights = [10.0, 5.0, 2.0, 1.0]
#
#     if len(feature_weights) != 4:
#         raise ValueError("feature_weights must be a list or array of length 4.")
#
#     # Generate features in their original ranges
#     X = generator.uniform(size=(n_samples, 4))
#     # X[:, 0] *= 100
#     # X[:, 1] *= 520 * np.pi
#     # X[:, 1] += 40 * np.pi
#     # X[:, 3] *= 10
#     # X[:, 3] += 1
#
#     # Create a weighted version of the features before they enter the formula
#     X_w = X * feature_weights
#
#     # Use the original formula structure with the weighted features
#     y = (
#                 X_w[:, 0] ** 2
#                 + (X_w[:, 1] * X_w[:, 2] - 1 / (X_w[:, 1] * X_w[:, 3])) ** 2
#         ) ** 0.5 #+ noise * generator.standard_normal(size=(n_samples))
#
#     return X, y

def make_friedman2_ranked(
        n_samples=100,
        *,
        feature_weights: List[float] = None,
        noise=0.0,
        random_state=None
):
    """
    Generate the "Friedman #2" regression problem with ranked features.

    The output `y` is created by first scaling the raw features to a common
    [0, 1] range, then applying weights, and finally using them in the original
    formula structure. This ensures weights are the primary driver of importance.

    Args:
        n_samples (int, optional): The number of samples. Defaults to 100.
        feature_weights (List[float], optional): A list of 4 weights to control the
            importance of each feature. Defaults to weights that rank features
            x1 > x2 > x3 > x4.
        noise (float, optional): Std of Gaussian noise applied to output. Defaults to 0.0.
        random_state (int, optional): Random state for reproducibility. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (samples), y (labels). The returned X
        contains the original, unscaled features.
    """
    generator = check_random_state(random_state)

    if feature_weights is None:
        # Weights now directly control importance due to pre-scaling
        feature_weights = [40.0, 15.0, 7.0, 2.0]

    if len(feature_weights) != 4:
        raise ValueError("feature_weights must be a list or array of length 4.")

    # 1. Generate features in their original, disparate ranges
    X = generator.uniform(size=(n_samples, 4))
    # X[:, 0] *= 100
    # X[:, 1] *= 520 * np.pi
    # X[:, 1] += 40 * np.pi
    # X[:, 3] *= 10
    # X[:, 3] += 1

    # 2. Pre-scale features to a common [0, 1] range before applying weights
    # X_scaled = minmax_scale(X, axis=0)

    # 3. Apply weights to the scaled features
    X_w = X * feature_weights

    # 4. Use the original formula structure with the weighted (and pre-scaled) features
    # A small epsilon is added to the denominator to prevent division by zero
    epsilon = 1e-6
    y = (
     X_w[:, 0] ** 2  + (X_w[:, 1] * X_w[:, 2] - 0.001 / (X_w[:, 3] + epsilon)) ** 2 ) ** 0.5 + noise * generator.standard_normal(size=(n_samples))

    # 5. Return the ORIGINAL, unscaled features as is standard for these benchmarks
    return X, y


def make_classification_ranked(
        n_samples=100,
        n_features=20,
        *,
        n_informative=2,
        n_clusters=None,
        feature_weights: List[float] = None,
        n_classes=2,
        class_sep=1.0,
        shuffle=True,
        random_state=None,
):

    """
    Generate a ranked classification dataset using a weighted hypercube method.

    This function modifies the logic of sklearn's make_classification to allow for
    direct control over the importance of informative features via `feature_weights`.
    It works by generating clusters at the vertices of a hypercube, then stretching
    the hypercube along each dimension according to the provided weights. A larger
    weight makes a feature more important for class separation.

    Args:
        n_samples (int): The number of samples.
        n_features (int): The total number of features.
        n_informative (int): The number of informative features.
        n_clusters (int, optional): The number of clusters per class. If None, it defaults
            to 2**n_informative, using all vertices of the hypercube.
        feature_weights (List[float]): A list of weights for the informative
            features, determining their rank and contribution. Must match
            n_informative. Defaults to descending weights.
        n_classes (int): The number of classes.
        class_sep (float): The factor multiplying the hypercube size, controlling
                           the separation between clusters.
        shuffle (bool): Whether to shuffle samples and features.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (samples), y (labels).
    """
    generator = check_random_state(random_state)

    if feature_weights is None:
        # Default to descending weights, e.g., 2^(n-1), 2^(n-2), ..., 1
        feature_weights = [2 ** i for i in range(n_informative)][::-1]

    if len(feature_weights) != n_informative:
        raise ValueError("Length of feature_weights must match n_informative.")

    if n_informative > n_features:
        raise ValueError("n_informative must be <= n_features.")

    max_possible_clusters = 2 ** n_informative
    if n_clusters is None:
        n_clusters = max_possible_clusters

    if n_clusters > max_possible_clusters:
        raise ValueError(
            f"n_clusters={n_clusters} cannot be greater than 2**n_informative={max_possible_clusters}."
        )
    if n_clusters < n_classes:
        raise ValueError(
            f"n_clusters={n_clusters} must be at least n_classes={n_classes}."
        )

    # --- Generate centroids based on a weighted hypercube ---
    # 1. Start with all possible vertices of a standard hypercube
    all_possible_centroids = np.array(list(product([-class_sep, class_sep], repeat=n_informative)))

    # 2. If n_clusters is less than the max possible, randomly select a subset
    if n_clusters < max_possible_clusters:
        choice_indices = generator.choice(max_possible_clusters, n_clusters, replace=False)
        centroids = all_possible_centroids[choice_indices]
    else:
        centroids = all_possible_centroids

    # 3. Stretch the hypercube by multiplying each dimension by its weight
    weights = np.array(feature_weights)
    # Normalize weights to have a mean of 1 to keep overall separation consistent
    weights = weights / np.mean(weights)
    centroids *= weights

    # --- Assign clusters to classes ---
    # y = np.zeros(n_samples, dtype=int)
    n_clusters_per_class = [n_clusters // n_classes] * n_classes
    for i in range(n_clusters % n_classes):
        n_clusters_per_class[i] += 1

    # Assign samples to each cluster, ensuring balance
    n_samples_per_cluster = [n_samples // n_clusters] * n_clusters
    for i in range(n_samples % n_clusters):
        n_samples_per_cluster[i] += 1

    cluster_labels = []
    for i, n_k in enumerate(n_clusters_per_class):
        cluster_labels.extend([i] * n_k)

    # --- Generate samples around the centroids ---
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    C_start = 0
    y_start = 0
    for k, n_k in enumerate(n_samples_per_cluster):
        if n_k == 0: continue

        C_stop = C_start + n_k
        # Generate points for the k-th cluster
        X[C_start:C_stop, :n_informative] = generator.multivariate_normal(
            centroids[k], np.identity(n_informative), n_k
        )
        # Assign labels for the k-th cluster
        y[C_start:C_stop] = cluster_labels[k]
        C_start = C_stop

    # --- Add noise features ---
    if n_features > n_informative:
        X[:, n_informative:] = generator.standard_normal(
            (n_samples, n_features - n_informative)
        )

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

        # Also shuffle the feature columns to mix informative and noise features
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]

    return X, y


def make_blobs_ranked(
        n_samples=100,
        n_features=2,
        *,
        cluster_std=1.0,
        center_sep=1.0,
        feature_weights: List[float] = None,
        shuffle=True,
        random_state=None,
        return_centers=False,
):
    """
    Generate isotropic Gaussian blobs around deterministically separated centers
    to create a robustly ranked classification problem.

    This robust version is specialized for binary classification (2 centers). It
    ensures feature importance by directly scaling the center separation by the
    feature_weights, while keeping the data blobs themselves circular (isotropic).

    Args:
        n_samples (int): The total number of points, split between the two blobs.
        n_features (int): The number of features for each sample.
        cluster_std (float): The standard deviation of the circular clusters. A smaller
            value makes the classification task easier.
        center_sep (float): A factor controlling the overall separation of the
            two cluster centers. A larger value makes the task easier.
        feature_weights (List[float], optional): A list of weights to control the
            importance of each feature. Must match n_features.
        shuffle (bool): Whether to shuffle the samples.
        random_state (int): Random state for reproducibility.
        return_centers (bool): If True, return the cluster centers.

    Returns:
        Tuple: X, y, and optionally centers.
    """
    generator = check_random_state(random_state)
    n_centers = 2  # This version is specialized for binary classification

    if feature_weights is None:
        feature_weights = [2 ** i for i in range(n_features)][::-1]

    if len(feature_weights) != n_features:
        raise ValueError("Length of feature_weights must match n_features.")

    weights = np.array(feature_weights)

    # --- Generate robustly separated centers ---
    # 1. Start with two base centers at opposite ends of a hypercube diagonal.
    base_centers = np.array([[-center_sep] * n_features, [center_sep] * n_features])

    # 2. Scale the center locations by the feature weights.
    # This is the key step: separation along each axis is now directly
    # proportional to the feature's weight.
    final_centers = base_centers * weights

    # --- Balance samples per center ---
    n_samples_per_center = [n_samples // n_centers] * n_centers
    for i in range(n_samples % n_centers):
        n_samples_per_center[i] += 1

    X = np.zeros((sum(n_samples_per_center), n_features))
    y = np.zeros(sum(n_samples_per_center), dtype=int)

    # --- Generate ISOTROPIC blobs around the WEIGHTED centers ---
    # The separation is weighted, but the blobs themselves are circular.
    # This makes the contribution cleaner and less complex.
    for i, n in enumerate(n_samples_per_center):
        start_idx = sum(n_samples_per_center[:i])
        end_idx = start_idx + n

        # Simple isotropic (circular) covariance matrix
        cov_matrix = np.identity(n_features) * (cluster_std ** 2)

        X[start_idx:end_idx] = generator.multivariate_normal(
            mean=final_centers[i], cov=cov_matrix, size=n
        )
        y[start_idx:end_idx] = i

    if shuffle:
        p = generator.permutation(len(X))
        X, y = X[p], y[p]

    if return_centers:
        return X, y, final_centers
    else:
        return X, y

# --- Dataset Generation Functions ---

def generate_ds0(size: int = 10000) -> SyntheticDataset:
    """
    DS0: Unrelated variables. All features are noise.
    """
    # In this case, all features are noise features.
    noise_features_df = _generate_uncorrelated_features(size, ALL_FEATURE_NAMES)

    # Target 'y' is generated from an independent random source to ensure no correlation.
    y_series = pd.Series(np.random.randint(0, 2, size), name='y')

    final_df = pd.concat([noise_features_df, y_series], axis=1)

    meta_data = pd.DataFrame({"imp_vars": [[""]] * size, "RGS": ["RGS0"] * size})
    return SyntheticDataset(
        data=final_df, meta_data=meta_data, feature_names=ALL_FEATURE_NAMES,
        name="DS0_Unrelated",
        description="Target is completely random and unrelated to any features."
    )


def generate_ds1(size: int = 10000) -> SyntheticDataset:
    """
    DS1: Target 'y' is correlated only to 'x1'.
    """
    important_feature_names = ['x1']
    noise_feature_names = [f for f in ALL_FEATURE_NAMES if f not in important_feature_names]

    # Generate important and noise features separately
    important_features_df = _generate_uncorrelated_features(size, important_feature_names)
    noise_features_df = _generate_uncorrelated_features(size, noise_feature_names)

    # Create target 'y' ONLY from the important feature(s)
    x1_modified = important_features_df['x1'] * 20
    median_point = x1_modified.median()
    y_series = pd.Series(x1_modified.apply(lambda x: 1 if x >= median_point else 0), name='y')

    # Combine into final DataFrame
    final_df = _combine_and_order_features(important_features_df, noise_features_df, y_series)

    meta_data = pd.DataFrame({"imp_vars": [important_feature_names] * size, "RGS": ["RGS1"] * size})
    return SyntheticDataset(
        data=final_df, meta_data=meta_data, feature_names=ALL_FEATURE_NAMES,
        name="DS1_Correlated_x1",
        description="Target 'y' is derived from x1. All other features are noise."
    )


# def generate_ds2(size: int = 10000) -> SyntheticDataset:
#     """
#     DS2: Target 'y' from sum of 'x2' and 'x3'.
#     """
#     important_feature_names = ['x2', 'x3']
#     noise_feature_names = [f for f in ALL_FEATURE_NAMES if f not in important_feature_names]
#
#     important_features_df = _generate_uncorrelated_features(size, important_feature_names)
#     noise_features_df = _generate_uncorrelated_features(size, noise_feature_names)
#     logger.info(important_features_df.describe())
#     # Create target 'y' ONLY from the important features
#     df_mod = important_features_df[['x2', 'x3']] * 20
#     logger.info(df_mod.describe())
#
#     df_mod = df_mod.apply( lambda row: (5 * row['x2']) + (1 * row['x3']), axis = 1)
#     df_mod = df_mod.apply( lambda row: (5 * row['x2']) + (1 * row['x3']), axis=1)
#
#
#     mid_point = ( df_mod['x2']  + df_mod['x3']).median()
#     y_series = pd.Series(( df_mod['x2']  +  df_mod['x3']).apply(lambda s: 1 if s >= mid_point else 0), name='y')
#
#     final_df = _combine_and_order_features(important_features_df, noise_features_df, y_series)
#
#     meta_data = pd.DataFrame({"imp_vars": [important_feature_names] * size, "RGS": ["RGS2"] * size})
#     return SyntheticDataset(
#         data=final_df, meta_data=meta_data, feature_names=ALL_FEATURE_NAMES,
#         name="DS2_Correlated_x2_x3_sum",
#         description="Target 'y' is derived from the sum of 5x2 and x3."
#     )
#

def generate_ds2(size: int = 10000) -> SyntheticDataset:
    """
    DS2: Target 'y' from a weighted sum of 'x2' and 'x3'.
    """
    important_feature_names = ['x2', 'x3']
    noise_feature_names = [f for f in ALL_FEATURE_NAMES if f not in important_feature_names]

    important_features_df = _generate_uncorrelated_features(size, important_feature_names)
    noise_features_df = _generate_uncorrelated_features(size, noise_feature_names)

    # Create target 'y' ONLY from the important features
    # First, apply a scaling factor to the base features (which are in [0, 1])
    df_mod = important_features_df[['x2', 'x3']] * 1

    # Calculate the weighted interaction term correctly
    interaction_series = df_mod.apply(lambda row: (5 * row['x2']) + (1 * row['x3']), axis=1)

    # Calculate the median of this new series
    mid_point = interaction_series.median()

    # Create the target variable by applying the threshold to the interaction series
    y_series = pd.Series(interaction_series.apply(lambda s: 1 if s >= mid_point else 0), name='y')

    final_df = _combine_and_order_features(important_features_df, noise_features_df, y_series)

    meta_data = pd.DataFrame({"imp_vars": [important_feature_names] * size, "RGS": ["RGS2"] * size})
    return SyntheticDataset(
        data=final_df, meta_data=meta_data, feature_names=ALL_FEATURE_NAMES,
        name="DS2_Correlated_5x2_plus_x3",
        description="Target 'y' is derived from the weighted sum of 5*x2 and x3."
    )


def generate_ds3(size: int = 10000) -> SyntheticDataset:
    """
    DS3: Target 'y' from a non-linear combination of 'x4', 'x5', 'x6'.
    """
    important_feature_names = ['x4', 'x5', 'x6']
    noise_feature_names = [f for f in ALL_FEATURE_NAMES if f not in important_feature_names]

    important_features_df = _generate_uncorrelated_features(size, important_feature_names)
    noise_features_df = _generate_uncorrelated_features(size, noise_feature_names)

    df_mod = important_features_df[['x4', 'x5', 'x6']] * 20
    interaction_series = df_mod.apply(
        lambda row: (np.power(row['x4'], 2) + (20 * row['x5']) + (10 * row['x6'])), axis=1
    )
    mid_point = interaction_series.median()
    y_series = pd.Series(interaction_series.apply(lambda val: 1 if val >= mid_point else 0), name='y')

    final_df = _combine_and_order_features(important_features_df, noise_features_df, y_series)

    meta_data = pd.DataFrame({"imp_vars": [important_feature_names] * size, "RGS": ["RGS3"] * size})
    return SyntheticDataset(
        data=final_df, meta_data=meta_data, feature_names=ALL_FEATURE_NAMES,
        name="DS3_Correlated_x4_x5_x6_nonlinear",
        description="Target 'y' is from a non-linear combination of x4, x5, x6."
    )



# def generate_ds4(size: int = 10000) -> SyntheticDataset:
#     return _generate_from_sklearn(
#         size,
#         lambda n_samples: make_gaussian_quantiles(n_features=2, n_classes=2, n_samples=n_samples, random_state=42),
#         ['x7', 'x8'],
#         "RGS4",
#         "Based on make_gaussian_quantiles; x7, x8 are informative."
#     )

def generate_ds4(size: int = 10000) -> SyntheticDataset:
    """
    DS14: Uses features from make_gaussian_quantiles but with a custom weighted
    target to ensure a clear feature ranking.
    """
    # Metadata defines the rank: x9 > x10 > x11 > x12 > x13
    important_feature_names = ['x7', 'x8']
    feature_weights = [60, 40]

    # Use the new custom generator that applies weights internally
    return _generate_from_sklearn(
        size,
        lambda n_samples: make_gaussian_quantiles_ranked(
            n_features=2,
            n_classes=2,
            n_samples=n_samples,
            feature_weights=feature_weights,
            random_state=42
        ),
        important_feature_names,
        "RGS04",
        "Based on make_gaussian_quantiles features with a custom ranked/weighted target."
    )


def generate_ds5(size: int = 10000) -> SyntheticDataset:
    """
    DS14: Uses features from make_gaussian_quantiles but with a custom weighted
    target to ensure a clear feature ranking.
    """
    # Metadata defines the rank: x9 > x10 > x11 > x12 > x13
    important_feature_names = ['x9', 'x10', 'x11', 'x12', 'x13']
    feature_weights = [60, 40, 30, 22, 15]

    # Use the new custom generator that applies weights internally
    return _generate_from_sklearn(
        size,
        lambda n_samples: make_gaussian_quantiles_ranked(
            n_features=5,
            n_classes=2,
            n_samples=n_samples,
            feature_weights=feature_weights,
            random_state=42
        ),
        important_feature_names,
        "RGS05",
        "Based on make_gaussian_quantiles features with a custom ranked/weighted target."
    )


# def generate_ds5(size: int = 10000) -> SyntheticDataset:
#     return _generate_from_sklearn(
#         size,
#         lambda n_samples: make_gaussian_quantiles(n_features=5, n_classes=2, n_samples=n_samples, random_state=42),
#         ['x9', 'x10', 'x11', 'x12', 'x13'],
#         "RGS5",
#         "Based on make_gaussian_quantiles; x9-x13 are informative."
#     )


# def generate_ds6(size: int = 10000) -> SyntheticDataset:
#     return _generate_from_sklearn(
#         size,
#         lambda n_samples: make_hastie_10_2(n_samples=n_samples, random_state=42),
#         [f'x{i}' for i in range(14, 24)],
#         "RGS6",
#         "Based on make_hastie_10_2; x14-x23 are informative."
#     )

def generate_ds6(size: int = 10000) -> SyntheticDataset:

    important_feature_names = [f'x{i}' for i in range(14, 24)]
    feature_weights = [250, 220, 190, 160, 130, 110, 80, 50, 35, 25]  # Example weights

    return _generate_from_sklearn(
        size,
        lambda n_samples: make_hastie_10_2_ranked(
            n_samples=n_samples,
            feature_weights=feature_weights,
            random_state=42
        ),
        important_feature_names,
        "RGS06",
        "Based on make_hastie_10_2 features with a custom ranked/weighted target."
    )



# def generate_ds7(size: int = 10000) -> SyntheticDataset:
#     def friedman_binary(n_samples):
#         X, y_reg = make_friedman1(n_features=5, n_samples=n_samples, random_state=42)
#         y_bin = np.where(y_reg > np.median(y_reg), 1, 0)
#         return X, y_bin
#
#     return _generate_from_sklearn(
#         size, friedman_binary, [f'x{i}' for i in range(24, 29)], "RGS7",
#         "Based on make_friedman1 (5 features); target binarized."
#     )

def generate_ds7(size: int = 10000) -> SyntheticDataset:
    """
       DS7: Uses features from a ranked version of make_friedman1, which are then
       used to create a binarized target.
       """
    important_feature_names = [f'x{i}' for i in range(24, 29)]
    # Weights to rank features as: x24 > x25 > x26 > x27 > x28
    feature_weights = [25.0, 20.0, 15.0, 10.0, 6.0]

    def friedman_binary_ranked(n_samples):
        # Call the ranked version of the generator
        X, y_reg = make_friedman1_ranked(
            n_features=5,
            n_samples=n_samples,
            feature_weights=feature_weights,
            random_state=42
        )
        # Binarize the regression output
        y_bin = np.where(y_reg > np.median(y_reg), 1, 0)
        return X, y_bin

    return _generate_from_sklearn(
        size,
        friedman_binary_ranked,
        important_feature_names,
        "RGS7",
        "Based on a ranked make_friedman1 (5 features); target binarized."
    )


# def generate_ds8(size: int = 10000) -> SyntheticDataset:
#     def friedman_binary(n_samples):
#         X, y_reg = make_friedman2(n_samples=n_samples, random_state=42)
#         y_bin = np.where(y_reg > np.median(y_reg), 1, 0)
#         return X, y_bin
#
#     return _generate_from_sklearn(
#         size, friedman_binary, [f'x{i}' for i in range(29, 33)], "RGS8",
#         "Based on make_friedman2 (4 features); target binarized."
#     )

def generate_ds8(size: int = 10000) -> SyntheticDataset:
    """
    DS8: Uses features from a ranked version of make_friedman2, which are then
    used to create a binarized target.
    """
    important_feature_names = [f'x{i}' for i in range(29, 33)]
    # Weights to rank features as: x29 > x30 > x31 > x32
    # feature_weights = [40.0, 1.5, 15.0, 100]

    # def friedman_binary_ranked(n_samples):
    #     # Call the ranked version of the generator
    #     X, y_reg = make_friedman2_ranked(
    #         n_samples=n_samples,
    #         feature_weights=feature_weights,
    #         random_state=42
    #     )
    #     # Binarize the regression output
    #     y_bin = np.where(y_reg > np.median(y_reg), 1, 0)
    #     return X, y_bin

    feature_weights = [500, 50.0, 20.0, 0.0001]


    def friedman_binary_ranked(n_samples):
        # Call the ranked version of the generator
        X, y_reg = make_friedman2_ranked(
            n_samples=n_samples,
            feature_weights=feature_weights,
            random_state=42
        )
        # Binarize the regression output
        y_bin = np.where(y_reg > np.median(y_reg), 1, 0)
        return X, y_bin

    return _generate_from_sklearn(
        size,
        friedman_binary_ranked,
        important_feature_names,
        "RGS8",
        "Based on a ranked make_friedman2 (4 features); target binarized."
    )

def generate_ds9(size: int = 10000) -> SyntheticDataset:
    def friedman_binary(n_samples):
        X, y_reg = make_friedman3(n_samples=n_samples, random_state=42)
        y_bin = np.where(y_reg > np.median(y_reg), 1, 0)
        return X, y_bin

    return _generate_from_sklearn(
        size, friedman_binary, [f'x{i}' for i in range(33, 37)], "RGS9",
        "Based on make_friedman3 (4 features); target binarized."
    )


# def generate_ds10(size: int = 10000) -> SyntheticDataset:
#     return _generate_from_sklearn(
#         size,
#         lambda n_samples: make_classification(n_samples=n_samples, n_features=4, n_informative=4, n_redundant=0,
#                                               n_repeated=0, class_sep=2.0, random_state=100),
#         [f'x{i}' for i in range(37, 41)], "RGS10",
#         "Based on make_classification (4 informative features)."
#     )

def generate_ds10(size: int = 10000) -> SyntheticDataset:

    """
   DS18: Uses a custom, ranked version of make_classification to generate a dataset
   with a clear feature hierarchy.
   """
    important_feature_names = [f'x{i}' for i in range(37, 42)]  # x37, x38, x39, x40, x41
    # Weights to rank features as: x1 > x2 > x3 > x4 > x5
    feature_weights = [ 80, 55, 33, 24, 15]

    return _generate_from_sklearn(
        size,
        lambda n_samples: make_classification_ranked(
            n_samples=n_samples,
            n_features=len(important_feature_names),  # Generate only informative for this wrapper
            n_informative=len(important_feature_names),
            feature_weights=feature_weights,
            n_classes=2,
            shuffle=False,  # Wrapper will handle shuffling if needed
            random_state=42,
            n_clusters=5
        ),
        important_feature_names,
        "RGS10",
        "Based on a custom ranked make_classification."
    )


# def generate_ds11(size: int = 10000) -> SyntheticDataset:
#     return _generate_from_sklearn(
#         size,
#         lambda n_samples: make_blobs(n_samples=n_samples, n_features=6, centers=2, cluster_std=8.0, random_state=100),
#         [f'x{i}' for i in range(41, 47)], "RGS11",
#         "Based on make_blobs (6 informative features)."
#     )

def generate_ds11(size: int = 10000) -> SyntheticDataset:
    """
    DS11: Uses a custom, ranked version of make_blobs to generate a dataset
    with a clear feature hierarchy. This is the robust version.
    """
    # Using 6 features to match the original generate_ds11
    important_feature_names = [f'x{i}' for i in range(41, 47)]
    # Weights to rank features as: x41 > x42 > ... > x46
    feature_weights = [32, 20, 16, 10, 7, 5]
    # feature_weights = [ , 8, 4, 3, 2, 0.1]

    return _generate_from_sklearn(
        size,
        lambda n_samples: make_blobs_ranked(
            n_samples=n_samples,
            n_features=len(important_feature_names),
            # centers=2,  # Fixed to 2 for a robust binary problem
            cluster_std=2.0,  # Controls tightness of clusters
            center_sep=.1,  # Controls overall separation
            feature_weights=feature_weights,
            shuffle=True,
            random_state=42
        ),
        important_feature_names,
        "RGS11",
        "Based on a custom ranked make_blobs (robust version)."
    )


def generate_ds12(size: int = 10000) -> SyntheticDataset:
    return _generate_from_sklearn(
        size,
        lambda n_samples: make_moons(n_samples=n_samples, noise=0.1, random_state=100),
        ['x47', 'x48'], "RGS12",
        "Based on make_moons (2 informative features)."
    )


# --- Registry for easy access ---
_GENERATOR_FUNCTIONS: Dict[str, Callable[[int], SyntheticDataset]] = {
    f"ds{i}": globals()[f"generate_ds{i}"] for i in range(0, 13)
}


def list_available_datasets() -> List[str]:
    return sorted(list(_GENERATOR_FUNCTIONS.keys()))


def generate_dataset_by_name(name: str, size: int = 10000) -> SyntheticDataset:
    generator_func = _GENERATOR_FUNCTIONS.get(name)
    if generator_func is None:
        raise ValueError(f"Dataset generator '{name}' not found. Available: {list_available_datasets()}")
    logger.info(f"Generating dataset '{name}' with size {size}...")
    dataset = generator_func(size=size)
    logger.info(f"Successfully generated dataset '{dataset.name}'.")
    return dataset

