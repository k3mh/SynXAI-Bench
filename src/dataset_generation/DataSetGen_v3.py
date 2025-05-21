# File: synthetic_data_generator.py

import logging
from typing import List, Dict, Callable, NamedTuple

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_gaussian_quantiles, make_hastie_10_2, make_classification,
    make_friedman1, make_friedman2, make_friedman3, make_blobs, make_moons
)
from sklearn.preprocessing import minmax_scale

logger = logging.getLogger(__name__)

# --- Configuration ---
NUM_BACKGROUND_FEATURES = 48
BACKGROUND_FEATURE_NAMES = [f'x{i}' for i in range(1, NUM_BACKGROUND_FEATURES + 1)]


# --- Data Structure for Output ---
class SyntheticDataset(NamedTuple):
    data: pd.DataFrame
    meta_data: pd.DataFrame
    feature_names: List[str]
    name: str
    description: str

    def __repr__(self):
        return (f"SyntheticDataset(name='{self.name}', shape={self.data.shape}, "
                f"features={len(self.feature_names)})")


# --- Helper Functions ---

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _generate_correlated_variables(var: np.ndarray, scale: float = 0.05) -> np.ndarray:
    """Generates variables correlated to 'var' by multiplicative noise."""
    size = var.shape[0]
    if var.ndim > 1 and var.shape[1] == 1:
        var = var.flatten()
    z = np.random.normal(size=size, loc=1, scale=scale)
    return var * z


def _logistic(x_vector: np.ndarray) -> np.ndarray:
    """Applies a logistic function centered at the median of x_vector."""
    x_node = np.median(x_vector)
    L = 1
    K = 1
    return L / (1 + np.exp(-K * (x_vector - x_node)))


def _generate_correlated_feature_set(samples: int, corr: float, num_vars: int) -> np.ndarray:
    """Generates a set of features with a specified pairwise correlation."""
    if num_vars <= 0:
        return np.empty((samples, 0))
    if num_vars == 1:
        return np.random.normal(0, 1, size=(samples, 1))

    mean_lst = [0] * num_vars
    cov_mtx = np.full((num_vars, num_vars), corr)
    np.fill_diagonal(cov_mtx, 1)

    try:
        data = np.random.multivariate_normal(mean_lst, cov_mtx, size=samples)
    except np.linalg.LinAlgError:
        logger.warning(
            f"Covariance matrix for {num_vars} vars with corr {corr} not positive semi-definite. Adding jitter to diagonal.")
        cov_mtx += np.eye(num_vars) * 1e-6  # Add jitter
        data = np.random.multivariate_normal(mean_lst, cov_mtx, size=samples)
    return data


def _create_base_dataframe(size: int, num_features: int = NUM_BACKGROUND_FEATURES, corr: float = 0.0) -> pd.DataFrame:
    """
    Creates a base DataFrame with a specified number of features,
    potentially correlated, and scaled to [0, 1].
    """
    if num_features == 0:
        return pd.DataFrame(index=range(size))

    feature_names_list = [f'x{i}' for i in range(1, num_features + 1)]
    data_array = _generate_correlated_feature_set(samples=size, corr=corr, num_vars=num_features)
    df = pd.DataFrame(data_array, columns=feature_names_list)

    # Scale only if there's data to scale
    if not df.empty:
        df_scaled = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=feature_names_list,
                                 dtype=float)
    else:
        df_scaled = df.astype(float)  # ensure correct dtype for empty df
    return df_scaled


# --- Dataset Generation Functions ---

def generate_RRs0(size: int = 10000) -> SyntheticDataset:
    """
    DS0: A monotonic single variable linear relationship.
    """
    num_total_initial_features = NUM_BACKGROUND_FEATURES + 1
    temp_feature_names = [f'x{i}' for i in range(1, num_total_initial_features + 1)]

    data_array = _generate_correlated_feature_set(samples=size, corr=0, num_vars=num_total_initial_features)
    df_initial = pd.DataFrame(data_array, columns=temp_feature_names)
    df_scaled = pd.DataFrame(minmax_scale(df_initial, axis=0, feature_range=(0, 1)), columns=temp_feature_names,
                             dtype=float)

    y_temp_col = temp_feature_names[-1]
    df_scaled['y'] = df_scaled[y_temp_col].apply(lambda x: 1 if x >= 0.5 else 0)

    final_feature_names = BACKGROUND_FEATURE_NAMES[:]
    df_final = df_scaled[final_feature_names + ['y']].copy()

    meta_data = pd.DataFrame({"imp_vars": [[""]] * size, "RGS": ["RGS0"] * size})
    return SyntheticDataset(
        data=df_final, meta_data=meta_data, feature_names=final_feature_names,
        name="RRs0",
        description="Target is related to a single variable with monotonic linear relationship."
    )


def generate_RRs1(size: int = 10000) -> SyntheticDataset:
    """
    DS1: Target 'y' is correlated to 'x1'.
    'x1' is amplified before a median split to create 'y'.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    df_processed['x1_modified'] = df_processed['x1'] * 20
    median_point = df_processed['x1_modified'].median()
    df_processed['y'] = df_processed['x1_modified'].apply(lambda x: 1 if x >= median_point else 0)

    final_df = base_df.join(df_processed['y'])
    meta_data = pd.DataFrame({"imp_vars": [["x1"]] * size, "RGS": ["RGS1"] * size})
    return SyntheticDataset(
        data=final_df, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS1_Correlated_x1",
        description="Target 'y' is derived from x1."
    )


def generate_RRs2(size: int = 10000) -> SyntheticDataset:
    """
    DS2: Target 'y' from sum of 'x2' and 'x3'.
    True when (x2_modified + x3_modified) >= median. x2, x3 are not correlated.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    df_processed[['x2_mod', 'x3_mod']] = df_processed[['x2', 'x3']] * 20
    mid_point = (df_processed['x2_mod'] + df_processed['x3_mod']).median()
    df_processed['y'] = (df_processed['x2_mod'] + df_processed['x3_mod']).apply(lambda s: 1 if s >= mid_point else 0)

    final_df = base_df.join(df_processed['y'])
    meta_data = pd.DataFrame({"imp_vars": [["x2", "x3"]] * size, "RGS": ["RGS2"] * size})
    return SyntheticDataset(
        data=final_df, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS2_Correlated_x2_x3_sum",
        description="Target 'y' is derived from the sum of modified x2 and x3."
    )


def generate_RRs3(size: int = 10000) -> SyntheticDataset:
    """
    DS3: Target 'y' from a non-linear combination of 'x4', 'x5', 'x6'.
    y=1 if (x4_mod^2 + 2*x5_mod + x6_mod) >= median.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    df_processed[['x4_mod', 'x5_mod', 'x6_mod']] = df_processed[['x4', 'x5', 'x6']] * 20

    interaction_series = df_processed.apply(
        lambda row: (np.power(row['x4_mod'], 2) + (2 * row['x5_mod']) + row['x6_mod']), axis=1
    )
    mid_point = interaction_series.median()
    df_processed['y'] = interaction_series.apply(lambda val: 1 if val >= mid_point else 0)

    final_df = base_df.join(df_processed['y'])
    meta_data = pd.DataFrame({"imp_vars": [["x4", "x5", "x6"]] * size, "RGS": ["RGS3"] * size})
    return SyntheticDataset(
        data=final_df, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS3_Correlated_x4_x5_x6_nonlinear",
        description="Target 'y' is from a non-linear combination of modified x4, x5, x6."
    )


def generate_RRs4(size: int = 10000) -> SyntheticDataset:
    """
    DS4: Based on sklearn.datasets.make_gaussian_quantiles (2 features).
    Features 'x7', 'x8' are replaced by the informative features.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    informative_features, y_target = make_gaussian_quantiles(n_features=2, n_classes=2, n_samples=size, random_state=42)

    df_processed['x7'] = informative_features[:, 0]
    df_processed['x8'] = informative_features[:, 1]
    df_processed['y'] = y_target

    # Rescale all features again *after* inserting new ones, to maintain [0,1] range for all
    feature_cols = BACKGROUND_FEATURE_NAMES[:]
    df_processed[feature_cols] = minmax_scale(df_processed[feature_cols], axis=0, feature_range=(0, 1))

    meta_data = pd.DataFrame({"imp_vars": [["x7", "x8"]] * size, "RGS": ["RGS4"] * size})
    return SyntheticDataset(
        data=df_processed, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS4_GaussianQuantiles_2Feat",
        description="Based on make_gaussian_quantiles; x7, x8 are informative."
    )


def generate_RRs5(size: int = 10000) -> SyntheticDataset:
    """
    DS5: Based on sklearn.datasets.make_gaussian_quantiles (5 features).
    Features 'x9'-'x13' are replaced by the informative features.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    informative_features, y_target = make_gaussian_quantiles(n_features=5, n_classes=2, n_samples=size, random_state=42)

    df_processed['x9'] = informative_features[:, 0]
    df_processed['x10'] = informative_features[:, 1]
    df_processed['x11'] = informative_features[:, 2]
    df_processed['x12'] = informative_features[:, 3]
    df_processed['x13'] = informative_features[:, 4]
    df_processed['y'] = y_target

    feature_cols = BACKGROUND_FEATURE_NAMES[:]
    df_processed[feature_cols] = minmax_scale(df_processed[feature_cols], axis=0, feature_range=(0, 1))

    meta_data = pd.DataFrame({"imp_vars": [["x9", "x10", "x11", "x12", "x13"]] * size, "RGS": ["RGS5"] * size})
    return SyntheticDataset(
        data=df_processed, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS5_GaussianQuantiles_5Feat",
        description="Based on make_gaussian_quantiles; x9-x13 are informative."
    )


def generate_RRs6(size: int = 10000) -> SyntheticDataset:
    """
    DS6: Based on sklearn.datasets.make_hastie_10_2 (10 features).
    Features 'x14'-'x23' are replaced by the informative features. Target y is mapped to 0/1.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    informative_features, y_target_hastie = make_hastie_10_2(n_samples=size, random_state=42)
    y_target = np.where(y_target_hastie == 1, 1, 0)  # Map from {-1, 1} to {0, 1}

    imp_vars_names = [f'x{i}' for i in range(14, 14 + 10)]
    for i, col_name in enumerate(imp_vars_names):
        df_processed[col_name] = informative_features[:, i]
    df_processed['y'] = y_target

    feature_cols = BACKGROUND_FEATURE_NAMES[:]
    df_processed[feature_cols] = minmax_scale(df_processed[feature_cols], axis=0, feature_range=(0, 1))

    meta_data = pd.DataFrame(
        {"imp_vars": [imp_vars_names] * size, "RGS": ["RGS6_Hastie"] * size})  # Corrected RGS name based on old code.
    return SyntheticDataset(
        data=df_processed, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS6_Hastie_10Feat",
        description="Based on make_hastie_10_2; x14-x23 are informative."
    )


def generate_RRs7(size: int = 10000) -> SyntheticDataset:
    """
    DS7: Based on sklearn.datasets.make_friedman1 (5 informative features).
    Features 'x24'-'x28' are replaced. Target y is binarized from regression target.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    # make_friedman1 by default uses 5 informative features out of 10. We'll use its first 5.
    informative_features, y_reg = make_friedman1(n_features=5, n_samples=size, random_state=42)

    median_point = np.median(y_reg)
    y_target = np.where(y_reg > median_point, 1, 0)

    imp_vars_names = [f'x{i}' for i in range(24, 24 + 5)]
    for i, col_name in enumerate(imp_vars_names):
        df_processed[col_name] = informative_features[:, i]
    df_processed['y'] = y_target

    feature_cols = BACKGROUND_FEATURE_NAMES[:]
    df_processed[feature_cols] = minmax_scale(df_processed[feature_cols], axis=0, feature_range=(0, 1))

    meta_data = pd.DataFrame(
        {"imp_vars": [imp_vars_names] * size, "RGS": ["RGS7_Friedman1"] * size})  # Corrected RGS in line with others
    return SyntheticDataset(
        data=df_processed, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS7_Friedman1_5Feat",
        description="Based on make_friedman1 (5 true features x24-x28); target binarized."
    )


def generate_RRs8(size: int = 10000) -> SyntheticDataset:
    """
    DS8: Based on sklearn.datasets.make_friedman2 (4 informative features).
    Features 'x29'-'x32' are replaced. Target y is binarized.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    informative_features, y_reg = make_friedman2(n_samples=size, random_state=42)

    median_point = np.median(y_reg)
    y_target = np.where(y_reg > median_point, 1, 0)

    imp_vars_names = [f'x{i}' for i in range(29, 29 + 4)]
    for i, col_name in enumerate(imp_vars_names):
        df_processed[col_name] = informative_features[:, i]
    df_processed['y'] = y_target

    feature_cols = BACKGROUND_FEATURE_NAMES[:]
    df_processed[feature_cols] = minmax_scale(df_processed[feature_cols], axis=0, feature_range=(0, 1))

    meta_data = pd.DataFrame({"imp_vars": [imp_vars_names] * size, "RGS": ["RGS8_Friedman2"] * size})
    return SyntheticDataset(
        data=df_processed, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS8_Friedman2_4Feat",
        description="Based on make_friedman2 (4 true features x29-x32); target binarized."
    )


def generate_RRs9(size: int = 10000) -> SyntheticDataset:
    """
    DS9: Based on sklearn.datasets.make_friedman3 (4 informative features).
    Features 'x33'-'x36' are replaced. Target y is binarized.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    informative_features, y_reg = make_friedman3(n_samples=size, random_state=42)

    median_point = np.median(y_reg)
    y_target = np.where(y_reg > median_point, 1, 0)

    imp_vars_names = [f'x{i}' for i in range(33, 33 + 4)]
    for i, col_name in enumerate(imp_vars_names):
        df_processed[col_name] = informative_features[:, i]
    df_processed['y'] = y_target

    feature_cols = BACKGROUND_FEATURE_NAMES[:]
    df_processed[feature_cols] = minmax_scale(df_processed[feature_cols], axis=0, feature_range=(0, 1))

    meta_data = pd.DataFrame(
        {"imp_vars": [imp_vars_names] * size, "RGS": ["RGS9_Friedman3"] * size})  # RGS was RGS8 in original
    return SyntheticDataset(
        data=df_processed, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS9_Friedman3_4Feat",
        description="Based on make_friedman3 (4 true features x33-x36); target binarized."
    )


def generate_RRs10(size: int = 10000) -> SyntheticDataset:
    """
    DS10: Based on sklearn.datasets.make_classification (4 informative features).
    Features 'x37'-'x40' are replaced.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    informative_features, y_target = make_classification(
        n_samples=size, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
        hypercube=False, class_sep=5.0,  # Increased class_sep from 2.0 to 5.0 like in original
        random_state=100  # Matched original random_state
    )

    imp_vars_names = [f'x{i}' for i in range(37, 37 + 4)]
    for i, col_name in enumerate(imp_vars_names):
        df_processed[col_name] = informative_features[:, i]
    df_processed['y'] = y_target

    feature_cols = BACKGROUND_FEATURE_NAMES[:]
    df_processed[feature_cols] = minmax_scale(df_processed[feature_cols], axis=0, feature_range=(0, 1))

    meta_data = pd.DataFrame(
        {"imp_vars": [imp_vars_names] * size, "RGS": ["RGS10_Classification"] * size})  # RGS was RGS5
    return SyntheticDataset(
        data=df_processed, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS10_Classification_4Feat",
        description="Based on make_classification (4 informative features x37-x40)."
    )


def generate_RRs11(size: int = 10000) -> SyntheticDataset:
    """
    DS11: Based on sklearn.datasets.make_blobs (6 features).
    Features 'x41'-'x46' are replaced.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    informative_features, y_target = make_blobs(
        n_samples=size, n_features=6, centers=2, cluster_std=8.0,  # Matched original cluster_std
        random_state=100  # Matched original random_state
    )

    imp_vars_names = [f'x{i}' for i in range(41, 41 + 6)]
    # Correcting imp_vars list from original '45', '46' string literals
    corrected_imp_vars_list = [f'x{i}' for i in range(41, 47)]

    for i, col_name in enumerate(imp_vars_names):
        df_processed[col_name] = informative_features[:, i]
    df_processed['y'] = y_target

    feature_cols = BACKGROUND_FEATURE_NAMES[:]
    df_processed[feature_cols] = minmax_scale(df_processed[feature_cols], axis=0, feature_range=(0, 1))

    meta_data = pd.DataFrame(
        {"imp_vars": [corrected_imp_vars_list] * size, "RGS": ["RGS11_Blobs"] * size})  # RGS was RGS5
    return SyntheticDataset(
        data=df_processed, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS11_Blobs_6Feat",
        description="Based on make_blobs (6 informative features x41-x46)."
    )


def generate_RRs12(size: int = 10000) -> SyntheticDataset:
    """
    DS12: Based on sklearn.datasets.make_moons (2 features).
    Features 'x47'-'x48' are replaced.
    """
    base_df = _create_base_dataframe(size=size, num_features=NUM_BACKGROUND_FEATURES, corr=0.0)
    df_processed = base_df.copy()

    informative_features, y_target = make_moons(n_samples=size, noise=0.1,
                                                random_state=100)  # Added some noise like typical usage

    imp_vars_names = [f'x{i}' for i in range(47, 47 + 2)]
    for i, col_name in enumerate(imp_vars_names):
        df_processed[col_name] = informative_features[:, i]
    df_processed['y'] = y_target

    feature_cols = BACKGROUND_FEATURE_NAMES[:]
    df_processed[feature_cols] = minmax_scale(df_processed[feature_cols], axis=0, feature_range=(0, 1))

    meta_data = pd.DataFrame({"imp_vars": [imp_vars_names] * size, "RGS": ["RGS12_Moons"] * size})  # RGS was RGS5
    return SyntheticDataset(
        data=df_processed, meta_data=meta_data, feature_names=BACKGROUND_FEATURE_NAMES[:],
        name="DS12_Moons_2Feat",
        description="Based on make_moons (2 informative features x47-x48)."
    )


# --- Registry for easy access ---
_GENERATOR_FUNCTIONS: Dict[str, Callable[[int], SyntheticDataset]] = {
    "ds0": generate_RRs0,
    "ds1": generate_RRs1,
    "ds2": generate_RRs2,
    "ds3": generate_RRs3,
    "ds4": generate_RRs4,
    "ds5": generate_RRs5,
    "ds6": generate_RRs6,
    "ds7": generate_RRs7,
    "ds8": generate_RRs8,
    "ds9": generate_RRs9,
    "ds10": generate_RRs10,
    "ds11": generate_RRs11,
    "ds12": generate_RRs12,
}


def list_available_datasets() -> List[str]:
    """Returns a list of keys for available synthetic dataset generators."""
    return sorted(list(_GENERATOR_FUNCTIONS.keys()))


def generate_dataset_by_name(name: str, size: int = 10000) -> SyntheticDataset:
    """
    Generates a specific synthetic dataset by its key/name.

    Args:
        name: The key of the dataset generator (e.g., "ds0", "ds1").
        size: The number of samples to generate.

    Returns:
        A SyntheticDataset object.

    Raises:
        ValueError: if the dataset name is not found.
    """
    generator_func = _GENERATOR_FUNCTIONS.get(name)
    if generator_func is None:
        raise ValueError(f"Dataset generator '{name}' not found. Available: {list_available_datasets()}")
    logger.info(f"Generating dataset '{name}' with size {size}...")
    dataset = generator_func(size=size)
    logger.info(f"Successfully generated dataset '{dataset.name}'.")
    return dataset


# if __name__ == '__main__':
#     # Example usage:
#     print("Available datasets:", list_available_datasets())
#
#     for ds_name in list_available_datasets()[:3]:  # Generate first 3 for demo
#         print(f"\n--- Generating {ds_name} ---")
#         try:
#             dataset = generate_dataset_by_name(name=ds_name, size=100)  # Smaller size for quick demo
#             print(dataset)
#             print("Data head:\n", dataset.data.head())
#             print("Meta data head:\n", dataset.meta_data.head())
#             print("Feature names:", dataset.feature_names)
#             if not dataset.data.empty:
#                 print("Target 'y' value counts:\n", dataset.data['y'].value_counts(normalize=True))
#         except Exception as e:
#             print(f"Error generating {ds_name}: {e}")
#             logger.exception(f"Failed to generate {ds_name}")
#
#     # Example of generating a specific dataset
#     # ds1_data = generate_RRs1(size=500)
#     # print("\nDS1 Data sample:")
#     # print(ds1_data.data.head())
