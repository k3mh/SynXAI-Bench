import logging

logger = logging.getLogger(__name__)
logger.debug("Loading Explaination_v2.py...")

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Callable, Optional

# For LIME
import lime
import lime.lime_tabular

# For Anchor
from anchor import anchor_tabular  # Assuming you have the 'anchor-exp' package

# For SHAP (will be imported if/when ShapExplainer is used)
# import shap

# For parallel processing
from joblib import Parallel, delayed
import multiprocessing




# --- Base Explainer Abstract Class ---
class BaseExplainer(ABC):
    """
    Abstract base class for XAI explainers.
    It standardizes the interface for fitting explainers and generating explanations.
    """

    def __init__(self, model: Any, explainer_name: str):
        """
        Initialize the base explainer.
        Args:
            model: The trained machine learning model to explain.
            explainer_name: Name of the explainer library (e.g., "lime", "anchor", "shap").
        """
        self.model = model
        self.explainer_name = explainer_name
        self.explainer_object: Optional[Any] = None  # The actual explainer instance from the library
        self.feature_names: Optional[List[str]] = None
        self.class_names: Optional[List[str]] = None

    @abstractmethod
    def fit(self,
            training_data: pd.DataFrame,
            feature_names: List[str],
            class_names: Optional[List[str]] = None,
            categorical_features: Optional[List[int]] = None,
            categorical_names: Optional[Dict[int, List[str]]] = None,  # For Anchor
            **kwargs) -> None:
        """
        Prepare or "fit" the explainer using training data or its characteristics.
        Not all XAI libraries use 'fit' in the scikit-learn sense, but this method
        serves to initialize the explainer with necessary data context.

        Args:
            training_data (pd.DataFrame): The data used to initialize/train the explainer.
            feature_names (List[str]): List of feature names.
            class_names (Optional[List[str]]): Names of the classes for classification problems.
            categorical_features (Optional[List[int]]): Indices of categorical features (for LIME).
            categorical_names (Optional[Dict[int, List[str]]]): Mapping for categorical features (for Anchor).
            **kwargs: Additional explainer-specific configuration.
        """
        self.feature_names = feature_names
        self.class_names = class_names
        logger.info(f"Fitting {self.explainer_name} explainer with {len(feature_names)} features.")

    @abstractmethod
    def explain_instance(self,
                         data_instance: pd.Series,
                         **kwargs) -> Dict[str, Any]:
        """
        Explain a single data instance.

        Args:
            data_instance (pd.Series): The instance to explain. Column names must match
                                       feature_names used in fit.
            **kwargs: Explainer-specific parameters for generating the explanation.

        Returns:
            Dict[str, Any]: A dictionary typically containing:
                           {'features': List[str],
                            'importance': List[float],
                            'raw_explanation': Any} # Original explanation object
                           For Anchor, 'features' might be rule conditions and 'importance' might be metrics.
        """
        if self.explainer_object is None:
            raise RuntimeError(f"{self.explainer_name} explainer not fitted. Call fit() first.")
        if self.feature_names is None:
            raise RuntimeError("Feature names not set. Call fit() first.")
        pass

    def explain_dataset(self,
                        dataset: pd.DataFrame,
                        parallel: bool = True,
                        n_jobs: int = -2,
                        **kwargs) -> pd.DataFrame:
        """
        Explains all instances in a dataset, potentially in parallel.

        Args:
            dataset (pd.DataFrame): The dataset (rows are instances) to explain.
                                    Columns must match feature_names used in fit.
            parallel (bool): Whether to run explanations in parallel.
            n_jobs (int): Number of jobs for parallel processing (if parallel=True).
                          -1 means using all processors, -2 means all but one.
            **kwargs: Additional arguments passed to each `explain_instance` call.

        Returns:
            pd.DataFrame: A DataFrame with columns:
                          ["explainer_lib", "instance_idx", "features", "importance", "raw_output" (optional)]
                          "instance_idx" refers to the index of the instance in the input `dataset`.
        """
        logger.info(f"Starting explanation for {len(dataset)} instances using {self.explainer_name}...")
        all_explanations_data = []

        # Ensure kwargs for explain_instance are correctly propagated
        explain_instance_kwargs = kwargs

        if parallel:
            logger.info("Using parallel processing.")
            if n_jobs == -1:
                effective_n_jobs = multiprocessing.cpu_count()
            elif n_jobs <= 0:  # e.g. -2
                effective_n_jobs = multiprocessing.cpu_count() + n_jobs + 1
                if effective_n_jobs < 1: effective_n_jobs = 1
            else:
                effective_n_jobs = n_jobs

            logger.info(f"Using parallel processing with n_jobs={effective_n_jobs}")

            def _explain_and_collect_safe(idx_instance_tuple):
                idx, instance_data = idx_instance_tuple
                try:
                    exp_data = self.explain_instance(instance_data, **explain_instance_kwargs)
                    return {
                        "explainer_lib": self.explainer_name,
                        "instance_idx": idx,
                        "features": exp_data.get("features"),
                        "importance": exp_data.get("importance"),
                        "raw_output": exp_data.get("raw_explanation")
                    }
                except Exception as e:
                    logger.error(f"Error explaining instance {idx} with {self.explainer_name}: {e}",
                                 exc_info=True)  # Set exc_info=True for full traceback
                    return {
                        "explainer_lib": self.explainer_name,
                        "instance_idx": idx,
                        "features": [f"Error: {e}"],
                        "importance": [0.0],
                        "raw_output": None
                    }

            # Prepare list of (index, Series) tuples
            tasks = [(idx, dataset.loc[idx]) for idx in dataset.index]

            results = Parallel(n_jobs=effective_n_jobs)(
                delayed(_explain_and_collect_safe)(task) for task in tasks
            )
            all_explanations_data.extend(results)

        else:
            logger.info("Using sequential processing.")
            for idx in dataset.index:
                instance_data = dataset.loc[idx]
                logger.debug(f"Explaining instance {idx}...")
                try:
                    exp_data = self.explain_instance(instance_data, **explain_instance_kwargs)
                    all_explanations_data.append({
                        "explainer_lib": self.explainer_name,
                        "instance_idx": idx,
                        "features": exp_data.get("features"),
                        "importance": exp_data.get("importance"),
                        "raw_output": exp_data.get("raw_explanation")
                    })
                except Exception as e:
                    logger.error(f"Error explaining instance {idx} with {self.explainer_name}: {e}", exc_info=False)
                    all_explanations_data.append({
                        "explainer_lib": self.explainer_name,
                        "instance_idx": idx,
                        "features": [f"Error: {e}"],
                        "importance": [0.0],
                        "raw_output": None
                    })

        logger.info(f"Finished explanation for {len(dataset)} instances.")
        return pd.DataFrame(all_explanations_data)


# --- LIME Explainer Implementation ---
class LimeTabularExplainerWrapper(BaseExplainer):
    def __init__(self, model: Any, predict_proba_fn: Optional[Callable] = None):
        super().__init__(model, "lime")
        if predict_proba_fn:
            self.predict_fn = predict_proba_fn
        elif hasattr(model, 'predict_proba'):
            self.predict_fn = model.predict_proba
        else:
            raise ValueError("LIME requires a model with 'predict_proba' method or a 'predict_proba_fn'.")

    def fit(self,
            training_data: pd.DataFrame,
            feature_names: List[str],
            class_names: Optional[List[str]] = None,
            categorical_features: Optional[List[int]] = None,  # LIME uses indices
            mode: str = "classification",
            **kwargs) -> None:  # `kwargs` for LimeTabularExplainer constructor
        super().fit(training_data, feature_names, class_names, categorical_features)

        training_data_np = training_data[self.feature_names].values if isinstance(training_data,
                                                                                  pd.DataFrame) else training_data

        self.explainer_object = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data_np,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=categorical_features,
            mode=mode,
            **kwargs
        )
        logger.info("LIME TabularExplainer initialized.")

    def explain_instance(self,
                         data_instance: pd.Series,
                         num_features: Optional[int] = None,
                         top_labels: int = 1,
                         **kwargs) -> Dict[str, Any]:  # `kwargs` for LIME's explain_instance
        super().explain_instance(data_instance)  # Basic checks

        instance_np = data_instance[self.feature_names].values.astype(float)

        # If num_features is not provided, use all features
        effective_num_features = num_features if num_features is not None else len(self.feature_names)

        lime_exp = self.explainer_object.explain_instance(
            data_row=instance_np,
            predict_fn=self.predict_fn,
            num_features=effective_num_features,
            top_labels=top_labels,
            **kwargs
        )

        # Process the explanation for the first label (or the primary label)
        # LIME's as_map() returns a dict {label_index: [(feat_idx, weight), ...]}.
        explanation_map = lime_exp.as_map()
        if not explanation_map:
            return {"features": [], "importance": [], "raw_explanation": lime_exp}

        first_label_key = list(explanation_map.keys())[0]
        lime_explanation_list = explanation_map[first_label_key]

        explained_features = [self.feature_names[item[0]] for item in lime_explanation_list]
        importances = [item[1] for item in lime_explanation_list]

        return {"features": explained_features, "importance": importances, "raw_explanation": lime_exp}


# --- Anchor Explainer Implementation ---
class AnchorTabularExplainerWrapper(BaseExplainer):
    def __init__(self, model: Any, predict_fn: Optional[Callable] = None):
        super().__init__(model, "anchor")
        if predict_fn:
            self.predict_fn = predict_fn  # Should return class labels
        elif hasattr(model, 'predict'):
            self.predict_fn = model.predict
        else:
            raise ValueError("Anchor requires a model with 'predict' method or a 'predict_fn'.")

    def fit(self,
            training_data: pd.DataFrame,
            feature_names: List[str],
            class_names: Optional[List[str]] = None,
            categorical_names: Optional[Dict[int, List[str]]] = None,  # Anchor uses dict {feat_idx: [val_names]}
            # ordinal_features: Optional[List[str]] = None,  # Names of ordinal features
            **kwargs) -> None:
        super().fit(training_data, feature_names, class_names, categorical_names=categorical_names)

        training_data_np = training_data[self.feature_names].values if isinstance(training_data,
                                                                                  pd.DataFrame) else training_data

        # ordinal_features_idx = [self.feature_names.index(of) for of in ordinal_features if
        #                         of in self.feature_names] if ordinal_features else []

        self.explainer_object = anchor_tabular.AnchorTabularExplainer(
            class_names=self.class_names if self.class_names else [str(c) for c in
                                                                   np.unique(training_data_np[:, -1])] if
            training_data_np.shape[1] > len(self.feature_names) else ['0', '1'],  # Anchor needs class names
            feature_names=self.feature_names,
            train_data=training_data_np,  # Anchor can infer types or use categorical_names
            categorical_names=categorical_names if categorical_names else {},
            # ordinal_features_idx=ordinal_features_idx
            # Anchor's constructor has fewer direct kwargs compared to LIME for general behavior
        )
        logger.info("Anchor TabularExplainer initialized.")


        class_names = self.class_names if self.class_names else [str(c) for c in np.unique(training_data_np[:, -1])] if training_data_np.shape[1] > len(self.feature_names) else ['0', '1']  # Anchor needs class names
        feature_names = self.feature_names
        train_data = training_data_np  # Anchor can infer types or use categorical_names
        categorical_names = categorical_names if categorical_names else {}
        print("========== debug 0=======")
        print(class_names)
        print(feature_names)
        print(train_data)
        print(categorical_names)

    def explain_instance(self,
                         data_instance: pd.Series,
                         threshold: float = 0.95,
                         **kwargs) -> Dict[str, Any]:  # `kwargs` for Anchor's explain_instance
        super().explain_instance(data_instance)

        instance_np = data_instance[self.feature_names].values if isinstance(data_instance, pd.Series) else data_instance# Anchor handles dtypes
        # instance_np = instance_np.reshape(1, -1)
        print("========== debug 1=======")
        print(instance_np)
        print(instance_np.ndim )
        print(dict(**kwargs))

        anchor_exp = self.explainer_object.explain_instance(
            data_row=instance_np,
            # classifier_fn=self.predict_fn,  # This function should return labels
            threshold=threshold,
            batch_size= 1,
            verbose= True,
            **kwargs
        )
        print("========== debug 2=======")
        print(anchor_exp)
        # Extract feature names involved in the rule conditions
        # anchor_exp.names() returns list of strings like "feature_name <= value"
        # anchor_exp.features() returns list of feature indices involved in the rule
        rule_conditions = anchor_exp.names()  # These are the "features" of the rule

        # Attempt to parse feature names from rule conditions for a more direct feature list
        # This is heuristic and might need refinement based on how Anchor formats rule conditions.
        parsed_features_from_rule = []
        if self.feature_names:
            for condition_str in rule_conditions:
                for fn in self.feature_names:
                    if fn in condition_str:  # Simple check; more robust parsing might be needed
                        if fn not in parsed_features_from_rule:
                            parsed_features_from_rule.append(fn)

        if not parsed_features_from_rule and rule_conditions:  # Fallback if parsing fails but rules exist
            parsed_features_from_rule = ["rule_condition_" + str(i + 1) for i in range(len(rule_conditions))]

        # For 'importance', we can use rule precision or just assign a dummy value.
        # Original code used: list(range(1, len(vars) + 1)).
        # Let's use precision if available, or 1s for each condition.
        importances = [anchor_exp.precision()] * len(parsed_features_from_rule) if parsed_features_from_rule else []
        if not importances and parsed_features_from_rule:
            importances = [1.0] * len(parsed_features_from_rule)

        return {
            "features": parsed_features_from_rule,
            "importance": importances,
            "rule_conditions_text": rule_conditions,  # Store the actual rule
            "precision": anchor_exp.precision(),
            "coverage": anchor_exp.coverage(),
            "raw_explanation": anchor_exp
        }


# --- SHAP Explainer (Wrapper Structure) ---
class ShapExplainerWrapper(BaseExplainer):
    def __init__(self, model: Any, explainer_type: str = "kernel", predict_fn: Optional[Callable] = None, **kwargs):
        super().__init__(model, "shap")
        self.explainer_type = explainer_type.lower()
        self.predict_fn = predict_fn  # SHAP often needs predict_proba for classification, or predict for regression
        self.shap_kwargs = kwargs  # Store additional SHAP specific kwargs

    def fit(self,
            training_data: pd.DataFrame,
            feature_names: List[str],
            class_names: Optional[List[str]] = None,
            **kwargs) -> None:  # `kwargs` for SHAP explainer constructor
        super().fit(training_data, feature_names, class_names)
        try:
            import shap
        except ImportError:
            logger.error("SHAP library not installed. Please install it using 'pip install shap'.")
            raise

        # Consolidate kwargs for SHAP explainer
        current_kwargs = {**self.shap_kwargs, **kwargs}

        # Prepare data for SHAP (often a summary or the data itself)
        # For KernelExplainer, a small summary is often used.
        # For TreeExplainer, the model itself is primary.
        data_for_shap = training_data[self.feature_names]

        if self.explainer_type == "kernel":
            if not self.predict_fn and hasattr(self.model, 'predict_proba'):
                self.predict_fn = self.model.predict_proba
            elif not self.predict_fn:
                raise ValueError("KernelSHAP requires a predict_fn (usually predict_proba).")

            # KernelSHAP's background data: use shap.sample or shap.kmeans
            # Using a sample of the training data as background
            background_data_size = current_kwargs.pop("background_data_size", 100)
            if len(data_for_shap) > background_data_size:
                background_data = shap.sample(data_for_shap, background_data_size,
                                              random_state=kwargs.get("random_state", 42))
            else:
                background_data = data_for_shap
            self.explainer_object = shap.KernelExplainer(self.predict_fn, background_data, **current_kwargs)

        elif self.explainer_type == "tree":
            # TreeExplainer can sometimes take data for background, or infer from model type
            self.explainer_object = shap.TreeExplainer(self.model, data=data_for_shap, **current_kwargs)

        elif self.explainer_type == "deep":
            # DeepExplainer typically needs the model and a sample of training data (often tensors)
            # This part requires more specific handling based on DL framework (TF, PyTorch)
            # For now, assuming data_for_shap is appropriately formatted (e.g., numpy array)
            self.explainer_object = shap.DeepExplainer(self.model, data_for_shap.values, **current_kwargs)

        elif self.explainer_type == "linear":
            self.explainer_object = shap.LinearExplainer(self.model, data_for_shap, **current_kwargs)
        else:
            raise ValueError(
                f"Unsupported SHAP explainer type: {self.explainer_type}. Supported: kernel, tree, deep, linear.")
        logger.info(f"SHAP {self.explainer_type} explainer initialized.")

    def explain_instance(self,
                         data_instance: pd.Series,
                         **kwargs) -> Dict[str, Any]:  # `kwargs` for SHAP's shap_values method
        super().explain_instance(data_instance)

        instance_for_shap = data_instance[self.feature_names].to_frame().T  # SHAP expects 2D array for tabular

        # `kwargs` here are for the .shap_values() call
        shap_values_output = self.explainer_object.shap_values(instance_for_shap, **kwargs)

        # SHAP values can be a list (for multi-class/multi-output) or a single array.
        # For classification with predict_proba, shap_values is often a list of arrays (one per class).
        # We typically want SHAP values for a specific class (e.g., the predicted class or positive class).

        shap_values_for_instance: np.ndarray
        if isinstance(shap_values_output, list):  # Multi-output (e.g., per class)
            # Heuristic: for binary classification, shap_values[1] often corresponds to the positive class.
            # For multi-class, one might need to specify which class's SHAP values are of interest.
            # Defaulting to the first set of SHAP values or values for class 1 if binary.
            idx_to_use = 0
            if len(shap_values_output) > 1: idx_to_use = 1  # common for positive class in binary

            shap_values_for_instance = shap_values_output[idx_to_use][0]  # [0] because we passed one instance
        else:  # Single output (e.g., regression or some TreeExplainer outputs)
            shap_values_for_instance = shap_values_output[0]  # [0] because we passed one instance

        # Sort features by absolute SHAP value magnitude for a ranked list
        abs_shap_values = np.abs(shap_values_for_instance)
        sorted_indices = np.argsort(abs_shap_values)[::-1]

        explained_features = [self.feature_names[i] for i in sorted_indices]
        importances = shap_values_for_instance[sorted_indices].tolist()

        return {
            "features": explained_features,
            "importance": importances,
            "raw_explanation": shap_values_output  # Store the original full SHAP values list/array
        }


# --- Factory Function to Get Explainer ---
def get_explainer_wrapper(explainer_name: str, model: Any, **init_kwargs) -> BaseExplainer:
    """
    Factory function to get an initialized explainer wrapper.

    Args:
        explainer_name (str): Name of the explainer ("lime", "anchor", "shap").
        model (Any): The trained machine learning model.
        **init_kwargs: Keyword arguments to pass to the specific explainer wrapper's constructor.
                       For SHAP, this can include `explainer_type` and `predict_fn`.
                       For LIME/Anchor, this can include `predict_proba_fn` or `predict_fn`.

    Returns:
        BaseExplainer: An instance of the requested explainer wrapper.
    """
    explainer_name_lower = explainer_name.lower()
    if explainer_name_lower == "lime":
        return LimeTabularExplainerWrapper(model, **init_kwargs)
    elif explainer_name_lower == "anchor":
        return AnchorTabularExplainerWrapper(model, **init_kwargs)
    elif explainer_name_lower == "shap":
        # SHAP explainer_type and other specific args should be in init_kwargs
        return ShapExplainerWrapper(model, **init_kwargs)
    else:
        raise ValueError(f"Unknown explainer: {explainer_name}. Supported: lime, anchor, shap.")