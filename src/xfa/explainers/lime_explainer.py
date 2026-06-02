"""LIME (LimeTabularExplainer) wrapper -> per-feature importance vector. Score-based.

Mirrors the setup in Experiments/Initial_experiment_v1.py; |weight| per feature from ``as_map()``.
The explainer's RNG is reset per instance (when a seed is given) so results are deterministic and
invariant to parallel call order.
"""
import numpy as np
import lime
import lime.lime_tabular

from .base import XaiExplainer, XaiOutput


class LimeExplainer(XaiExplainer):
    name = "lime"
    is_score_based = True

    def __init__(self, model, training_data: np.ndarray, feature_names, seed: int = 42):
        self._model = model
        self._n = len(feature_names)
        self._explainer = lime.lime_tabular.LimeTabularExplainer(
            np.asarray(training_data, dtype=float),
            feature_names=list(feature_names),
            class_names=["0", "1"],
            mode="classification",
            random_state=seed,
        )

    def explain(self, x_row: np.ndarray, seed=None) -> XaiOutput:
        if seed is not None:
            # LIME draws from the explainer RNG, the global RNG, and the discretizer's own RNG;
            # reset all three so an instance is scored identically regardless of call order.
            np.random.seed(seed)
            self._explainer.random_state = np.random.RandomState(seed)
            if getattr(self._explainer, "discretizer", None) is not None:
                self._explainer.discretizer.random_state = np.random.RandomState(seed)
        exp = self._explainer.explain_instance(
            np.asarray(x_row, dtype=float), self._model.predict_proba, num_features=self._n)
        vec = np.zeros(self._n, dtype=float)
        for fi, w in exp.as_map()[1]:
            vec[fi] = abs(w)
        return XaiOutput(is_score_based=True, vector=vec)
