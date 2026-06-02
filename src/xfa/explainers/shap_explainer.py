"""SHAP (KernelExplainer) wrapper -> per-feature importance vector. Score-based.

Mirrors the setup in Experiments/Initial_experiment_v1.py (KernelExplainer over a 50-row background
sample; positive-class attributions). The background sample and the per-instance coalition sampling
are seeded so results are deterministic and parallel-invariant.
"""
import numpy as np
import shap

from .base import XaiExplainer, XaiOutput


class ShapExplainer(XaiExplainer):
    name = "shap"
    is_score_based = True

    def __init__(self, model, background: np.ndarray, n_background: int = 50, seed: int = 42):
        background = np.asarray(background, dtype=float)
        np.random.seed(seed)  # deterministic background sample
        bg = shap.sample(background, n_background) if len(background) > n_background else background
        self._explainer = shap.KernelExplainer(model.predict_proba, bg)

    def explain(self, x_row: np.ndarray, seed=None) -> XaiOutput:
        if seed is not None:
            np.random.seed(seed)  # fix coalition sampling for this instance
        sv = self._explainer.shap_values(np.asarray(x_row, dtype=float), silent=True)
        if isinstance(sv, list):                       # [class0, class1]
            arr = np.asarray(sv[1])
        else:
            arr = np.asarray(sv)
            if arr.ndim == 2 and arr.shape[-1] == 2:   # (n_features, n_classes)
                arr = arr[:, 1]
        return XaiOutput(is_score_based=True, vector=np.abs(arr).ravel())
