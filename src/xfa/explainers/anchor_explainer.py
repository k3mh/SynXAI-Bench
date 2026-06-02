"""Anchor (anchor-exp) wrapper -> rule feature set. Rule-based (Tier-1 only).

AnchorTabularExplainer(class_names, feature_names, train_data); explain_instance returns an anchor
whose ``.features()`` are the indices of the features in the rule. Per-instance seeding makes the
anchor search deterministic and parallel-invariant.
"""
import numpy as np
from anchor import anchor_tabular

from .base import XaiExplainer, XaiOutput


class AnchorExplainer(XaiExplainer):
    name = "anchor"
    is_score_based = False

    def __init__(self, model, training_data: np.ndarray, feature_names, threshold: float = 0.90):
        self._model = model
        self._threshold = threshold
        # Full training data (no subsample): the perturbation distribution must reflect the real data
        # so the test stays honest. Speed comes only from parallelism, which is result-neutral.
        self._explainer = anchor_tabular.AnchorTabularExplainer(
            ["0", "1"], list(feature_names), np.asarray(training_data, dtype=float))

    def explain(self, x_row: np.ndarray, seed=None) -> XaiOutput:
        if seed is not None:
            np.random.seed(seed)
        exp = self._explainer.explain_instance(
            np.asarray(x_row, dtype=float), self._model.predict, threshold=self._threshold)
        return XaiOutput(is_score_based=False, feature_set={int(i) for i in exp.features()})
