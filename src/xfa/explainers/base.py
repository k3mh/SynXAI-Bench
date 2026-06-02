"""Common explainer interface for XFA.

A wrapper produces, per instance, either a full per-feature importance ``vector`` (score-based tools:
SHAP, LIME) or a ``feature_set`` of flagged feature indices (rule-based tools: Anchor, LORE).
``explain`` accepts an optional ``seed`` so each instance is scored deterministically regardless of
call order (this makes results invariant to parallel chunking and reproducible run-to-run).
"""
from dataclasses import dataclass
from typing import Optional, Set

import numpy as np


@dataclass
class XaiOutput:
    is_score_based: bool
    vector: Optional[np.ndarray] = None       # length-n |importance|, for score-based tools
    feature_set: Optional[Set[int]] = None    # feature indices, for rule-based tools


class XaiExplainer:
    """Base class. Subclasses set ``name`` / ``is_score_based`` and implement ``explain``."""
    name: str = "base"
    is_score_based: bool = True

    def explain(self, x_row: np.ndarray, seed: Optional[int] = None) -> XaiOutput:
        raise NotImplementedError
