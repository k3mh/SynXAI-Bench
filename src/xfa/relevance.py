"""Per-instance ground-truth relevance vectors, derived from metadata (imp_vars + RGS).

A rule's important features are stored most-important-first in ``imp_vars`` (the AFR protocol relies
on the same convention: it restores least->most via ``reversed(imp_vars)``). XFA needs two views over
the n-feature space:

  - membership: 1 for ground-truth features, 0 otherwise (recall / FPR / PR / ROC / relevance-mass);
  - graded relevance: a rank-based gain (most-important feature highest), 0 otherwise (NDCG).

Only the designed *rank* is used (already encoded by imp_vars order) -- no absolute weights.
"""
from typing import List, Sequence

import numpy as np


def clean_imp_vars(imp_vars: Sequence[str]) -> List[str]:
    """Drop the empty-string marker the noise-only rule (RGS0) uses for 'no ground truth'."""
    return [f for f in imp_vars if f]


def gt_membership(imp_vars: Sequence[str], all_features: Sequence[str]) -> np.ndarray:
    """Binary relevance vector aligned to ``all_features`` (1.0 = ground-truth feature)."""
    gt = set(clean_imp_vars(imp_vars))
    return np.array([1.0 if f in gt else 0.0 for f in all_features], dtype=float)


def rank_relevance(imp_vars: Sequence[str], all_features: Sequence[str]) -> np.ndarray:
    """Graded relevance for NDCG: the i-th most-important GT feature gets gain (k - i), non-GT 0.

    k = number of ground-truth features, so imp_vars[0] gets gain k and imp_vars[-1] gets gain 1.
    This rewards explainers that place the heaviest designed features at the top of their ranking.
    """
    gt = clean_imp_vars(imp_vars)
    k = len(gt)
    gain = {f: (k - i) for i, f in enumerate(gt)}  # most-important-first
    return np.array([float(gain.get(f, 0)) for f in all_features], dtype=float)
