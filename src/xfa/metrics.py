"""XFA fidelity metrics for a single explained instance, over the n-feature space.

Tier 1 (set-based, all explainers):           recall, 1 - FPR.
Tier 2 (vector-based, score-based explainers): AP / AUC-PR, AUC-ROC, NDCG (graded), relevance-mass.

All components live in [0, 1]. AP and AUC-ROC are baseline-normalized (against their random
baselines of prevalence and 0.5) so the equal-weight tier composites are not distorted.

Score-based -> set conversion (Tier 1 for SHAP/LIME): normalize |attribution| to unit mass and flag
features above the uniform-attribution baseline 1/n -- a non-circular, comparability-oriented set
(the threshold-free Tier 2 is the primary assessment for score-based tools). Rule-based tools
(Anchor/LORE) emit a feature set directly and use ``score_set``.

Note on NDCG ties: ``ndcg_score`` breaks ties in the score vector by position; for continuous
SHAP/LIME attributions exact ties are rare, so this is acceptable for v1.
"""
from typing import Dict, Set

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


# ---------------------------------------------------------------- helpers
def _abs_unit_mass(attr: np.ndarray) -> np.ndarray:
    a = np.abs(np.asarray(attr, dtype=float))
    s = a.sum()
    return a / s if s > 0 else a


def vector_to_set(attr: np.ndarray, n: int) -> Set[int]:
    """Score-based -> flagged feature indices: unit-mass normalize, keep features above 1/n."""
    norm = _abs_unit_mass(attr)
    return {int(i) for i in np.where(norm > 1.0 / n)[0]}


# ---------------------------------------------------------------- tier 1 (set)
def recall(flagged: Set[int], gt: Set[int]) -> float:
    if not gt:
        return float("nan")
    return len(flagged & gt) / len(gt)


def one_minus_fpr(flagged: Set[int], gt: Set[int], n: int) -> float:
    neg = n - len(gt)
    if neg <= 0:
        return float("nan")
    fp = len(flagged - gt)
    return 1.0 - fp / neg


# ---------------------------------------------------------------- tier 2 (vector)
def average_precision(membership: np.ndarray, attr: np.ndarray) -> float:
    return float(average_precision_score(membership, np.abs(attr)))


def auc_roc(membership: np.ndarray, attr: np.ndarray) -> float:
    return float(roc_auc_score(membership, np.abs(attr)))


def ndcg(graded: np.ndarray, attr: np.ndarray) -> float:
    return float(ndcg_score(np.asarray(graded).reshape(1, -1), np.abs(attr).reshape(1, -1)))


def relevance_mass(gt: Set[int], attr: np.ndarray) -> float:
    a = np.abs(np.asarray(attr, dtype=float))
    s = a.sum()
    if s <= 0 or not gt:
        return 0.0
    return float(a[list(gt)].sum() / s)


# ---------------------------------------------------------------- baseline normalization
def norm_ap(ap: float, gt_size: int, n: int) -> float:
    prev = gt_size / n
    if prev >= 1.0:
        return float("nan")
    return max(0.0, (ap - prev) / (1.0 - prev))


def norm_aucroc(auc: float) -> float:
    return max(0.0, 2.0 * auc - 1.0)


# ---------------------------------------------------------------- composites
def tier1_composite(rec: float, omf: float) -> float:
    return float(np.nanmean([rec, omf]))


def tier2_composite(nap: float, nauc: float, ndcg_v: float, relmass: float) -> float:
    return float(np.nanmean([nap, nauc, ndcg_v, relmass]))


# ---------------------------------------------------------------- instance entry points
def score_vector(attr, membership, graded, n: int) -> Dict[str, float]:
    """Full battery for a score-based explainer's per-feature importance vector.

    Assumes a non-empty ground truth (callers skip the noise-only RGS0).
    """
    attr = np.asarray(attr, dtype=float)
    membership = np.asarray(membership, dtype=float)
    gt = {int(i) for i in np.where(membership > 0)[0]}
    gt_size = len(gt)
    flagged = vector_to_set(attr, n)
    rec = recall(flagged, gt)
    omf = one_minus_fpr(flagged, gt, n)
    ap = average_precision(membership, attr)
    auc = auc_roc(membership, attr)
    nd = ndcg(graded, attr)
    rm = relevance_mass(gt, attr)
    nap = norm_ap(ap, gt_size, n)
    nauc = norm_aucroc(auc)
    return {
        "recall": rec, "one_minus_fpr": omf,
        "ap": ap, "auc_roc": auc, "ndcg": nd, "relevance_mass": rm,
        "ap_norm": nap, "auc_roc_norm": nauc,
        "tier1": tier1_composite(rec, omf),
        "tier2": tier2_composite(nap, nauc, nd, rm),
        "n_flagged": float(len(flagged)), "gt_size": float(gt_size),
    }


def score_set(flagged: Set[int], gt: Set[int], n: int) -> Dict[str, float]:
    """Tier-1-only battery for a rule-based explainer that emits a feature set."""
    rec = recall(flagged, gt)
    omf = one_minus_fpr(flagged, gt, n)
    return {
        "recall": rec, "one_minus_fpr": omf,
        "tier1": tier1_composite(rec, omf),
        "n_flagged": float(len(flagged)), "gt_size": float(len(gt)),
    }
