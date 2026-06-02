"""Worker-side scoring for the XFA runner.

Kept in its own importable module (not the ``__main__`` runner script) so ``score_chunk`` is picklable
by joblib/loky when running in parallel. Each chunk builds its explainer once, then scores its
instances; per-instance seeding makes the output independent of how instances are chunked across
workers, so results are identical for any ``n_jobs``.

Per-instance explainer failures are RECORDED as a degenerate explanation (empty feature set / zero
vector) with ``failed=True`` rather than dropped — dropping would bias a tool's scores upward by
silently excluding the hard instances it cannot explain.
"""
import numpy as np

from xfa import metrics, relevance


def build_explainer(name, model, X_np, feats, anchor_threshold=0.90):
    if name == "shap":
        from xfa.explainers.shap_explainer import ShapExplainer
        return ShapExplainer(model, X_np)
    if name == "lime":
        from xfa.explainers.lime_explainer import LimeExplainer
        return LimeExplainer(model, X_np, feats)
    if name == "anchor":
        from xfa.explainers.anchor_explainer import AnchorExplainer
        return AnchorExplainer(model, X_np, feats, threshold=anchor_threshold)
    if name == "lore":
        from xfa.explainers.lore_explainer import LoreExplainer
        return LoreExplainer(model, X_np, feats)
    raise ValueError(f"unknown tool: {name}")


def score_chunk(tool, model, X_np, feats, items, rank, seed_base=42, anchor_threshold=0.90):
    """Score a chunk of instances with one explainer build.

    items: list of (instance_idx:int, x_row:1d-array, imp_vars:list[str], rgs:str).
    Returns a list of per-instance metric dicts (with tool/rank/rgs/instance/failed attached).
    A build failure propagates (the whole tool is unavailable); a per-instance explain failure is
    recorded as a degenerate explanation with failed=True.
    """
    ex = build_explainer(tool, model, X_np, feats, anchor_threshold=anchor_threshold)
    n = len(feats)
    rows = []
    for idx, x_row, imp, rgs in items:
        mem = relevance.gt_membership(imp, feats)
        gt_idx = {int(i) for i in np.where(mem > 0)[0]}
        failed = False
        raw = ""  # the raw explanation, persisted so new metrics never need re-running the explainer
        try:
            out = ex.explain(np.asarray(x_row, dtype=float), seed=seed_base + int(idx))
        except Exception:
            out, failed = None, True

        if out is not None and out.is_score_based:
            graded = relevance.rank_relevance(imp, feats)
            sc = metrics.score_vector(out.vector, mem, graded, n)
            raw = "|".join(map(str, np.asarray(out.vector, dtype=float).tolist()))   # |attr| per feature
        elif out is not None:
            sc = metrics.score_set(out.feature_set or set(), gt_idx, n)
            raw = "|".join(feats[i] for i in sorted(out.feature_set or set()))        # flagged features
        elif ex.is_score_based:  # failed score-based -> zero attribution vector
            graded = relevance.rank_relevance(imp, feats)
            sc = metrics.score_vector(np.zeros(n), mem, graded, n)
        else:                    # failed rule-based -> empty feature set
            sc = metrics.score_set(set(), gt_idx, n)

        sc.update({"tool": tool, "rank": rank, "rgs": rgs, "instance": int(idx),
                   "failed": failed, "raw": raw})
        rows.append(sc)
    return rows
