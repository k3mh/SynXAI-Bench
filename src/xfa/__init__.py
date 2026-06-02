"""XFA — Explainability Fidelity Assessor.

A two-tier fidelity battery that scores how well a local explainer recovers each instance's
ground-truth feature set/ranking on the SynXAI-Data benchmark. See relevance.py (ground-truth
vectors), metrics.py (the metric battery), aggregate.py (roll-ups + complexity weighting),
and explainers/ (per-tool wrappers).
"""
from . import relevance, metrics, aggregate  # noqa: F401
