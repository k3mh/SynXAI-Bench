"""Explainer wrappers for XFA.

Each wrapper exposes a uniform ``explain(x_row) -> XaiOutput`` (see base.py). Import the specific
wrapper you need (e.g. ``from xfa.explainers.shap_explainer import ShapExplainer``) so a missing
optional dependency for one tool never breaks the others.
"""
