"""Roll up per-instance XFA scores to per-dataset and per-tool, with an AUC-derived
complexity-weighted aggregate across the benchmark's 11 datasets."""
from pathlib import Path

import numpy as np
import pandas as pd

TIER1 = ["recall", "one_minus_fpr"]
TIER2 = ["ap", "auc_roc", "ndcg", "relevance_mass", "ap_norm", "auc_roc_norm"]
COMPOSITES = ["tier1", "tier2"]


def per_dataset(per_instance: pd.DataFrame) -> pd.DataFrame:
    """Mean each metric over instances, grouped by (tool, rank, rgs) then averaged to (tool, rank)."""
    metric_cols = [c for c in per_instance.columns if c in TIER1 + TIER2 + COMPOSITES]
    by_rgs = per_instance.groupby(["tool", "rank", "rgs"], as_index=False)[metric_cols].mean()
    by_ds = by_rgs.groupby(["tool", "rank"], as_index=False)[metric_cols].mean()
    return by_ds


def complexity_weights(perf_csv: Path) -> dict:
    """rank -> normalized (1 - AUC) difficulty weight, from model_performance_summary.csv."""
    df = pd.read_csv(perf_csv)
    df["rank"] = df["dataset_id"].astype(str).str.extract(r"(\d+)").astype(int)
    w = 1.0 - df["auc"]
    w = w / w.sum()
    return dict(zip(df["rank"], w))


def per_tool(by_ds: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Per tool: unweighted mean across ranks (all metrics) + complexity-weighted composites."""
    rows = []
    for tool, g in by_ds.groupby("tool"):
        row = {"tool": tool}
        for c in COMPOSITES + TIER1 + TIER2:
            if c in g.columns:
                row[f"{c}_mean"] = float(g[c].mean())
        for c in COMPOSITES:
            if c in g.columns:
                sub = g[["rank", c]].dropna()
                w = sub["rank"].map(weights).fillna(0.0).to_numpy(dtype=float)
                if w.sum() > 0:
                    w = w / w.sum()
                    row[f"{c}_cxw"] = float(np.sum(sub[c].to_numpy(dtype=float) * w))
        rows.append(row)
    return pd.DataFrame(rows)
