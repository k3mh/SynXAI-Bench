"""Run configuration, loaders and helpers shared by the XFA validation battery.

A *run* names one validation input: the benchmark suite (datasets, metadata, models) plus the XFA
output directories produced on it. Runs are registered in ``RUNS``; every pillar accepts
``--run <name>`` (default ``1K_v2``) and writes its tables under ``results/<run>/<pillar>/``, so
different inputs never overwrite each other. To validate a new XFA run (another suite, model family
or budget), register one more ``RunConfig`` entry.

All paths are derived from this file's location, so the battery is portable.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2]                       # .../src
RES = SRC / "my_benchmark_results"
_RESULTS = Path(__file__).resolve().parent / "results"

TOOLS = ["shap", "lime", "lore", "anchor"]                     # expected fidelity order (SHAP best)


@dataclass(frozen=True)
class RunConfig:
    name: str            # results namespace: results/<name>/
    suite: Path          # benchmark suite dir (datasets, metadata, models, AUC summary)
    xfa_dirs: tuple      # XFA output dirs; their per-instance/explanation files are concatenated

    @property
    def expl_files(self):
        return [d / "xfa_explanations.csv" for d in self.xfa_dirs]


RUNS = {
    "1K_v2": RunConfig(
        name="1K_v2",
        suite=RES / "SynXAI-DB_run_49feat_v2",
        xfa_dirs=(
            RES / "SynXAI-DB_run_49feat_v2__xfa_2026-06-18__shap-lime",
            RES / "SynXAI-DB_run_49feat_v2__xfa_2026-06-19__anchor",
            RES / "SynXAI-DB_run_49feat_v2__xfa_2026-06-19__lore",
        ),
    ),
}
DEFAULT_RUN = "1K_v2"


def get_run() -> RunConfig:
    """Parse ``--run <name>``, the one CLI argument every pillar shares."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN, choices=sorted(RUNS),
                    help="registered validation input (see RUNS in xfa/validation/_common.py)")
    return RUNS[ap.parse_args().run]


def results_dir(run: RunConfig, pillar: str) -> Path:
    """The output folder for one pillar of one run, e.g. results/1K_v2/p3_discriminant."""
    d = _RESULTS / run.name / pillar
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_per_instance(run: RunConfig, cols=None) -> pd.DataFrame:
    """All tools' per-instance scores for a run, failed explanations dropped. Tier-2 is NaN for the
    rule-based tools (Anchor/LORE), which only produce Tier-1."""
    frames = []
    for d in run.xfa_dirs:
        df = pd.read_csv(d / "xfa_per_instance.csv")
        df = df[~df["failed"].astype(bool)]
        frames.append(df if cols is None else df[[c for c in cols if c in df.columns]])
    return pd.concat(frames, ignore_index=True)


def load_score_instances(run: RunConfig) -> pd.DataFrame:
    """Per-instance rows for the score-based tools only (all Tier-1 + Tier-2 components present)."""
    frames = []
    for d in run.xfa_dirs:
        df = pd.read_csv(d / "xfa_per_instance.csv")
        df = df[df["tool"].isin(["shap", "lime"]) & ~df["failed"].astype(bool)]
        if len(df):
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_auc_by_rank(run: RunConfig) -> pd.Series:
    """Model AUC per config, indexed by complexity rank (1 = easiest / highest AUC)."""
    perf = pd.read_csv(run.suite / "model_performance_summary.csv")
    perf["rank"] = perf["dataset_id"].str.replace("config_", "").astype(int)
    return perf.set_index("rank")["auc"]


def load_explanations(run: RunConfig) -> pd.DataFrame:
    """Raw stored explanations for all tools: a per-feature vector (score-based) or a '|'-joined
    feature-name set (rule-based), one row per (tool, rank, instance)."""
    df = pd.concat([pd.read_csv(f) for f in run.expl_files], ignore_index=True)
    return df.drop_duplicates(["tool", "rank", "instance"])


def parse_vector(raw) -> np.ndarray:
    """Parse a score-based explanation's stored '|'-joined importance vector."""
    return np.fromstring(str(raw), sep="|")


def bootstrap_ci(x, n_boot: int = 2000, rng=None):
    """Percentile 95% CI of the mean by resampling with replacement."""
    rng = rng if rng is not None else np.random.default_rng(42)
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
