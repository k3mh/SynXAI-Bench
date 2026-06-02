"""XFA runner: score explainers (SHAP/LIME/Anchor/LORE) on the SynXAI-DB benchmark and aggregate.

Loads a benchmark run dir (datasets, metadata, XGBoost models), samples correctly-predicted instances
evenly across each dataset's rule groups (RGS), runs each explainer per instance, computes the two-tier
XFA metric battery, and writes per-instance metrics, the raw explanations, and per-dataset / per-tool
roll-ups (the latter with the AUC-derived complexity-weighted aggregate).

Robust to interruption: results are written/appended **per dataset**, and a relaunch with the same
``--out`` **resumes** (datasets already present are skipped). A ``run_info.json`` (timestamp, source
dataset, params) stamps each run for organisation across multiple runs.

Parallelism is optional and OFF by default. ``--n_jobs 1`` is sequential; ``-1`` uses the physical core
count (capped at 8). Per-instance seeding makes results identical for any ``n_jobs``.

Run via the package so workers can import it:
  cd src && python -m xfa.runner [--tools ...] [--budget N] [--ndatasets K] [--n_jobs J] [--out DIR]
"""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from xfa import aggregate  # noqa: E402
from xfa._worker import score_chunk  # noqa: E402

SEED = 42
DEFAULT_RUN = REPO / "src" / "my_benchmark_results" / "SynXAI-DB_run_49feat"


def _default_out(run_dir: Path, tools) -> Path:
    """Dated, source-named output dir, e.g. SynXAI-DB_run_49feat__xfa_2026-06-07__shap-lime."""
    date = datetime.date.today().isoformat()
    tag = "-".join(tools)
    return run_dir.parent / f"{run_dir.name}__xfa_{date}__{tag}"


def load_dataset(run_dir: Path, cid: int):
    """Load a benchmark dataset + metadata. Prefers the numpy-2 pickles; falls back to portable CSV
    (so a numpy<2 env, e.g. the LORE env, can read the same datasets — imp_vars is '|'-joined there)."""
    try:
        data = pd.read_pickle(run_dir / f"dataset_config_{cid}.pkl")
        meta = pd.read_pickle(run_dir / f"metadata_config_{cid}.pkl")
    except Exception:
        data = pd.read_csv(run_dir / f"dataset_config_{cid}.csv")
        meta = pd.read_csv(run_dir / f"metadata_config_{cid}.csv")
        meta["imp_vars"] = meta["imp_vars"].fillna("").apply(lambda s: s.split("|") if s else [""])
    feats = [c for c in data.columns if c != "y"]
    return data[feats], data["y"], meta, feats


def load_model(run_dir: Path, cid: int):
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(str(run_dir / f"model_config_{cid}.json"))
    m.set_params(n_jobs=1)  # tiny per-call batches; avoids thread-spawn overhead and oversubscription
    return m


def sample_correct_across_rgs(model, X, y, meta, budget, seed=SEED):
    """Indices of correctly-predicted instances, sampled evenly across the dataset's RGS groups."""
    correct = X.index[model.predict(X.values) == y.values]
    cm = meta.loc[correct]
    rules = sorted(cm["RGS"].unique())
    nr = len(rules)
    per, rem = budget // nr, budget % nr
    sel = []
    for i, r in enumerate(rules):
        ri = list(cm[cm["RGS"] == r].index)
        k = min(per + (1 if i < rem else 0), len(ri))
        if k > 0:
            sel += list(pd.Series(ri).sample(n=k, random_state=seed))
    return sel


def _append(df: pd.DataFrame, pi_path: Path, expl_path: Path):
    """Append one dataset's rows: raw explanations to a companion file, metrics to per-instance."""
    if "raw" in df.columns:
        df[["tool", "rank", "rgs", "instance", "raw"]].to_csv(
            expl_path, mode="a", header=not expl_path.exists(), index=False)
        df = df.drop(columns=["raw"])
    df.to_csv(pi_path, mode="a", header=not pi_path.exists(), index=False)


def run(run_dir: Path, out_dir: Path, tools, budget: int, ndatasets: int, n_jobs: int,
        anchor_threshold: float = 0.90):
    out_dir.mkdir(parents=True, exist_ok=True)
    eff = (min(8, os.cpu_count() or 1) if n_jobs == -1 else max(1, n_jobs))
    Parallel = delayed = None
    if eff != 1:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = "1"  # loky workers inherit -> no BLAS oversubscription
        from joblib import Parallel, delayed

    # stamp the run for organisation across multiple runs
    (out_dir / "run_info.json").write_text(json.dumps({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_dataset": run_dir.name,
        "tools": tools, "budget": budget, "ndatasets": ndatasets,
        "n_jobs": eff, "anchor_threshold": anchor_threshold, "seed": SEED,
    }, indent=2))

    pi_path = out_dir / "xfa_per_instance.csv"
    expl_path = out_dir / "xfa_explanations.csv"

    # resume: skip datasets already written
    done_ranks = set()
    if pi_path.exists():
        done_ranks = set(pd.read_csv(pi_path, usecols=["rank"])["rank"].unique())
        if done_ranks:
            print(f"[resume] datasets already done: {sorted(done_ranks)}", flush=True)

    for cid in range(1, ndatasets + 1):
        if cid in done_ranks:
            print(f"[cfg {cid}] already done, skipping", flush=True)
            continue
        X, y, meta, feats = load_dataset(run_dir, cid)
        model = load_model(run_dir, cid)
        sel = sample_correct_across_rgs(model, X, y, meta, budget)
        items = [(int(idx), X.loc[idx].values, meta.loc[idx, "imp_vars"], meta.loc[idx, "RGS"])
                 for idx in sel]
        X_np = X.values
        cid_rows = []
        for tool in tools:
            t0 = time.time()
            try:
                if eff == 1:
                    chunk_rows = score_chunk(tool, model, X_np, feats, items, cid, SEED, anchor_threshold)
                else:
                    chunks = [items[i::eff] for i in range(eff)]
                    res = Parallel(n_jobs=eff, backend="loky")(
                        delayed(score_chunk)(tool, model, X_np, feats, ch, cid, SEED, anchor_threshold)
                        for ch in chunks if ch)
                    chunk_rows = [r for sub in res for r in sub]
            except Exception as e:
                print(f"[cfg {cid}] {tool} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            cid_rows += chunk_rows
            print(f"[cfg {cid}] {tool}: {len(chunk_rows)}/{len(items)} instances "
                  f"({time.time()-t0:.0f}s, n_jobs={eff})", flush=True)
        if cid_rows:
            _append(pd.DataFrame(cid_rows), pi_path, expl_path)  # checkpoint this dataset

    # final aggregates from the full per-instance file
    if pi_path.exists():
        full = pd.read_csv(pi_path)
        by_ds = aggregate.per_dataset(full)
        by_ds.to_csv(out_dir / "xfa_per_dataset.csv", index=False)
        perf = run_dir / "model_performance_summary.csv"
        weights = aggregate.complexity_weights(perf) if perf.exists() else {}
        by_tool = aggregate.per_tool(by_ds, weights)
        if "failed" in full.columns:
            fr = (full.groupby("tool")["failed"].mean()
                  .reset_index().rename(columns={"failed": "failed_rate"}))
            by_tool = by_tool.merge(fr, on="tool", how="left")
        by_tool.to_csv(out_dir / "xfa_per_tool.csv", index=False)
        print("\n=== XFA per-tool summary ===", flush=True)
        print(by_tool.to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--out", default=None, help="output dir (default: dated/source-named)")
    ap.add_argument("--tools", default="shap,lime,anchor,lore")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--ndatasets", type=int, default=11)
    ap.add_argument("--n_jobs", type=int, default=1, help="1=sequential (default); -1=physical cores (cap 8)")
    ap.add_argument("--anchor_threshold", type=float, default=0.90, help="Anchor precision threshold")
    args = ap.parse_args()
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else _default_out(run_dir, tools)
    run(run_dir, out_dir, tools, args.budget, args.ndatasets, args.n_jobs, args.anchor_threshold)
