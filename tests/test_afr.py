"""
Guard tests for the AFR (Additive Feature Restoration) validation, focused on the
multi-shuffle option added to model_validation.

Checks:
  - seeded runs are reproducible;
  - n_shuffles > 1 runs and reports the shuffle count;
  - the validation -> analysis pipeline yields one Kendall's-tau row per rule group.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import dataset_generation.DataSetGen_v4 as sdg  # noqa: E402
import model_validation as mv  # noqa: E402

SEED = 42


def _small_benchmark():
    """A small multi-rule dataset + trained model + aligned correct-sample test split."""
    cfg, size = [2, 4, 10], 4000          # RGS2/RGS4/RGS10 (multi-feature rules)
    n = len(cfg); per = size // n
    data, meta = [], []
    for j, idx in enumerate(cfg):
        sz = per if j < n - 1 else size - per * (n - 1)
        ds = sdg.generate_dataset_by_name(f"ds{idx}", sz)
        data.append(ds.data); meta.append(ds.meta_data)
    d = pd.concat(data, ignore_index=True)
    m = pd.concat(meta, ignore_index=True)
    feats = sdg.ALL_FEATURE_NAMES
    Xtr, Xte, ytr, yte = train_test_split(
        d[feats], d["y"], train_size=0.7, random_state=SEED, stratify=d["y"])
    mt = m.loc[Xte.index].reset_index(drop=True)
    Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)
    model = RandomForestClassifier(n_jobs=-1, random_state=SEED).fit(Xtr.values, ytr)
    return model, Xte, yte, mt


@pytest.fixture(scope="module")
def benchmark():
    return _small_benchmark()


def test_seeded_run_is_reproducible(benchmark):
    model, Xte, yte, mt = benchmark
    a = mv.validate_model_additive_impact_on_correct_samples(
        model, Xte, yte, mt, n_shuffles=4, random_state=SEED)
    b = mv.validate_model_additive_impact_on_correct_samples(
        model, Xte, yte, mt, n_shuffles=4, random_state=SEED)
    assert np.allclose(a["performance_gain"].values, b["performance_gain"].values)


def test_n_shuffles_is_recorded(benchmark):
    model, Xte, yte, mt = benchmark
    v = mv.validate_model_additive_impact_on_correct_samples(
        model, Xte, yte, mt, n_shuffles=5, random_state=SEED)
    assert (v["n_shuffles"] == 5).all()


def test_analysis_one_tau_per_group(benchmark):
    model, Xte, yte, mt = benchmark
    v = mv.validate_model_additive_impact_on_correct_samples(
        model, Xte, yte, mt, n_shuffles=3, random_state=SEED)
    a = mv.analyze_validation_results(v, mt)
    # one row per multi-feature RGS present, tau within valid range
    assert len(a) == v["rgs_group"].nunique()
    tau = a["kendalls_tau_correlation"]
    assert tau[tau.notna()].between(-1.0, 1.0).all()


def test_single_feature_rule_tau_is_nan():
    """A k=1 rule has no defined rank correlation: it must report NaN, never the -1 sentinel.

    -1 is a valid tau (perfect anti-correlation), so returning it for 'not applicable' quietly
    corrupts any statistic averaged over rules.
    """
    cfg, size = [1, 2], 3000              # RGS1 is the single-feature rule
    n = len(cfg); per = size // n
    data, meta = [], []
    for j, idx in enumerate(cfg):
        sz = per if j < n - 1 else size - per * (n - 1)
        ds = sdg.generate_dataset_by_name(f"ds{idx}", sz)
        data.append(ds.data); meta.append(ds.meta_data)
    d = pd.concat(data, ignore_index=True)
    m = pd.concat(meta, ignore_index=True)
    feats = sdg.ALL_FEATURE_NAMES
    Xtr, Xte, ytr, yte = train_test_split(
        d[feats], d["y"], train_size=0.7, random_state=SEED, stratify=d["y"])
    mt = m.loc[Xte.index].reset_index(drop=True)
    Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)
    model = RandomForestClassifier(n_jobs=-1, random_state=SEED).fit(Xtr.values, ytr)

    v = mv.validate_model_additive_impact_on_correct_samples(
        model, Xte, yte, mt, n_shuffles=2, random_state=SEED)
    a = mv.analyze_validation_results(v, mt)
    rgs1 = a[a["rgs_group"] == "RGS1"]
    assert len(rgs1) == 1
    assert np.isnan(rgs1["kendalls_tau_correlation"].iloc[0])
    assert (a[a["rgs_group"] != "RGS1"]["kendalls_tau_correlation"] != -1).all()
