"""Unit tests for the XFA metric core (relevance + metrics), on hand-computed cases."""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from xfa import metrics as m  # noqa: E402
from xfa import relevance as rel  # noqa: E402

FEATS = [f"x{i}" for i in range(1, 50)]  # x1..x49
N = 49


def test_membership_and_rank_from_imp_vars():
    imp = ["x37", "x38", "x39", "x40", "x41"]  # stored most-important-first
    mem = rel.gt_membership(imp, FEATS)
    assert mem.sum() == 5
    assert mem[FEATS.index("x37")] == 1.0 and mem[FEATS.index("x1")] == 0.0
    g = rel.rank_relevance(imp, FEATS)
    assert g[FEATS.index("x37")] == 5.0  # most important -> highest gain
    assert g[FEATS.index("x41")] == 1.0  # least important
    assert g[FEATS.index("x1")] == 0.0   # non-GT


def test_rgs0_noise_marker_has_no_ground_truth():
    assert rel.gt_membership([""], FEATS).sum() == 0


def test_vector_to_set_unit_mass_threshold():
    attr = np.zeros(N)
    attr[0] = 10.0    # dominant -> above 1/n after normalization
    attr[1] = 0.0001  # negligible -> below 1/n
    s = m.vector_to_set(attr, N)
    assert 0 in s and 1 not in s


def test_set_metrics_recall_and_fpr():
    gt, flagged = {0, 1, 2}, {0, 1, 9}  # 2 hits, 1 false positive
    assert m.recall(flagged, gt) == pytest.approx(2 / 3)
    assert m.one_minus_fpr(flagged, gt, N) == pytest.approx(1 - 1 / (N - 3))


def test_perfect_attribution_scores_one():
    imp = ["x37", "x38", "x39", "x40", "x41"]
    mem, g = rel.gt_membership(imp, FEATS), rel.rank_relevance(imp, FEATS)
    attr = np.zeros(N)
    for i, f in enumerate(imp):           # attribution follows the designed rank
        attr[FEATS.index(f)] = len(imp) - i
    r = m.score_vector(attr, mem, g, N)
    assert r["recall"] == 1.0
    assert r["ap"] == pytest.approx(1.0)
    assert r["auc_roc"] == pytest.approx(1.0)
    assert r["ndcg"] == pytest.approx(1.0)
    assert r["relevance_mass"] == pytest.approx(1.0)
    assert r["tier2"] == pytest.approx(1.0)


def test_uniform_attribution_hits_baselines():
    imp = ["x1", "x2", "x3", "x4", "x5"]
    mem, g = rel.gt_membership(imp, FEATS), rel.rank_relevance(imp, FEATS)
    r = m.score_vector(np.ones(N), mem, g, N)
    assert r["auc_roc"] == pytest.approx(0.5, abs=1e-9)
    assert r["relevance_mass"] == pytest.approx(5 / 49, abs=1e-9)
    assert r["ap"] == pytest.approx(5 / 49, abs=0.02)
    assert r["auc_roc_norm"] == 0.0
    assert r["ap_norm"] == pytest.approx(0.0, abs=0.03)


def test_ndcg_penalizes_reversed_ranking():
    imp = ["x1", "x2", "x3", "x4", "x5"]
    g = rel.rank_relevance(imp, FEATS)
    attr = np.zeros(N)
    for i, f in enumerate(imp):           # reverse the designed importance order
        attr[FEATS.index(f)] = i + 1
    assert m.ndcg(g, attr) < 1.0


def test_score_set_tier1_only():
    out = m.score_set({0, 1}, {0, 1, 2}, N)
    assert set(out) == {"recall", "one_minus_fpr", "tier1", "n_flagged", "gt_size"}
    assert out["recall"] == pytest.approx(2 / 3)
