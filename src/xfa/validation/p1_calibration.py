"""Pillar 1 - diagnosticity / calibration.

Feed synthetic explanations of known, graded fidelity into the real XFA scorer and check the metric is
monotone in true fidelity, with the expected anchors:

    oracle (p=0) -> metric maximum,   random (p=1) -> chance,   anti-oracle -> below chance.

Ground-truth relevance is constructed directly (no model, no explainer), so this isolates the metric's
own response. The score-based path exercises both tiers (Tier-2 is primary for SHAP/LIME); the
rule-based path exercises Tier-1 over a flagged feature set (Anchor/LORE).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xfa.metrics import score_set, score_vector

from ._common import get_run, results_dir

N = 49                                  # suite-v2 feature space
CARDS = [1, 2, 3, 4, 5, 6, 10]          # single-rule ground-truth cardinalities present in SynXAI-DB
PS = np.round(np.linspace(0.0, 1.0, 11), 2)   # corruption level p: 0 (oracle) .. 1 (random)
SEEDS = 40                              # draws per (k, p) for confidence intervals


def membership_graded(k: int, n: int):
    """Binary and rank-graded relevance for a rule with k features (WLOG the first k)."""
    membership = np.zeros(n); membership[:k] = 1.0
    graded = np.zeros(n); graded[:k] = np.arange(k, 0, -1)     # k, k-1, ..., 1 (rank_relevance convention)
    return membership, graded


def oracle_signal(k: int, n: int) -> np.ndarray:
    """Perfect attribution: decreasing mass on the k ground-truth features, zero elsewhere."""
    s = np.zeros(n); s[:k] = np.arange(k, 0, -1)
    return s / s.sum()


def synth_vector(k: int, n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Graded-fidelity attribution: (1-p) * oracle + p * uniform noise over all features."""
    noise = rng.random(n); noise = noise / noise.sum()
    return (1.0 - p) * oracle_signal(k, n) + p * noise


def anti_oracle_vector(k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Mass entirely on non-ground-truth features: a below-chance anchor."""
    a = np.zeros(n); a[k:] = rng.random(n - k)
    return a


def synth_flagged(k: int, n: int, p: float, rng: np.random.Generator):
    """Fidelity-p feature set for a rule-based tool: recover (1-p)*k true features, add p*k false ones."""
    n_true, n_fp = int(round((1.0 - p) * k)), int(round(p * k))
    fps = rng.choice(range(k, n), size=min(n_fp, n - k), replace=False) if n_fp else []
    return {int(i) for i in range(n_true)} | {int(i) for i in fps}


def run_score_based() -> pd.DataFrame:
    rows = []
    for k in CARDS:
        membership, graded = membership_graded(k, N)
        for p in PS:
            for seed in range(SEEDS):
                rng = np.random.default_rng([k, int(round(p * 100)), seed])
                m = score_vector(synth_vector(k, N, float(p), rng), membership, graded, N)
                rows.append({"k": k, "p": float(p), "tier1": m["tier1"], "tier2": m["tier2"]})
    return pd.DataFrame(rows)


def run_anti_oracle() -> pd.DataFrame:
    rows = []
    for k in CARDS:
        membership, graded = membership_graded(k, N)
        for seed in range(SEEDS):
            rng = np.random.default_rng([k, 999, seed])
            m = score_vector(anti_oracle_vector(k, N, rng), membership, graded, N)
            rows.append({"k": k, "tier2": m["tier2"]})
    return pd.DataFrame(rows)


def run_rule_based() -> pd.DataFrame:
    rows = []
    for k in CARDS:
        gt = set(range(k))
        for p in PS:
            for seed in range(SEEDS):
                rng = np.random.default_rng([k, int(round(p * 100)), seed, 7])
                m = score_set(synth_flagged(k, N, float(p), rng), gt, N)
                rows.append({"k": k, "p": float(p), "tier1": m["tier1"]})
    return pd.DataFrame(rows)


def _spearman_by_k(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby(["k", "p"])[col].mean().reset_index()
    return g.groupby("k").apply(lambda d: spearmanr(d["p"], d[col]).correlation)


def main():
    # input-independent (synthetic explanations); --run only namespaces the results folder
    out = results_dir(get_run(), "p1_calibration")
    sb, anti, rb = run_score_based(), run_anti_oracle(), run_rule_based()
    sb.to_csv(out / "score_based_raw.csv", index=False)
    rb.to_csv(out / "rule_based_raw.csv", index=False)

    rho_t2 = _spearman_by_k(sb, "tier2")
    rho_t1 = _spearman_by_k(sb, "tier1")
    rho_rb = _spearman_by_k(rb, "tier1")
    oracle_t2 = sb[sb.p == 0.0].groupby("k")["tier2"].mean()
    random_t2 = sb[sb.p == 1.0].groupby("k")["tier2"].mean()
    anti_t2 = anti.groupby("k")["tier2"].mean()
    spread_t2 = oracle_t2 - random_t2
    spread_t1 = sb[sb.p == 0.0].groupby("k")["tier1"].mean() - sb[sb.p == 1.0].groupby("k")["tier1"].mean()

    summary = pd.DataFrame({
        "k": CARDS,
        "spearman_tier2": [rho_t2.get(k, np.nan) for k in CARDS],
        "spearman_tier1": [rho_t1.get(k, np.nan) for k in CARDS],
        "spearman_rule_tier1": [rho_rb.get(k, np.nan) for k in CARDS],
        "oracle_tier2": [oracle_t2.get(k, np.nan) for k in CARDS],
        "random_tier2": [random_t2.get(k, np.nan) for k in CARDS],
        "anti_tier2": [anti_t2.get(k, np.nan) for k in CARDS],
        "spread_tier2": [spread_t2.get(k, np.nan) for k in CARDS],
        "spread_tier1": [spread_t1.get(k, np.nan) for k in CARDS],
    })
    summary.to_csv(out / "summary.csv", index=False)

    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print("\n=== P1 calibration summary (per cardinality k) ===")
    print(summary.to_string(index=False))
    print("\n--- Acceptance ---")
    print(f"[monotone] Tier-2 Spearman(p,XFA) <= -0.95 for all k : "
          f"{bool((summary.spearman_tier2 <= -0.95).all())}  (min {summary.spearman_tier2.min():.3f})")
    print(f"[monotone] rule Tier-1 Spearman(p,XFA) <= -0.95 all k : "
          f"{bool((summary.spearman_rule_tier1 <= -0.95).all())}  (min {summary.spearman_rule_tier1.min():.3f})")
    ok_order = bool((oracle_t2 > random_t2).all() and (random_t2.reindex(anti_t2.index) > anti_t2).all())
    print(f"[anchors]  oracle > random > anti-oracle (Tier-2)      : {ok_order}")
    print(f"           mean oracle/random/anti Tier-2 = {oracle_t2.mean():.3f} / {random_t2.mean():.3f} / {anti_t2.mean():.3f}")
    print(f"[sensitiv] Tier-2 spread > Tier-1 spread for all k     : "
          f"{bool((spread_t2 > spread_t1).all())}  (mean T2 {spread_t2.mean():.3f} vs T1 {spread_t1.mean():.3f})")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
