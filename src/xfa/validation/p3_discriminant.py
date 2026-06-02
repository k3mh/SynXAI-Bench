"""Pillar 3 - discriminant validity.

XFA is distinct from - though correlated with - model learnability (AUC). The headline is the fixed-AUC
test: all four explainers score the same model on the same dataset, so a config's AUC is identical across
them, yet XFA separates them. AUC therefore cannot explain the separation (its within-config variance is
zero). Discriminant here means non-redundant / dissociable, not zero-correlation - XFA legitimately
co-moves with AUC between configs.

  1. Per-config tool means with bootstrap CIs (Tier-1, common to all four tools).
  2. Friedman test (treatments = tools, blocks = configs): a tool effect with config blocked is a tool
     effect independent of AUC.
  3. Ranking consistency across the 11 fixed-AUC configs.
  4. Variance decomposition of the 11x4 mean matrix: config (~complexity/AUC) vs tool (~fidelity) vs
     interaction. The tool component is XFA's AUC-invisible signal.
  5. Shared backbone: between-config correlation of mean XFA with AUC.
"""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr

from ._common import TOOLS, bootstrap_ci, get_run, load_auc_by_rank, load_per_instance, results_dir


def main():
    run = get_run()
    out = results_dir(run, "p3_discriminant")
    inst = load_per_instance(run, cols=["tool", "rank", "instance", "tier1", "tier2"])
    auc = load_auc_by_rank(run)
    rng = np.random.default_rng(42)

    rows = []
    for rank in range(1, 12):
        for tool in TOOLS:
            x = inst[(inst["rank"] == rank) & (inst["tool"] == tool)]["tier1"].dropna().values
            if len(x) == 0:
                continue
            m, lo, hi = bootstrap_ci(x, rng=rng)
            rows.append({"rank": rank, "auc": float(auc[rank]), "tool": tool,
                         "tier1_mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(x)})
    pc = pd.DataFrame(rows)
    pc.to_csv(out / "per_config_tool_tier1.csv", index=False)
    M = pc.pivot(index="rank", columns="tool", values="tier1_mean")[TOOLS]

    # fixed-AUC tool effect
    fr = friedmanchisquare(*[M[t].values for t in TOOLS])
    full_hold = int(sum(list(M.loc[r].sort_values(ascending=False).index) == TOOLS for r in M.index))
    shap_top = int((M.idxmax(axis=1) == "shap").sum())
    anchor_bot = int((M.idxmin(axis=1) == "anchor").sum())

    # variance decomposition of the 11x4 mean matrix
    Mv = M.values
    grand = Mv.mean()
    ce, te = Mv.mean(axis=1) - grand, Mv.mean(axis=0) - grand
    resid = Mv - grand - ce[:, None] - te[None, :]
    ss_c, ss_t, ss_r = Mv.shape[1] * (ce ** 2).sum(), Mv.shape[0] * (te ** 2).sum(), (resid ** 2).sum()
    ss_tot = ss_c + ss_t + ss_r
    pct_c, pct_t, pct_r = 100 * ss_c / ss_tot, 100 * ss_t / ss_tot, 100 * ss_r / ss_tot

    # shared backbone
    cfg_mean = M.mean(axis=1)
    aucs = np.array([auc[r] for r in M.index])
    pear = float(np.corrcoef(cfg_mean.values, aucs)[0, 1])
    spear = spearmanr(cfg_mean.values, aucs).correlation

    overall = {t: bootstrap_ci(inst[inst["tool"] == t]["tier1"].dropna().values, rng=rng) for t in TOOLS}
    t2 = {t: bootstrap_ci(inst[inst["tool"] == t]["tier2"].dropna().values, rng=rng) for t in ("shap", "lime")}

    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    show = M.copy(); show.insert(0, "AUC", aucs)
    print("\n=== P3 per-config Tier-1 means (fixed AUC within each row) ===")
    print(show.to_string())
    print("\n--- Fixed-AUC separation (Friedman: tools as treatments, configs as blocks) ---")
    print(f"Friedman chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.2e}  "
          f"-> tool effect {'IS' if fr.pvalue < 0.05 else 'NOT'} significant with AUC held constant")
    print("overall Tier-1 (mean [95% CI]): " +
          " | ".join(f"{t} {overall[t][0]:.3f} [{overall[t][1]:.3f},{overall[t][2]:.3f}]" for t in TOOLS))
    print(f"SHAP vs LIME Tier-2: shap {t2['shap'][0]:.3f} [{t2['shap'][1]:.3f},{t2['shap'][2]:.3f}]  "
          f"lime {t2['lime'][0]:.3f} [{t2['lime'][1]:.3f},{t2['lime'][2]:.3f}]")
    print("\n--- Ranking consistency across 11 fixed-AUC configs ---")
    print(f"full SHAP>LIME>LORE>Anchor: {full_hold}/11 | SHAP top: {shap_top}/11 | Anchor bottom: {anchor_bot}/11")
    print("\n--- Variance decomposition of XFA (11x4 mean matrix) ---")
    print(f"config(~complexity/AUC): {pct_c:.1f}%   tool(~fidelity, AUC-invisible): {pct_t:.1f}%   interaction: {pct_r:.1f}%")
    print("\n--- Shared backbone (between-config) ---")
    print(f"corr(mean XFA, AUC): Pearson {pear:.3f} | Spearman {spear:.3f}")
    print("\n--- Acceptance ---")
    print(f"[PRIMARY] tool effect significant at fixed AUC (Friedman p<0.05): {bool(fr.pvalue < 0.05)}")
    print(f"[PRIMARY] SHAP top & Anchor bottom in all 11 configs: {bool(shap_top == 11 and anchor_bot == 11)}")
    print(f"[confirm] tool variance component >= 15%: {bool(pct_t >= 15)}  ({pct_t:.1f}%)")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
