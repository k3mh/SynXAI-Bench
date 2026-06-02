"""Pillar 5 - reliability (determinism + rank stability).

The metric is a pure function of the explanation (``xfa.metrics`` has no RNG), so it is deterministic;
the run is seed=42 end-to-end. What remains to check is whether the SHAP > LIME > LORE > Anchor ranking
survives resampling. Two bootstraps: instance-level (resample each tool's instances) and, more
conservatively, config-level (resample the 11 configs).

Hyperparameter sensitivity (Tier-1 threshold, SHAP background, LIME sample count, anchor precision) is a
separate leg that needs explainer re-runs and is not covered here.
"""
import numpy as np
import pandas as pd

from ._common import TOOLS, get_run, load_per_instance, results_dir

N_BOOT = 2000
RNG = np.random.default_rng(42)


def main():
    run = get_run()
    out = results_dir(run, "p5_reliability")
    df = load_per_instance(run, cols=["tool", "rank", "tier1"])
    arrs = {t: df[df["tool"] == t]["tier1"].dropna().values for t in TOOLS}
    per_cfg = {t: df[df["tool"] == t].groupby("rank")["tier1"].mean() for t in TOOLS}

    # instance-level bootstrap
    full_i = shaptop_i = anchbot_i = 0
    for _ in range(N_BOOT):
        means = {t: arrs[t][RNG.integers(0, len(arrs[t]), len(arrs[t]))].mean() for t in TOOLS}
        order = sorted(TOOLS, key=lambda t: means[t], reverse=True)
        full_i += order == TOOLS
        shaptop_i += order[0] == "shap"
        anchbot_i += order[-1] == "anchor"

    # config-level block bootstrap (resample the 11 configs)
    full_c = 0
    for _ in range(N_BOOT):
        chosen = RNG.integers(1, 12, 11)
        means = {t: np.mean([per_cfg[t][c] for c in chosen]) for t in TOOLS}
        full_c += sorted(TOOLS, key=lambda t: means[t], reverse=True) == TOOLS

    point = {t: arrs[t].mean() for t in TOOLS}
    pd.DataFrame(per_cfg).to_csv(out / "per_config_tool_tier1.csv")
    print("=== P5 reliability ===")
    print("point Tier-1 (overall): " + " | ".join(f"{t} {point[t]:.3f}" for t in TOOLS))
    print("(a) determinism: metrics.py is RNG-free -> XFA is a pure function of the explanation; run seed=42.")
    print("(b) rank stability (SHAP>LIME>LORE>Anchor):")
    print(f"    instance bootstrap : full order {100*full_i/N_BOOT:.1f}%  | SHAP top {100*shaptop_i/N_BOOT:.1f}%"
          f"  | Anchor bottom {100*anchbot_i/N_BOOT:.1f}%")
    print(f"    config  bootstrap  : full order {100*full_c/N_BOOT:.1f}%  (conservative, resamples 11 configs)")
    print("\n--- Acceptance ---")
    print(f"[reliab] full ranking holds >= 95% (instance bootstrap): {bool(100*full_i/N_BOOT >= 95)}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
