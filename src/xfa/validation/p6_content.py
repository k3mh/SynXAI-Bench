"""Pillar 6 - content validity.

Shows XFA's components cover distinct fidelity facets without redundancy, from the real SHAP/LIME
per-instance components. Two pieces of evidence:
  (a) the component correlation matrix is not saturated (no pair with |r| > 0.95) -> nothing redundant;
  (b) recall and (1 - FPR) dissociate -> over-attribution (finds the ground truth but also flags noise)
      and under-attribution are distinct, tool-dependent failure modes, so both Tier-1 components matter.

Facet map: recall = coverage of relevant; one_minus_fpr = avoidance of irrelevant; ap_norm = ranking of
relevant mass; auc_roc_norm = separation; ndcg = graded ranking; relevance_mass = concentration on GT.
"""
import numpy as np

from ._common import get_run, load_score_instances, results_dir

COMPONENTS = ["recall", "one_minus_fpr", "ap_norm", "auc_roc_norm", "ndcg", "relevance_mass"]


def main():
    run = get_run()
    out = results_dir(run, "p6_content")
    df = load_score_instances(run)

    redundant_any = False
    for tool in ("shap", "lime"):
        d = df[df["tool"] == tool]
        corr = d[COMPONENTS].corr()
        corr.to_csv(out / f"component_corr_{tool}.csv")
        off = corr.where(~np.eye(len(COMPONENTS), dtype=bool))
        n_red = int((np.abs(off.values) > 0.95).sum() // 2)
        redundant_any = redundant_any or n_red > 0
        print(f"\n=== {tool.upper()} component correlations (n={len(d)}) ===")
        print(corr.round(3).to_string())
        print(f"max |off-diagonal r| = {np.nanmax(np.abs(off.values)):.3f}   redundant pairs (|r|>0.95): {n_red}")

    print("\n--- Tier-1 dissociation (recall vs 1-FPR are distinct facets) ---")
    over_any = under_any = False
    for tool in ("shap", "lime"):
        d = df[df["tool"] == tool]
        over = float(((d["recall"] >= 0.8) & (d["one_minus_fpr"] <= 0.7)).mean())
        under = float(((d["recall"] <= 0.5) & (d["one_minus_fpr"] >= 0.95)).mean())
        over_any, under_any = over_any or over > 0, under_any or under > 0
        print(f"  {tool}: corr(recall,1-FPR)={d['recall'].corr(d['one_minus_fpr']):.3f}"
              f" | over-attribution(recall>=.8 & 1-FPR<=.7) {over*100:.1f}%"
              f" | under-attribution(recall<=.5 & 1-FPR>=.95) {under*100:.1f}%")

    print("\n--- Acceptance ---")
    print(f"[content] no component redundant (all |r| <= 0.95): {not redundant_any}")
    print(f"[content] both failure modes present across tools : {bool(over_any and under_any)}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
