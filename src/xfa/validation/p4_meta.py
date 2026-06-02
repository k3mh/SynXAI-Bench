"""Pillar 4 - meta-evaluation (MetaQuantus-style).

Validates XFA as a reliable instrument by perturbing the explanation and checking the metric responds
correctly. Self-implemented (MetaQuantus is torch/TF-first; cited for the protocol). Two perturbations:

  - noise (meaning-preserving): attr + small Gaussian -> a reliable metric barely moves (resilience).
  - adversary (meaning-destroying): permute the attribution across features (keeps the value distribution
    but destroys the feature -> importance mapping) -> a reliable metric must drop (reactivity).

Reliable = resilient to noise AND reactive to adversary AND reaction >> drift. This argument never invokes
the "model-uses-GT" (AFR) premise, so it is the AFR-independent leg (holds even if the AFR paper is
revised). XFA still uses GT to score; only the validity argument is AFR-free.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xfa import relevance
from xfa.metrics import score_set, score_vector
from xfa.runner import load_dataset

from ._common import get_run, load_explanations, parse_vector, results_dir

M_PER_CFG = 200
NOISE_ALPHA = 0.05          # noise std = 5% of the max |attribution| (meaning-preserving)
RNG = np.random.default_rng(42)


def tiers(attr, mem, graded, n):
    d = score_vector(attr, mem, graded, n)
    return d["tier1"], d["tier2"]


def main():
    run = get_run()
    out = results_dir(run, "p4_meta")
    expl = load_explanations(run)

    vrows, srows = [], []
    for cid in range(1, 12):
        X, y, meta, feats = load_dataset(run.suite, cid)
        n = len(feats)
        name_to_idx = {f: i for i, f in enumerate(feats)}
        e_cid = expl[expl["rank"] == cid]
        for tool in ("shap", "lime", "lore", "anchor"):
            et = e_cid[e_cid["tool"] == tool]
            if len(et) == 0:
                continue
            for _, r in et.sample(n=min(M_PER_CFG, len(et)), random_state=int(cid)).iterrows():
                inst = int(r["instance"])
                if inst not in meta.index:
                    continue
                imp = meta.loc[inst, "imp_vars"]
                mem = relevance.gt_membership(imp, feats)
                graded = relevance.rank_relevance(imp, feats)
                gt = {int(i) for i in np.where(mem > 0)[0]}
                if not gt:
                    continue
                if tool in ("shap", "lime"):
                    attr = parse_vector(r["raw"])
                    if attr.size != n:
                        continue
                    t1, t2 = tiers(attr, mem, graded, n)
                    sd = NOISE_ALPHA * max(float(np.abs(attr).max()), 1e-9)
                    t1n, t2n = tiers(np.abs(attr + RNG.normal(0, sd, n)), mem, graded, n)   # noise
                    t1a, t2a = tiers(RNG.permutation(attr), mem, graded, n)                 # adversary
                    vrows.append({"tool": tool, "t1": t1, "t2": t2, "t1_noise": t1n, "t2_noise": t2n,
                                  "t1_adv": t1a, "t2_adv": t2a})
                else:  # rule-based Tier-1 set: adversary = random same-size set (no natural "noise")
                    S = {name_to_idx[s] for s in str(r["raw"]).split("|") if s in name_to_idx}
                    if not S:
                        continue
                    R = set(int(i) for i in RNG.choice(n, size=len(S), replace=False))
                    srows.append({"tool": tool, "t1": score_set(S, gt, n)["tier1"],
                                  "t1_adv": score_set(R, gt, n)["tier1"]})
        print(f"[cfg {cid}] done", flush=True)

    v = pd.DataFrame(vrows)
    s = pd.DataFrame(srows)
    v.to_csv(out / "score_based.csv", index=False)
    s.to_csv(out / "rule_based.csv", index=False)

    def report(df, orig, pert, label):
        rho = spearmanr(df[orig], df[pert]).correlation
        mad = float((df[pert] - df[orig]).abs().mean())
        drop = float((df[orig] - df[pert]).mean())
        frac = float((df[pert] < df[orig]).mean())
        print(f"  {label:26s} rho={rho:.3f}  mean_abs_change={mad:.3f}  mean_drop={drop:+.3f}  frac(pert<orig)={frac:.2f}")
        return rho, mad, drop, frac

    print("\n=== P4 meta-evaluation (score-based, Tier-2 primary) ===")
    print("-- noise resilience (want rho~1, small change) --")
    nr_rho, nr_mad, _, _ = report(v, "t2", "t2_noise", "Tier-2 noise (overall)")
    report(v, "t1", "t1_noise", "Tier-1 noise (overall)")
    print("-- adversary reactivity (want big drop, frac~1, low rho) --")
    ar_rho, ar_mad, ar_drop, ar_frac = report(v, "t2", "t2_adv", "Tier-2 adversary (overall)")
    report(v, "t1", "t1_adv", "Tier-1 adversary (overall)")
    for tool in ("shap", "lime"):
        d = v[v["tool"] == tool]
        print(f"  [{tool}] T2 noise rho={spearmanr(d['t2'], d['t2_noise']).correlation:.3f} | "
              f"T2 adv drop={(d['t2'] - d['t2_adv']).mean():+.3f} frac={(d['t2_adv'] < d['t2']).mean():.2f}")

    print("\n=== rule-based Tier-1 adversary (Anchor/LORE) ===")
    for tool in ("lore", "anchor"):
        d = s[s["tool"] == tool]
        if len(d):
            print(f"  {tool}: drop={(d['t1'] - d['t1_adv']).mean():+.3f}  frac(adv<orig)={(d['t1_adv'] < d['t1']).mean():.2f}")

    ratio = ar_drop / (nr_mad + 1e-9)
    print("\n--- Acceptance ---")
    print(f"[PRIMARY] Tier-2 noise-resilient (rho>=0.95): {bool(nr_rho >= 0.95)}  (rho={nr_rho:.3f})")
    print(f"[PRIMARY] Tier-2 adversary-reactive (frac>=0.90): {bool(ar_frac >= 0.90)}  (frac={ar_frac:.2f}, drop={ar_drop:+.3f})")
    print(f"[PRIMARY] reaction >> drift (drop/change >= 5): {bool(ratio >= 5)}  (ratio={ratio:.1f})")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
