"""Pillar 2 - convergent validity (AFR-licensed).

Cross-checks XFA against an independent faithfulness metric that uses neither the ground truth nor the
XFA code - only the model's forward passes. This is what answers the "self-referential" objection. The
comparator is self-implemented (Quantus is torch/TF-first; we cite it for the definitions):

  - insertion / deletion (RISE-style): rank features by |attribution|; insertion reveals them
    baseline -> true most-important-first, deletion removes them true -> baseline. A faithful explanation
    puts the truly influential features first -> high insertion probability. (On tabular tree models
    deletion-to-baseline is noisy / off-manifold, so insertion is the primary instance-level comparator.)
  - size-normalized comprehensiveness (ERASER-style): comp(S) minus the mean comp of random same-size
    sets, so tools that flag more features (Anchor ~15 vs SHAP ~4) are compared on flagging the RIGHT
    features, not how many.

P2a (instance-level, SHAP/LIME): Spearman(XFA Tier-2, insertion faithfulness).
P2b (tool-level, all four): does norm-comprehensiveness rank the tools like XFA, and do the per-config
    tool means correlate?

AFR licenses the interpretation (the model relies on the GT features, so perturbation-faithfulness ~
GT-recovery); the correlation itself stands regardless. No GT is used here.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xfa.metrics import vector_to_set
from xfa.runner import load_dataset, load_model

from ._common import TOOLS, get_run, load_explanations, load_per_instance, parse_vector, results_dir

M_PER_CFG = 150            # instances per (tool, config)
N_RANDSETS = 3            # random same-size sets for the comprehensiveness baseline
RNG = np.random.default_rng(42)


def ins_del(model, x, baseline, attr, c):
    n = len(x)
    order = np.argsort(-np.abs(attr))
    Xins = np.repeat(baseline[None, :], n + 1, axis=0)
    Xdel = np.repeat(x[None, :], n + 1, axis=0)
    for j, f in enumerate(order):
        Xins[j + 1:, f] = x[f]
        Xdel[j + 1:, f] = baseline[f]
    return float(model.predict_proba(Xins)[:, c].mean()), float(model.predict_proba(Xdel)[:, c].mean())


def _mask_prob(model, x, baseline, idx, c):
    xm = x.copy(); xm[list(idx)] = baseline[list(idx)]
    return float(model.predict_proba(xm[None, :])[:, c][0])


def comp_and_norm(model, x, baseline, flagged, c, p_full, n):
    """Comprehensiveness of the flagged set, and its size-normalized version (minus a random same-size set)."""
    if not flagged:
        return 0.0, 0.0
    comp = p_full - _mask_prob(model, x, baseline, flagged, c)
    rand = [p_full - _mask_prob(model, x, baseline, RNG.choice(n, len(flagged), replace=False), c)
            for _ in range(N_RANDSETS)]
    return float(comp), float(comp - np.mean(rand))


def main():
    run = get_run()
    out = results_dir(run, "p2_convergent")
    expl = load_explanations(run)
    xfa = load_per_instance(run, cols=["tool", "rank", "instance", "tier1", "tier2"])

    di_rows, comp_rows = [], []
    for cid in range(1, 12):
        X, y, meta, feats = load_dataset(run.suite, cid)
        model = load_model(run.suite, cid)
        baseline = X.values.astype(float).mean(axis=0)
        name_to_idx = {f: i for i, f in enumerate(feats)}
        n = len(feats)
        e_cid = expl[expl["rank"] == cid]
        for tool in TOOLS:
            et = e_cid[e_cid["tool"] == tool]
            if len(et) == 0:
                continue
            for _, r in et.sample(n=min(M_PER_CFG, len(et)), random_state=int(cid)).iterrows():
                inst = int(r["instance"])
                if inst not in X.index:
                    continue
                x = X.loc[inst].values.astype(float)
                c = int(model.predict(x[None, :])[0])
                p_full = float(model.predict_proba(x[None, :])[:, c][0])
                if tool in ("shap", "lime"):
                    attr = parse_vector(r["raw"])
                    if attr.size != n:
                        continue
                    flagged = vector_to_set(attr, n)
                else:
                    flagged = {name_to_idx[s] for s in str(r["raw"]).split("|") if s in name_to_idx}
                comp, ncomp = comp_and_norm(model, x, baseline, flagged, c, p_full, n)
                comp_rows.append({"tool": tool, "rank": cid, "instance": inst,
                                  "comp": comp, "norm_comp": ncomp, "set_size": len(flagged)})
                if tool in ("shap", "lime"):
                    ins, dele = ins_del(model, x, baseline, attr, c)
                    di_rows.append({"tool": tool, "rank": cid, "instance": inst,
                                    "insertion": ins, "deletion": dele, "faith_delins": ins - dele})
        print(f"[cfg {cid}] done", flush=True)

    di = pd.DataFrame(di_rows).merge(xfa, on=["tool", "rank", "instance"], how="left")
    comp = pd.DataFrame(comp_rows)
    di.to_csv(out / "delins_instance.csv", index=False)
    comp.to_csv(out / "comprehensiveness_instance.csv", index=False)

    # P2a instance-level (primary comparator = insertion; deletion / (ins-del) reported for transparency)
    print("\n=== P2a instance-level convergent: Spearman(XFA Tier-2, independent faithfulness) ===")
    for tool in ("shap", "lime"):
        d = di[di["tool"] == tool].dropna(subset=["tier2"])
        print(f"  {tool} (n={len(d)}): insertion rho={spearmanr(d['tier2'], d['insertion']).correlation:.3f}"
              f" | deletion rho={spearmanr(d['tier2'], d['deletion']).correlation:.3f}"
              f" | (ins-del) rho={spearmanr(d['tier2'], d['faith_delins']).correlation:.3f}")
    dall = di.dropna(subset=["tier2"])
    rho_ins = spearmanr(dall["tier2"], dall["insertion"]).correlation

    # P2b tool-level (size-normalized comprehensiveness vs XFA)
    ncomp_tool = comp.groupby("tool")["norm_comp"].mean().reindex(TOOLS)
    comp_tool = comp.groupby("tool")["comp"].mean().reindex(TOOLS)
    size_tool = comp.groupby("tool")["set_size"].mean().reindex(TOOLS)
    ncomp_order = list(ncomp_tool.sort_values(ascending=False).index)

    ncomp_cfg = comp.groupby(["rank", "tool"])["norm_comp"].mean().reset_index()
    xfa_cfg = xfa.groupby(["rank", "tool"])["tier1"].mean().reset_index()
    paired = ncomp_cfg.merge(xfa_cfg, on=["rank", "tool"], how="inner")
    rho_tool = spearmanr(paired["norm_comp"], paired["tier1"]).correlation

    print("\n=== P2b tool-level convergent: size-normalized comprehensiveness vs XFA ===")
    print("            " + " | ".join(TOOLS))
    print("mean set size " + " | ".join(f"{size_tool[t]:.1f}" for t in TOOLS))
    print("raw comp      " + " | ".join(f"{comp_tool[t]:.3f}" for t in TOOLS))
    print("norm comp     " + " | ".join(f"{ncomp_tool[t]:.3f}" for t in TOOLS))
    print(f"norm-comp tool order : {ncomp_order}   (XFA order: {TOOLS})")
    print(f"paired per-config corr (norm-comp vs XFA Tier-1), n={len(paired)}: Spearman {rho_tool:.3f}")

    print("\n--- Acceptance (primary comparator = insertion) ---")
    print(f"[PRIMARY] instance-level Spearman(XFA T2, insertion) > 0.5: {bool(rho_ins > 0.5)}  (rho={rho_ins:.3f})")
    print(f"[confirm] norm-comp ranks SHAP>LIME>LORE>Anchor like XFA : {bool(ncomp_order == TOOLS)}")
    print(f"[confirm] tool-level paired corr > 0.5                   : {bool(rho_tool > 0.5)}  (rho={rho_tool:.3f})")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
