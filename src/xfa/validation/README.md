# XFA validation battery

A construct-validity battery for the XFA fidelity metric. Each module is one validity pillar.

## Running

From `src/` with the SHAP/LIME environment (numpy >= 2, has sklearn/scipy/xgboost):

```
cd src
../.venv_xfa/Scripts/python.exe -m xfa.validation.p3_discriminant             # default input (1K_v2)
../.venv_xfa/Scripts/python.exe -m xfa.validation.p3_discriminant --run 1K_v2
```

## Inputs and outputs

A **run** names one validation input: the benchmark suite (datasets, metadata, models, AUC summary)
plus the XFA output directories produced on it. Runs are registered in `RUNS` in `_common.py`; to
validate a new XFA run (another suite, model family or budget), add one `RunConfig` entry:

```python
RUNS["1K_v2_rf"] = RunConfig(
    name="1K_v2_rf",
    suite=RES / "SynXAI-DB_run_49feat_v2_rf",
    xfa_dirs=(RES / "SynXAI-DB_run_49feat_v2_rf__xfa_2026-xx-xx__shap-lime", ...),
)
```

Every pillar writes its tables to `results/<run>/<pillar>/` (regenerable; not tracked), so different
inputs never overwrite each other.

| Module | Pillar | What it checks | Uses the model? |
|---|---|---|---|
| `p1_calibration`  | Diagnosticity / calibration | XFA is monotone in true fidelity (synthetic explanations of graded fidelity; input-independent) | no |
| `p2_convergent`   | Convergent validity | XFA agrees with an independent insertion / comprehensiveness metric | yes |
| `p3_discriminant` | Discriminant validity | XFA separates the explainers with the model AUC held constant | no |
| `p4_meta`         | Meta-evaluation | XFA is resilient to noise yet reactive to an adversarial explanation | no |
| `p5_reliability`  | Reliability | the SHAP > LIME > LORE > Anchor ranking is stable under resampling | no |
| `p6_content`      | Content validity | the components are non-redundant and cover distinct fidelity facets | no |

`_common.py` holds the run registry, shared loaders and helpers (all paths derived from the file
location, so the battery is portable).
