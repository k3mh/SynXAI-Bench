# archive/

Superseded scripts kept for reference. These are **not** part of the live pipeline and are not
imported by any live module. They were moved here during the Track-1 cleanup so the active code is
easier to read; nothing here is expected to run as-is.

| Archived file | Superseded by (live) |
|---|---|
| `start.py`, `start_v1.py`, `start_v2.py` | `src/run_xai_benchmark_v2.py` |
| `DataSetGen.py`, `DataSetGen_v1.py`, `DataSetGen_v2.py` | `src/dataset_generation/DataSetGen_v4.py` |
| `Evaluation.py` | `src/Evaluation_v2.py` |
| `Evaluation_plots.py` | `src/Evaluation_plots_v2.py` |
| `Explaination.py` | `src/Explaination_v2.py` |

Note: `src/dataset_generation/DataSetGen_v3.py` was **left in place** (not archived) because it is a
transitive live dependency — `run_xai_benchmark_v2.py` imports a constant from
`src/run_xai_benchmark.py`, which imports `DataSetGen_v3`.
