# archive/

Superseded scripts kept for reference. These are **not** part of the live pipeline and are not
imported by any live module. They were moved here during the Track-1 cleanup so the active code is
easier to read; nothing here is expected to run as-is.

| Archived file | Superseded by (live) |
|---|---|
| `start.py`, `start_v1.py`, `start_v2.py` | `src/run_xai_benchmark_v2.py` |
| `DataSetGen.py`, `DataSetGen_v1.py`, `DataSetGen_v2.py` | `src/dataset_generation/DataSetGen_v4.py` |
| `DataSetGen_v3.py` | `src/dataset_generation/DataSetGen_v4.py` |
| `Evaluation.py` | `src/Evaluation_v2.py` |
| `Evaluation_plots.py` | `src/Evaluation_plots_v2.py` |
| `Explaination.py` | `src/Explaination_v2.py` |
| `run_xai_benchmark.py` | `src/run_xai_benchmark_v2.py` |
| `Other_graphs.py` | — (standalone exploratory plots, no live importer) |

`run_xai_benchmark.py` and `DataSetGen_v3.py` were previously kept as a transitive dependency:
`run_xai_benchmark_v2.py` imported `DEFAULT_DATASETS_SEQUENCES` from the former, which imported the
latter. That import was dead — the v2 driver redefines the same constant further down the module, so
the imported value was always shadowed. Removing the import severed the chain and freed both files.
