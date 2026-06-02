"""Formal validation battery for the XFA fidelity metric.

Each module is one validity pillar, runnable standalone from ``src`` with the SHAP/LIME environment:

    python -m xfa.validation.p1_calibration

    p1_calibration   diagnosticity / calibration        p2_convergent    convergent validity (AFR-licensed)
    p3_discriminant  discriminant validity              p4_meta          meta-evaluation (noise / adversary)
    p5_reliability   reliability / rank stability        p6_content       content validity

Inputs are selected with ``--run <name>`` (registered in ``_common.RUNS``; default ``1K_v2``); each
pillar writes its tables to ``results/<run>/<pillar>/`` (regenerable, not tracked), so different
inputs never overwrite each other. Shared loaders and helpers live in ``_common``.
"""
