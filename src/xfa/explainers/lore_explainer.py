"""LORE (lore_sa >= 1.1.1) wrapper -> rule feature set. Rule-based (Tier-1 only).

Assembles the lore_sa pipeline once (bbox + dataset + encoder + genetic generator + decision-tree
surrogate); per instance, returns the set of features appearing in the local rule's premises
(`exp["rule"]["premises"][i]["attr"]`). Self-contained from (model, training_data): the dataset's
categorical target is the model's own predictions. Per-instance seeding (deap uses the stdlib
``random`` module + numpy) makes the genetic neighborhood deterministic regardless of call order.

NOTE: lore_sa 1.1.1 pins numpy<2, which conflicts with the SHAP stack — run LORE in an environment
where that is acceptable (LORE needs neither SHAP nor numpy>=2).
"""
import random

import numpy as np

from .base import XaiExplainer, XaiOutput


class _StrModel:
    """Expose string-label predict so lore_sa's (string) target encoder matches the bbox output."""
    def __init__(self, model):
        self._m = model
        self.classes_ = np.array(["0", "1"])

    def predict(self, X):
        return self._m.predict(np.asarray(X)).astype(str)

    def predict_proba(self, X):
        return self._m.predict_proba(np.asarray(X))


class LoreExplainer(XaiExplainer):
    name = "lore"
    is_score_based = False

    def __init__(self, model, training_data, feature_names, seed: int = 42):
        import pandas as pd
        from lore_sa.bbox import sklearnBBox
        from lore_sa.dataset import TabularDataset
        from lore_sa.encoder_decoder import ColumnTransformerEnc
        from lore_sa.neighgen.genetic import GeneticGenerator
        from lore_sa.surrogate import DecisionTreeSurrogate
        from lore_sa.lore import Lore

        self._feats = list(feature_names)
        # Full training data (no subsample) for an honest test; only parallelism is used for speed.
        X = np.asarray(training_data, dtype=float)
        df = pd.DataFrame(X, columns=self._feats)
        df["__target__"] = model.predict(X).astype(str)   # categorical target = model predictions
        dataset = TabularDataset(df, class_name="__target__")
        bbox = sklearnBBox(_StrModel(model))
        enc = ColumnTransformerEnc(dataset.descriptor)
        gen = GeneticGenerator(bbox, dataset, enc, random_seed=seed)
        self._lore = Lore(bbox, dataset, enc, gen, DecisionTreeSurrogate())

    def explain(self, x_row, seed=None) -> XaiOutput:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        exp = self._lore.explain(np.asarray(x_row, dtype=float))
        premises = exp.get("rule", {}).get("premises", []) if isinstance(exp, dict) else []
        feat_set = {self._feats.index(p["attr"]) for p in premises if p.get("attr") in self._feats}
        return XaiOutput(is_score_based=False, feature_set=feat_set)
