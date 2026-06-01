"""
Guard tests for the ground-truth generation logic in DataSetGen_v4.

This logic is load-bearing for both the SynXAI-Data paper and the downstream XFA
component, so these light checks protect the two properties that recently broke:
  1. every rule's declared ground-truth features (imp_vars) exist as data columns
     (regression guard for the NUM_TOTAL_FEATURES / dropped-x49 class of bug);
  2. generation is reproducible (same dataset bytes across runs);
  3. the data has the expected x1..x49 + y shape with well-formed RGS labels.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import dataset_generation.DataSetGen_v4 as sdg  # noqa: E402

ALL_DS = [f"ds{i}" for i in range(0, 13)]
SIZE = 300


@pytest.mark.parametrize("name", ALL_DS)
def test_imp_vars_are_real_columns(name):
    """Declared ground-truth features must exist in the generated data."""
    ds = sdg.generate_dataset_by_name(name, SIZE)
    cols = set(ds.data.columns)
    imp = ds.meta_data["imp_vars"].iloc[0]
    gt = {f for f in imp if f}  # RGS0 uses [""]; drop the empty marker
    missing = gt - cols
    assert not missing, f"{name}: ground-truth features missing from data: {sorted(missing)}"


@pytest.mark.parametrize("name", ALL_DS)
def test_generation_is_reproducible(name):
    """Same generator + size must produce byte-identical data across runs."""
    a = sdg.generate_dataset_by_name(name, SIZE).data
    b = sdg.generate_dataset_by_name(name, SIZE).data
    assert np.array_equal(a.values, b.values), f"{name}: data differs across runs"


@pytest.mark.parametrize("name", ALL_DS)
def test_shape_and_rgs(name):
    """Data is x1..x49 + y; RGS metadata is a single well-formed group per dataset."""
    ds = sdg.generate_dataset_by_name(name, SIZE)
    expected_cols = sdg.ALL_FEATURE_NAMES + ["y"]
    assert list(ds.data.columns) == expected_cols
    assert ds.data.shape == (SIZE, sdg.NUM_TOTAL_FEATURES + 1)
    rgs = ds.meta_data["RGS"].unique()
    assert len(rgs) == 1 and str(rgs[0]).startswith("RGS")


def test_feature_space_is_49():
    """The rule library spans exactly x1..x49 (RR12 uses x49)."""
    assert sdg.NUM_TOTAL_FEATURES == 49
    assert sdg.ALL_FEATURE_NAMES[-1] == "x49"
