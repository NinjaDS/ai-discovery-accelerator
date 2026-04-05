"""
tests/test_data_profiler.py — Unit tests for DataProfiler
"""

import pytest
import pandas as pd
import numpy as np

from core.data_loader import DataLoader
from core.agents.data_profiler import DataProfiler


@pytest.fixture(scope="module")
def churn_df():
    loader = DataLoader(seed=42)
    return loader.load_sample_data("churn_prediction")


@pytest.fixture(scope="module")
def churn_profile(churn_df):
    profiler = DataProfiler()
    return profiler.profile(churn_df, source_name="churn_test")


class TestDataProfilerBasics:
    def test_profile_returns_dict(self, churn_profile):
        assert isinstance(churn_profile, dict)

    def test_required_keys(self, churn_profile):
        required = {
            "source_name", "n_rows", "n_cols", "memory_mb",
            "overall_null_pct", "column_profiles", "warnings",
            "type_distribution",
        }
        assert required.issubset(set(churn_profile.keys()))

    def test_row_count(self, churn_df, churn_profile):
        assert churn_profile["n_rows"] == len(churn_df)

    def test_col_count(self, churn_df, churn_profile):
        assert churn_profile["n_cols"] == len(churn_df.columns)

    def test_column_profiles_all_columns(self, churn_df, churn_profile):
        assert set(churn_profile["column_profiles"].keys()) == set(churn_df.columns)

    def test_overall_null_pct_is_float(self, churn_profile):
        assert isinstance(churn_profile["overall_null_pct"], float)
        assert 0.0 <= churn_profile["overall_null_pct"] <= 100.0

    def test_memory_mb_positive(self, churn_profile):
        assert churn_profile["memory_mb"] > 0


class TestColumnProfiles:
    def test_tenure_months_is_numeric(self, churn_profile):
        col = churn_profile["column_profiles"]["tenure_months"]
        assert col["inferred_type"] in ("numeric_continuous", "numeric_discrete")

    def test_churn_is_boolean_or_discrete(self, churn_profile):
        col = churn_profile["column_profiles"]["churn"]
        assert col["inferred_type"] in ("boolean", "numeric_discrete")

    def test_customer_id_has_high_unique_ratio(self, churn_profile):
        col = churn_profile["column_profiles"]["customer_id"]
        assert col["unique_ratio"] > 0.9

    def test_numeric_cols_have_stats(self, churn_profile):
        for col_name in ("age", "monthly_charge", "tenure_months"):
            col = churn_profile["column_profiles"][col_name]
            if col["inferred_type"] in ("numeric_continuous", "numeric_discrete"):
                assert "stats" in col
                assert "mean" in col["stats"]

    def test_null_pct_in_total_charges(self, churn_profile):
        # DataLoader inserts ~2% nulls in total_charges
        col = churn_profile["column_profiles"]["total_charges"]
        assert col["null_pct"] >= 0.0

    def test_type_distribution_keys(self, churn_profile):
        td = churn_profile["type_distribution"]
        assert isinstance(td, dict)
        total = sum(td.values())
        assert total == churn_profile["n_cols"]


class TestWarnings:
    def test_warnings_is_list(self, churn_profile):
        assert isinstance(churn_profile["warnings"], list)

    def test_warnings_are_strings(self, churn_profile):
        for w in churn_profile["warnings"]:
            assert isinstance(w, str)
