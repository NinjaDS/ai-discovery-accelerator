"""
tests/test_feature_intelligence.py — Unit tests for FeatureIntelligence
"""

import pytest
import pandas as pd

from core.data_loader import DataLoader
from core.agents.feature_intelligence import FeatureIntelligence


@pytest.fixture(scope="module")
def churn_df():
    loader = DataLoader(seed=42)
    return loader.load_sample_data("churn_prediction")


@pytest.fixture(scope="module")
def fi_result(churn_df):
    fi = FeatureIntelligence(target_col="churn", task_type="classification")
    return fi.analyse(churn_df)


class TestFeatureIntelligenceBasics:
    def test_returns_dict(self, fi_result):
        assert isinstance(fi_result, dict)

    def test_required_keys(self, fi_result):
        required = {
            "target_col", "task_type", "n_features_analysed",
            "feature_importance", "mutual_information",
            "correlation_with_target", "high_correlations",
            "transformation_recommendations",
        }
        assert required.issubset(set(fi_result.keys()))

    def test_target_col_set(self, fi_result):
        assert fi_result["target_col"] == "churn"

    def test_task_type_set(self, fi_result):
        assert fi_result["task_type"] == "classification"

    def test_n_features_analysed(self, fi_result, churn_df):
        # n_features = all cols - 1 (target)
        assert fi_result["n_features_analysed"] == len(churn_df.columns) - 1


class TestFeatureImportance:
    def test_is_list(self, fi_result):
        assert isinstance(fi_result["feature_importance"], list)

    def test_not_empty(self, fi_result):
        assert len(fi_result["feature_importance"]) > 0

    def test_has_required_keys(self, fi_result):
        for entry in fi_result["feature_importance"]:
            assert "feature" in entry
            assert "importance" in entry
            assert "rank" in entry

    def test_importances_sum_to_one(self, fi_result):
        total = sum(e["importance"] for e in fi_result["feature_importance"])
        assert abs(total - 1.0) < 0.01

    def test_ranked_descending(self, fi_result):
        importances = [e["importance"] for e in fi_result["feature_importance"]]
        assert importances == sorted(importances, reverse=True)


class TestMutualInformation:
    def test_is_list(self, fi_result):
        assert isinstance(fi_result["mutual_information"], list)

    def test_mi_scores_non_negative(self, fi_result):
        for entry in fi_result["mutual_information"]:
            assert entry["mi_score"] >= 0.0


class TestCorrelations:
    def test_correlation_list(self, fi_result):
        assert isinstance(fi_result["correlation_with_target"], list)

    def test_pearson_r_in_range(self, fi_result):
        for entry in fi_result["correlation_with_target"]:
            assert -1.0 <= entry["pearson_r"] <= 1.0


class TestTransformationRecs:
    def test_is_list(self, fi_result):
        assert isinstance(fi_result["transformation_recommendations"], list)

    def test_each_has_column_and_suggestions(self, fi_result):
        for rec in fi_result["transformation_recommendations"]:
            assert "column" in rec
            assert "suggestions" in rec
            assert isinstance(rec["suggestions"], list)


class TestMissingTarget:
    def test_missing_target_minimal_result(self, churn_df):
        fi = FeatureIntelligence(target_col="nonexistent_target")
        result = fi.analyse(churn_df)
        assert result["feature_importance"] == []
        assert result["mutual_information"] == []
