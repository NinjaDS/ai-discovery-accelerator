"""
tests/test_data_loader.py — Unit tests for DataLoader
"""

import pytest
import pandas as pd

from core.data_loader import DataLoader


@pytest.fixture
def loader():
    return DataLoader(seed=42)


class TestLoadSampleData:
    def test_churn_prediction_shape(self, loader):
        df = loader.load_sample_data("churn_prediction")
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 900, f"Expected ~1000 rows, got {len(df)}"
        assert len(df) <= 1100

    def test_churn_prediction_columns(self, loader):
        df = loader.load_sample_data("churn_prediction")
        expected_cols = {
            "customer_id", "age", "tenure_months", "monthly_charge",
            "total_charges", "num_products", "support_calls", "churn",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_churn_prediction_target(self, loader):
        df = loader.load_sample_data("churn_prediction")
        assert set(df["churn"].dropna().unique()).issubset({0, 1})

    def test_fraud_detection_shape(self, loader):
        df = loader.load_sample_data("fraud_detection")
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 900
        assert len(df) <= 1100

    def test_fraud_detection_columns(self, loader):
        df = loader.load_sample_data("fraud_detection")
        expected_cols = {
            "transaction_id", "amount", "merchant_category",
            "time_of_day", "distance_from_home", "is_fraud",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_fraud_detection_target(self, loader):
        df = loader.load_sample_data("fraud_detection")
        assert set(df["is_fraud"].unique()).issubset({0, 1})

    def test_sales_forecasting_shape(self, loader):
        df = loader.load_sample_data("sales_forecasting")
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 400
        assert len(df) <= 600

    def test_sales_forecasting_columns(self, loader):
        df = loader.load_sample_data("sales_forecasting")
        expected_cols = {
            "date", "product_id", "category",
            "units_sold", "unit_price", "region", "revenue",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_sales_forecasting_revenue(self, loader):
        df = loader.load_sample_data("sales_forecasting")
        assert (df["revenue"] > 0).all()

    def test_unknown_use_case_raises(self, loader):
        with pytest.raises(ValueError, match="Unknown sample use case"):
            loader.load_sample_data("nonexistent_use_case")


class TestDetectFormat:
    def test_csv(self, loader):
        assert loader.detect_format("data.csv") == ".csv"

    def test_json(self, loader):
        assert loader.detect_format("data.json") == ".json"

    def test_excel(self, loader):
        assert loader.detect_format("data.xlsx") == ".xlsx"

    def test_parquet(self, loader):
        assert loader.detect_format("data.parquet") == ".parquet"

    def test_case_insensitive(self, loader):
        assert loader.detect_format("DATA.CSV") == ".csv"
