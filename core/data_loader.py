"""
core/data_loader.py — Data Loading and Synthetic Data Generation

Handles loading datasets from CSV, JSON, Excel files and generating
synthetic sample data for demonstration and testing.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.utils import get_logger

logger = get_logger(__name__)

# Attempt to import Faker — soft dependency
try:
    from faker import Faker
    _FAKER_AVAILABLE = True
except ImportError:
    _FAKER_AVAILABLE = False
    logger.warning("Faker not installed — synthetic data will use numpy/random only")


class DataLoader:
    """
    Handles dataset loading from multiple formats and synthetic data generation.

    Methods
    -------
    load_csv(path)              → pd.DataFrame
    load_json(path)             → pd.DataFrame
    load_excel(path)            → pd.DataFrame
    load_sample_data(use_case)  → pd.DataFrame (synthetic)
    detect_format(path)         → str (file extension)
    load(path)                  → pd.DataFrame (auto-detect format)
    """

    SUPPORTED_FORMATS = {".csv", ".json", ".xlsx", ".xls", ".parquet"}

    SAMPLE_USE_CASES = {
        "churn_prediction",
        "fraud_detection",
        "sales_forecasting",
    }

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        if _FAKER_AVAILABLE:
            self.fake = Faker()
            Faker.seed(seed)
        else:
            self.fake = None

    # ------------------------------------------------------------------
    # File loaders
    # ------------------------------------------------------------------

    def load_csv(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""
        path = Path(path)
        logger.info(f"Loading CSV: {path}")
        df = pd.read_csv(path, **kwargs)
        logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")
        return df

    def load_json(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load a JSON file into a DataFrame (supports records and lines format)."""
        path = Path(path)
        logger.info(f"Loading JSON: {path}")
        try:
            df = pd.read_json(path, **kwargs)
        except ValueError:
            # Try lines format
            df = pd.read_json(path, lines=True, **kwargs)
        logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")
        return df

    def load_excel(self, path: str | Path, sheet_name: int | str = 0, **kwargs: Any) -> pd.DataFrame:
        """Load an Excel file (.xlsx or .xls) into a DataFrame."""
        path = Path(path)
        logger.info(f"Loading Excel: {path} (sheet={sheet_name})")
        df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
        logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")
        return df

    def load_parquet(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load a Parquet file into a DataFrame."""
        path = Path(path)
        logger.info(f"Loading Parquet: {path}")
        df = pd.read_parquet(path, **kwargs)
        logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")
        return df

    def load(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Auto-detect format and load the file."""
        fmt = self.detect_format(path)
        loaders = {
            ".csv": self.load_csv,
            ".json": self.load_json,
            ".xlsx": self.load_excel,
            ".xls": self.load_excel,
            ".parquet": self.load_parquet,
        }
        loader = loaders.get(fmt)
        if loader is None:
            raise ValueError(
                f"Unsupported format '{fmt}'. "
                f"Supported: {sorted(self.SUPPORTED_FORMATS)}"
            )
        return loader(path, **kwargs)

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------

    def detect_format(self, path: str | Path) -> str:
        """
        Auto-detect file type from extension.

        Returns
        -------
        str : lower-case extension including dot, e.g. '.csv'
        """
        ext = Path(path).suffix.lower()
        logger.debug(f"Detected format: '{ext}' for '{path}'")
        return ext

    # ------------------------------------------------------------------
    # Synthetic data generators
    # ------------------------------------------------------------------

    def load_sample_data(self, use_case: str) -> pd.DataFrame:
        """
        Generate synthetic data for a given use case.

        Parameters
        ----------
        use_case : str
            One of: 'churn_prediction', 'fraud_detection', 'sales_forecasting'

        Returns
        -------
        pd.DataFrame
        """
        generators = {
            "churn_prediction": self._generate_churn_data,
            "fraud_detection": self._generate_fraud_data,
            "sales_forecasting": self._generate_sales_data,
        }
        if use_case not in generators:
            raise ValueError(
                f"Unknown sample use case '{use_case}'. "
                f"Available: {sorted(self.SAMPLE_USE_CASES)}"
            )
        logger.info(f"Generating synthetic data for use case: {use_case}")
        df = generators[use_case]()
        logger.info(f"  Generated {len(df):,} rows × {len(df.columns)} cols")
        return df

    def _generate_churn_data(self, n: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic churn prediction dataset.
        Columns: customer_id, age, tenure_months, monthly_charge, total_charges,
                 num_products, support_calls, churn (0/1)
        """
        rng = np.random.default_rng(self.seed)

        customer_ids = [f"CUST-{i:05d}" for i in range(1, n + 1)]
        age = rng.integers(18, 75, size=n)
        tenure_months = rng.integers(1, 72, size=n)
        monthly_charge = rng.uniform(20, 120, size=n).round(2)
        total_charges = (monthly_charge * tenure_months * rng.uniform(0.9, 1.1, size=n)).round(2)
        num_products = rng.integers(1, 6, size=n)
        support_calls = rng.integers(0, 15, size=n)

        # Churn probability: higher with short tenure, high support calls, low products
        churn_prob = (
            0.05
            + 0.3 * (tenure_months < 12)
            + 0.2 * (support_calls > 5)
            - 0.1 * (num_products > 3)
            + 0.1 * (monthly_charge > 80)
        )
        churn_prob = np.clip(churn_prob, 0.02, 0.85)
        churn = rng.binomial(1, churn_prob, size=n)

        # Introduce ~2% nulls for realism
        total_charges_with_nulls = total_charges.astype(float)
        null_indices = rng.choice(n, size=int(n * 0.02), replace=False)
        total_charges_with_nulls[null_indices] = np.nan

        return pd.DataFrame({
            "customer_id": customer_ids,
            "age": age,
            "tenure_months": tenure_months,
            "monthly_charge": monthly_charge,
            "total_charges": total_charges_with_nulls,
            "num_products": num_products,
            "support_calls": support_calls,
            "churn": churn,
        })

    def _generate_fraud_data(self, n: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic fraud detection dataset.
        Columns: transaction_id, amount, merchant_category, time_of_day,
                 distance_from_home, is_fraud (0/1)
        """
        rng = np.random.default_rng(self.seed)

        merchant_categories = [
            "grocery", "gas_station", "restaurant", "online_retail",
            "travel", "entertainment", "healthcare", "atm",
        ]

        transaction_ids = [f"TXN-{i:07d}" for i in range(1, n + 1)]
        amount = rng.exponential(scale=80, size=n).round(2)
        amount = np.clip(amount, 0.5, 5000.0)
        merchant_category = rng.choice(merchant_categories, size=n)
        time_of_day = rng.uniform(0, 24, size=n).round(2)
        distance_from_home = rng.exponential(scale=15, size=n).round(2)

        # Fraud: higher for large amounts, late night, far from home
        fraud_prob = (
            0.02
            + 0.15 * (amount > 500)
            + 0.10 * ((time_of_day < 5) | (time_of_day > 22))
            + 0.15 * (distance_from_home > 50)
        )
        fraud_prob = np.clip(fraud_prob, 0.01, 0.60)
        is_fraud = rng.binomial(1, fraud_prob, size=n)

        return pd.DataFrame({
            "transaction_id": transaction_ids,
            "amount": amount,
            "merchant_category": merchant_category,
            "time_of_day": time_of_day,
            "distance_from_home": distance_from_home,
            "is_fraud": is_fraud,
        })

    def _generate_sales_data(self, n: int = 500) -> pd.DataFrame:
        """
        Generate synthetic sales forecasting dataset.
        Columns: date, product_id, category, units_sold, unit_price, region, revenue
        """
        rng = np.random.default_rng(self.seed)

        categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Food & Beverage"]
        regions = ["North", "South", "East", "West", "Central"]
        products_per_category = 5

        # Generate a date range
        start_date = datetime(2023, 1, 1)
        dates = [
            start_date + timedelta(days=int(d))
            for d in rng.integers(0, 365, size=n)
        ]
        dates = sorted(dates)

        product_ids = [
            f"{cat[:3].upper()}-{pid:03d}"
            for cat in categories
            for pid in range(1, products_per_category + 1)
        ]

        selected_products = rng.choice(product_ids, size=n)
        category = [p.split("-")[0] for p in selected_products]

        # Map 3-letter prefix back to full category name
        cat_map = {cat[:3].upper(): cat for cat in categories}
        category_full = [cat_map.get(c, c) for c in category]

        unit_price = rng.uniform(5, 500, size=n).round(2)
        units_sold = rng.integers(1, 200, size=n)

        # Apply seasonality boost for Nov/Dec
        seasonal_boost = np.array([
            1.4 if d.month in (11, 12) else 1.0 for d in dates
        ])
        units_sold = (units_sold * seasonal_boost).astype(int)

        revenue = (unit_price * units_sold).round(2)
        selected_regions = rng.choice(regions, size=n)

        return pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "product_id": selected_products,
            "category": category_full,
            "units_sold": units_sold,
            "unit_price": unit_price,
            "region": selected_regions,
            "revenue": revenue,
        })
