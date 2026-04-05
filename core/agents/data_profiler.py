"""
core/agents/data_profiler.py — Data Profiling Agent

Ingests a pandas DataFrame (plus optional schema metadata) and returns a
comprehensive data profile: null rates, type distribution, outlier flags,
distribution statistics, and an enterprise multi-source flag.

Inspired by:
  - DREAMER data readiness framework (Kolachalama Lab, BU)
  - AutoEDA (arXiv 2510.04023)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from core.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature type inference
# ---------------------------------------------------------------------------

def _infer_feature_type(series: pd.Series) -> str:
    """
    Classify a column into one of: numeric_continuous, numeric_discrete,
    categorical_low, categorical_high, boolean, datetime, text, or unknown.
    """
    dtype = series.dtype
    n_unique = series.nunique(dropna=True)
    n_total = len(series.dropna())

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        if n_unique <= 2:
            return "boolean"
        if n_unique <= 15:
            return "numeric_discrete"
        return "numeric_continuous"

    if pd.api.types.is_object_dtype(series):
        # Attempt datetime parse on a sample
        try:
            sample = series.dropna().head(50)
            pd.to_datetime(sample, infer_datetime_format=True)
            return "datetime"
        except Exception:
            pass

        if n_total > 0:
            unique_ratio = n_unique / n_total
            avg_len = series.dropna().str.len().mean()
            if avg_len and avg_len > 50:
                return "text"
            if n_unique <= 20 or unique_ratio < 0.05:
                return "categorical_low"
            return "categorical_high"

    return "unknown"


# ---------------------------------------------------------------------------
# Outlier detection (IQR method)
# ---------------------------------------------------------------------------

def _detect_outliers_iqr(series: pd.Series) -> dict[str, Any]:
    """
    Compute IQR-based outlier bounds and count extreme values.
    Returns a dict with q1, q3, iqr, lower_bound, upper_bound, n_outliers.
    """
    clean = series.dropna()
    if len(clean) < 10:
        return {"method": "iqr", "n_outliers": 0, "pct_outliers": 0.0}

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_out = int(((clean < lower) | (clean > upper)).sum())

    return {
        "method": "iqr",
        "q1": round(float(q1), 4),
        "q3": round(float(q3), 4),
        "iqr": round(float(iqr), 4),
        "lower_fence": round(float(lower), 4),
        "upper_fence": round(float(upper), 4),
        "n_outliers": n_out,
        "pct_outliers": round(n_out / len(clean) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Column-level profiling
# ---------------------------------------------------------------------------

def _profile_column(col: str, series: pd.Series) -> dict[str, Any]:
    """Build a complete profile dict for a single column."""
    n_total = len(series)
    n_null = int(series.isna().sum())
    n_unique = int(series.nunique(dropna=True))
    ftype = _infer_feature_type(series)

    profile: dict[str, Any] = {
        "name": col,
        "dtype": str(series.dtype),
        "inferred_type": ftype,
        "n_total": n_total,
        "n_null": n_null,
        "null_pct": round(n_null / n_total * 100, 2) if n_total else 0.0,
        "n_unique": n_unique,
        "unique_ratio": round(n_unique / (n_total - n_null), 4) if (n_total - n_null) > 0 else 0.0,
    }

    # Numeric stats
    if ftype in ("numeric_continuous", "numeric_discrete"):
        clean = series.dropna()
        profile["stats"] = {
            "min": round(float(clean.min()), 4),
            "max": round(float(clean.max()), 4),
            "mean": round(float(clean.mean()), 4),
            "median": round(float(clean.median()), 4),
            "std": round(float(clean.std()), 4),
            "skewness": round(float(stats.skew(clean)), 4),
            "kurtosis": round(float(stats.kurtosis(clean)), 4),
        }
        profile["outliers"] = _detect_outliers_iqr(series)

    # Categorical stats
    elif ftype in ("categorical_low", "categorical_high", "boolean"):
        vc = series.value_counts(dropna=True)
        profile["top_values"] = vc.head(10).to_dict()
        profile["mode"] = str(vc.index[0]) if len(vc) > 0 else None

    # Datetime stats
    elif ftype == "datetime":
        dt_series = pd.to_datetime(series, errors="coerce")
        profile["date_range"] = {
            "min": str(dt_series.min()),
            "max": str(dt_series.max()),
        }

    return profile


# ---------------------------------------------------------------------------
# Main DataProfiler agent
# ---------------------------------------------------------------------------

class DataProfiler:
    """
    Agent that profiles one or more DataFrames and returns a structured
    profile dict suitable for downstream Gap Analysis and Feature Intelligence.

    Parameters
    ----------
    schema_metadata : dict, optional
        Pre-parsed schema metadata (from SchemaIngester) providing expected
        columns, types, and business descriptions.
    multi_source : bool
        Set True when data comes from multiple enterprise systems (CRM, billing,
        support, etc.). Included in the profile and used in gap scoring.
    """

    def __init__(
        self,
        schema_metadata: dict | None = None,
        multi_source: bool = False,
    ) -> None:
        self.schema_metadata = schema_metadata or {}
        self.multi_source = multi_source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, df: pd.DataFrame, source_name: str = "primary") -> dict[str, Any]:
        """
        Profile a DataFrame and return a structured profile dict.

        Returns
        -------
        dict with keys:
          source_name, n_rows, n_cols, memory_mb, columns,
          overall_null_pct, multi_source, schema_coverage, warnings
        """
        logger.info(f"Profiling source '{source_name}': {df.shape[0]} rows × {df.shape[1]} cols")

        col_profiles = {}
        warnings = []

        for col in df.columns:
            try:
                col_profiles[col] = _profile_column(col, df[col])
            except Exception as e:
                logger.warning(f"  Could not profile column '{col}': {e}")
                warnings.append(f"Column '{col}' profiling failed: {e}")

        # Overall null rate
        total_cells = df.shape[0] * df.shape[1]
        total_nulls = int(df.isna().sum().sum())
        overall_null_pct = round(total_nulls / total_cells * 100, 2) if total_cells else 0.0

        # Schema coverage (if metadata provided)
        schema_coverage = self._compute_schema_coverage(df)

        # High-null warnings
        for col, cp in col_profiles.items():
            if cp["null_pct"] > 20:
                warnings.append(f"High nulls in '{col}': {cp['null_pct']}%")

        # Duplicate rows
        n_dupes = int(df.duplicated().sum())
        if n_dupes > 0:
            warnings.append(f"{n_dupes} duplicate rows detected ({round(n_dupes/len(df)*100,1)}%)")

        profile = {
            "source_name": source_name,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "n_duplicate_rows": n_dupes,
            "overall_null_pct": overall_null_pct,
            "multi_source": self.multi_source,
            "schema_coverage": schema_coverage,
            "column_profiles": col_profiles,
            "warnings": warnings,
            # Summary counts by inferred type
            "type_distribution": self._type_distribution(col_profiles),
        }

        logger.info(
            f"  Profile complete — null rate: {overall_null_pct}%, "
            f"warnings: {len(warnings)}"
        )
        return profile

    def profile_multiple(
        self, sources: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """
        Profile multiple DataFrames (representing enterprise systems).
        Returns combined profile with per-source profiles and a
        cross-source relationship hint.
        """
        per_source = {
            name: self.profile(df, source_name=name)
            for name, df in sources.items()
        }

        # Basic cross-source relationship detection (shared column names)
        all_cols = {name: set(df.columns) for name, df in sources.items()}
        relationships = []
        source_names = list(all_cols.keys())
        for i, s1 in enumerate(source_names):
            for s2 in source_names[i + 1 :]:
                shared = all_cols[s1] & all_cols[s2]
                if shared:
                    relationships.append(
                        {
                            "source_a": s1,
                            "source_b": s2,
                            "shared_columns": list(shared),
                            "likely_join_key": list(shared)[0] if shared else None,
                        }
                    )

        return {
            "multi_source": True,
            "n_sources": len(sources),
            "source_names": source_names,
            "per_source": per_source,
            "detected_relationships": relationships,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_schema_coverage(self, df: pd.DataFrame) -> dict[str, Any]:
        """Check DataFrame columns against provided schema metadata."""
        if not self.schema_metadata:
            return {"available": False}

        expected = set(self.schema_metadata.get("columns", {}).keys())
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected

        return {
            "available": True,
            "expected_columns": len(expected),
            "present_columns": len(expected & actual),
            "missing_columns": list(missing),
            "extra_columns": list(extra),
            "coverage_pct": round(
                len(expected & actual) / len(expected) * 100, 1
            ) if expected else 100.0,
        }

    @staticmethod
    def _type_distribution(col_profiles: dict) -> dict[str, int]:
        """Count columns by inferred type."""
        dist: dict[str, int] = {}
        for cp in col_profiles.values():
            t = cp["inferred_type"]
            dist[t] = dist.get(t, 0) + 1
        return dist
