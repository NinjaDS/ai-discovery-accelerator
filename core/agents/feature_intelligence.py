"""
core/agents/feature_intelligence.py — Feature Intelligence Agent

Runs statistical feature analysis on a DataFrame:
  - Correlation matrix against target
  - Mutual information scores
  - Quick RandomForest feature importance
  - Transformation recommendations per column

Inspired by AutoEDA (arXiv 2510.04023) and CrewAI EDA agent patterns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from core.utils import get_logger

logger = get_logger(__name__)


class FeatureIntelligence:
    """
    Statistical intelligence layer: identifies which features matter,
    how they relate, and what transformations would help.

    Parameters
    ----------
    target_col : str
        Name of the target / label column.
    task_type : str
        'classification' or 'regression'.
    max_features : int
        Top-N features to include in importance report.
    """

    def __init__(
        self,
        target_col: str,
        task_type: str = "classification",
        max_features: int = 20,
    ) -> None:
        self.target_col = target_col
        self.task_type = task_type
        self.max_features = max_features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Run full feature intelligence on a DataFrame.

        Returns
        -------
        dict with keys:
          target_col, task_type, n_features_analysed,
          feature_importance, mutual_information,
          correlation_with_target, high_correlations,
          transformation_recommendations, summary
        """
        logger.info(
            f"Feature intelligence — target: '{self.target_col}', "
            f"task: {self.task_type}, shape: {df.shape}"
        )

        if self.target_col not in df.columns:
            logger.warning(f"Target '{self.target_col}' not found. Skipping MI / importance.")
            return self._minimal_result(df)

        # Prepare encoded version of the DataFrame
        df_enc, encoders = self._encode_dataframe(df)

        feature_cols = [c for c in df_enc.columns if c != self.target_col]

        results: dict[str, Any] = {
            "target_col": self.target_col,
            "task_type": self.task_type,
            "n_features_analysed": len(feature_cols),
        }

        # --- Correlation with target (numeric) ---
        results["correlation_with_target"] = self._target_correlation(df_enc, feature_cols)

        # --- High pairwise correlations ---
        results["high_correlations"] = self._pairwise_high_corr(df_enc, feature_cols)

        # --- Mutual information ---
        results["mutual_information"] = self._mutual_info(df_enc, feature_cols)

        # --- RandomForest feature importance ---
        results["feature_importance"] = self._rf_importance(df_enc, feature_cols)

        # --- Transformation recommendations ---
        results["transformation_recommendations"] = self._transform_recommendations(df)

        # --- Natural language summary ---
        results["summary"] = self._build_summary(results)

        logger.info(
            f"  Feature intelligence done — "
            f"top feature: {results['feature_importance'][0]['feature'] if results['feature_importance'] else 'N/A'}"
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_dataframe(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        """
        Return a fully numeric copy of the DataFrame suitable for sklearn.
        Categorical columns are label-encoded; nulls are median-imputed.
        """
        df_enc = df.copy()
        encoders: dict[str, LabelEncoder] = {}

        # Encode object/category columns (pandas 3.x compatible)
        for col in df_enc.select_dtypes(include=["object", "category", "string"]).columns:
            le = LabelEncoder()
            mask = df_enc[col].notna()
            # Build a plain Python object array to avoid StringDtype restrictions
            encoded = df_enc[col].to_numpy(dtype=object, na_value=np.nan)
            encoded[mask.to_numpy()] = le.fit_transform(
                df_enc.loc[mask, col].astype(str).to_numpy()
            )
            df_enc[col] = pd.to_numeric(
                pd.array(encoded, dtype=object), errors="coerce"
            )
            encoders[col] = le

        # Encode booleans
        for col in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[col] = df_enc[col].astype(int)

        # Impute remaining nulls with median
        imputer = SimpleImputer(strategy="median")
        df_enc = pd.DataFrame(
            imputer.fit_transform(df_enc),
            columns=df_enc.columns,
        )

        return df_enc, encoders

    def _target_correlation(
        self, df_enc: pd.DataFrame, feature_cols: list[str]
    ) -> list[dict]:
        """Pearson correlation of each feature with the target."""
        target = df_enc[self.target_col]
        corr_list = []
        for col in feature_cols:
            corr_val = df_enc[col].corr(target)
            corr_list.append({
                "feature": col,
                "pearson_r": round(float(corr_val), 4) if not np.isnan(corr_val) else 0.0,
                "abs_r": round(abs(float(corr_val)), 4) if not np.isnan(corr_val) else 0.0,
            })
        return sorted(corr_list, key=lambda x: x["abs_r"], reverse=True)

    def _pairwise_high_corr(
        self, df_enc: pd.DataFrame, feature_cols: list[str], threshold: float = 0.85
    ) -> list[dict]:
        """Return pairs of features with |correlation| > threshold (multicollinearity warning)."""
        if len(feature_cols) < 2:
            return []
        corr_matrix = df_enc[feature_cols].corr()
        pairs = []
        for i, c1 in enumerate(feature_cols):
            for c2 in feature_cols[i + 1 :]:
                val = corr_matrix.loc[c1, c2]
                if not np.isnan(val) and abs(val) >= threshold:
                    pairs.append({
                        "feature_a": c1,
                        "feature_b": c2,
                        "correlation": round(float(val), 4),
                        "note": "Potential multicollinearity — consider dropping one.",
                    })
        return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)

    def _mutual_info(
        self, df_enc: pd.DataFrame, feature_cols: list[str]
    ) -> list[dict]:
        """Compute mutual information scores between each feature and the target."""
        X = df_enc[feature_cols].values
        y = df_enc[self.target_col].values

        try:
            if self.task_type == "classification":
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)
        except Exception as e:
            logger.warning(f"  Mutual info failed: {e}")
            return []

        mi_list = [
            {"feature": col, "mi_score": round(float(score), 5)}
            for col, score in zip(feature_cols, mi_scores)
        ]
        return sorted(mi_list, key=lambda x: x["mi_score"], reverse=True)[: self.max_features]

    def _rf_importance(
        self, df_enc: pd.DataFrame, feature_cols: list[str]
    ) -> list[dict]:
        """Fit a fast RandomForest and return feature importances."""
        X = df_enc[feature_cols].values
        y = df_enc[self.target_col].values

        try:
            if self.task_type == "classification":
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1,
                )
            model.fit(X, y)
            importances = model.feature_importances_
        except Exception as e:
            logger.warning(f"  RF importance failed: {e}")
            return []

        imp_list = [
            {"feature": col, "importance": round(float(imp), 5), "rank": rank + 1}
            for rank, (col, imp) in enumerate(
                sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
            )
        ]
        return imp_list[: self.max_features]

    @staticmethod
    def _transform_recommendations(df: pd.DataFrame) -> list[dict]:
        """
        Suggest column-level transformations based on distribution stats.
        """
        from scipy import stats as scipy_stats

        recs = []
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 10:
                continue

            skew = float(scipy_stats.skew(series))
            nunique = series.nunique()
            pct_zero = float((series == 0).mean() * 100)

            suggestions = []
            if abs(skew) > 1.5 and series.min() >= 0:
                suggestions.append(f"Log-transform (skewness={skew:.2f})")
            if nunique <= 10 and df[col].dtype in [np.int64, np.int32, int]:
                suggestions.append("Consider treating as categorical / one-hot encode")
            if pct_zero > 30:
                suggestions.append(f"High zero-inflation ({pct_zero:.0f}%) — consider binary indicator")
            if series.max() > 1000 and series.std() / series.mean() > 2:
                suggestions.append("High variance — consider standardisation (StandardScaler)")

            if suggestions:
                recs.append({"column": col, "suggestions": suggestions})

        # Categorical encoding recommendations
        for col in df.select_dtypes(include=["object", "category"]).columns:
            nunique = df[col].nunique()
            if nunique == 2:
                recs.append({"column": col, "suggestions": ["Binary encode (0/1)"]})
            elif nunique <= 10:
                recs.append({"column": col, "suggestions": [f"One-hot encode ({nunique} categories)"]})
            elif nunique <= 50:
                recs.append({"column": col, "suggestions": ["Target encode or frequency encode (high-cardinality)"]})
            else:
                recs.append({"column": col, "suggestions": ["Very high cardinality — consider embedding or hashing"]})

        return recs

    def _build_summary(self, results: dict) -> str:
        """Build a human-readable summary string."""
        top_feat = results["feature_importance"][:3] if results.get("feature_importance") else []
        top_names = ", ".join(f["feature"] for f in top_feat)
        hi_corr = len(results.get("high_correlations", []))
        n_transform = len(results.get("transformation_recommendations", []))

        return (
            f"Analysed {results['n_features_analysed']} features. "
            f"Top predictors by RF importance: [{top_names}]. "
            f"{hi_corr} high-correlation pair(s) detected (multicollinearity risk). "
            f"{n_transform} column(s) flagged for transformation."
        )

    def _minimal_result(self, df: pd.DataFrame) -> dict[str, Any]:
        """Fallback when target column is missing."""
        return {
            "target_col": self.target_col,
            "task_type": self.task_type,
            "n_features_analysed": len(df.columns),
            "feature_importance": [],
            "mutual_information": [],
            "correlation_with_target": [],
            "high_correlations": [],
            "transformation_recommendations": self._transform_recommendations(df),
            "summary": f"Target '{self.target_col}' not found — only transformation analysis available.",
        }
