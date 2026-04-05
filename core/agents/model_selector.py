"""
core/agents/model_selector.py — Model Selection Agent

Recommends the top 3 ML approaches given dataset characteristics,
class balance, use case type, and feature composition.
Each recommendation includes: pros, cons, expected performance range,
and effort estimate in engineering days.
"""

from __future__ import annotations

from typing import Any

from core.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

MODEL_CATALOGUE: dict[str, dict] = {
    # ----- Classification -----
    "xgboost_classifier": {
        "name": "XGBoost (Gradient Boosted Trees)",
        "family": "ensemble",
        "task": "classification",
        "pros": [
            "State-of-the-art on tabular data",
            "Handles missing values natively",
            "Built-in feature importance",
            "Robust to outliers and skewed distributions",
        ],
        "cons": [
            "Requires hyperparameter tuning (learning rate, depth, subsample)",
            "Less interpretable than linear models",
            "Can overfit on small datasets",
        ],
        "expected_auc_range": "0.82–0.92",
        "expected_f1_range": "0.70–0.85",
        "effort_days": {"poc": 2, "production": 8},
        "tags": ["tabular", "structured", "medium-large-data", "classification"],
    },
    "logistic_regression": {
        "name": "Logistic Regression (with regularisation)",
        "family": "linear",
        "task": "classification",
        "pros": [
            "Highly interpretable — coefficients as feature weights",
            "Fast to train and serve",
            "Great baseline / benchmark",
            "Works well when features are informative and clean",
        ],
        "cons": [
            "Assumes linear decision boundary — misses complex interactions",
            "Sensitive to class imbalance",
            "Requires manual feature engineering for non-linear patterns",
        ],
        "expected_auc_range": "0.74–0.85",
        "expected_f1_range": "0.62–0.78",
        "effort_days": {"poc": 1, "production": 4},
        "tags": ["tabular", "interpretable", "baseline", "classification"],
    },
    "random_forest_classifier": {
        "name": "Random Forest Classifier",
        "family": "ensemble",
        "task": "classification",
        "pros": [
            "Robust to noise and outliers",
            "Low risk of overfitting",
            "Good out-of-the-box performance",
            "Feature importance built in",
        ],
        "cons": [
            "Slower inference vs single trees",
            "Memory intensive for large forests",
            "Less powerful than gradient boosting on most benchmarks",
        ],
        "expected_auc_range": "0.80–0.90",
        "expected_f1_range": "0.68–0.82",
        "effort_days": {"poc": 1, "production": 5},
        "tags": ["tabular", "ensemble", "medium-data", "classification"],
    },
    "lightgbm_classifier": {
        "name": "LightGBM Classifier",
        "family": "ensemble",
        "task": "classification",
        "pros": [
            "Faster training than XGBoost on large datasets",
            "Lower memory footprint",
            "Excellent with high-cardinality categoricals",
        ],
        "cons": [
            "Can overfit on small datasets",
            "Less popular — fewer resources than XGBoost",
        ],
        "expected_auc_range": "0.83–0.93",
        "expected_f1_range": "0.71–0.86",
        "effort_days": {"poc": 2, "production": 8},
        "tags": ["tabular", "large-data", "classification"],
    },
    "neural_net_tabular": {
        "name": "Tabular Neural Network (TabNet / MLP)",
        "family": "deep_learning",
        "task": "classification",
        "pros": [
            "Can capture complex feature interactions automatically",
            "Scales well with very large datasets",
            "TabNet provides attention-based interpretability",
        ],
        "cons": [
            "Requires large dataset (>50k rows) to outperform boosting",
            "Longer training and tuning time",
            "Overkill for small/medium tabular problems",
        ],
        "expected_auc_range": "0.80–0.91",
        "expected_f1_range": "0.68–0.83",
        "effort_days": {"poc": 4, "production": 15},
        "tags": ["tabular", "large-data", "deep-learning", "classification"],
    },
    # ----- Regression -----
    "xgboost_regressor": {
        "name": "XGBoost Regressor",
        "family": "ensemble",
        "task": "regression",
        "pros": [
            "Best-in-class for tabular regression",
            "Handles non-linear relationships",
        ],
        "cons": ["Needs tuning", "Less interpretable than linear models"],
        "expected_rmse_note": "Typically 10-20% improvement over linear baseline",
        "effort_days": {"poc": 2, "production": 8},
        "tags": ["tabular", "regression"],
    },
    "linear_regression_ridge": {
        "name": "Ridge Regression (regularised linear)",
        "family": "linear",
        "task": "regression",
        "pros": ["Interpretable", "Fast", "Good baseline"],
        "cons": ["Linear only", "Poor on non-linear relationships"],
        "effort_days": {"poc": 1, "production": 3},
        "tags": ["tabular", "regression", "interpretable"],
    },
}


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

class ModelSelector:
    """
    Recommends the top 3 ML models based on dataset characteristics.

    Parameters
    ----------
    use_case : str
        High-level use case (e.g. 'churn_prediction', 'fraud_detection').
    task_type : str
        'classification' or 'regression'.
    """

    def __init__(self, use_case: str = "churn_prediction", task_type: str = "classification") -> None:
        self.use_case = use_case
        self.task_type = task_type

    def recommend(
        self,
        data_profile: dict,
        gap_report: dict,
        feature_intelligence: dict,
    ) -> dict[str, Any]:
        """
        Return top-3 model recommendations with full rationale.

        Parameters
        ----------
        data_profile    : output of DataProfiler.profile()
        gap_report      : output of GapAnalyst.analyse()
        feature_intelligence : output of FeatureIntelligence.analyse()

        Returns
        -------
        dict with keys:
          task_type, use_case, data_characteristics, recommendations (top 3),
          primary_recommendation, training_strategy, evaluation_strategy
        """
        logger.info(f"Model selection — use case: {self.use_case}, task: {self.task_type}")

        # Extract characteristics
        chars = self._extract_characteristics(data_profile, feature_intelligence, gap_report)
        logger.info(f"  Data characteristics: {chars}")

        # Score models
        scored = self._score_models(chars)
        top3 = scored[:3]

        primary = top3[0]["model_key"] if top3 else None

        result = {
            "task_type": self.task_type,
            "use_case": self.use_case,
            "data_characteristics": chars,
            "recommendations": top3,
            "primary_recommendation": primary,
            "training_strategy": self._training_strategy(chars),
            "evaluation_strategy": self._evaluation_strategy(chars),
            "estimated_total_effort": self._total_effort(top3),
        }

        logger.info(f"  Top recommendation: {primary}")
        return result

    # ------------------------------------------------------------------

    def _extract_characteristics(
        self, data_profile: dict, feature_intel: dict, gap_report: dict | None = None
    ) -> dict[str, Any]:
        """Summarise the key data characteristics that drive model selection."""
        if gap_report is None:
            gap_report = {}
        n_rows = data_profile.get("n_rows", 0)

        # Class balance (for classification)
        class_balance = "unknown"
        col_profiles = data_profile.get("column_profiles", {})
        for col, cp in col_profiles.items():
            if "churn" in col.lower() or "target" in col.lower() or "label" in col.lower():
                top = cp.get("top_values", {})
                if top:
                    vals = list(top.values())
                    if len(vals) >= 2:
                        ratio = min(vals) / max(vals)
                        class_balance = "balanced" if ratio > 0.3 else "imbalanced"
                break

        # Feature types
        type_dist = data_profile.get("type_distribution", {})
        has_categoricals = (
            type_dist.get("categorical_low", 0) + type_dist.get("categorical_high", 0)
        ) > 0
        n_numeric = (
            type_dist.get("numeric_continuous", 0) + type_dist.get("numeric_discrete", 0)
        )

        # High-correlation pairs (multicollinearity)
        n_high_corr = len(feature_intel.get("high_correlations", []))

        return {
            "n_rows": n_rows,
            "data_size": (
                "large" if n_rows > 50_000
                else "medium" if n_rows > 5_000
                else "small"
            ),
            "class_balance": class_balance,
            "has_categoricals": has_categoricals,
            "n_numeric_features": n_numeric,
            "n_high_corr_pairs": n_high_corr,
            "null_pct": data_profile.get("overall_null_pct", 0),
            "readiness_score": gap_report.get("readiness_score", 0),
        }

    def _score_models(self, chars: dict) -> list[dict]:
        """Score each applicable model and return sorted list."""
        scored = []

        for key, model in MODEL_CATALOGUE.items():
            if model["task"] != self.task_type:
                continue

            score = 0

            # Data size fit
            if chars["data_size"] == "large":
                if "large-data" in model["tags"]:
                    score += 20
                elif "medium-data" in model["tags"]:
                    score += 10
            elif chars["data_size"] == "medium":
                if "medium-data" in model["tags"] or "tabular" in model["tags"]:
                    score += 20
                if "large-data" in model["tags"]:
                    score += 5  # still ok
            else:  # small
                if "baseline" in model["tags"] or "interpretable" in model["tags"]:
                    score += 20
                if "large-data" in model["tags"] or "deep-learning" in model["tags"]:
                    score -= 10  # penalise for small data

            # Categorical features
            if chars["has_categoricals"] and key in (
                "xgboost_classifier", "lightgbm_classifier", "xgboost_regressor"
            ):
                score += 10

            # Class imbalance
            if chars["class_balance"] == "imbalanced":
                if key in ("xgboost_classifier", "lightgbm_classifier"):
                    score += 10  # scale_pos_weight support

            # Multicollinearity
            if chars["n_high_corr_pairs"] > 3:
                if key in ("logistic_regression", "linear_regression_ridge"):
                    score -= 5  # linear models suffer more

            # Always add baseline
            if key in ("logistic_regression", "linear_regression_ridge"):
                score += 5  # always worth including as baseline

            # Use case affinity
            if self.use_case == "churn_prediction" and key == "xgboost_classifier":
                score += 10
            if self.use_case == "fraud_detection" and key == "lightgbm_classifier":
                score += 10

            scored.append({
                "model_key": key,
                "model_name": model["name"],
                "score": score,
                "pros": model["pros"],
                "cons": model["cons"],
                "effort_days": model["effort_days"],
                "expected_performance": {
                    k: v for k, v in model.items()
                    if k.startswith("expected_")
                },
                "rationale": self._rationale(key, chars),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        # Add rank
        for i, s in enumerate(scored):
            s["rank"] = i + 1

        return scored

    @staticmethod
    def _rationale(model_key: str, chars: dict) -> str:
        """One-sentence rationale for the recommendation."""
        rationale_map = {
            "xgboost_classifier": (
                f"XGBoost is the industry standard for tabular classification — "
                f"excellent fit for {chars['data_size']} datasets with mixed feature types."
            ),
            "logistic_regression": (
                "Logistic Regression provides an interpretable baseline; "
                "stakeholder-friendly coefficients for compliance and explainability."
            ),
            "random_forest_classifier": (
                "Random Forest is robust and low-maintenance — "
                "good balance of performance and reliability for production."
            ),
            "lightgbm_classifier": (
                "LightGBM excels at speed and high-cardinality categoricals; "
                "ideal if dataset grows beyond 50k rows."
            ),
            "neural_net_tabular": (
                "Tabular neural networks (TabNet) capture complex feature interactions "
                "— worth exploring if simpler models plateau."
            ),
        }
        return rationale_map.get(
            model_key,
            f"Solid choice for {chars['data_size']} {chars['class_balance']} datasets.",
        )

    @staticmethod
    def _training_strategy(chars: dict) -> dict:
        """Recommend training / validation strategy."""
        if chars["n_rows"] < 1000:
            return {
                "strategy": "k-fold cross-validation (k=5)",
                "reason": "Small dataset — maximise data usage with cross-validation.",
                "train_split": "80/20 with stratification",
            }
        elif chars["n_rows"] < 10_000:
            return {
                "strategy": "stratified train/val/test split (70/15/15)",
                "reason": "Medium dataset — hold-out test set for unbiased evaluation.",
                "train_split": "70/15/15 stratified",
            }
        else:
            return {
                "strategy": "time-based split (if temporal) or random 80/10/10",
                "reason": "Large dataset — simple split is statistically robust.",
                "train_split": "80/10/10",
            }

    @staticmethod
    def _evaluation_strategy(chars: dict) -> dict:
        """Recommend evaluation metrics."""
        if chars["class_balance"] == "imbalanced":
            return {
                "primary_metric": "AUC-ROC",
                "secondary_metrics": ["F1-score (macro)", "Precision-Recall curve", "Cohen's Kappa"],
                "note": "Avoid accuracy — misleading with imbalanced classes.",
                "threshold": "Optimise threshold via F1 or business cost matrix.",
            }
        return {
            "primary_metric": "AUC-ROC",
            "secondary_metrics": ["F1-score", "Precision", "Recall"],
            "note": "Also track business metrics: cost per FP/FN.",
            "threshold": "Default 0.5 — adjust based on business tolerance for false positives.",
        }

    @staticmethod
    def _total_effort(top3: list[dict]) -> dict:
        """Estimate total PoC + production effort for the primary recommendation."""
        if not top3:
            return {}
        primary = top3[0]
        effort = primary.get("effort_days", {})
        return {
            "poc_days": effort.get("poc", "N/A"),
            "production_days": effort.get("production", "N/A"),
            "note": (
                "PoC includes: data prep, training, initial evaluation. "
                "Production includes: feature pipeline, serving API, monitoring."
            ),
        }
