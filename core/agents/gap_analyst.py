"""
core/agents/gap_analyst.py — Gap Analysis Agent

Compares the data profile against feature requirements for a given use case.
Produces a scored gap report: overall data readiness 0-100, plus breakdowns
by critical vs. nice-to-have fields and actionable recommendations.

Inspired by DREAMER (Kolachalama Lab) data readiness scoring methodology.
"""

from __future__ import annotations

from typing import Any

from core.utils import get_logger, readiness_bar

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature requirements catalogue
# ---------------------------------------------------------------------------
# Each use case defines:
#   critical   — must-have features; absence heavily penalises score
#   important  — significant impact if missing
#   nice       — marginal improvement; absence only lightly penalises
#
# Each entry is a dict:
#   "field_patterns" : list of regex-like substrings to match column names
#   "description"    : human-readable label
#   "weight"         : contribution to the 100-point scale

USE_CASE_REQUIREMENTS: dict[str, dict] = {
    "churn_prediction": {
        "critical": [
            {"field_patterns": ["tenure", "months_with_company", "customer_age_months"],
             "description": "Customer tenure / time with company", "weight": 15},
            {"field_patterns": ["churn", "churned", "attrition", "target"],
             "description": "Churn / attrition target label", "weight": 15},
            {"field_patterns": ["monthly_charge", "monthly_fee", "monthly_revenue", "arpu"],
             "description": "Monthly billing / charges", "weight": 10},
        ],
        "important": [
            {"field_patterns": ["contract", "plan_type", "subscription"],
             "description": "Contract type / subscription tier", "weight": 8},
            {"field_patterns": ["support_call", "ticket", "complaint", "interaction"],
             "description": "Support interactions / complaints", "weight": 8},
            {"field_patterns": ["login", "last_active", "days_since", "activity"],
             "description": "Engagement / recency signal", "weight": 7},
            {"field_patterns": ["nps", "csat", "satisfaction", "rating", "score"],
             "description": "Customer satisfaction / NPS score", "weight": 7},
            {"field_patterns": ["product", "num_product", "num_service", "feature_used"],
             "description": "Product / feature usage breadth", "weight": 6},
        ],
        "nice": [
            {"field_patterns": ["payment_method", "payment_type", "billing_type"],
             "description": "Payment method", "weight": 4},
            {"field_patterns": ["internet", "broadband", "service_type"],
             "description": "Service/internet type", "weight": 3},
            {"field_patterns": ["tech_support", "has_support", "support_flag"],
             "description": "Tech support subscription", "weight": 3},
            {"field_patterns": ["age", "gender", "demographics", "segment"],
             "description": "Customer demographics / segment", "weight": 4},
            {"field_patterns": ["region", "location", "state", "country"],
             "description": "Geography / region", "weight": 3},
            {"field_patterns": ["referral", "promo", "discount", "coupon"],
             "description": "Promotional / acquisition channel", "weight": 3},
            {"field_patterns": ["total_charge", "ltv", "lifetime_value", "clv"],
             "description": "Total charges / LTV", "weight": 4},
        ],
    },

    "customer_support_chatbot": {
        "critical": [
            {"field_patterns": ["ticket", "issue", "case", "request"],
             "description": "Support ticket / issue records", "weight": 20},
            {"field_patterns": ["category", "type", "topic", "intent"],
             "description": "Issue category / intent label", "weight": 15},
            {"field_patterns": ["resolution", "resolved", "solution", "answer"],
             "description": "Resolution / answer text", "weight": 15},
        ],
        "important": [
            {"field_patterns": ["knowledge", "kb", "article", "faq", "document"],
             "description": "Knowledge base / FAQ corpus", "weight": 10},
            {"field_patterns": ["sentiment", "emotion", "satisfaction"],
             "description": "Customer sentiment signal", "weight": 8},
            {"field_patterns": ["resolution_time", "handle_time", "sla"],
             "description": "Resolution time / SLA data", "weight": 7},
            {"field_patterns": ["escalat", "priority", "severity"],
             "description": "Escalation / priority flags", "weight": 6},
        ],
        "nice": [
            {"field_patterns": ["channel", "source", "medium"],
             "description": "Contact channel (email/chat/phone)", "weight": 4},
            {"field_patterns": ["agent_id", "agent_name", "handler"],
             "description": "Agent / handler identifier", "weight": 3},
            {"field_patterns": ["keyword", "tag", "label"],
             "description": "Keywords / tags", "weight": 4},
            {"field_patterns": ["auto_resolv", "bot_handled", "automated"],
             "description": "Auto-resolvable flag", "weight": 4},
            {"field_patterns": ["product", "version", "sku"],
             "description": "Product / version context", "weight": 4},
        ],
    },

    "fraud_detection": {
        "critical": [
            {"field_patterns": ["fraud", "is_fraud", "fraudulent", "label", "target"],
             "description": "Fraud label / target", "weight": 20},
            {"field_patterns": ["amount", "transaction_value", "value"],
             "description": "Transaction amount", "weight": 15},
            {"field_patterns": ["timestamp", "date", "time", "created_at"],
             "description": "Transaction timestamp", "weight": 10},
        ],
        "important": [
            {"field_patterns": ["merchant", "vendor", "retailer"],
             "description": "Merchant / vendor info", "weight": 8},
            {"field_patterns": ["ip", "device", "browser", "user_agent"],
             "description": "Device / IP fingerprint", "weight": 8},
            {"field_patterns": ["location", "country", "city", "geo"],
             "description": "Transaction location", "weight": 7},
            {"field_patterns": ["user_id", "account_id", "customer_id"],
             "description": "Customer / account identifier", "weight": 7},
        ],
        "nice": [
            {"field_patterns": ["velocity", "freq", "count_last"],
             "description": "Transaction velocity features", "weight": 5},
            {"field_patterns": ["card_type", "payment_method"],
             "description": "Payment instrument", "weight": 5},
        ],
    },
}


# ---------------------------------------------------------------------------
# Gap Analyst
# ---------------------------------------------------------------------------

class GapAnalyst:
    """
    Compares data profile against feature requirements for a use case.

    Parameters
    ----------
    use_case : str
        Key into USE_CASE_REQUIREMENTS (e.g. 'churn_prediction').
    custom_requirements : dict, optional
        Provide your own requirements dict in the same schema to override.
    """

    def __init__(
        self,
        use_case: str,
        custom_requirements: dict | None = None,
    ) -> None:
        self.use_case = use_case
        if custom_requirements:
            self.requirements = custom_requirements
        elif use_case in USE_CASE_REQUIREMENTS:
            self.requirements = USE_CASE_REQUIREMENTS[use_case]
        else:
            logger.warning(
                f"Unknown use case '{use_case}'. "
                f"Known: {list(USE_CASE_REQUIREMENTS.keys())}. "
                "Using empty requirements — all fields will be 'extra'."
            )
            self.requirements = {"critical": [], "important": [], "nice": []}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, data_profile: dict) -> dict[str, Any]:
        """
        Run gap analysis against the data profile.

        Parameters
        ----------
        data_profile : dict
            Output from DataProfiler.profile() or DataProfiler.profile_multiple().

        Returns
        -------
        dict with keys:
          use_case, readiness_score, readiness_bar, tier_scores,
          present_fields, missing_critical, missing_important, missing_nice,
          field_details, recommendations, risks
        """
        logger.info(f"Running gap analysis for use case: {self.use_case}")

        # Flatten all column names from profile
        available_cols = self._extract_columns(data_profile)
        logger.info(f"  Available columns: {sorted(available_cols)}")

        # Score each tier
        tier_results = {}
        for tier in ("critical", "important", "nice"):
            tier_results[tier] = self._score_tier(
                tier, self.requirements.get(tier, []), available_cols
            )

        # Aggregate readiness score (weighted sum of present weights)
        total_possible = sum(
            req["weight"]
            for tier in ("critical", "important", "nice")
            for req in self.requirements.get(tier, [])
        )
        total_earned = sum(tr["earned_weight"] for tr in tier_results.values())
        readiness_score = round(
            (total_earned / total_possible * 100) if total_possible else 0.0, 1
        )

        # Collect missing/present field lists
        missing_critical = [
            r["description"]
            for r in tier_results["critical"]["missing"]
        ]
        missing_important = [
            r["description"]
            for r in tier_results["important"]["missing"]
        ]
        missing_nice = [
            r["description"]
            for r in tier_results["nice"]["missing"]
        ]
        present_fields = [
            r["description"]
            for tier in ("critical", "important", "nice")
            for r in tier_results[tier]["present"]
        ]

        # Penalise further for data quality issues
        quality_penalty = self._quality_penalty(data_profile)
        adjusted_score = max(0.0, round(readiness_score - quality_penalty, 1))

        recommendations = self._build_recommendations(
            tier_results, data_profile, adjusted_score
        )
        risks = self._build_risks(tier_results, data_profile)

        result = {
            "use_case": self.use_case,
            "readiness_score": adjusted_score,
            "readiness_bar": readiness_bar(adjusted_score),
            "raw_feature_score": readiness_score,
            "quality_penalty": quality_penalty,
            "tier_scores": {
                t: {
                    "score": tr["score"],
                    "earned_weight": tr["earned_weight"],
                    "max_weight": tr["max_weight"],
                }
                for t, tr in tier_results.items()
            },
            "present_fields": present_fields,
            "missing_critical": missing_critical,
            "missing_important": missing_important,
            "missing_nice": missing_nice,
            "recommendations": recommendations,
            "risks": risks,
        }

        logger.info(
            f"  Gap analysis complete — "
            f"readiness: {adjusted_score}/100, "
            f"missing critical: {len(missing_critical)}"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_columns(data_profile: dict) -> set[str]:
        """Extract all column names from a profile dict (single or multi-source)."""
        cols: set[str] = set()

        if data_profile.get("multi_source") and "per_source" in data_profile:
            for source_profile in data_profile["per_source"].values():
                cols.update(source_profile.get("column_profiles", {}).keys())
        else:
            cols.update(data_profile.get("column_profiles", {}).keys())

        return {c.lower() for c in cols}

    @staticmethod
    def _matches(col_name: str, patterns: list[str]) -> bool:
        """Check if a column name contains any of the given substrings."""
        col_lower = col_name.lower()
        return any(p.lower() in col_lower for p in patterns)

    def _score_tier(
        self,
        tier_name: str,
        requirements: list[dict],
        available_cols: set[str],
    ) -> dict[str, Any]:
        """Score a single tier (critical / important / nice)."""
        present = []
        missing = []
        earned = 0
        total = 0

        for req in requirements:
            total += req["weight"]
            matched_col = next(
                (c for c in available_cols if self._matches(c, req["field_patterns"])),
                None,
            )
            if matched_col:
                present.append({**req, "matched_column": matched_col})
                earned += req["weight"]
            else:
                missing.append(req)

        score = round(earned / total * 100, 1) if total else 100.0
        return {
            "tier": tier_name,
            "score": score,
            "earned_weight": earned,
            "max_weight": total,
            "present": present,
            "missing": missing,
        }

    @staticmethod
    def _quality_penalty(data_profile: dict) -> float:
        """
        Compute a penalty (0-20 pts) based on data quality issues:
          - overall null rate
          - number of high-outlier columns
          - duplicate rows
        """
        penalty = 0.0

        # Null rate penalty: up to 10 pts
        null_pct = data_profile.get("overall_null_pct", 0)
        if null_pct > 30:
            penalty += 10.0
        elif null_pct > 15:
            penalty += 6.0
        elif null_pct > 8:
            penalty += 3.0
        elif null_pct > 3:
            penalty += 1.0

        # Outlier penalty: up to 5 pts
        for cp in data_profile.get("column_profiles", {}).values():
            out = cp.get("outliers", {})
            if out.get("pct_outliers", 0) > 10:
                penalty += 1.0

        # Duplicate rows penalty: up to 5 pts
        n_rows = data_profile.get("n_rows", 1)
        n_dupes = data_profile.get("n_duplicate_rows", 0)
        if n_rows > 0 and n_dupes / n_rows > 0.05:
            penalty += 5.0
        elif n_rows > 0 and n_dupes / n_rows > 0.01:
            penalty += 2.0

        return min(penalty, 20.0)

    def _build_recommendations(
        self, tier_results: dict, data_profile: dict, score: float
    ) -> list[str]:
        """Generate actionable recommendations based on gaps."""
        recs = []

        # Critical gaps
        for r in tier_results["critical"]["missing"]:
            recs.append(
                f"🔴 CRITICAL: Source '{r['description']}' data. "
                f"This is required for {self.use_case.replace('_', ' ')}."
            )

        # Important gaps
        for r in tier_results["important"]["missing"]:
            recs.append(
                f"🟡 IMPORTANT: Obtain '{r['description']}' — "
                "significant impact on model performance."
            )

        # Quality recommendations
        null_pct = data_profile.get("overall_null_pct", 0)
        if null_pct > 8:
            recs.append(
                f"🟡 DATA QUALITY: Null rate is {null_pct}%. "
                "Implement imputation strategy (median for numeric, mode for categorical)."
            )

        for col, cp in data_profile.get("column_profiles", {}).items():
            if cp.get("outliers", {}).get("pct_outliers", 0) > 5:
                recs.append(
                    f"⚠️  OUTLIERS: Column '{col}' has "
                    f"{cp['outliers']['pct_outliers']}% outliers — "
                    "apply IQR capping or investigate."
                )

        # Score-based overall recommendation
        if score >= 80:
            recs.append("✅ Data readiness is HIGH — proceed to feature engineering.")
        elif score >= 60:
            recs.append(
                "🟡 Data readiness is MODERATE — address important gaps before modelling."
            )
        elif score >= 40:
            recs.append(
                "🔴 Data readiness is LOW — resolve critical gaps before proceeding."
            )
        else:
            recs.append(
                "❌ Data readiness is VERY LOW — significant data sourcing effort required."
            )

        return recs

    @staticmethod
    def _build_risks(tier_results: dict, data_profile: dict) -> list[dict]:
        """Build structured risk list."""
        risks = []

        n_critical_missing = len(tier_results["critical"]["missing"])
        if n_critical_missing > 0:
            risks.append({
                "severity": "HIGH",
                "risk": "Missing critical features",
                "detail": f"{n_critical_missing} critical field(s) not found in dataset.",
                "mitigation": "Source data from CRM, billing, or support systems.",
            })

        null_pct = data_profile.get("overall_null_pct", 0)
        if null_pct > 15:
            risks.append({
                "severity": "HIGH",
                "risk": "High null rate",
                "detail": f"Overall null rate: {null_pct}%",
                "mitigation": "Implement data collection improvements and imputation pipeline.",
            })

        n_rows = data_profile.get("n_rows", 0)
        if n_rows < 500:
            risks.append({
                "severity": "HIGH",
                "risk": "Insufficient data volume",
                "detail": f"Only {n_rows} rows. Minimum 1,000 recommended for ML.",
                "mitigation": "Expand data collection window or augment with synthetic data.",
            })
        elif n_rows < 2000:
            risks.append({
                "severity": "MEDIUM",
                "risk": "Limited data volume",
                "detail": f"{n_rows} rows may limit model generalisation.",
                "mitigation": "Consider collecting more historical data.",
            })

        return risks
