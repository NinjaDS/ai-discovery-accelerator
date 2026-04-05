"""
core/agents/insight_synthesiser.py — Insight Synthesis Agent

Takes all previous agent outputs and uses Amazon Bedrock Claude to generate a
comprehensive natural language discovery report. Falls back to a template-based
report if Bedrock is unavailable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """You are an expert ML and data science advisor. You have just completed a thorough
analysis of a dataset for the use case: **{use_case}** (task type: {task_type}).

Here is the complete analysis output from our multi-agent system:

## Data Profile Summary
- Rows: {n_rows} | Columns: {n_cols}
- Overall null rate: {null_pct}%
- Memory: {memory_mb} MB
- Type distribution: {type_distribution}
- Warnings: {profile_warnings}

## Gap Analysis Report
- Data readiness score: {readiness_score}/100
- Missing critical fields: {missing_critical}
- Missing important fields: {missing_important}
- Key recommendations: {gap_recommendations}
- Key risks: {gap_risks}

## Feature Intelligence
- Top predictive features: {top_features}
- High-correlation pairs: {high_correlations}
- Dimensionality reduction recommended: {dimensionality_reduction}
- Key feature engineering suggestions: {feature_engineering}

## Model Recommendations
- Primary recommendation: {primary_model}
- Top 3 models: {top_models}
- Training strategy: {training_strategy}
- Evaluation strategy: {evaluation_strategy}

---

Based on this complete analysis, write a comprehensive **ML Discovery Report** with these exact sections:

1. **Executive Summary** (3-4 sentences, business-friendly, no jargon)
2. **Key Findings** (5-8 bullet points, mix of data quality, feature gaps, and opportunities)
3. **Data Readiness Assessment** (paragraph on overall data quality and readiness)
4. **Top ML Opportunities** (3-5 specific, actionable opportunities with expected business impact)
5. **Recommended Next Steps** (5-7 prioritised action items with rough effort estimates)
6. **Full Discovery Report** (comprehensive markdown, ~500-800 words, covering all dimensions)

Format your response as valid JSON with these exact keys:
- executive_summary: string
- key_findings: array of strings
- data_readiness_assessment: string
- top_opportunities: array of strings
- recommended_next_steps: array of strings
- full_report: markdown string

Write in a confident, actionable tone for a technical audience. Be specific — reference actual numbers from the analysis.
"""


# ---------------------------------------------------------------------------
# Insight Synthesiser agent
# ---------------------------------------------------------------------------

class InsightSynthesiser:
    """
    LLM-powered agent that synthesises all previous agent outputs into a
    comprehensive natural language discovery report using Amazon Bedrock Claude.

    Parameters
    ----------
    use_case : str
        High-level use case label (e.g. 'churn_prediction').
    task_type : str
        'classification' or 'regression'.
    aws_region : str
        AWS region where Bedrock is available.
    model_id : str
        Bedrock model ID to invoke.
    """

    MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def __init__(
        self,
        use_case: str = "churn_prediction",
        task_type: str = "classification",
        aws_region: str = "us-east-1",
        model_id: str | None = None,
    ) -> None:
        self.use_case = use_case
        self.task_type = task_type
        self.aws_region = aws_region
        self.model_id = model_id or self.MODEL_ID

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesise(
        self,
        data_profile: dict,
        gap_report: dict,
        feature_intelligence: dict,
        model_recommendations: dict,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive discovery report from all agent outputs.

        Parameters
        ----------
        data_profile         : output from DataProfiler.profile()
        gap_report           : output from GapAnalyst.analyse()
        feature_intelligence : output from FeatureIntelligence.analyse()
        model_recommendations: output from ModelSelector.recommend()

        Returns
        -------
        dict with keys:
          executive_summary, key_findings, data_readiness_assessment,
          top_opportunities, recommended_next_steps, full_report
        """
        logger.info("Synthesising insights from all agent outputs …")

        prompt = self._build_prompt(
            data_profile, gap_report, feature_intelligence, model_recommendations
        )

        try:
            result = self._invoke_bedrock(prompt)
            logger.info("  Bedrock synthesis complete ✓")
            return result
        except Exception as exc:
            logger.warning(f"  Bedrock unavailable ({exc}), using template fallback")
            return self._template_fallback(
                data_profile, gap_report, feature_intelligence, model_recommendations
            )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        data_profile: dict,
        gap_report: dict,
        feature_intelligence: dict,
        model_recs: dict,
    ) -> str:
        """Fill in the structured prompt template."""
        # Data profile fields
        n_rows = data_profile.get("n_rows", "N/A")
        n_cols = data_profile.get("n_cols", "N/A")
        null_pct = data_profile.get("overall_null_pct", 0)
        memory_mb = data_profile.get("memory_mb", "N/A")
        type_distribution = json.dumps(data_profile.get("type_distribution", {}))
        profile_warnings = "; ".join(data_profile.get("warnings", [])) or "None"

        # Gap report fields
        readiness_score = gap_report.get("readiness_score", 0)
        missing_critical = ", ".join(gap_report.get("missing_critical", [])) or "None"
        missing_important = ", ".join(gap_report.get("missing_important", [])) or "None"
        gap_recommendations = "; ".join(gap_report.get("recommendations", [])[:5]) or "None"
        gap_risks_raw = gap_report.get("risks", [])
        gap_risks = "; ".join(
            f"{r.get('severity','?')}: {r.get('risk','?')}" for r in gap_risks_raw
        ) or "None"

        # Feature intelligence fields
        top_features_raw = feature_intelligence.get("top_features", [])
        top_features = ", ".join(
            f.get("feature", str(f)) if isinstance(f, dict) else str(f)
            for f in top_features_raw[:6]
        ) or "N/A"
        high_correlations = len(feature_intelligence.get("high_correlations", []))
        dim_rec = feature_intelligence.get("dimensionality_recommendation", "N/A")
        fe_suggestions_raw = feature_intelligence.get("feature_engineering_suggestions", [])
        feature_engineering = "; ".join(
            s if isinstance(s, str) else str(s) for s in fe_suggestions_raw[:5]
        ) or "None"

        # Model recommendation fields
        primary_model = model_recs.get("primary_recommendation", "N/A")
        top_models = ", ".join(
            r.get("model_name", r.get("model_key", "")) 
            for r in model_recs.get("recommendations", [])[:3]
        )
        training_strategy = json.dumps(model_recs.get("training_strategy", {}))
        evaluation_strategy = json.dumps(model_recs.get("evaluation_strategy", {}))

        return _PROMPT_TEMPLATE.format(
            use_case=self.use_case,
            task_type=self.task_type,
            n_rows=n_rows,
            n_cols=n_cols,
            null_pct=null_pct,
            memory_mb=memory_mb,
            type_distribution=type_distribution,
            profile_warnings=profile_warnings,
            readiness_score=readiness_score,
            missing_critical=missing_critical,
            missing_important=missing_important,
            gap_recommendations=gap_recommendations,
            gap_risks=gap_risks,
            top_features=top_features,
            high_correlations=high_correlations,
            dimensionality_reduction=dim_rec,
            feature_engineering=feature_engineering,
            primary_model=primary_model,
            top_models=top_models,
            training_strategy=training_strategy,
            evaluation_strategy=evaluation_strategy,
        )

    # ------------------------------------------------------------------
    # Bedrock invocation
    # ------------------------------------------------------------------

    def _invoke_bedrock(self, prompt: str) -> dict[str, Any]:
        """Call Amazon Bedrock Claude and parse the JSON response."""
        import boto3  # noqa: PLC0415

        client = boto3.client("bedrock-runtime", region_name=self.aws_region)

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        })

        response = client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        response_body = json.loads(response["body"].read())
        raw_text = response_body["content"][0]["text"]

        return self._parse_llm_response(raw_text)

    def _parse_llm_response(self, raw_text: str) -> dict[str, Any]:
        """Extract and parse JSON from the LLM response text."""
        # Try to find JSON block in the response
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return self._validate_and_fill(parsed)
            except json.JSONDecodeError:
                pass

        # If direct parse fails, try to extract from code block
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
        if code_match:
            try:
                parsed = json.loads(code_match.group(1))
                return self._validate_and_fill(parsed)
            except json.JSONDecodeError:
                pass

        # Last resort: build minimal response from raw text
        logger.warning("  Could not parse JSON from LLM response; building minimal output")
        return {
            "executive_summary": raw_text[:500],
            "key_findings": ["See full report for details"],
            "data_readiness_assessment": "",
            "top_opportunities": [],
            "recommended_next_steps": [],
            "full_report": raw_text,
        }

    @staticmethod
    def _validate_and_fill(parsed: dict) -> dict[str, Any]:
        """Ensure all required keys are present with appropriate types."""
        defaults: dict[str, Any] = {
            "executive_summary": "",
            "key_findings": [],
            "data_readiness_assessment": "",
            "top_opportunities": [],
            "recommended_next_steps": [],
            "full_report": "",
        }
        for key, default in defaults.items():
            if key not in parsed:
                parsed[key] = default
            # Normalise lists
            if isinstance(default, list) and isinstance(parsed[key], str):
                parsed[key] = [parsed[key]]
        return parsed

    # ------------------------------------------------------------------
    # Template fallback (no LLM)
    # ------------------------------------------------------------------

    def _template_fallback(
        self,
        data_profile: dict,
        gap_report: dict,
        feature_intelligence: dict,
        model_recs: dict,
    ) -> dict[str, Any]:
        """Generate a structured report without calling Bedrock."""
        use_case_label = self.use_case.replace("_", " ").title()
        n_rows = data_profile.get("n_rows", 0)
        n_cols = data_profile.get("n_cols", 0)
        null_pct = data_profile.get("overall_null_pct", 0)
        readiness_score = gap_report.get("readiness_score", 0)
        missing_critical = gap_report.get("missing_critical", [])
        primary_model = model_recs.get("primary_recommendation", "xgboost_classifier")

        # Readiness tier
        if readiness_score >= 80:
            readiness_tier = "HIGH"
            readiness_note = "Data is well-suited for immediate modelling."
        elif readiness_score >= 60:
            readiness_tier = "MODERATE"
            readiness_note = "Address important gaps before full model training."
        elif readiness_score >= 40:
            readiness_tier = "LOW"
            readiness_note = "Significant data gaps must be resolved first."
        else:
            readiness_tier = "VERY LOW"
            readiness_note = "Substantial data sourcing effort is required."

        executive_summary = (
            f"Analysis of {n_rows:,} rows × {n_cols} columns for {use_case_label} "
            f"reveals a data readiness score of {readiness_score}/100 ({readiness_tier}). "
            f"The dataset has a null rate of {null_pct}% and "
            f"{'no critical gaps' if not missing_critical else f'{len(missing_critical)} critical gap(s)'}. "
            f"{readiness_note}"
        )

        key_findings = [
            f"Dataset contains {n_rows:,} rows and {n_cols} columns",
            f"Overall null rate: {null_pct}% — {'acceptable' if null_pct < 5 else 'needs attention'}",
            f"Data readiness score: {readiness_score}/100 ({readiness_tier})",
            f"Critical missing fields: {', '.join(missing_critical) if missing_critical else 'None'}",
            f"Recommended primary model: {primary_model.replace('_', ' ').title()}",
        ]
        for w in data_profile.get("warnings", [])[:3]:
            key_findings.append(w)

        top_features = feature_intelligence.get("top_features", [])
        if top_features:
            key_findings.append(
                f"Top predictive features: "
                + ", ".join(
                    f.get("feature", str(f)) if isinstance(f, dict) else str(f)
                    for f in top_features[:3]
                )
            )

        data_readiness_assessment = (
            f"The dataset achieves a readiness score of {readiness_score}/100. "
            f"With {null_pct}% null rate across {n_cols} features, "
            f"the data quality is {'good' if null_pct < 5 else 'moderate' if null_pct < 15 else 'poor'}. "
            + ("There are no critical missing fields, indicating strong data coverage. "
               if not missing_critical
               else f"Critical missing fields ({', '.join(missing_critical)}) must be sourced. ")
            + readiness_note
        )

        top_models_list = [
            r.get("model_name", r.get("model_key", "")) 
            for r in model_recs.get("recommendations", [])[:3]
        ]
        top_opportunities = [
            f"Build a {use_case_label} model using {top_models_list[0] if top_models_list else primary_model}",
            f"Feature engineering: create interaction features from top predictors",
            f"Automated retraining pipeline triggered by data drift detection",
        ]
        if not missing_critical:
            top_opportunities.append(
                "Expand to multi-class or probabilistic scoring for richer insights"
            )

        recommended_next_steps = []
        if missing_critical:
            recommended_next_steps.append(
                f"[Week 1] Source critical missing data: {', '.join(missing_critical)}"
            )
        recommended_next_steps += [
            "[Week 1-2] Clean nulls and apply imputation strategy",
            f"[Week 2] Engineer top features and encode categoricals",
            f"[Week 2-3] Train baseline {primary_model.replace('_', ' ').title()} model",
            "[Week 3] Evaluate with AUC-ROC, F1, and business cost metrics",
            "[Week 4] Package best model into a REST API for production scoring",
            "[Ongoing] Monitor model drift and schedule monthly retraining",
        ]

        full_report = self._build_markdown_report(
            use_case_label, n_rows, n_cols, null_pct, readiness_score,
            readiness_tier, missing_critical, gap_report, feature_intelligence,
            model_recs, executive_summary, key_findings, recommended_next_steps,
        )

        return {
            "executive_summary": executive_summary,
            "key_findings": key_findings,
            "data_readiness_assessment": data_readiness_assessment,
            "top_opportunities": top_opportunities,
            "recommended_next_steps": recommended_next_steps,
            "full_report": full_report,
        }

    @staticmethod
    def _build_markdown_report(
        use_case_label, n_rows, n_cols, null_pct, readiness_score,
        readiness_tier, missing_critical, gap_report, feature_intelligence,
        model_recs, executive_summary, key_findings, recommended_next_steps,
    ) -> str:
        """Build a formatted Markdown discovery report."""
        top_models = [
            f"- **{r.get('model_name', r.get('model_key', ''))}** — {r.get('rationale', '')}"
            for r in model_recs.get("recommendations", [])[:3]
        ]
        top_features = [
            f.get("feature", str(f)) if isinstance(f, dict) else str(f)
            for f in (
                feature_intelligence.get("feature_importance", [])
                or feature_intelligence.get("top_features", [])
            )[:6]
        ]
        fe_suggestions = (
            feature_intelligence.get("transformation_recommendations", [])
            or feature_intelligence.get("feature_engineering_suggestions", [])
        )
        eval_strategy = model_recs.get("evaluation_strategy", {})
        train_strategy = model_recs.get("training_strategy", {})

        return f"""# ML Discovery Report — {use_case_label}

## Executive Summary
{executive_summary}

## Key Findings
{"".join(f'- {f}{chr(10)}' for f in key_findings)}

## Dataset Overview
| Metric | Value |
|--------|-------|
| Rows | {n_rows:,} |
| Columns | {n_cols} |
| Null Rate | {null_pct}% |
| Readiness Score | {readiness_score}/100 ({readiness_tier}) |
| Readiness Bar | {gap_report.get('readiness_bar', '')} |

## Gap Analysis
**Score:** {readiness_score}/100

**Critical Missing Fields:** {', '.join(missing_critical) if missing_critical else '✅ None'}

**Recommendations:**
{"".join(f'- {r}{chr(10)}' for r in gap_report.get('recommendations', []))}

**Risks:**
{"".join(f'- **{r.get("severity","?")}**: {r.get("risk","?")} — {r.get("detail","")}{chr(10)}' for r in gap_report.get('risks', []))}

## Feature Intelligence
**Top Predictive Features:** {", ".join(top_features) if top_features else "N/A"}

**High-Correlation Pairs:** {len(feature_intelligence.get("high_correlations", []))}

**Feature Engineering Suggestions:**
{"".join(("- " + r.get("column", "") + ": " + ", ".join(r.get("suggestions", [])) + chr(10)) if isinstance(r, dict) else ("- " + str(r) + chr(10)) for r in fe_suggestions[:5])}

## Model Recommendations
{chr(10).join(top_models)}

**Training Strategy:** {train_strategy.get("strategy", "N/A")} — {train_strategy.get("reason", "")}

**Evaluation Strategy:** Primary metric: {eval_strategy.get("primary_metric", "AUC-ROC")}
Secondary: {", ".join(eval_strategy.get("secondary_metrics", []))}

## Recommended Next Steps
{"".join(f'{i+1}. {step}{chr(10)}' for i, step in enumerate(recommended_next_steps))}

---
*Generated by AI Discovery Accelerator v1.0.0*
"""
