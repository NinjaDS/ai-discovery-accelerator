"""
core/orchestrator.py — Discovery Orchestrator

Simple sequential orchestrator that runs all agents in order, collects
results, and saves the final report to disk.
"""

from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from core.agents.data_profiler import DataProfiler
from core.agents.gap_analyst import GapAnalyst
from core.agents.feature_intelligence import FeatureIntelligence
from core.agents.model_selector import ModelSelector
from core.agents.insight_synthesiser import InsightSynthesiser
from core.utils import (
    get_logger,
    ensure_dir,
    save_json,
    save_markdown,
    build_metadata,
    print_progress,
)

logger = get_logger(__name__)


class DiscoveryOrchestrator:
    """
    Sequential orchestrator that runs the full ML discovery pipeline:

        DataProfiler → GapAnalyst → FeatureIntelligence
        → ModelSelector → InsightSynthesiser → Report

    Parameters
    ----------
    use_case : str
        Use case label (e.g. 'churn_prediction').
    task_type : str
        'classification' or 'regression'.
    output_dir : str | Path
        Directory where JSON and Markdown reports will be saved.
    aws_region : str
        AWS region for Bedrock (used by InsightSynthesiser).
    multi_source : bool
        Whether the dataset comes from multiple enterprise sources.
    """

    def __init__(
        self,
        use_case: str = "churn_prediction",
        task_type: str = "classification",
        output_dir: str | Path = "./output",
        aws_region: str = "us-east-1",
        multi_source: bool = False,
    ) -> None:
        self.use_case = use_case
        self.task_type = task_type
        self.output_dir = Path(output_dir)
        self.aws_region = aws_region
        self.multi_source = multi_source
        # Map use cases to their typical target column name
        self._target_col_map: dict[str, str] = {
            "churn_prediction": "churn",
            "fraud_detection": "is_fraud",
            "sales_forecasting": "revenue",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: "pd.DataFrame") -> dict[str, Any]:  # noqa: F821
        """
        Execute the full discovery pipeline on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to analyse.

        Returns
        -------
        dict with keys:
          metadata, data_profile, gap_report, feature_intelligence,
          model_recommendations, insights, run_status
        """
        logger.info(
            f"Starting discovery pipeline — use case: {self.use_case}, "
            f"task: {self.task_type}"
        )
        ensure_dir(self.output_dir)

        results: dict[str, Any] = {
            "metadata": build_metadata(self.use_case, self.task_type),
            "run_status": {},
        }

        # ── Step 1: Data Profiling ──────────────────────────────────────
        print_progress("Data Profiling", "running")
        try:
            profiler = DataProfiler(multi_source=self.multi_source)
            results["data_profile"] = profiler.profile(df, source_name=self.use_case)
            print_progress("Data Profiling", "done")
            results["run_status"]["data_profiler"] = "success"
        except Exception as exc:
            logger.error(f"DataProfiler failed: {exc}\n{traceback.format_exc()}")
            results["data_profile"] = {}
            results["run_status"]["data_profiler"] = f"error: {exc}"
            print_progress("Data Profiling", "error")

        # ── Step 2: Gap Analysis ────────────────────────────────────────
        print_progress("Gap Analysis", "running")
        try:
            analyst = GapAnalyst(use_case=self.use_case)
            results["gap_report"] = analyst.analyse(results["data_profile"])
            print_progress("Gap Analysis", "done")
            results["run_status"]["gap_analyst"] = "success"
        except Exception as exc:
            logger.error(f"GapAnalyst failed: {exc}\n{traceback.format_exc()}")
            results["gap_report"] = {}
            results["run_status"]["gap_analyst"] = f"error: {exc}"
            print_progress("Gap Analysis", "error")

        # ── Step 3: Feature Intelligence ────────────────────────────────
        print_progress("Feature Intelligence", "running")
        try:
            target_col = self._target_col_map.get(self.use_case, "target")
            fi = FeatureIntelligence(
                target_col=target_col,
                task_type=self.task_type,
            )
            results["feature_intelligence"] = fi.analyse(df)
            print_progress("Feature Intelligence", "done")
            results["run_status"]["feature_intelligence"] = "success"
        except Exception as exc:
            logger.error(f"FeatureIntelligence failed: {exc}\n{traceback.format_exc()}")
            results["feature_intelligence"] = {}
            results["run_status"]["feature_intelligence"] = f"error: {exc}"
            print_progress("Feature Intelligence", "error")

        # ── Step 4: Model Selection ─────────────────────────────────────
        print_progress("Model Selection", "running")
        try:
            selector = ModelSelector(use_case=self.use_case, task_type=self.task_type)
            results["model_recommendations"] = selector.recommend(
                results["data_profile"],
                results["gap_report"],
                results["feature_intelligence"],
            )
            print_progress("Model Selection", "done")
            results["run_status"]["model_selector"] = "success"
        except Exception as exc:
            logger.error(f"ModelSelector failed: {exc}\n{traceback.format_exc()}")
            results["model_recommendations"] = {}
            results["run_status"]["model_selector"] = f"error: {exc}"
            print_progress("Model Selection", "error")

        # ── Step 5: Insight Synthesis ────────────────────────────────────
        print_progress("Insight Synthesis", "running")
        try:
            synthesiser = InsightSynthesiser(
                use_case=self.use_case,
                task_type=self.task_type,
                aws_region=self.aws_region,
            )
            results["insights"] = synthesiser.synthesise(
                data_profile=results["data_profile"],
                gap_report=results["gap_report"],
                feature_intelligence=results["feature_intelligence"],
                model_recommendations=results["model_recommendations"],
            )
            print_progress("Insight Synthesis", "done")
            results["run_status"]["insight_synthesiser"] = "success"
        except Exception as exc:
            logger.error(f"InsightSynthesiser failed: {exc}\n{traceback.format_exc()}")
            results["insights"] = {}
            results["run_status"]["insight_synthesiser"] = f"error: {exc}"
            print_progress("Insight Synthesis", "error")

        # ── Save reports ─────────────────────────────────────────────────
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._save_reports(results, timestamp)

        logger.info("Discovery pipeline complete ✓")
        return results

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------

    def _save_reports(self, results: dict, timestamp: str) -> None:
        """Save JSON and Markdown reports to output_dir."""
        json_path = self.output_dir / f"report_{timestamp}.json"
        md_path = self.output_dir / f"report_{timestamp}.md"

        # JSON report (full)
        try:
            save_json(results, json_path)
            logger.info(f"  JSON report saved → {json_path}")
        except Exception as exc:
            logger.error(f"Failed to save JSON report: {exc}")

        # Markdown report (from insights or fallback)
        try:
            md_content = results.get("insights", {}).get("full_report", "")
            if not md_content:
                md_content = self._build_fallback_markdown(results, timestamp)
            save_markdown(md_content, md_path)
            logger.info(f"  Markdown report saved → {md_path}")
        except Exception as exc:
            logger.error(f"Failed to save Markdown report: {exc}")

        results["output_paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
        }

    @staticmethod
    def _build_fallback_markdown(results: dict, timestamp: str) -> str:
        """Build a minimal Markdown report when InsightSynthesiser output is missing."""
        use_case = results.get("metadata", {}).get("use_case", "Unknown")
        dp = results.get("data_profile", {})
        gr = results.get("gap_report", {})
        mr = results.get("model_recommendations", {})

        lines = [
            f"# ML Discovery Report — {use_case.replace('_', ' ').title()}",
            f"\n_Generated: {timestamp}_\n",
            "## Dataset Summary",
            f"- Rows: {dp.get('n_rows', 'N/A'):,}",
            f"- Columns: {dp.get('n_cols', 'N/A')}",
            f"- Null Rate: {dp.get('overall_null_pct', 'N/A')}%",
            "",
            "## Readiness",
            f"- Score: {gr.get('readiness_score', 'N/A')}/100",
            f"- {gr.get('readiness_bar', '')}",
            "",
            "## Primary Model Recommendation",
            f"- {mr.get('primary_recommendation', 'N/A')}",
            "",
            "## Run Status",
        ]
        for agent, status in results.get("run_status", {}).items():
            lines.append(f"- {agent}: {status}")

        return "\n".join(lines)
