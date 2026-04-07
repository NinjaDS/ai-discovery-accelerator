"""
main.py — AI Discovery Accelerator CLI Entry Point

Usage:
    python main.py --dataset path/to/data.csv --use-case churn_prediction --task-type classification
    python main.py --sample churn_prediction
    python main.py --sample fraud_detection --output ./reports
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-discovery-accelerator",
        description="Multi-agent ML opportunity discovery system powered by Amazon Bedrock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --sample churn_prediction
  python main.py --dataset data.csv --use-case churn_prediction --task-type classification
  python main.py --dataset txns.csv --use-case fraud_detection --task-type classification --output ./output
        """,
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset",
        metavar="PATH",
        help="Path to dataset file (CSV, JSON, Excel, Parquet)",
    )
    input_group.add_argument(
        "--sample",
        metavar="USE_CASE",
        choices=["churn_prediction", "fraud_detection", "sales_forecasting"],
        help="Generate synthetic sample data for the given use case",
    )

    parser.add_argument(
        "--use-case",
        default="churn_prediction",
        dest="use_case",
        help="Use case label (default: churn_prediction)",
    )
    parser.add_argument(
        "--task-type",
        default="classification",
        dest="task_type",
        choices=["classification", "regression"],
        help="ML task type (default: classification)",
    )
    parser.add_argument(
        "--output",
        default="./output",
        metavar="DIR",
        help="Output directory for reports (default: ./output)",
    )
    parser.add_argument(
        "--aws-region",
        default="us-west-2",
        dest="aws_region",
        help="AWS region for Bedrock (default: us-west-2)",
    )
    parser.add_argument(
        "--multi-source",
        action="store_true",
        dest="multi_source",
        help="Flag dataset as multi-source enterprise data",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Rich output (JSON paths only)",
    )

    return parser


def print_summary(results: dict, quiet: bool = False) -> None:
    """Print a Rich-formatted summary of the discovery results."""
    if quiet:
        out_paths = results.get("output_paths", {})
        print(out_paths.get("json", ""))
        print(out_paths.get("markdown", ""))
        return

    meta = results.get("metadata", {})
    dp = results.get("data_profile", {})
    gr = results.get("gap_report", {})
    mr = results.get("model_recommendations", {})
    insights = results.get("insights", {})

    # Header
    console.print()
    console.print(Panel(
        f"[bold cyan]AI Discovery Accelerator[/bold cyan]\n"
        f"Use Case: [yellow]{meta.get('use_case', 'N/A')}[/yellow]  |  "
        f"Task: [yellow]{meta.get('track', 'N/A')}[/yellow]  |  "
        f"Generated: [dim]{meta.get('generated_at', 'N/A')}[/dim]",
        border_style="cyan",
    ))

    # Dataset overview table
    overview = Table(title="📊 Dataset Overview", box=box.ROUNDED, border_style="blue")
    overview.add_column("Metric", style="bold")
    overview.add_column("Value")
    overview.add_row("Rows", f"{dp.get('n_rows', 'N/A'):,}" if isinstance(dp.get('n_rows'), int) else "N/A")
    overview.add_row("Columns", str(dp.get("n_cols", "N/A")))
    overview.add_row("Null Rate", f"{dp.get('overall_null_pct', 'N/A')}%")
    overview.add_row("Memory", f"{dp.get('memory_mb', 'N/A')} MB")
    overview.add_row("Duplicate Rows", str(dp.get("n_duplicate_rows", 0)))
    console.print(overview)

    # Readiness
    score = gr.get("readiness_score", 0)
    bar = gr.get("readiness_bar", "")
    score_color = "green" if score >= 75 else "yellow" if score >= 50 else "red"
    console.print(Panel(
        f"[{score_color}]{bar}[/{score_color}]  Score: [{score_color}]{score}/100[/{score_color}]",
        title="🎯 Data Readiness",
        border_style=score_color,
    ))

    # Missing fields
    missing_critical = gr.get("missing_critical", [])
    if missing_critical:
        console.print(f"  [red]❌ Missing critical fields:[/red] {', '.join(missing_critical)}")
    else:
        console.print("  [green]✅ No critical fields missing[/green]")

    # Top model
    primary = mr.get("primary_recommendation", "N/A")
    top3 = mr.get("recommendations", [])[:3]
    models_table = Table(title="🤖 Model Recommendations", box=box.SIMPLE, border_style="magenta")
    models_table.add_column("Rank", width=6)
    models_table.add_column("Model")
    models_table.add_column("Rationale")
    for rec in top3:
        models_table.add_row(
            f"#{rec.get('rank', '?')}",
            rec.get("model_name", rec.get("model_key", "N/A")),
            rec.get("rationale", "")[:80],
        )
    console.print(models_table)

    # Executive summary
    exec_summary = insights.get("executive_summary", "")
    if exec_summary:
        console.print(Panel(exec_summary, title="📝 Executive Summary", border_style="green"))

    # Next steps
    next_steps = insights.get("recommended_next_steps", [])
    if next_steps:
        console.print("\n[bold]📋 Recommended Next Steps:[/bold]")
        for i, step in enumerate(next_steps[:5], 1):
            console.print(f"  {i}. {step}")

    # Output paths
    out_paths = results.get("output_paths", {})
    if out_paths:
        console.print()
        console.print(Panel(
            f"JSON: [cyan]{out_paths.get('json', 'N/A')}[/cyan]\n"
            f"Markdown: [cyan]{out_paths.get('markdown', 'N/A')}[/cyan]",
            title="💾 Saved Reports",
            border_style="dim",
        ))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Lazy imports so errors surface cleanly
    try:
        from core.data_loader import DataLoader
        from core.orchestrator import DiscoveryOrchestrator
    except ImportError as exc:
        console.print(f"[red]Import error:[/red] {exc}")
        console.print("[yellow]Run:[/yellow] pip install -r requirements.txt")
        return 1

    # ── Load data ────────────────────────────────────────────────────────
    loader = DataLoader()
    try:
        if args.sample:
            use_case = args.sample
            if not args.quiet:
                console.print(f"[cyan]Generating synthetic data for:[/cyan] {use_case}")
            df = loader.load_sample_data(use_case)
        else:
            use_case = args.use_case
            if not args.quiet:
                console.print(f"[cyan]Loading dataset:[/cyan] {args.dataset}")
            df = loader.load(args.dataset)
    except Exception as exc:
        console.print(f"[red]Failed to load data:[/red] {exc}")
        return 1

    if not args.quiet:
        console.print(f"  Loaded [bold]{len(df):,}[/bold] rows × [bold]{len(df.columns)}[/bold] columns")

    # ── Run orchestrator ─────────────────────────────────────────────────
    orchestrator = DiscoveryOrchestrator(
        use_case=use_case,
        task_type=args.task_type,
        output_dir=args.output,
        aws_region=args.aws_region,
        multi_source=args.multi_source,
    )

    try:
        results = orchestrator.run(df)
    except Exception as exc:
        console.print(f"[red]Pipeline failed:[/red] {exc}")
        import traceback
        console.print(traceback.format_exc())
        return 1

    # ── Print summary ────────────────────────────────────────────────────
    print_summary(results, quiet=args.quiet)

    return 0


if __name__ == "__main__":
    sys.exit(main())
