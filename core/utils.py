"""
core/utils.py — Shared utilities for the AI Discovery Accelerator

Common helpers for logging, formatting, file I/O, and markdown rendering
used across all agents and orchestrators.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a Rich-formatted logger for a given module name."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

console = Console()


def print_panel(title: str, content: str, style: str = "cyan") -> None:
    """Print a rich panel with title and content."""
    console.print(Panel(content, title=title, border_style=style))


def print_progress(step: str, status: str = "running") -> None:
    """Print a progress indicator."""
    icons = {"running": "⚙️ ", "done": "✅", "error": "❌", "warn": "⚠️ "}
    icon = icons.get(status, "•")
    console.print(f"  {icon}  {step}")


# ---------------------------------------------------------------------------
# Data readiness visual bar
# ---------------------------------------------------------------------------

def readiness_bar(score: float, width: int = 30) -> str:
    """
    Return an ASCII progress bar for a 0-100 readiness score.

    Example:  [█████████████░░░░░░░░░░░░░░░░░]  43%
    """
    filled = int(round(score / 100 * width))
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return f"[{bar}] {score:.0f}%"


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> dict:
    """Load a JSON file and return the parsed dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data as formatted JSON."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def save_markdown(content: str, path: str | Path) -> None:
    """Save a Markdown string to a file."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def timestamp_str() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def build_metadata(use_case: str, track: str) -> dict:
    """Build standard run metadata block included in every report."""
    return {
        "generated_at": timestamp_str(),
        "accelerator_version": "1.0.0",
        "use_case": use_case,
        "track": track,
    }
