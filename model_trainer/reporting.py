"""Markdown reports for model comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_comparison_report(
    path: Path,
    *,
    title: str,
    selection_rule: str,
    rows: list[dict[str, Any]],
    best_run_name: str | None = None,
) -> None:
    """
    Write a markdown table comparing training runs.

    Each row should include at least: ``model``, ``target_mode``, ``mae``, ``rmse``, ``r2``,
    optionally ``cv_rmse_mean``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## Selection rule",
        "",
        selection_rule,
        "",
        "## Results",
        "",
        "| model | target | MAE | RMSE | R² | CV RMSE (mean) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        cv = r.get("cv_rmse_mean")
        cv_s = f"{cv:.4f}" if cv is not None else "—"
        lines.append(
            f"| {r.get('model', '')} | {r.get('target_mode', '')} | "
            f"{r.get('mae', float('nan')):.4f} | {r.get('rmse', float('nan')):.4f} | "
            f"{r.get('r2', float('nan')):.4f} | {cv_s} |",
        )
    lines.extend(
        [
            "",
            "## Best run",
            "",
            f"`{best_run_name}`" if best_run_name else "_None_",
            "",
        ],
    )
    path.write_text("\n".join(lines), encoding="utf-8")
