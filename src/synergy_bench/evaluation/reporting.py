"""Raw-result persistence and dependency-free Phase 1 reports."""

from __future__ import annotations

import csv
from html import escape
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .contracts import RunResult
from .core import aggregate_results


def write_results(path: str | Path, results: Sequence[RunResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def append_results(path: str | Path, results: Sequence[RunResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def read_results(path: str | Path) -> list[RunResult]:
    results = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                values = json.loads(line)
                values["modalities"] = tuple(values["modalities"])
                results.append(RunResult(**values))
    return results


def write_summary_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("summary rows must be non-empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metric_svg(path: str | Path, rows: Sequence[dict[str, Any]], metric: str) -> None:
    selected = [row for row in rows if row["metric"] == metric]
    if not selected:
        raise ValueError(f"no rows found for metric {metric!r}")
    width, height, margin = 720, 360, 50
    labels = [escape(f"{row['family']} / {row['head']}") for row in selected]
    values = [float(row["mean"]) for row in selected]
    deviations = [float(row["std"]) for row in selected]
    upper = max(1.0, max(value + deviation for value, deviation in zip(values, deviations)))
    step = (width - 2 * margin) / len(values)
    bars = []
    texts = []
    for index, (label, value, deviation) in enumerate(zip(labels, values, deviations)):
        x = margin + index * step + step * 0.15
        bar_width = step * 0.7
        bar_height = (height - 2 * margin) * value / upper
        y = height - margin - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="#4c78a8" />'
        )
        error = (height - 2 * margin) * deviation / upper
        center = x + bar_width / 2
        bars.append(
            f'<line x1="{center:.1f}" y1="{y - error:.1f}" x2="{center:.1f}" '
            f'y2="{y + error:.1f}" stroke="#222" />'
        )
        texts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 28}" '
            f'text-anchor="middle" font-size="10">{label}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="16">'
        f'{escape(metric)}</text>'
        + "".join(bars + texts)
        + "</svg>"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def build_report(results_path: str | Path, table_path: str | Path, figure_path: str | Path) -> None:
    rows = aggregate_results(read_results(results_path))
    write_summary_csv(table_path, rows)
    write_metric_svg(figure_path, rows, "accuracy")
