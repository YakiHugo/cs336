#!/usr/bin/env python3
"""Render validation-loss JSONL logs as a dependency-free SVG learning curve."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


COLORS = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2")


def validation_points(path: Path, x_axis: str) -> list[tuple[float, float]]:
    points = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            metric = json.loads(line)
            if metric.get("event") == "validation":
                if x_axis == "steps":
                    x_value = float(metric["iteration"])
                else:
                    # Initial validation happens before the timed training loop.
                    wall_seconds = metric.get("wall_seconds", 0 if metric["iteration"] == 0 else None)
                    if wall_seconds is None:
                        raise ValueError(f"validation event lacks wall_seconds in {path}")
                    x_value = float(wall_seconds) / 60
                points.append((x_value, float(metric["validation_loss"])))
    if not points:
        raise ValueError(f"no validation events in {path}")
    return points


def parse_series(specification: str) -> tuple[str, Path]:
    try:
        label, path = specification.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("series must be LABEL=METRICS_PATH") from error
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", action="append", type=parse_series, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Validation learning curves")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--x-axis", choices=("steps", "wall-minutes"), default="steps")
    args = parser.parse_args()

    # Repeated labels join checkpoint/resume log segments into one trajectory.
    grouped: dict[str, dict[float, float]] = {}
    for label, path in args.series:
        points_by_step = grouped.setdefault(label, {})
        points_by_step.update(validation_points(path, args.x_axis))
    series = [(label, sorted(points.items())) for label, points in grouped.items()]
    all_points = [point for _, points in series for point in points]
    max_x = max(step for step, _ in all_points)
    transform_y = math.log10 if args.log_y else lambda value: value
    raw_min_y = min(transform_y(loss) for _, loss in all_points)
    raw_max_y = max(transform_y(loss) for _, loss in all_points)
    padding_y = max((raw_max_y - raw_min_y) * 0.08, 0.05)
    min_y, max_y = raw_min_y - padding_y, raw_max_y + padding_y

    width, height = 960, 600
    left, right, top, bottom = 86, 30, 120, 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_coordinate(x_value: float) -> float:
        return left + plot_width * x_value / max_x

    def y_coordinate(loss: float) -> float:
        return top + plot_height * (max_y - transform_y(loss)) / (max_y - min_y)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="600">{html.escape(args.title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x = left + plot_width * fraction
        x_value = max_x * fraction
        tick_label = f"{x_value:.1f}" if args.x_axis == "wall-minutes" else str(round(x_value))
        svg.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e5e7eb"/>',
                f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#374151">{tick_label}</text>',
            ]
        )
    for tick in range(6):
        fraction = tick / 5
        y = top + plot_height * fraction
        transformed_value = max_y - (max_y - min_y) * fraction
        value = 10**transformed_value if args.log_y else transformed_value
        label = f"{value:.1f}" if value < 100 else f"{value:.0f}"
        svg.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#374151">{label}</text>',
            ]
        )
    svg.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="15">{"wall-clock minutes" if args.x_axis == "wall-minutes" else "optimizer steps"}</text>',
            f'<text transform="translate(22 {top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="15">validation loss{" (log scale)" if args.log_y else ""}</text>',
        ]
    )
    for index, (label, points) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(f"{x_coordinate(x_value):.1f},{y_coordinate(loss):.1f}" for x_value, loss in points)
        svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x_value, loss in points:
            svg.append(f'<circle cx="{x_coordinate(x_value):.1f}" cy="{y_coordinate(loss):.1f}" r="4" fill="{color}"/>')
        legend_x = left + (index % 2) * 410
        legend_y = 66 + 28 * (index // 2)
        svg.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 26}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 34}" y="{legend_y + 5}" font-family="sans-serif" font-size="13">{html.escape(label)}</text>',
            ]
        )
    svg.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
