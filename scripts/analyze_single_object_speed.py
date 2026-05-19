#!/usr/bin/env python3
"""Plot single-object speed comparison using selected experiment IDs.

Default comparison:
  - exp 016 -> low
  - exp 001 -> normal
  - exp 015 -> high

Outputs:
  - speed_mean_sampling_exp016_001_015.png
  - speed_mean_sampling_exp016_001_015.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_task_specs(specs: str) -> List[Tuple[int, str]]:
    parsed: List[Tuple[int, str]] = []
    for item in specs.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid task spec '{token}'. Expected format '<exp_id>:<label>'.")
        left, right = token.split(":", 1)
        parsed.append((int(left.strip()), right.strip()))
    if not parsed:
        raise ValueError("No task specs provided.")
    return parsed


def _load_samples(result_dir: Path, exp_id: int, task_idx: int) -> Dict[str, List[float]]:
    tel_path = result_dir / f"exp_{exp_id:03d}" / f"telemetry_task{task_idx}.csv"
    if not tel_path.exists():
        raise FileNotFoundError(f"Telemetry not found: {tel_path}")

    steps: List[float] = []
    abs_v: List[float] = []
    abs_w: List[float] = []
    with tel_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(_safe_float(row.get("step", "0"), 0.0))
            abs_v.append(abs(_safe_float(row.get("v_cmd", "0"), 0.0)))
            abs_w.append(abs(_safe_float(row.get("w_cmd", "0"), 0.0)))
    if not steps:
        raise RuntimeError(f"No telemetry rows in: {tel_path}")
    return {"steps": steps, "abs_v": abs_v, "abs_w": abs_w}


def _write_summary_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    fieldnames = [
        "exp_id",
        "label",
        "n_samples",
        "mean_abs_v_cmd",
        "std_abs_v_cmd",
        "mean_abs_w_cmd",
        "std_abs_w_cmd",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(
    result_dir: Path,
    task_specs: List[Tuple[int, str]],
    task_idx: int,
    out_png: Path,
    out_csv: Path,
) -> None:
    data: List[Dict[str, object]] = []
    for exp_id, label in task_specs:
        samples = _load_samples(result_dir, exp_id, task_idx)
        data.append({
            "exp_id": exp_id,
            "label": label,
            "steps": samples["steps"],
            "abs_v": samples["abs_v"],
            "abs_w": samples["abs_w"],
        })

    summary_rows: List[Dict[str, float]] = []
    for item in data:
        abs_v = np.array(item["abs_v"], dtype=float)
        abs_w = np.array(item["abs_w"], dtype=float)
        summary_rows.append({
            "exp_id": int(item["exp_id"]),
            "label": str(item["label"]),
            "n_samples": int(len(abs_v)),
            "mean_abs_v_cmd": float(np.mean(abs_v)),
            "std_abs_v_cmd": float(np.std(abs_v)),
            "mean_abs_w_cmd": float(np.mean(abs_w)),
            "std_abs_w_cmd": float(np.std(abs_w)),
        })
    _write_summary_csv(out_csv, summary_rows)

    labels = [str(d["label"]) for d in data]
    exp_ids = [int(d["exp_id"]) for d in data]
    x = np.arange(len(data), dtype=float)
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    metric_defs = [
        ("abs_v", "Mean |Linear Velocity| (m/s)"),
        ("abs_w", "Mean |Angular Velocity| (rad/s)"),
    ]

    for ax, (metric_key, title) in zip(axes, metric_defs):
        means: List[float] = []
        for idx, item in enumerate(data):
            vals = np.array(item[metric_key], dtype=float)
            vals = vals[~np.isnan(vals)]
            color = colors[idx % len(colors)]
            jitter = rng.normal(0.0, 0.04, size=len(vals))
            ax.scatter(
                np.full(len(vals), x[idx]) + jitter,
                vals,
                s=12,
                alpha=0.22,
                color=color,
                edgecolors="none",
                zorder=1,
            )

            mean_val = float(np.mean(vals)) if len(vals) else math.nan
            means.append(mean_val)
            ax.bar(x[idx], mean_val, color=color, alpha=0.35, width=0.55, zorder=2)
            ax.hlines(
                mean_val,
                x[idx] - 0.27,
                x[idx] + 0.27,
                color=color,
                linestyle="--",
                linewidth=1.3,
                zorder=3,
            )

        ax.plot(x, means, color="black", marker="o", linewidth=1.8, label="Mean trend", zorder=4)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lbl}\n(exp {eid})" for lbl, eid in zip(labels, exp_ids)])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Single Object Speed Comparison (Sampled Commands + Mean Trend)\n"
        "Tasks: 16 (low), 1 (normal), 15 (high)",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    default_result_dir = repo_root.parent / "outputs" / "experiments" / "single_object"

    parser = argparse.ArgumentParser(
        description="Plot sampled and mean linear/angular command speed for selected single-object tasks."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=default_result_dir,
        help="Directory containing exp_XXX folders (default: outputs/experiments/single_object).",
    )
    parser.add_argument(
        "--task-specs",
        type=str,
        default="16:low,1:normal,15:high",
        help="Comma-separated list: '<exp_id>:<label>' (default: 16:low,1:normal,15:high).",
    )
    parser.add_argument(
        "--task-idx",
        type=int,
        default=0,
        help="Task index inside each experiment (default: 0).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="speed_mean_sampling_exp016_001_015.png",
        help="Output PNG file name.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="speed_mean_sampling_exp016_001_015.csv",
        help="Output CSV file name.",
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    task_specs = _parse_task_specs(args.task_specs)
    out_png = result_dir / args.output_name
    out_csv = result_dir / args.output_csv

    _plot(result_dir, task_specs, args.task_idx, out_png, out_csv)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
