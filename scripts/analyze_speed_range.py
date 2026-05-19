#!/usr/bin/env python3
"""Plot speed-configuration behavior metrics for a contiguous experiment ID range.

Outputs:
  - speed_behavior_exp<start>_<end>.png
  - speed_behavior_exp<start>_<end>_per_experiment.csv
  - speed_behavior_exp<start>_<end>_group_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


RESULT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "experiments" / "testing_ground"

SPEED_ORDER = ["very_slow", "slow", "normal", "fast", "very_fast"]
SPEED_COLORS = {
    "very_slow": "#7f8c8d",
    "slow": "#3498db",
    "normal": "#2ecc71",
    "fast": "#f39c12",
    "very_fast": "#e74c3c",
}


def _infer_speed_label(exp: Dict[str, Any]) -> str:
    llm = exp.get("llm_output") or {}
    tasks = llm.get("tasks") or []
    if tasks:
        behavior = tasks[0].get("behavior") or {}
        s = str(behavior.get("speed", "")).strip().lower()
        if s in SPEED_ORDER:
            return s

    instr = str(exp.get("instruction", "")).lower()
    if "very slow" in instr:
        return "very_slow"
    if "slow" in instr:
        return "slow"
    if "very fast" in instr:
        return "very_fast"
    if "fast" in instr:
        return "fast"
    return "normal"


def _load_telemetry(exp_id: int, task_idx: int) -> List[Dict[str, Any]]:
    p = RESULT_BASE / f"exp_{exp_id:03d}" / f"telemetry_task{task_idx}.csv"
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _collect_metrics(start_id: int, end_id: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for exp_id in range(start_id, end_id + 1):
        result_path = RESULT_BASE / f"exp_{exp_id:03d}" / "result.json"
        if not result_path.exists():
            continue
        with result_path.open() as f:
            exp = json.load(f)

        task_results = exp.get("task_results", [])
        if not task_results:
            continue

        speed = _infer_speed_label(exp)
        durations = []
        steps = []
        v_samples: List[float] = []
        w_samples: List[float] = []

        for tr in task_results:
            durations.append(_safe_float(tr.get("duration_sec"), 0.0))
            steps.append(_safe_float(tr.get("total_steps"), 0.0))
            tidx = int(_safe_float(tr.get("task_idx"), 0))
            tel = _load_telemetry(exp_id, tidx)
            for row in tel:
                v_samples.append(abs(_safe_float(row.get("v_cmd"), 0.0)))
                w_samples.append(abs(_safe_float(row.get("w_cmd"), 0.0)))

        rows.append({
            "exp_id": exp_id,
            "speed": speed,
            "success": bool(exp.get("overall_success", False)),
            "mean_duration_sec": float(np.mean(durations)) if durations else math.nan,
            "mean_steps": float(np.mean(steps)) if steps else math.nan,
            "mean_abs_v_cmd": float(np.mean(v_samples)) if v_samples else math.nan,
            "mean_abs_w_cmd": float(np.mean(w_samples)) if w_samples else math.nan,
        })
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot(rows: List[Dict[str, Any]], start_id: int, end_id: int) -> Path:
    if not rows:
        raise RuntimeError("No experiment rows found in requested range.")

    grouped: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPEED_ORDER}
    for r in rows:
        if r["speed"] in grouped:
            grouped[r["speed"]].append(r)
        else:
            grouped["normal"].append(r)

    metrics = [
        ("mean_duration_sec", "Mean Time (s)"),
        ("mean_steps", "Mean Steps"),
        ("mean_abs_v_cmd", "Mean |Linear Velocity| (m/s)"),
        ("mean_abs_w_cmd", "Mean |Angular Velocity| (rad/s)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    x = np.arange(len(SPEED_ORDER))

    for ax, (key, title) in zip(axes, metrics):
        means = []
        for s in SPEED_ORDER:
            vals = [r[key] for r in grouped[s] if not math.isnan(r[key])]
            means.append(float(np.mean(vals)) if vals else 0.0)

        # Keep bars, but remove vertical error lines (no yerr).
        ax.bar(
            x,
            means,
            color=[SPEED_COLORS[s] for s in SPEED_ORDER],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.8,
            zorder=1,
        )

        # All per-experiment dots aligned on the same vertical line for each
        # speed bucket (no jitter).
        for i, s in enumerate(SPEED_ORDER):
            vals_ok = [
                r[key] for r in grouped[s]
                if (not math.isnan(r[key])) and r["success"]
            ]
            vals_fail = [
                r[key] for r in grouped[s]
                if (not math.isnan(r[key])) and (not r["success"])
            ]
            if not vals_ok and not vals_fail:
                continue
            if vals_ok:
                ax.scatter(
                    np.full(len(vals_ok), i),
                    vals_ok,
                    color="black",
                    s=26,
                    alpha=0.7,
                    marker="o",
                    zorder=3,
                )
            if vals_fail:
                ax.scatter(
                    np.full(len(vals_fail), i),
                    vals_fail,
                    color="black",
                    s=36,
                    alpha=0.9,
                    marker="x",
                    zorder=3,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(SPEED_ORDER, rotation=20)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Speed Behavior Summary (Exp {start_id}-{end_id})\n"
        "Bars: mean per speed, dots: per-experiment values",
        fontsize=12,
        fontweight="bold",
    )
    legend_handles = [
        Patch(facecolor="#888888", edgecolor="white", alpha=0.7, label="Group Mean (bar)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="black", alpha=0.7, markersize=6, label="Successful Experiment"),
        Line2D([0], [0], marker="x", color="black", markersize=7, label="Failed Experiment"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()

    out_png = RESULT_BASE / f"speed_behavior_exp{start_id:03d}_{end_id:03d}.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def main() -> None:
    p = argparse.ArgumentParser(description="Plot speed behavior metrics for experiment ID range.")
    p.add_argument("--start-id", type=int, default=43, help="Start experiment ID (inclusive).")
    p.add_argument("--end-id", type=int, default=57, help="End experiment ID (inclusive).")
    args = p.parse_args()

    rows = _collect_metrics(args.start_id, args.end_id)
    if not rows:
        print("No results found for requested range.")
        return

    out_png = _plot(rows, args.start_id, args.end_id)

    per_exp_csv = RESULT_BASE / f"speed_behavior_exp{args.start_id:03d}_{args.end_id:03d}_per_experiment.csv"
    _write_csv(
        per_exp_csv,
        rows,
        [
            "exp_id",
            "speed",
            "success",
            "mean_duration_sec",
            "mean_steps",
            "mean_abs_v_cmd",
            "mean_abs_w_cmd",
        ],
    )

    # Group summary CSV
    group_rows: List[Dict[str, Any]] = []
    for s in SPEED_ORDER:
        vals = [r for r in rows if r["speed"] == s]
        if not vals:
            continue
        def agg(key: str) -> tuple[float, float]:
            arr = np.array([r[key] for r in vals if not math.isnan(r[key])], dtype=float)
            if arr.size == 0:
                return (math.nan, math.nan)
            return float(arr.mean()), float(arr.std())

        d_m, d_s = agg("mean_duration_sec")
        st_m, st_s = agg("mean_steps")
        v_m, v_s = agg("mean_abs_v_cmd")
        w_m, w_s = agg("mean_abs_w_cmd")
        group_rows.append({
            "speed": s,
            "n_experiments": len(vals),
            "n_success": sum(1 for r in vals if r["success"]),
            "mean_duration_sec": d_m,
            "std_duration_sec": d_s,
            "mean_steps": st_m,
            "std_steps": st_s,
            "mean_abs_v_cmd": v_m,
            "std_abs_v_cmd": v_s,
            "mean_abs_w_cmd": w_m,
            "std_abs_w_cmd": w_s,
        })

    group_csv = RESULT_BASE / f"speed_behavior_exp{args.start_id:03d}_{args.end_id:03d}_group_summary.csv"
    _write_csv(
        group_csv,
        group_rows,
        [
            "speed",
            "n_experiments",
            "n_success",
            "mean_duration_sec",
            "std_duration_sec",
            "mean_steps",
            "std_steps",
            "mean_abs_v_cmd",
            "std_abs_v_cmd",
            "mean_abs_w_cmd",
            "std_abs_w_cmd",
        ],
    )

    print(f"Saved: {out_png}")
    print(f"Saved: {per_exp_csv}")
    print(f"Saved: {group_csv}")


if __name__ == "__main__":
    main()
