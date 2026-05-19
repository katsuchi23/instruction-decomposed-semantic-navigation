#!/usr/bin/env python3
"""Post-experiment analysis and visualization.

Reads experiment results from outputs/experiments/testing_ground/ and generates:
  - Per-experiment path plots (global plan vs actual trajectory)
  - Per-category summary bar charts
  - Telemetry time-series plots (v, w, distance, costs)
  - Overall summary dashboard

Usage:
    python analyze_experiments.py                  # analyze all
    python analyze_experiments.py --ids 1,2,5      # specific experiments
    python analyze_experiments.py --category 3     # one category
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_visualization import plot_task_path


DEFAULT_RESULT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "experiments" / "testing_ground"
RESULT_BASE = DEFAULT_RESULT_BASE
VIZ_MAP_PGM: Optional[Path] = None
VIZ_MAP_YAML: Optional[Path] = None
VIZ_ROTATE_90: bool = True
UNITREE_REAL_PGM = (
    Path(__file__).resolve().parent
    / "outputs"
    / "experiments"
    / "unitree"
    / "main"
    / "rrc2_clean_crop_x-2.5_2.5_y-1.0_2.5_for_constraint.pgm"
)
UNITREE_REAL_YAML = (
    Path(__file__).resolve().parent
    / "outputs"
    / "experiments"
    / "unitree"
    / "main"
    / "rrc2_crop_y-1.0_2.5_for_constraint.yaml"
)

# Colors per category
CAT_COLORS = {
    1: "#4C72B0",  # blue
    2: "#55A868",  # green
    3: "#C44E52",  # red
    4: "#8172B2",  # purple
    5: "#CCB974",  # yellow
    6: "#64B5CD",  # cyan
    7: "#D68910",  # orange
}

CAT_NAMES = {
    1: "Baseline Only",
    2: "Preference Only",
    3: "Constraint Only",
    4: "Combination",
}


def _effective_category(cat: Any) -> int:
    """Normalize category IDs for analysis.

    Business rule: Category 8 should be treated as Baseline Only (Category 1).
    """
    try:
        c = int(cat)
    except (TypeError, ValueError):
        return 0
    return 1 if c == 8 else c


def _effective_category_name(cat: Any) -> str:
    c = _effective_category(cat)
    return CAT_NAMES.get(c, f"Cat {c}")


def _is_unitree_main_dir(result_base: Path) -> bool:
    parts = [p.lower() for p in result_base.parts]
    return "unitree" in parts and "main" in parts


def _default_rotate_for_result_dir(result_base: Path) -> bool:
    # Real-world Unitree maps should be displayed in native map orientation.
    return not _is_unitree_main_dir(result_base)


def _default_map_for_result_dir(result_base: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if _is_unitree_main_dir(result_base):
        return UNITREE_REAL_PGM, UNITREE_REAL_YAML

    # Prefer the map that was actually loaded during the experiment run.
    # This avoids subtle coordinate offsets when stress-test datasets use
    # scene-specific maps (e.g., scene2_* with slightly different origins).
    run_time: Optional[datetime] = None
    for rp in sorted(result_base.glob("exp_*/result.json")):
        try:
            ts = (json.loads(rp.read_text(encoding="utf-8")).get("timestamp") or "").strip()
            if ts:
                run_time = datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")
                break
        except Exception:
            continue

    log_dirs = [result_base, result_base.parent, result_base.parent.parent]
    seen_logs = set()
    ros_logs: List[Path] = []
    for d in log_dirs:
        if not d.exists():
            continue
        for lp in d.glob("ros_launch_*.log"):
            if lp not in seen_logs:
                ros_logs.append(lp)
                seen_logs.add(lp)

    def _log_dt(p: Path) -> Optional[datetime]:
        m = re.search(r"ros_launch_(\d{8}_\d{6})\.log$", p.name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            return None

    if run_time is not None:
        ros_logs.sort(key=lambda p: abs(((_log_dt(p) or run_time) - run_time).total_seconds()))
    else:
        ros_logs.sort(reverse=True)

    yaml_pat = re.compile(r"Loading yaml file:\s*(\S+\.yaml)")
    for log_path in ros_logs:
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = yaml_pat.findall(text)
        for y in reversed(hits):
            ypath = Path(y).expanduser()
            if ypath.exists():
                return None, ypath

    maps_dir = Path(__file__).resolve().parent / "ros2_ws" / "src" / "my_nav2_bringup" / "maps"
    key = result_base.name.strip().lower()
    if key == "reference_object":
        pgm = maps_dir / "reference_object.pgm"
        return pgm, pgm.with_suffix(".yaml")
    if key == "single_object":
        pgm = maps_dir / "single_object.pgm"
        return pgm, pgm.with_suffix(".yaml")
    return None, None


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_experiment(exp_id: int) -> Optional[Dict[str, Any]]:
    result_path = RESULT_BASE / f"exp_{exp_id:03d}" / "result.json"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        return json.load(f)


def load_all_experiments() -> List[Dict[str, Any]]:
    results = []
    if not RESULT_BASE.exists():
        return results
    for d in sorted(RESULT_BASE.iterdir()):
        if d.is_dir() and d.name.startswith("exp_"):
            rp = d / "result.json"
            if rp.exists():
                with open(rp) as f:
                    results.append(json.load(f))
    return results


def load_trajectory_csv(exp_id: int, task_idx: int = 0) -> Optional[np.ndarray]:
    csv_path = RESULT_BASE / f"exp_{exp_id:03d}" / f"trajectory_task{task_idx}.csv"
    if not csv_path.exists():
        return None
    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append([float(row["x"]), float(row["y"])])
    return np.array(data) if data else None


def load_global_path_csv(exp_id: int, task_idx: int = 0) -> Optional[np.ndarray]:
    csv_path = RESULT_BASE / f"exp_{exp_id:03d}" / f"global_path_task{task_idx}.csv"
    if not csv_path.exists():
        return None
    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append([float(row["x"]), float(row["y"])])
    return np.array(data) if data else None


def load_telemetry_csv(exp_id: int, task_idx: int = 0) -> Optional[List[Dict]]:
    csv_path = RESULT_BASE / f"exp_{exp_id:03d}" / f"telemetry_task{task_idx}.csv"
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for k in row:
                try:
                    row[k] = float(row[k])
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows if rows else None


# ═════════════════════════════════════════════════════════════════════════════
# Per-experiment path plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_experiment_path(exp: Dict[str, Any], save_dir: Optional[Path] = None) -> None:
    exp_id = exp["experiment_id"]
    if save_dir is None:
        save_dir = RESULT_BASE / f"exp_{exp_id:03d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for tr in exp.get("task_results", []):
        tidx = tr["task_idx"]
        plot_task_path(
            exp,
            tr,
            save_dir / f"path_task{tidx}.png",
            map_pgm=VIZ_MAP_PGM,
            map_yaml=VIZ_MAP_YAML,
            rotate_display_90=VIZ_ROTATE_90,
        )
        plot_task_path(
            exp,
            tr,
            save_dir / f"path_map_task{tidx}.png",
            map_pgm=VIZ_MAP_PGM,
            map_yaml=VIZ_MAP_YAML,
            rotate_display_90=VIZ_ROTATE_90,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Per-experiment telemetry plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_experiment_telemetry(exp: Dict[str, Any], save_dir: Optional[Path] = None) -> None:
    exp_id = exp["experiment_id"]
    if save_dir is None:
        save_dir = RESULT_BASE / f"exp_{exp_id:03d}"

    for tr in exp.get("task_results", []):
        tidx = tr["task_idx"]
        tel = load_telemetry_csv(exp_id, tidx)
        if not tel:
            continue

        steps = [r["step"] for r in tel]
        v = [r["v_cmd"] for r in tel]
        w = [r["w_cmd"] for r in tel]
        dist = [r["distance_to_target"] for r in tel]
        hdg = [r["heading_error_deg"] for r in tel]
        phi = [r["phase_error_deg"] for r in tel]
        sigma = [r["satisfaction_sigma"] for r in tel]
        cost_clear = [r.get("cost_clear", 0) for r in tel]
        cost_constr = [r.get("cost_constr", 0) for r in tel]
        cost_pref = [r.get("cost_pref", 0) for r in tel]

        fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

        # Panel 1: velocity commands
        axes[0].plot(steps, v, "b-", label="v (m/s)", linewidth=0.8)
        axes[0].plot(steps, w, "r-", label="ω (rad/s)", linewidth=0.8)
        axes[0].set_ylabel("Velocity")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Panel 2: distance and errors
        axes[1].plot(steps, dist, "k-", label="Distance to target (m)", linewidth=1)
        axes[1].axhline(y=tr.get("control_params", {}).get("r_min", 0.4),
                         color="g", linestyle="--", alpha=0.5, label="r_min")
        axes[1].axhline(y=tr.get("control_params", {}).get("r_max", 0.8),
                         color="g", linestyle="--", alpha=0.5, label="r_max")
        axes[1].set_ylabel("Distance (m)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # Panel 3: heading & phase error
        axes[2].plot(steps, hdg, "b-", label="Heading error (°)", linewidth=0.8)
        axes[2].plot(steps, phi, "r-", label="Phase error (°)", linewidth=0.8)
        axes[2].set_ylabel("Error (°)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        # Panel 4: costs
        axes[3].plot(steps, sigma, "k-", label="σ (satisfaction)", linewidth=0.8)
        axes[3].plot(steps, cost_clear, "b-", alpha=0.6, label="Clearance cost", linewidth=0.8)
        axes[3].plot(steps, cost_constr, "r-", alpha=0.6, label="Constraint cost", linewidth=0.8)
        axes[3].plot(steps, cost_pref, "g-", alpha=0.6, label="Preference cost", linewidth=0.8)
        axes[3].set_ylabel("Cost")
        axes[3].set_xlabel("Step")
        axes[3].legend(loc="upper right")
        axes[3].grid(True, alpha=0.3)

        fig.suptitle(
            f"Exp {exp_id} Task {tidx}: {exp['instruction'][:60]}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(save_dir / f"telemetry_task{tidx}.png", dpi=150)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Category summary charts
# ═════════════════════════════════════════════════════════════════════════════

def plot_category_summary(results: List[Dict[str, Any]]) -> None:
    """Bar charts comparing metrics across categories."""

    categories = sorted(set(_effective_category(r.get("category")) for r in results))

    # Aggregate per category
    cat_data = {}
    for cat in categories:
        cat_exps = [r for r in results if _effective_category(r.get("category")) == cat]
        successes = sum(1 for r in cat_exps if r["overall_success"])
        dist_errs = []
        hdg_errs = []
        phase_errs = []
        durations = []
        for r in cat_exps:
            llm_tasks = ((r.get("llm_output") or {}).get("tasks") or [])
            for tidx, tr in enumerate(r.get("task_results", [])):
                # Keep category metric plots robust by excluding failed tasks.
                if not tr.get("success", False):
                    continue
                final_dist = tr.get("final_distance_error", -1)
                if final_dist is not None and final_dist >= 0:
                    requested_dist = None
                    if 0 <= tidx < len(llm_tasks):
                        requested_dist = (
                            (((llm_tasks[tidx] or {}).get("main") or {}).get("termination") or {})
                            .get("distance_m")
                        )
                    if requested_dist is None:
                        cparams = tr.get("control_params") or {}
                        rmin = cparams.get("r_min")
                        rmax = cparams.get("r_max")
                        if rmin is not None and rmax is not None:
                            requested_dist = 0.5 * (float(rmin) + float(rmax))
                    if requested_dist is not None:
                        dist_errs.append(abs(float(final_dist) - float(requested_dist)))
                # final_heading_error_deg is signed (can be negative).
                # Only skip sentinel -1 used for missing data.
                hdg_val = tr.get("final_heading_error_deg", -1)
                if hdg_val != -1:
                    hdg_errs.append(abs(hdg_val))
                # final_phase_error_deg is signed (can be negative).
                # Only skip sentinel -1 used for missing data.
                phase_val = tr.get("final_phase_error_deg", -1)
                if phase_val != -1:
                    phase_errs.append(abs(phase_val))
                durations.append(tr.get("duration_sec", 0))

        cat_data[cat] = {
            "name": _effective_category_name(cat),
            "n": len(cat_exps),
            "success_rate": successes / max(1, len(cat_exps)),
            "mean_dist_err": np.mean(dist_errs) if dist_errs else 0,
            "std_dist_err": np.std(dist_errs) if dist_errs else 0,
            "mean_hdg_err": np.mean(hdg_errs) if hdg_errs else 0,
            "std_hdg_err": np.std(hdg_errs) if hdg_errs else 0,
            "mean_phase_err": np.mean(phase_errs) if phase_errs else 0,
            "std_phase_err": np.std(phase_errs) if phase_errs else 0,
            "mean_duration": np.mean(durations) if durations else 0,
        }

    names = [cat_data[c]["name"] for c in categories]
    colors = [CAT_COLORS.get(c, "#999999") for c in categories]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    # Success rate
    sr = [cat_data[c]["success_rate"] * 100 for c in categories]
    axes[0].bar(names, sr, color=colors)
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate by Category")
    axes[0].set_ylim(0, 110)
    for i, v in enumerate(sr):
        axes[0].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)

    # Distance error
    de = [cat_data[c]["mean_dist_err"] for c in categories]
    de_std = [cat_data[c]["std_dist_err"] for c in categories]
    axes[1].bar(names, de, yerr=de_std, color=colors, capsize=4)
    axes[1].set_ylabel("Distance Error (m)")
    axes[1].set_title("Mean Distance Error")

    # Heading error
    he = [cat_data[c]["mean_hdg_err"] for c in categories]
    he_std = [cat_data[c]["std_hdg_err"] for c in categories]
    axes[2].bar(names, he, yerr=he_std, color=colors, capsize=4)
    axes[2].set_ylabel("Heading Error (°)")
    axes[2].set_title("Mean Heading Error")

    # Phase error
    pe = [cat_data[c]["mean_phase_err"] for c in categories]
    pe_std = [cat_data[c]["std_phase_err"] for c in categories]
    axes[3].bar(names, pe, yerr=pe_std, color=colors, capsize=4)
    axes[3].set_ylabel("Phase Error (°)")
    axes[3].set_title("Mean Phase Error")

    # Duration (successful tasks only)
    dur = [cat_data[c]["mean_duration"] for c in categories]
    axes[4].bar(names, dur, color=colors)
    axes[4].set_ylabel("Duration (s)")
    axes[4].set_title("Mean Task Duration")

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Experiment Results by Category", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULT_BASE / "category_summary.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {RESULT_BASE / 'category_summary.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# Per-experiment success overview
# ═════════════════════════════════════════════════════════════════════════════

def plot_experiment_overview(results: List[Dict[str, Any]]) -> None:
    """Horizontal bar chart showing each experiment's outcome."""

    ids = [r["experiment_id"] for r in results]
    labels = [f"E{r['experiment_id']:02d}: {r['instruction'][:40]}" for r in results]
    success = [r["overall_success"] for r in results]
    colors = ["#55A868" if s else "#C44E52" for s in success]

    fig, ax = plt.subplots(figsize=(12, max(6, len(results) * 0.4)))
    y_pos = range(len(results))
    ax.barh(y_pos, [1] * len(results), color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("")
    ax.set_title("Experiment Outcomes (Green=Success, Red=Fail)")
    ax.set_xlim(0, 1.2)
    ax.get_xaxis().set_visible(False)
    ax.invert_yaxis()

    # Annotate with category + failure reason
    for i, r in enumerate(results):
        cat = _effective_category(r.get("category"))
        cat_label = _effective_category_name(cat)
        # Collect first task failure_reason if any
        fail_r = ""
        for tr in r.get("task_results", []):
            fr = tr.get("failure_reason", "")
            if fr:
                fail_r = fr[:30]
                break
        if not r["overall_success"] and r.get("error_message"):
            fail_r = fail_r or r["error_message"][:30]
        note = f"{cat_label}  {fail_r}" if fail_r else cat_label
        ax.text(1.05, i, note, fontsize=7, va="center",
                color=CAT_COLORS.get(cat, "#999"))

    fig.tight_layout()
    fig.savefig(RESULT_BASE / "experiment_overview.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {RESULT_BASE / 'experiment_overview.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global RESULT_BASE, VIZ_MAP_PGM, VIZ_MAP_YAML, VIZ_ROTATE_90
    parser = argparse.ArgumentParser(description="Analyze experiment results.")
    parser.add_argument("--result-dir", type=str, default=str(DEFAULT_RESULT_BASE),
                        help="Directory containing experiment outputs (exp_*/result.json).")
    parser.add_argument("--viz-map-pgm", type=str, default=None,
                        help="Optional map PGM for path/costmap overlays. Defaults by result-dir name.")
    parser.add_argument("--viz-map-yaml", type=str, default=None,
                        help="Optional map YAML. If omitted, uses <viz-map-pgm>.yaml.")
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma-separated experiment IDs.")
    parser.add_argument("--category", type=int, default=None,
                        help="Analyze only this category.")
    parser.add_argument("--skip-per-exp", action="store_true",
                        help="Skip per-experiment plots (only summary).")
    args = parser.parse_args()
    RESULT_BASE = Path(args.result_dir).expanduser()
    VIZ_ROTATE_90 = _default_rotate_for_result_dir(RESULT_BASE)
    if args.viz_map_pgm:
        VIZ_MAP_PGM = Path(args.viz_map_pgm).expanduser()
        VIZ_MAP_YAML = Path(args.viz_map_yaml).expanduser() if args.viz_map_yaml else VIZ_MAP_PGM.with_suffix(".yaml")
    else:
        VIZ_MAP_PGM, VIZ_MAP_YAML = _default_map_for_result_dir(RESULT_BASE)
    if VIZ_MAP_PGM is not None:
        print(f"Using map for overlays: {VIZ_MAP_PGM}")
    print(f"Display rotation: {'+90° CCW' if VIZ_ROTATE_90 else 'native map orientation'}")

    results = load_all_experiments()
    if not results:
        print("No experiment results found. Run experiments first.")
        return

    if args.ids:
        id_set = set(int(x) for x in args.ids.split(","))
        results = [r for r in results if r["experiment_id"] in id_set]
    elif args.category:
        results = [r for r in results if _effective_category(r.get("category")) == args.category]

    print(f"Loaded {len(results)} experiment results.\n")

    # Per-experiment plots
    if not args.skip_per_exp:
        for r in results:
            eid = r["experiment_id"]
            print(f"  Plotting experiment {eid}...")
            plot_experiment_path(r)
            plot_experiment_telemetry(r)

    # Summary plots
    plot_category_summary(results)
    plot_experiment_overview(results)

    print("\nDone. All plots saved.")


if __name__ == "__main__":
    main()
