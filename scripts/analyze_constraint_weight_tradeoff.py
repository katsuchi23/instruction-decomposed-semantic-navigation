#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


WEIGHT_DIR_RE = re.compile(r"^w_constraint_(\d+)p(\d+)$")


@dataclass
class RunMetrics:
    weight: float
    weight_dir: Path
    exp_dir: Path
    path_points: np.ndarray
    path_hash: str
    path_id: str
    goal_cost: float
    constraint_cost_raw: float
    constraint_cost_weighted: float
    total_proxy: float
    constraints_xy: Tuple[Tuple[float, float], ...]


@dataclass
class FamilyMetrics:
    path_id: str
    path_hash: str
    goal_cost: float
    constraint_cost_raw: float
    representative_points: np.ndarray
    weight_min: float
    weight_max: float


def parse_weight_from_dir_name(name: str) -> float:
    match = WEIGHT_DIR_RE.match(name)
    if not match:
        raise ValueError(f"Invalid weight directory name: {name}")
    major, minor = match.groups()
    return float(f"{major}.{minor}")


def discover_weight_dirs(base_dir: Path) -> List[Path]:
    dirs = [p for p in base_dir.iterdir() if p.is_dir() and WEIGHT_DIR_RE.match(p.name)]
    return sorted(dirs, key=lambda p: parse_weight_from_dir_name(p.name))


def discover_experiment_dir(weight_dir: Path) -> Path:
    candidates = sorted(p for p in weight_dir.iterdir() if p.is_dir() and p.name.startswith("exp_"))
    for candidate in candidates:
        if (candidate / "global_path_task0.csv").exists() and (candidate / "result.json").exists():
            return candidate
    raise FileNotFoundError(f"No valid experiment directory found under {weight_dir}")


def load_path_points(path_csv: Path) -> np.ndarray:
    points: List[Tuple[float, float]] = []
    with path_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append((float(row["x"]), float(row["y"])))
    if len(points) < 2:
        raise ValueError(f"Path file {path_csv} has fewer than 2 points")
    return np.asarray(points, dtype=float)


def load_constraint_locations(result_json: Path) -> Tuple[Tuple[float, float], ...]:
    with result_json.open() as handle:
        payload = json.load(handle)
    task_results = payload.get("task_results", [])
    if not task_results:
        return ()
    raw_locations = task_results[0].get("constraint_locations", [])
    locations: List[Tuple[float, float]] = []
    for item in raw_locations:
        if isinstance(item, list) and len(item) == 2:
            locations.append((float(item[0]), float(item[1])))
    return tuple(locations)


def polyline_length(points: np.ndarray) -> float:
    deltas = np.diff(points, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def unweighted_constraint_cost(
    points: np.ndarray,
    constraints_xy: Iterable[Tuple[float, float]],
    radius_m: float,
) -> float:
    radius = max(radius_m, 1e-6)
    constraints = list(constraints_xy)
    if not constraints:
        return 0.0

    total = 0.0
    # A* adds penalty at each entered neighbor cell; skip start node to mirror that.
    for x, y in points[1:]:
        for cx, cy in constraints:
            distance = math.hypot(x - cx, y - cy)
            if distance < radius:
                total += 1.0 - distance / radius
    return total


def path_signature(points: np.ndarray) -> str:
    rounded = np.round(points, 4)
    content = "\n".join(f"{x:.4f},{y:.4f}" for x, y in rounded)
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]


def collect_metrics(base_dir: Path, radius_m: float) -> List[RunMetrics]:
    metrics: List[RunMetrics] = []
    weight_dirs = discover_weight_dirs(base_dir)
    for weight_dir in weight_dirs:
        weight = parse_weight_from_dir_name(weight_dir.name)
        exp_dir = discover_experiment_dir(weight_dir)

        path_points = load_path_points(exp_dir / "global_path_task0.csv")
        constraints_xy = load_constraint_locations(exp_dir / "result.json")

        goal_cost = polyline_length(path_points)
        constr_raw = unweighted_constraint_cost(path_points, constraints_xy, radius_m=radius_m)
        constr_weighted = weight * constr_raw
        total_proxy = goal_cost + constr_weighted

        metrics.append(
            RunMetrics(
                weight=weight,
                weight_dir=weight_dir,
                exp_dir=exp_dir,
                path_points=path_points,
                path_hash=path_signature(path_points),
                path_id="",
                goal_cost=goal_cost,
                constraint_cost_raw=constr_raw,
                constraint_cost_weighted=constr_weighted,
                total_proxy=total_proxy,
                constraints_xy=constraints_xy,
            )
        )

    # Stable path-family IDs ordered by first appearance over ascending weight.
    hash_to_id: Dict[str, str] = {}
    for row in metrics:
        if row.path_hash not in hash_to_id:
            hash_to_id[row.path_hash] = f"P{len(hash_to_id) + 1}"
        row.path_id = hash_to_id[row.path_hash]

    return metrics


def _path_id_index(path_id: str) -> int:
    return int(path_id[1:])


def summarize_families(metrics: List[RunMetrics]) -> List[FamilyMetrics]:
    by_path: Dict[str, List[RunMetrics]] = {}
    for row in metrics:
        by_path.setdefault(row.path_id, []).append(row)

    families: List[FamilyMetrics] = []
    for path_id, rows in by_path.items():
        rows_sorted = sorted(rows, key=lambda r: r.weight)
        representative = rows_sorted[0]
        families.append(
            FamilyMetrics(
                path_id=path_id,
                path_hash=representative.path_hash,
                goal_cost=representative.goal_cost,
                constraint_cost_raw=representative.constraint_cost_raw,
                representative_points=representative.path_points,
                weight_min=rows_sorted[0].weight,
                weight_max=rows_sorted[-1].weight,
            )
        )

    return sorted(families, key=lambda f: _path_id_index(f.path_id))


def write_summary_csv(metrics: List[RunMetrics], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "weight",
                "path_id",
                "path_hash",
                "goal_cost_path_length",
                "constraint_cost_raw",
                "constraint_cost_weighted",
                "total_proxy_goal_plus_weighted_constraint",
                "num_points",
                "exp_dir",
            ]
        )
        for row in metrics:
            writer.writerow(
                [
                    f"{row.weight:.4f}",
                    row.path_id,
                    row.path_hash,
                    f"{row.goal_cost:.6f}",
                    f"{row.constraint_cost_raw:.6f}",
                    f"{row.constraint_cost_weighted:.6f}",
                    f"{row.total_proxy:.6f}",
                    len(row.path_points),
                    str(row.exp_dir),
                ]
            )


def infer_grid_resolution_from_paths(metrics: List[RunMetrics]) -> float:
    deltas: List[float] = []
    for row in metrics:
        pts = row.path_points
        if len(pts) < 2:
            continue
        diff = np.abs(np.diff(pts, axis=0))
        for value in diff.reshape(-1):
            if value > 1e-6:
                deltas.append(float(value))
    if not deltas:
        return 0.05
    deltas.sort()
    # Robust small-step estimate; typical grid resolution appears frequently.
    candidate = float(np.median(deltas[: max(5, min(50, len(deltas)))]))
    if candidate <= 0.0:
        return 0.05
    return candidate


def write_dominance_csv(metrics: List[RunMetrics], output_csv: Path, resolution_m: float) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "weight",
                "goal_cost_m",
                "goal_cost_planner_units",
                "weighted_constraint_cost",
                "constraint_to_goal_ratio",
                "dominant_term",
            ]
        )
        for row in metrics:
            goal_planner = row.goal_cost / max(1e-9, resolution_m)
            ratio = row.constraint_cost_weighted / max(1e-9, goal_planner)
            dominant = "constraint" if row.constraint_cost_weighted >= goal_planner else "goal"
            writer.writerow(
                [
                    f"{row.weight:.6f}",
                    f"{row.goal_cost:.6f}",
                    f"{goal_planner:.6f}",
                    f"{row.constraint_cost_weighted:.6f}",
                    f"{ratio:.6f}",
                    dominant,
                ]
            )


def _find_first_dominance_crossing(weights: np.ndarray, diff: np.ndarray) -> Optional[float]:
    # diff = weighted_constraint - goal
    for i in range(len(weights) - 1):
        d0 = float(diff[i])
        d1 = float(diff[i + 1])
        if d0 == 0.0:
            return float(weights[i])
        if d0 < 0.0 <= d1:
            w0, w1 = float(weights[i]), float(weights[i + 1])
            if abs(d1 - d0) < 1e-12:
                return w0
            t = -d0 / (d1 - d0)
            return w0 + t * (w1 - w0)
    if len(weights) > 0 and diff[0] >= 0.0:
        return float(weights[0])
    return None


def plot_dominance_comparison(
    metrics: List[RunMetrics],
    output_plot: Path,
    resolution_m: float,
) -> Optional[float]:
    weights = np.asarray([m.weight for m in metrics], dtype=float)
    goal = np.asarray([m.goal_cost / max(1e-9, resolution_m) for m in metrics], dtype=float)
    constr_weighted = np.asarray([m.constraint_cost_weighted for m in metrics], dtype=float)
    diff = constr_weighted - goal
    crossing_w = _find_first_dominance_crossing(weights, diff)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(weights, goal, "o-", linewidth=2.4, color="#1f77b4", label="Goal cost (planner units)")
    ax.plot(
        weights,
        constr_weighted,
        "s-",
        linewidth=2.4,
        color="#d62728",
        label="Weighted constraint cost",
    )

    # Shade dominant region directly for quick interpretation.
    ax.fill_between(
        weights,
        goal,
        constr_weighted,
        where=(constr_weighted >= goal),
        interpolate=True,
        color="#d62728",
        alpha=0.14,
        label="Constraint-dominant region",
    )
    ax.fill_between(
        weights,
        goal,
        constr_weighted,
        where=(constr_weighted < goal),
        interpolate=True,
        color="#1f77b4",
        alpha=0.12,
        label="Goal-dominant region",
    )

    if crossing_w is not None:
        ax.axvline(crossing_w, color="0.35", linestyle="--", linewidth=1.6)
        y_mid = float(np.interp(crossing_w, weights, goal))
        ax.scatter([crossing_w], [y_mid], color="black", s=50, zorder=6)
        ax.annotate(
            f"Dominance threshold\nw ≈ {crossing_w:.3f}",
            xy=(crossing_w, y_mid),
            xytext=(10, 18),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75"),
        )

    ax.set_title("When Weighted Constraint Cost Overtakes Goal Cost")
    ax.set_xlabel("Constraint weight w (W_CONSTRAINT_PATH)")
    ax.set_ylabel("Cost contribution (planner-comparable units)")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left")
    fig.tight_layout()

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return crossing_w


def _marker_for_path(path_id: str) -> str:
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    idx = max(0, int(path_id[1:]) - 1)
    return marker_cycle[idx % len(marker_cycle)]


def _color_for_path(path_id: str):
    cmap = plt.cm.get_cmap("tab10")
    idx = max(0, _path_id_index(path_id) - 1)
    return cmap(idx % 10)


def _compute_objective_envelope(
    families: List[FamilyMetrics],
    w_min: float,
    w_max: float,
    resolution_m: float,
    samples: int = 1200,
):
    ws = np.linspace(w_min, w_max, samples)
    objective_by_family = []
    for fam in families:
        goal_planner = fam.goal_cost / max(1e-9, resolution_m)
        objective_by_family.append(goal_planner + ws * fam.constraint_cost_raw)
    values = np.vstack(objective_by_family)  # [num_families, samples]
    winner_idx = np.argmin(values, axis=0)
    envelope = values[winner_idx, np.arange(samples)]

    switches: List[Tuple[float, str, str]] = []
    prev = winner_idx[0]
    for i in range(1, len(ws)):
        cur = winner_idx[i]
        if cur == prev:
            continue
        from_family = families[prev]
        to_family = families[cur]
        denom = from_family.constraint_cost_raw - to_family.constraint_cost_raw
        if abs(denom) > 1e-9:
            from_goal = from_family.goal_cost / max(1e-9, resolution_m)
            to_goal = to_family.goal_cost / max(1e-9, resolution_m)
            w_cross = (to_goal - from_goal) / denom
        else:
            w_cross = ws[i]
        # Clip to visible domain for robust annotations.
        w_cross = float(np.clip(w_cross, w_min, w_max))
        switches.append((w_cross, from_family.path_id, to_family.path_id))
        prev = cur

    return ws, values, winner_idx, envelope, switches


def _plot_objective_envelope_on_axis(
    ax,
    metrics: List[RunMetrics],
    families: List[FamilyMetrics],
    resolution_m: float,
) -> None:
    weights = np.asarray([m.weight for m in metrics], dtype=float)
    totals = np.asarray(
        [
            (m.goal_cost / max(1e-9, resolution_m)) + m.constraint_cost_weighted
            for m in metrics
        ],
        dtype=float,
    )
    w_min = float(weights.min())
    w_max = float(weights.max())
    ws, values, _, envelope, switches = _compute_objective_envelope(
        families=families,
        w_min=w_min,
        w_max=w_max,
        resolution_m=resolution_m,
    )

    # Candidate family lines.
    for idx, fam in enumerate(families):
        color = _color_for_path(fam.path_id)
        ax.plot(
            ws,
            values[idx],
            color=color,
            linewidth=1.6,
            alpha=0.85,
            label=f"{fam.path_id}: J=L'+{fam.constraint_cost_raw:.2f}w",
        )

    # Lower envelope (winner at each weight).
    ax.plot(ws, envelope, color="black", linewidth=2.8, label="Lower envelope (selected path)")

    # Observed run points.
    for i, row in enumerate(metrics):
        color = _color_for_path(row.path_id)
        marker = _marker_for_path(row.path_id)
        ax.scatter(
            row.weight,
            totals[i],
            s=72,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )

    # Mark switching thresholds.
    ymin, ymax = ax.get_ylim()
    text_y = ymin + 0.92 * (ymax - ymin)
    for w_cross, from_id, to_id in switches:
        ax.axvline(w_cross, color="0.45", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.text(
            w_cross,
            text_y,
            f"{from_id}\u2192{to_id}\n@w={w_cross:.2f}",
            fontsize=8,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.8),
        )

    ax.set_xlabel("Constraint weight w (W_CONSTRAINT_PATH)")
    ax.set_ylabel("Planner objective J = Goal' + w\u00b7Constraint")
    ax.set_title("Objective Envelope and Path-Switch Thresholds")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


def _plot_path_families_on_axis(
    ax,
    metrics: List[RunMetrics],
    families: List[FamilyMetrics],
    radius_m: float,
) -> None:
    for fam in families:
        pts = fam.representative_points
        color = _color_for_path(fam.path_id)
        label = (
            f"{fam.path_id}: w={fam.weight_min:g}-{fam.weight_max:g}, "
            f"L={fam.goal_cost:.2f}, C={fam.constraint_cost_raw:.2f}"
        )
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.4, label=label)

    if metrics and metrics[0].constraints_xy:
        for i, (cx, cy) in enumerate(metrics[0].constraints_xy):
            label = "Constraint anchor" if i == 0 else None
            ax.scatter([cx], [cy], color="red", marker="x", s=90, linewidths=2.0, label=label, zorder=6)
            circle = plt.Circle((cx, cy), radius_m, color="red", fill=False, linestyle="--", alpha=0.35)
            ax.add_patch(circle)

    start = metrics[0].path_points[0]
    end = metrics[0].path_points[-1]
    ax.scatter([start[0]], [start[1]], color="black", marker="o", s=40, label="Start")
    ax.scatter([end[0]], [end[1]], color="black", marker="*", s=55, label="Goal ring point")

    handles, labels = ax.get_legend_handles_labels()
    dedup: Dict[str, Line2D] = {}
    for handle, label in zip(handles, labels):
        if label not in dedup:
            dedup[label] = handle
    ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Chosen Path Families and Weight Regimes")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def plot_metrics(metrics: List[RunMetrics], radius_m: float, output_plot: Path) -> None:
    weights = np.asarray([m.weight for m in metrics], dtype=float)
    goal = np.asarray([m.goal_cost for m in metrics], dtype=float)
    constr_raw = np.asarray([m.constraint_cost_raw for m in metrics], dtype=float)
    constr_weighted = np.asarray([m.constraint_cost_weighted for m in metrics], dtype=float)
    total_proxy = np.asarray([m.total_proxy for m in metrics], dtype=float)
    path_index = np.asarray([int(m.path_id[1:]) for m in metrics], dtype=int)

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(weights.min()), vmax=float(weights.max()))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_cost, ax_tradeoff, ax_switch, ax_paths = axes.flatten()

    # Panel 1: cost components vs weight
    ax_cost.plot(weights, goal, "o-", linewidth=2.0, label="Goal cost (path length)")
    ax_cost.plot(weights, constr_raw, "s--", linewidth=2.0, label="Constraint cost (raw)")
    ax_cost.plot(weights, constr_weighted, "d-", linewidth=2.0, label="Constraint cost (weighted)")
    ax_cost.plot(weights, total_proxy, "x-.", linewidth=2.0, label="Total proxy")
    ax_cost.set_xlabel("W_CONSTRAINT_PATH")
    ax_cost.set_ylabel("Cost")
    ax_cost.set_title("Cost Terms Across Constraint Weight Sweep")
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(loc="best")

    # Panel 2: goal vs raw constraint tradeoff
    ax_tradeoff.plot(goal, constr_raw, color="0.75", linewidth=1.2, zorder=1)
    for row in metrics:
        marker = _marker_for_path(row.path_id)
        color = cmap(norm(row.weight))
        ax_tradeoff.scatter(
            row.goal_cost,
            row.constraint_cost_raw,
            s=85,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=2,
        )
        ax_tradeoff.annotate(
            f"{row.weight:g}",
            (row.goal_cost, row.constraint_cost_raw),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax_tradeoff.set_xlabel("Goal cost (path length)")
    ax_tradeoff.set_ylabel("Constraint cost (raw)")
    ax_tradeoff.set_title("Goal vs Constraint Tradeoff (Chosen Path per Weight)")
    ax_tradeoff.grid(True, alpha=0.3)

    unique_path_ids = sorted({m.path_id for m in metrics}, key=lambda s: int(s[1:]))
    path_handles = [
        Line2D(
            [0],
            [0],
            marker=_marker_for_path(pid),
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            label=f"{pid}",
        )
        for pid in unique_path_ids
    ]
    ax_tradeoff.legend(handles=path_handles, title="Path family", loc="best")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_tradeoff, fraction=0.046, pad=0.04, label="W_CONSTRAINT_PATH")

    # Panel 3: selected path family vs weight
    ax_switch.step(weights, path_index, where="mid", color="0.45", linewidth=1.5)
    ax_switch.scatter(
        weights,
        path_index,
        c=weights,
        cmap=cmap,
        norm=norm,
        s=90,
        edgecolors="black",
        linewidths=0.6,
    )
    for row in metrics:
        ax_switch.annotate(
            row.path_id,
            (row.weight, int(row.path_id[1:])),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax_switch.set_xlabel("W_CONSTRAINT_PATH")
    ax_switch.set_ylabel("Selected path family")
    ax_switch.set_yticks([int(pid[1:]) for pid in unique_path_ids])
    ax_switch.set_yticklabels(unique_path_ids)
    ax_switch.set_title("Where Path Switching Happens")
    ax_switch.grid(True, alpha=0.3)

    # Panel 4: path overlay by family
    family_representative: Dict[str, RunMetrics] = {}
    family_weights: Dict[str, List[float]] = {}
    for row in metrics:
        family_weights.setdefault(row.path_id, []).append(row.weight)
        family_representative.setdefault(row.path_id, row)

    for path_id in unique_path_ids:
        rep = family_representative[path_id]
        ws = sorted(family_weights[path_id])
        label = f"{path_id} (w={ws[0]:g}-{ws[-1]:g})" if len(ws) > 1 else f"{path_id} (w={ws[0]:g})"
        pts = rep.path_points
        ax_paths.plot(pts[:, 0], pts[:, 1], linewidth=2.0, label=label)

    if metrics and metrics[0].constraints_xy:
        for cx, cy in metrics[0].constraints_xy:
            ax_paths.scatter([cx], [cy], color="red", marker="x", s=80, linewidths=2.0, label="Constraint anchor")
            circle = plt.Circle((cx, cy), radius_m, color="red", fill=False, linestyle="--", alpha=0.45)
            ax_paths.add_patch(circle)

    start = metrics[0].path_points[0]
    end = metrics[0].path_points[-1]
    ax_paths.scatter([start[0]], [start[1]], color="black", s=40, marker="o", label="Start")
    ax_paths.scatter([end[0]], [end[1]], color="black", s=50, marker="*", label="Goal ring point")

    # Deduplicate legend entries on path panel.
    handles, labels = ax_paths.get_legend_handles_labels()
    dedup: Dict[str, Line2D] = {}
    for handle, label in zip(handles, labels):
        if label not in dedup:
            dedup[label] = handle
    ax_paths.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8)
    ax_paths.set_xlabel("x [m]")
    ax_paths.set_ylabel("y [m]")
    ax_paths.set_title("Chosen Global Path Families")
    ax_paths.grid(True, alpha=0.3)
    ax_paths.set_aspect("equal", adjustable="box")

    fig.suptitle("Constraint Stress Test: Goal Cost vs Constraint Cost", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_report_figures(
    metrics: List[RunMetrics],
    radius_m: float,
    resolution_m: float,
    output_envelope: Path,
    output_paths: Path,
    output_compact: Path,
) -> None:
    families = summarize_families(metrics)

    # Figure 1: objective envelope only (single-panel, threshold-focused).
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    _plot_objective_envelope_on_axis(
        ax1,
        metrics=metrics,
        families=families,
        resolution_m=resolution_m,
    )
    fig1.tight_layout()
    output_envelope.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(output_envelope, dpi=260, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: path geometry only (single-panel, spatial interpretation).
    fig2, ax2 = plt.subplots(figsize=(8.5, 6.8))
    _plot_path_families_on_axis(ax2, metrics=metrics, families=families, radius_m=radius_m)
    fig2.tight_layout()
    output_paths.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(output_paths, dpi=260, bbox_inches="tight")
    plt.close(fig2)

    # Optional compact two-panel image for report space constraints.
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5.6))
    _plot_objective_envelope_on_axis(
        ax3,
        metrics=metrics,
        families=families,
        resolution_m=resolution_m,
    )
    _plot_path_families_on_axis(ax4, metrics=metrics, families=families, radius_m=radius_m)
    fig3.suptitle("Constraint Weight Effect: Why and Where Path Switches", fontsize=13)
    fig3.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_compact.parent.mkdir(parents=True, exist_ok=True)
    fig3.savefig(output_compact, dpi=260, bbox_inches="tight")
    plt.close(fig3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze constraint stress-testing outputs and plot planner-level "
            "goal-cost vs constraint-cost tradeoffs."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/experiments/stress_testing/constraint"),
        help="Directory containing w_constraint_* experiment folders.",
    )
    parser.add_argument(
        "--constraint-radius-m",
        type=float,
        default=1.0,
        help="Constraint influence radius used by planner penalty field.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <base-dir>/constraint_goal_vs_constraint_cost.csv",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <base-dir>/constraint_goal_vs_constraint_cost.png",
    )
    parser.add_argument(
        "--output-report-envelope",
        type=Path,
        default=None,
        help="Compact report figure 1 (objective envelope). "
        "Defaults to <base-dir>/constraint_report_figure_1_envelope.png",
    )
    parser.add_argument(
        "--output-report-paths",
        type=Path,
        default=None,
        help="Compact report figure 2 (path families). "
        "Defaults to <base-dir>/constraint_report_figure_2_paths.png",
    )
    parser.add_argument(
        "--output-report-compact",
        type=Path,
        default=None,
        help="Compact 2-panel report figure. "
        "Defaults to <base-dir>/constraint_report_compact_2panel.png",
    )
    parser.add_argument(
        "--output-dominance-plot",
        type=Path,
        default=None,
        help="Single report-friendly dominance plot. "
        "Defaults to <base-dir>/constraint_goal_vs_weighted_constraint.png",
    )
    parser.add_argument(
        "--output-dominance-csv",
        type=Path,
        default=None,
        help="CSV for dominance analysis. "
        "Defaults to <base-dir>/constraint_goal_vs_weighted_constraint.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    if args.output_csv is None:
        output_csv = base_dir / "constraint_goal_vs_constraint_cost.csv"
    else:
        output_csv = args.output_csv.resolve()
    if args.output_plot is None:
        output_plot = base_dir / "constraint_goal_vs_constraint_cost.png"
    else:
        output_plot = args.output_plot.resolve()
    if args.output_report_envelope is None:
        output_report_envelope = base_dir / "constraint_report_figure_1_envelope.png"
    else:
        output_report_envelope = args.output_report_envelope.resolve()
    if args.output_report_paths is None:
        output_report_paths = base_dir / "constraint_report_figure_2_paths.png"
    else:
        output_report_paths = args.output_report_paths.resolve()
    if args.output_report_compact is None:
        output_report_compact = base_dir / "constraint_report_compact_2panel.png"
    else:
        output_report_compact = args.output_report_compact.resolve()
    if args.output_dominance_plot is None:
        output_dominance_plot = base_dir / "constraint_goal_vs_weighted_constraint.png"
    else:
        output_dominance_plot = args.output_dominance_plot.resolve()
    if args.output_dominance_csv is None:
        output_dominance_csv = base_dir / "constraint_goal_vs_weighted_constraint.csv"
    else:
        output_dominance_csv = args.output_dominance_csv.resolve()

    metrics = collect_metrics(base_dir, radius_m=args.constraint_radius_m)
    if not metrics:
        raise RuntimeError(f"No constraint runs found in {base_dir}")

    grid_resolution_m = infer_grid_resolution_from_paths(metrics)
    write_summary_csv(metrics, output_csv)
    write_dominance_csv(metrics, output_dominance_csv, resolution_m=grid_resolution_m)
    dominance_threshold = plot_dominance_comparison(
        metrics,
        output_plot=output_dominance_plot,
        resolution_m=grid_resolution_m,
    )
    plot_metrics(metrics, radius_m=args.constraint_radius_m, output_plot=output_plot)
    plot_report_figures(
        metrics,
        radius_m=args.constraint_radius_m,
        resolution_m=grid_resolution_m,
        output_envelope=output_report_envelope,
        output_paths=output_report_paths,
        output_compact=output_report_compact,
    )

    print(f"Wrote CSV:  {output_csv}")
    print(f"Wrote plot: {output_plot}")
    print(f"Wrote dominance CSV:  {output_dominance_csv}")
    print(f"Wrote dominance plot: {output_dominance_plot}")
    print(f"Wrote report fig 1 (envelope): {output_report_envelope}")
    print(f"Wrote report fig 2 (paths):    {output_report_paths}")
    print(f"Wrote report compact (2-panel): {output_report_compact}")
    print(f"Inferred planner grid resolution: {grid_resolution_m:.4f} m/cell")
    if dominance_threshold is not None:
        print(f"Constraint starts dominating at approximately w = {dominance_threshold:.4f}")
    else:
        print("No goal-to-constraint dominance crossing found in the provided weight range.")
    print("Per-weight summary:")
    for row in metrics:
        print(
            f"  w={row.weight:>4g}  path={row.path_id:<3}  "
            f"goal={row.goal_cost:.4f}  constr_raw={row.constraint_cost_raw:.4f}  "
            f"weighted={row.constraint_cost_weighted:.4f}  total_proxy={row.total_proxy:.4f}"
        )


if __name__ == "__main__":
    main()
