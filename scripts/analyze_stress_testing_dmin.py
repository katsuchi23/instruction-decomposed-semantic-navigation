#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SceneMetrics:
    scene_name: str
    scene_suffix: int
    obstacle_x: float
    obstacle_y: float
    d_min: float
    constraint_deviation: float
    preference_deviation: float
    constraint_mean_distance: float
    preference_mean_distance: float


@dataclass(frozen=True)
class WeightMetrics:
    weight: float
    deviation: float
    mean_distance: float


@dataclass(frozen=True)
class CombinedWeightDeviation:
    weight: float
    constraint_deviation: float
    preference_deviation: float


def load_xy_csv(path: Path) -> np.ndarray:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        points = [(float(row["x"]), float(row["y"])) for row in reader]
    if not points:
        raise ValueError(f"No path points found in {path}")
    return np.asarray(points, dtype=float)


def segment_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(0, dtype=float)
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def polyline_length(points: np.ndarray) -> float:
    return float(segment_lengths(points).sum())


def mean_distance_to_point(points: np.ndarray, point: np.ndarray) -> float:
    return float(np.linalg.norm(points - point[None, :], axis=1).mean())


def resample_polyline(points: np.ndarray, samples: int = 200) -> np.ndarray:
    if len(points) == 1:
        return np.repeat(points, samples, axis=0)

    seg_lengths = segment_lengths(points)
    cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total = cumulative[-1]
    if total == 0.0:
        return np.repeat(points[:1], samples, axis=0)

    targets = np.linspace(0.0, total, samples)
    result = np.empty((samples, 2), dtype=float)

    seg_index = 0
    for i, target in enumerate(targets):
        while seg_index < len(seg_lengths) - 1 and cumulative[seg_index + 1] < target:
            seg_index += 1
        start = points[seg_index]
        end = points[seg_index + 1]
        length = seg_lengths[seg_index]
        if length == 0.0:
            result[i] = start
            continue
        alpha = (target - cumulative[seg_index]) / length
        result[i] = start + alpha * (end - start)

    return result


def mean_nearest_neighbor_distance(source: np.ndarray, target: np.ndarray) -> float:
    distances = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2)
    return float(np.min(distances, axis=1).mean())


def symmetric_path_deviation(path_a: np.ndarray, path_b: np.ndarray, samples: int = 200) -> float:
    a = resample_polyline(path_a, samples=samples)
    b = resample_polyline(path_b, samples=samples)
    return 0.5 * (
        mean_nearest_neighbor_distance(a, b) + mean_nearest_neighbor_distance(b, a)
    )


def path_length_difference(path_a: np.ndarray, path_b: np.ndarray) -> float:
    return abs(polyline_length(path_a) - polyline_length(path_b))


def mean_object_distance_difference(
    altered_path: np.ndarray, normal_path: np.ndarray, object_point: np.ndarray, samples: int = 200
) -> float:
    altered = resample_polyline(altered_path, samples=samples)
    normal = resample_polyline(normal_path, samples=samples)
    return abs(
        mean_distance_to_point(altered, object_point) - mean_distance_to_point(normal, object_point)
    )


def mean_object_distance(path: np.ndarray, object_point: np.ndarray, samples: int = 200) -> float:
    resampled = resample_polyline(path, samples=samples)
    return mean_distance_to_point(resampled, object_point)


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom == 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / denom)
    t = min(1.0, max(0.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def min_distance_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    if len(polyline) == 1:
        return float(np.linalg.norm(point - polyline[0]))
    return min(
        point_to_segment_distance(point, polyline[i], polyline[i + 1])
        for i in range(len(polyline) - 1)
    )


def scene_suffix(scene_dir: Path) -> int:
    return int(scene_dir.name.split("_")[-1])


def discover_scene_dirs(base_dir: Path) -> list[Path]:
    scenes = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("scene2_")]
    return sorted((path for path in scenes if scene_suffix(path) >= 30), key=scene_suffix)


def build_metrics(scene_dir: Path) -> SceneMetrics:
    suffix = scene_suffix(scene_dir)
    obstacle = np.array([0.5, suffix / 100.0], dtype=float)

    normal_path = load_xy_csv(scene_dir / "exp_003" / "trajectory_task0.csv")
    constraint_path = load_xy_csv(scene_dir / "exp_001" / "trajectory_task0.csv")
    preference_path = load_xy_csv(scene_dir / "exp_002" / "trajectory_task0.csv")

    return SceneMetrics(
        scene_name=scene_dir.name,
        scene_suffix=suffix,
        obstacle_x=float(obstacle[0]),
        obstacle_y=float(obstacle[1]),
        d_min=min_distance_to_polyline(obstacle, normal_path),
        constraint_deviation=mean_object_distance_difference(constraint_path, normal_path, obstacle),
        preference_deviation=mean_object_distance_difference(preference_path, normal_path, obstacle),
        constraint_mean_distance=mean_object_distance(constraint_path, obstacle),
        preference_mean_distance=mean_object_distance(preference_path, obstacle),
    )


def write_metrics_csv(metrics: Iterable[SceneMetrics], output_path: Path) -> None:
    rows = list(metrics)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scene_name",
                "scene_suffix",
                "obstacle_x",
                "obstacle_y",
                "d_min",
                "constraint_deviation",
                "preference_deviation",
                "constraint_mean_distance",
                "preference_mean_distance",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.scene_name,
                    row.scene_suffix,
                    row.obstacle_x,
                    row.obstacle_y,
                    row.d_min,
                    row.constraint_deviation,
                    row.preference_deviation,
                    row.constraint_mean_distance,
                    row.preference_mean_distance,
                ]
            )


def write_weight_metrics_csv(metrics: Iterable[WeightMetrics], output_path: Path) -> None:
    rows = list(metrics)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["weight", "deviation", "mean_distance"])
        for row in rows:
            writer.writerow([row.weight, row.deviation, row.mean_distance])


def plot_series(
    metrics: list[SceneMetrics],
    value_getter,
    title: str,
    ylabel: str,
    color: str,
    output_path: Path,
) -> None:
    x = np.array([row.d_min for row in metrics], dtype=float)
    y = np.array([value_getter(row) for row in metrics], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linewidth=2, color=color)
    ax.scatter(x, y, s=50, color=color)
    ax.set_title(title)
    ax.set_xlabel("d_min to normal path (m)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_weight_series(
    metrics: list[WeightMetrics],
    title: str,
    ylabel: str,
    color: str,
    output_path: Path,
) -> None:
    x = np.array([row.weight for row in metrics], dtype=float)
    y = np.array([row.deviation for row in metrics], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    log_x = np.log10(x)
    slope, intercept = np.polyfit(log_x, y, deg=1)
    fit_log_x = np.linspace(log_x[0], log_x[-1], 300)
    fit_y = slope * fit_log_x + intercept
    fit_x = np.power(10.0, fit_log_x)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fit_x, fit_y, linewidth=2.5, color=color, linestyle="--")
    ax.scatter(x, y, s=50, color=color)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Weight parameter")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def combine_weight_deviation_metrics(
    constraint_metrics: list[WeightMetrics],
    preference_metrics: list[WeightMetrics],
) -> list[CombinedWeightDeviation]:
    constraint_by_weight = {row.weight: row for row in constraint_metrics}
    preference_by_weight = {row.weight: row for row in preference_metrics}
    common_weights = sorted(set(constraint_by_weight.keys()) & set(preference_by_weight.keys()))
    return [
        CombinedWeightDeviation(
            weight=w,
            constraint_deviation=constraint_by_weight[w].deviation,
            preference_deviation=preference_by_weight[w].deviation,
        )
        for w in common_weights
    ]


def write_combined_weight_deviation_csv(
    rows: list[CombinedWeightDeviation], output_path: Path
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["weight", "constraint_deviation", "preference_deviation"])
        for row in rows:
            writer.writerow([row.weight, row.constraint_deviation, row.preference_deviation])


def plot_constraint_preference_deviation_vs_weight(
    rows: list[CombinedWeightDeviation], output_path: Path
) -> None:
    x = np.array([row.weight for row in rows], dtype=float)
    y_constraint = np.array([row.constraint_deviation for row in rows], dtype=float)
    y_preference = np.array([row.preference_deviation for row in rows], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y_constraint = y_constraint[order]
    y_preference = y_preference[order]

    # Keep the same regression model as existing plots: linear in log10(weight).
    log_x = np.log10(x)
    c_slope, c_intercept = np.polyfit(log_x, y_constraint, deg=1)
    p_slope, p_intercept = np.polyfit(log_x, y_preference, deg=1)
    fit_log_x = np.linspace(log_x[0], log_x[-1], 300)
    fit_x = np.power(10.0, fit_log_x)
    fit_constraint = c_slope * fit_log_x + c_intercept
    fit_preference = p_slope * fit_log_x + p_intercept

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        fit_x,
        fit_constraint,
        linewidth=2.0,
        color="#c0392b",
        linestyle="--",
        label=f"Constraint fit (slope={c_slope:.3f})",
    )
    ax.plot(
        fit_x,
        fit_preference,
        linewidth=2.0,
        color="#1f618d",
        linestyle="--",
        label=f"Preference fit (slope={p_slope:.3f})",
    )
    ax.plot(x, y_constraint, marker="o", linewidth=1.8, color="#c0392b", label="Constraint points")
    ax.plot(x, y_preference, marker="s", linewidth=1.8, color="#1f618d", label="Preference points")

    for xi, yi in zip(x, y_constraint):
        ax.annotate(f"{xi:g}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    for xi, yi in zip(x, y_preference):
        ax.annotate(f"{xi:g}", (xi, yi), textcoords="offset points", xytext=(0, -10), ha="center", fontsize=7)

    # Linear x-axis makes all sampled weights explicit and easy to read in report.
    ax.set_xticks(x)
    ax.set_xlabel("Weight parameter")
    ax.set_ylabel("Path deviation from normal path (m)")
    ax.set_title("Path Deviation vs Weight (Constraint vs Preference, All Weights)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_single_behavior_deviation_vs_weight(
    metrics: list[WeightMetrics],
    behavior_name: str,
    color: str,
    marker: str,
    output_path: Path,
) -> None:
    x = np.array([row.weight for row in metrics], dtype=float)
    y = np.array([row.deviation for row in metrics], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    log_x = np.log10(x)
    slope, intercept = np.polyfit(log_x, y, deg=1)
    fit_log_x = np.linspace(log_x[0], log_x[-1], 300)
    fit_x = np.power(10.0, fit_log_x)
    fit_y = slope * fit_log_x + intercept

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ax.plot(
        fit_x,
        fit_y,
        linewidth=2.3,
        color=color,
        linestyle="--",
        label=f"{behavior_name} fit (slope={slope:.3f})",
    )
    ax.plot(
        x,
        y,
        marker=marker,
        linewidth=2.0,
        color=color,
        label=f"{behavior_name} points",
    )
    ax.scatter(x, y, s=48, color=color)

    ax.set_xticks(x)
    ax.set_xlabel("Weight parameter")
    ax.set_ylabel("Path deviation from normal path (m)")
    ax.set_title(f"{behavior_name} Path Deviation vs Weight (All Weights)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_weight_dir_name(path: Path) -> float:
    suffix = path.name
    if suffix.startswith("w_constraint_"):
        raw = suffix.removeprefix("w_constraint_")
    elif suffix.startswith("w_"):
        raw = suffix.removeprefix("w_")
    else:
        raise ValueError(f"Unexpected weight directory name: {path.name}")
    return float(raw.replace("p", "."))


def build_weight_metrics(
    weight_dir: Path, normal_path: np.ndarray, object_point: np.ndarray
) -> WeightMetrics:
    weight = parse_weight_dir_name(weight_dir)
    experiment_path = weight_dir / "exp_001" / "trajectory_task0.csv"
    weighted_path = load_xy_csv(experiment_path)
    return WeightMetrics(
        weight=weight,
        deviation=mean_object_distance_difference(weighted_path, normal_path, object_point),
        mean_distance=mean_object_distance(weighted_path, object_point),
    )


def discover_weight_dirs(base_dir: Path) -> list[Path]:
    return sorted(
        [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("w_")],
        key=parse_weight_dir_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze stress-testing path deviation against d_min from the normal path."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/experiments/stress_testing"),
        help="Stress-testing experiment directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/stress_testing/analysis"),
        help="Directory for plots and metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dirs = discover_scene_dirs(args.base_dir)
    if not scene_dirs:
        raise SystemExit(f"No scene2_* directories found under {args.base_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [build_metrics(scene_dir) for scene_dir in scene_dirs]

    write_metrics_csv(metrics, args.output_dir / "dmin_path_deviation_metrics.csv")
    plot_series(
        metrics,
        value_getter=lambda row: row.constraint_deviation,
        title="Constraint Mean Object-Distance Difference vs d_min",
        ylabel="Mean distance to object minus normal (m)",
        color="#c0392b",
        output_path=args.output_dir / "constraint_vs_dmin.png",
    )
    plot_series(
        metrics,
        value_getter=lambda row: row.preference_deviation,
        title="Preference Mean Object-Distance Difference vs d_min",
        ylabel="Mean distance to object minus normal (m)",
        color="#1f618d",
        output_path=args.output_dir / "preference_vs_dmin.png",
    )
    plot_series(
        metrics,
        value_getter=lambda row: row.constraint_mean_distance,
        title="Constraint Mean Distance to Object vs d_min",
        ylabel="Mean distance to object (m)",
        color="#a93226",
        output_path=args.output_dir / "constraint_mean_distance_vs_dmin.png",
    )
    plot_series(
        metrics,
        value_getter=lambda row: row.preference_mean_distance,
        title="Preference Mean Distance to Object vs d_min",
        ylabel="Mean distance to object (m)",
        color="#21618c",
        output_path=args.output_dir / "preference_mean_distance_vs_dmin.png",
    )

    normal_path = load_xy_csv(args.base_dir / "scene2_50" / "exp_003" / "trajectory_task0.csv")
    weight_object = np.array([0.5, 0.5], dtype=float)

    constraint_weight_dirs = discover_weight_dirs(args.base_dir / "constraint")
    preference_weight_dirs = discover_weight_dirs(args.base_dir / "preference")

    constraint_weight_metrics = [
        build_weight_metrics(weight_dir, normal_path, weight_object) for weight_dir in constraint_weight_dirs
    ]
    preference_weight_metrics = [
        build_weight_metrics(weight_dir, normal_path, weight_object) for weight_dir in preference_weight_dirs
    ]

    write_weight_metrics_csv(
        constraint_weight_metrics,
        args.output_dir / "constraint_weight_path_deviation_metrics.csv",
    )
    write_weight_metrics_csv(
        preference_weight_metrics,
        args.output_dir / "preference_weight_path_deviation_metrics.csv",
    )
    combined_weight_deviation = combine_weight_deviation_metrics(
        constraint_weight_metrics,
        preference_weight_metrics,
    )
    write_combined_weight_deviation_csv(
        combined_weight_deviation,
        args.output_dir / "constraint_preference_weight_path_deviation_metrics.csv",
    )
    plot_weight_series(
        constraint_weight_metrics,
        title="Constraint Mean Object-Distance Difference vs Weight",
        ylabel="Mean distance to object minus normal (m)",
        color="#c0392b",
        output_path=args.output_dir / "constraint_vs_weight.png",
    )
    plot_weight_series(
        preference_weight_metrics,
        title="Preference Mean Object-Distance Difference vs Weight",
        ylabel="Mean distance to object minus normal (m)",
        color="#1f618d",
        output_path=args.output_dir / "preference_vs_weight.png",
    )
    plot_weight_series(
        constraint_weight_metrics,
        title="Constraint Mean Distance to Object vs Weight",
        ylabel="Mean distance to object (m)",
        color="#a93226",
        output_path=args.output_dir / "constraint_mean_distance_vs_weight.png",
    )
    plot_weight_series(
        preference_weight_metrics,
        title="Preference Mean Distance to Object vs Weight",
        ylabel="Mean distance to object (m)",
        color="#21618c",
        output_path=args.output_dir / "preference_mean_distance_vs_weight.png",
    )
    plot_constraint_preference_deviation_vs_weight(
        combined_weight_deviation,
        output_path=args.output_dir / "constraint_preference_vs_weight.png",
    )
    plot_single_behavior_deviation_vs_weight(
        constraint_weight_metrics,
        behavior_name="Constraint",
        color="#c0392b",
        marker="o",
        output_path=args.output_dir / "constraint_only_vs_weight.png",
    )
    plot_single_behavior_deviation_vs_weight(
        preference_weight_metrics,
        behavior_name="Preference",
        color="#1f618d",
        marker="s",
        output_path=args.output_dir / "preference_only_vs_weight.png",
    )

    for row in metrics:
        print(
            f"{row.scene_name}: d_min={row.d_min:.4f}, "
            f"constraint_dev={row.constraint_deviation:.4f}, "
            f"preference_dev={row.preference_deviation:.4f}, "
            f"constraint_mean={row.constraint_mean_distance:.4f}, "
            f"preference_mean={row.preference_mean_distance:.4f}"
        )

    for row in constraint_weight_metrics:
        print(
            f"constraint_weight={row.weight:.1f}: deviation={row.deviation:.4f}, "
            f"mean_distance={row.mean_distance:.4f}"
        )

    for row in preference_weight_metrics:
        print(
            f"preference_weight={row.weight:.1f}: deviation={row.deviation:.4f}, "
            f"mean_distance={row.mean_distance:.4f}"
        )


if __name__ == "__main__":
    main()
