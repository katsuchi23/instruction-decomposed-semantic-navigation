"""Shared path visualization helpers for semantic-navigation experiments."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from planning.intent_costs import (
    get_constraint_radius_m,
    get_preference_min_dist_m,
    get_preference_radius_m,
)
from utils.config import get_runtime_value


_FALLBACK_MAP_PGM = Path(__file__).resolve().parent.parent / "assets" / "maps" / "testing_ground.pgm"


def _default_map_pgm() -> Path:
    """Read viz_map_pgm from config at call time so import order doesn't matter."""
    configured = get_runtime_value(("paths", "viz_map_pgm"), "")
    if configured:
        return Path(str(configured))
    return _FALLBACK_MAP_PGM
DEFAULT_MAP_RESOLUTION = 0.05
DEFAULT_MAP_ORIGIN_X = -2.39
DEFAULT_MAP_ORIGIN_Y = -2.39
GLOBAL_COST_SCALING_FACTOR = 5.0
GLOBAL_INSCRIBED_RADIUS_M = 0.2
LETHAL_OBSTACLE = 254.0
INSCRIBED_INFLATED_OBSTACLE = 253.0


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class MapVizConfig:
    pgm_path: Path
    resolution: float
    origin_x: float
    origin_y: float


def _parse_ros_map_yaml(yaml_path: Path) -> Tuple[Optional[str], Optional[float], Optional[Tuple[float, float]]]:
    image_path: Optional[str] = None
    resolution: Optional[float] = None
    origin_xy: Optional[Tuple[float, float]] = None
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("image:"):
            image_path = line.split(":", 1)[1].strip().strip("'\"")
            continue
        if line.startswith("resolution:"):
            try:
                resolution = float(line.split(":", 1)[1].strip())
            except ValueError:
                resolution = None
            continue
        if line.startswith("origin:"):
            payload = line.split(":", 1)[1].strip()
            if payload.startswith("[") and payload.endswith("]"):
                vals = [x.strip() for x in payload[1:-1].split(",")]
                if len(vals) >= 2:
                    try:
                        origin_xy = (float(vals[0]), float(vals[1]))
                    except ValueError:
                        origin_xy = None
    return image_path, resolution, origin_xy


def resolve_map_viz_config(
    map_pgm: Optional[Path] = None,
    map_yaml: Optional[Path] = None,
) -> MapVizConfig:
    default_pgm = _default_map_pgm()
    pgm_path = Path(map_pgm).expanduser() if map_pgm is not None else default_pgm
    if map_yaml is not None:
        yaml_path = Path(map_yaml).expanduser()
    elif map_pgm is None:
        yaml_path = default_pgm.with_suffix(".yaml")
    else:
        yaml_path = pgm_path.with_suffix(".yaml")

    resolution = DEFAULT_MAP_RESOLUTION
    origin_x = DEFAULT_MAP_ORIGIN_X
    origin_y = DEFAULT_MAP_ORIGIN_Y

    if yaml_path.exists():
        image_ref, yaml_resolution, yaml_origin = _parse_ros_map_yaml(yaml_path)
        if map_pgm is None and image_ref:
            image_path = Path(image_ref).expanduser()
            if not image_path.is_absolute():
                image_path = (yaml_path.parent / image_path)
            pgm_path = image_path
        if yaml_resolution is not None:
            resolution = yaml_resolution
        if yaml_origin is not None:
            origin_x, origin_y = yaml_origin
    elif map_yaml is not None:
        print(f"  [WARN] Visualization map YAML not found at {yaml_path}; using default origin/resolution.")

    return MapVizConfig(
        pgm_path=pgm_path,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
    )


def _global_inflation_radius(caution: str) -> float:
    key = (caution or "normal").strip().lower()
    if key == "low":
        return 0.20
    if key == "high":
        return 0.40
    return 0.30


def _map_extent(width: int, height: int, map_cfg: MapVizConfig) -> Tuple[float, float, float, float]:
    wx_min = map_cfg.origin_x
    wy_min = map_cfg.origin_y
    wx_max = map_cfg.origin_x + width * map_cfg.resolution
    wy_max = map_cfg.origin_y + height * map_cfg.resolution
    return (wx_min, wx_max, wy_min, wy_max)


def world_to_plot(wx: float, wy: float, *, rotate_display_90: bool = True) -> Tuple[float, float]:
    """World (x, y) -> plot coordinates in chosen display frame."""
    if rotate_display_90:
        # +90° CCW frame rotation: (u, v)=(-y, x)
        return -wy, wx
    return wx, wy


def world_vec_to_plot(dx: float, dy: float, *, rotate_display_90: bool = True) -> Tuple[float, float]:
    """World vector transform in chosen display frame."""
    if rotate_display_90:
        return -dy, dx
    return dx, dy


def _extract_behavior(exp: Dict[str, Any], task_idx: int) -> Dict[str, Any]:
    tasks = ((exp.get("llm_output") or {}).get("tasks") or [])
    if 0 <= task_idx < len(tasks):
        return tasks[task_idx].get("behavior") or {}
    return {}


def _cell_world_grid(height: int, width: int, map_cfg: MapVizConfig) -> Tuple[np.ndarray, np.ndarray]:
    xs = map_cfg.origin_x + (np.arange(width) + 0.5) * map_cfg.resolution
    # Image row 0 is top, while map origin is bottom-left.
    # Flip row indexing so world-Y mapping aligns with rendered map overlays.
    ys = map_cfg.origin_y + ((height - 1 - np.arange(height)) + 0.5) * map_cfg.resolution
    xw, yw = np.meshgrid(xs, ys)
    return xw, yw


def _build_obstacle_cost(map_img: np.ndarray, inflation_radius_m: float, resolution: float) -> np.ndarray:
    occupied = map_img < 200
    if not np.any(occupied):
        return np.zeros_like(map_img, dtype=np.float64)

    occ_idx = np.argwhere(occupied)
    grid_y, grid_x = np.indices(map_img.shape)
    dist_pix = np.full(map_img.shape, np.inf, dtype=np.float64)
    for oy, ox in occ_idx:
        dist_pix = np.minimum(dist_pix, np.hypot(grid_x - ox, grid_y - oy))

    dist_m = dist_pix * resolution
    cost = np.zeros_like(dist_m, dtype=np.float64)
    cost[occupied] = LETHAL_OBSTACLE

    inscribed_mask = (~occupied) & (dist_m <= GLOBAL_INSCRIBED_RADIUS_M)
    cost[inscribed_mask] = INSCRIBED_INFLATED_OBSTACLE

    inflated_mask = (
        (~occupied)
        & (dist_m > GLOBAL_INSCRIBED_RADIUS_M)
        & (dist_m <= inflation_radius_m)
    )
    cost[inflated_mask] = (
        (INSCRIBED_INFLATED_OBSTACLE - 1.0)
        * np.exp(
            -GLOBAL_COST_SCALING_FACTOR
            * (dist_m[inflated_mask] - GLOBAL_INSCRIBED_RADIUS_M)
        )
    )
    return cost / LETHAL_OBSTACLE


def _build_constraint_overlay(
    shape: Tuple[int, int],
    constraint_xys: Tuple[Tuple[float, float], ...],
    map_cfg: MapVizConfig,
) -> np.ndarray:
    height, width = shape
    if not constraint_xys:
        return np.zeros((height, width), dtype=np.float64)

    xw, yw = _cell_world_grid(height, width, map_cfg)
    overlay = np.zeros((height, width), dtype=np.float64)
    for cx, cy in constraint_xys:
        dist = np.hypot(xw - cx, yw - cy)
        overlay = np.maximum(overlay, np.clip(1.0 - dist / get_constraint_radius_m(), 0.0, 1.0))
    return overlay


def _build_preference_overlay(
    shape: Tuple[int, int],
    preference_xys: Tuple[Tuple[float, float], ...],
    map_cfg: MapVizConfig,
) -> np.ndarray:
    height, width = shape
    if not preference_xys:
        return np.zeros((height, width), dtype=np.float64)

    xw, yw = _cell_world_grid(height, width, map_cfg)
    overlay = np.zeros((height, width), dtype=np.float64)
    preference_radius = get_preference_radius_m()
    preference_min_dist = get_preference_min_dist_m()
    span = max(preference_radius - preference_min_dist, 1e-6)
    for px, py in preference_xys:
        dist = np.hypot(xw - px, yw - py)
        basin = 1.0 - np.clip((dist - preference_min_dist) / span, 0.0, 1.0)
        overlay = np.maximum(overlay, basin)
    return overlay



def plot_task_path(
    exp: Dict[str, Any],
    task_result: Dict[str, Any],
    out_path: Path,
    *,
    map_pgm: Optional[Path] = None,
    map_yaml: Optional[Path] = None,
    rotate_display_90: bool = True,
    show_title: bool = True,
    show_axes: bool = True,
    title_text: Optional[str] = None,
    legend_label_style: str = "default",
    crop_world_x: Optional[Tuple[float, float]] = None,
    crop_world_y: Optional[Tuple[float, float]] = None,
) -> bool:
    """Render one task's path over the map and instruction-conditioned cost layers."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    map_cfg = resolve_map_viz_config(map_pgm=map_pgm, map_yaml=map_yaml)
    if not map_cfg.pgm_path.exists():
        print(f"  [WARN] Map PGM not found at {map_cfg.pgm_path} — skipping.")
        return False

    task_idx = int(task_result.get("task_idx", 0))
    behavior = _extract_behavior(exp, task_idx)
    symbol_labels = legend_label_style == "symbolic"
    caution = behavior.get("caution", "normal")
    cparams = task_result.get("control_params") or {}
    expected_d_safe_viz = _global_inflation_radius(caution)
    d_safe = cparams.get("d_safe")
    try:
        d_safe_viz = float(d_safe)
    except (TypeError, ValueError):
        d_safe_viz = expected_d_safe_viz
    if not math.isfinite(d_safe_viz) or d_safe_viz <= 0.0:
        d_safe_viz = expected_d_safe_viz

    # Old experiment logs may embed stale safety-distance values in
    # control_params. For report overlays, prefer the current caution mapping
    # so regenerated figures stay aligned with Appendix B defaults.
    if str(caution).strip().lower() in {"low", "normal", "high"}:
        if abs(d_safe_viz - expected_d_safe_viz) > 1e-9:
            d_safe_viz = expected_d_safe_viz

    map_img = mpimg.imread(str(map_cfg.pgm_path))
    height, width = map_img.shape[:2]
    x_min, x_max, y_min, y_max = _map_extent(width, height, map_cfg)
    if rotate_display_90:
        # Rotate the displayed map frame +90° CCW: (u, v)=(-y, x).
        extent = (-y_max, -y_min, x_min, x_max)
    else:
        extent = (x_min, x_max, y_min, y_max)

    constraint_xys = tuple(tuple(p) for p in (task_result.get("constraint_locations") or []))
    preference_xys = tuple(tuple(p) for p in (task_result.get("preference_locations") or []))

    obstacle_cost = _build_obstacle_cost(map_img, d_safe_viz, map_cfg.resolution)
    constraint_cost = _build_constraint_overlay(map_img.shape, constraint_xys, map_cfg)
    preference_gain = _build_preference_overlay(map_img.shape, preference_xys, map_cfg)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    # Keep gray overlay for obstacle inflation only. Constraint/preference
    # intent fields are rendered with dedicated colors to avoid looking like
    # phantom map obstacles.
    hot_cost = np.clip(obstacle_cost, 0.0, 1.0)
    if rotate_display_90:
        # Rotate raster layers to match the +90° CCW display frame.
        map_xy = np.rot90(map_img, k=1)
        hot_xy = np.rot90(hot_cost, k=1)
        constraint_xy = np.rot90(constraint_cost, k=1)
        pref_xy = np.rot90(preference_gain, k=1)
    else:
        map_xy = map_img
        hot_xy = hot_cost
        constraint_xy = constraint_cost
        pref_xy = preference_gain

    ax.imshow(
        map_xy,
        cmap="gray",
        origin="upper",
        extent=extent,
        alpha=1.0,
        interpolation="nearest",
        zorder=0,
    )
    ax.imshow(
        hot_xy,
        cmap="Greys",
        origin="upper",
        extent=extent,
        alpha=np.clip(hot_xy * 0.4, 0.0, 0.5),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        zorder=1,
    )
    if constraint_xys:
        ax.imshow(
            constraint_xy,
            cmap="Reds",
            origin="upper",
            extent=extent,
            alpha=np.clip(constraint_xy * 0.18, 0.0, 0.30),
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            zorder=2,
        )
    if preference_xys:
        ax.imshow(
            pref_xy,
            cmap="Greens",
            origin="upper",
            extent=extent,
            alpha=np.clip(pref_xy * 0.08, 0.0, 0.12),
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            zorder=3,
        )

    r_min = cparams.get("r_min")
    r_max = cparams.get("r_max")
    target = task_result.get("target_location")
    if target and r_min is not None and r_max is not None:
        tx_p, ty_p = world_to_plot(target[0], target[1], rotate_display_90=rotate_display_90)
        band_fill = plt.Circle((tx_p, ty_p), r_max, fill=True, facecolor="#4CAF50", alpha=0.08, zorder=3)
        band_outer = plt.Circle(
            (tx_p, ty_p),
            r_max,
            fill=False,
            edgecolor="#4CAF50",
            linewidth=1.5,
            linestyle="--",
            label="Maximum stopping radius" if symbol_labels else f"r_max={r_max:.2f}m",
            zorder=4,
        )
        band_inner = plt.Circle(
            (tx_p, ty_p),
            r_min,
            fill=False,
            edgecolor="#FF9800",
            linewidth=1.5,
            linestyle="--",
            label="Minimum stopping radius" if symbol_labels else f"r_min={r_min:.2f}m",
            zorder=4,
        )
        ax.add_patch(band_fill)
        ax.add_patch(band_outer)
        ax.add_patch(band_inner)

    attempt_results = task_result.get("attempt_results") or []
    if not attempt_results:
        attempt_results = [{
            "attempt_idx": 1,
            "global_path_initial": task_result.get("global_path_initial") or [],
            "trajectory": task_result.get("trajectory") or [],
            "target_location": task_result.get("target_location"),
            "semantic_retargeted": False,
            "post_goal_target_shift_m": 0.0,
        }]

    traj_colors = ["#F4511E", "#E53935", "#6D4C41"]
    target_colors = ["#8E24AA", "#D81B60", "#3949AB"]

    first_traj_pt: Optional[Tuple[float, float]] = None
    last_traj_pt: Optional[Tuple[float, float]] = None
    initial_yaw_deg: Optional[float] = None

    for i, attempt in enumerate(attempt_results):
        attempt_idx = int(attempt.get("attempt_idx", i + 1))
        traj_color = traj_colors[i % len(traj_colors)]
        target_color = target_colors[i % len(target_colors)]

        traj = attempt.get("trajectory") or []
        if traj:
            tpx = [world_to_plot(x, y, rotate_display_90=rotate_display_90)[0] for x, y in traj]
            tpy = [world_to_plot(x, y, rotate_display_90=rotate_display_90)[1] for x, y in traj]
            ax.plot(
                tpx,
                tpy,
                color=traj_color,
                linewidth=2.0,
                label="Executed path" if symbol_labels and i == 0 else f"Trajectory A{attempt_idx}",
                zorder=7,
            )
            if first_traj_pt is None:
                first_traj_pt = (tpx[0], tpy[0])
                attempt_steps = attempt.get("steps") or []
                if attempt_steps:
                    try:
                        initial_yaw_deg = float(attempt_steps[0].get("yaw_deg"))
                    except Exception:
                        initial_yaw_deg = None
            last_traj_pt = (tpx[-1], tpy[-1])
            if attempt.get("semantic_retargeted"):
                ax.plot(
                    tpx[-1],
                    tpy[-1],
                    marker="X",
                    color="black",
                    markersize=10,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    label="Semantic Recheck" if i == 0 else None,
                    zorder=9,
                )

        attempt_target = attempt.get("target_location")
        if attempt_target:
            px, py = world_to_plot(
                attempt_target[0],
                attempt_target[1],
                rotate_display_90=rotate_display_90,
            )
            shift_m = float(attempt.get("post_goal_target_shift_m", 0.0) or 0.0)
            label = f"Target A{attempt_idx}"
            if symbol_labels and i == 0:
                label = "Grounded target"
            if attempt.get("semantic_retargeted"):
                label += f" (shift {shift_m:.2f} m)"
            ax.plot(
                px,
                py,
                marker="D",
                color=target_color,
                markersize=11,
                markeredgecolor="white",
                markeredgewidth=1.3,
                label=label,
                zorder=9,
            )

    if first_traj_pt is not None:
        ax.plot(
            first_traj_pt[0],
            first_traj_pt[1],
            marker="o",
            color="#4CAF50",
            markersize=10,
            label="Start pose" if symbol_labels else "Start",
            zorder=10,
        )
        if initial_yaw_deg is None:
            task_steps = task_result.get("steps") or []
            if task_steps:
                try:
                    initial_yaw_deg = float(task_steps[0].get("yaw_deg"))
                except Exception:
                    initial_yaw_deg = None
        if initial_yaw_deg is not None:
            yaw = math.radians(initial_yaw_deg)
            dx_w = math.cos(yaw)
            dy_w = math.sin(yaw)
            dpx, dpy = world_vec_to_plot(dx_w, dy_w, rotate_display_90=rotate_display_90)
            arrow_len_m = 0.20
            ax.arrow(
                first_traj_pt[0],
                first_traj_pt[1],
                dpx * arrow_len_m,
                dpy * arrow_len_m,
                width=0.015,
                head_width=0.08,
                head_length=0.10,
                fc="#2E7D32",
                ec="white",
                linewidth=1.0,
                alpha=0.95,
                length_includes_head=True,
                zorder=11,
            )
            ax.plot(
                [],
                [],
                color="#2E7D32",
                linewidth=2.0,
                label="Initial heading" if symbol_labels else "Start heading",
            )
    if last_traj_pt is not None:
        ax.plot(
            last_traj_pt[0],
            last_traj_pt[1],
            marker="*",
            color="#FF5722",
            markersize=14,
            label="Final pose" if symbol_labels else "End",
            zorder=10,
        )

    for i, cloc in enumerate(constraint_xys):
        px, py = world_to_plot(cloc[0], cloc[1], rotate_display_90=rotate_display_90)
        ax.add_patch(plt.Circle((px, py), get_constraint_radius_m(), color="#F44336", alpha=0.12, zorder=3))
        ax.plot(
            px,
            py,
            marker="X",
            color="#F44336",
            markersize=11,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label="Constraint (avoid)" if i == 0 else None,
            zorder=9,
        )

    for i, ploc in enumerate(preference_xys):
        px, py = world_to_plot(ploc[0], ploc[1], rotate_display_90=rotate_display_90)
        ax.add_patch(plt.Circle((px, py), get_preference_radius_m(), color="#00BCD4", alpha=0.05, zorder=3))
        ax.plot(
            px,
            py,
            marker="P",
            color="#00BCD4",
            markersize=11,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label="Preference (near)" if i == 0 else None,
            zorder=9,
        )

    if rotate_display_90:
        # Horizontal axis stores -Y after rotation; show world-Y values directly.
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{-v:g}"))
        if show_axes:
            ax.set_xlabel("Y (m)", fontsize=11)
            ax.set_ylabel("X (m)", fontsize=11)
    elif show_axes:
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_aspect("equal")
    plot_xlim = [extent[0], extent[1]]
    plot_ylim = [extent[2], extent[3]]
    if crop_world_x is not None:
        crop_x_min, crop_x_max = crop_world_x
        if rotate_display_90:
            plot_ylim = [crop_x_min, crop_x_max]
        else:
            plot_xlim = [crop_x_min, crop_x_max]
    if crop_world_y is not None:
        crop_y_min, crop_y_max = crop_world_y
        if rotate_display_90:
            plot_xlim = [-crop_y_max, -crop_y_min]
        else:
            plot_ylim = [crop_y_min, crop_y_max]
    ax.set_xlim(plot_xlim[0], plot_xlim[1])
    ax.set_ylim(plot_ylim[0], plot_ylim[1])
    ax.legend_ = None

    if show_title:
        instruction = exp.get("instruction", "")
        if title_text is not None:
            ax.set_title(title_text, fontsize=11, pad=10)
        else:
            ax.set_title(f'"{instruction}"', fontsize=11, pad=10)


    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True
