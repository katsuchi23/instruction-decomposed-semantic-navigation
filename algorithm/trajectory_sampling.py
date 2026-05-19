"""Trajectory sampling and path-guidance utilities."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from planning.types import PathGuidance
from utils.math_helpers import clamp, wrap_angle


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _project_point_to_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> Tuple[float, float, float, float]:
    """Project point P onto segment AB.

    Returns (t, cx, cy, d2) where *t* is the clamped interpolation factor,
    *(cx, cy)* is the closest point, and *d2* the squared distance.
    """
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        cx, cy = ax, ay
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        return 0.0, cx, cy, d2

    t = clamp((apx * abx + apy * aby) / denom, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    d2 = (px - cx) ** 2 + (py - cy) ** 2
    return t, cx, cy, d2


# ---------------------------------------------------------------------------
# Path guidance
# ---------------------------------------------------------------------------

def compute_path_guidance(
    x: Tuple[float, float, float],
    path_xy: List[Tuple[float, float]],
    *,
    last_s0_index: int = 0,
    lookahead_dist: float = 0.6,
    search_back: int = 5,
) -> PathGuidance:
    """Convert a polyline *path_xy* into a :class:`PathGuidance` bundle."""
    px, py, _ = x

    if not path_xy:
        return PathGuidance(s0_index=0, lookahead_xy=(px, py), seg_p0=(px, py), seg_p1=(px, py), path_xy=[])

    if len(path_xy) == 1:
        gx, gy = path_xy[0]
        return PathGuidance(s0_index=0, lookahead_xy=(gx, gy), seg_p0=(gx, gy), seg_p1=(gx, gy), path_xy=path_xy)

    # cumulative arc-lengths
    cum = [0.0]
    for i in range(len(path_xy) - 1):
        x0, y0 = path_xy[i]
        x1, y1 = path_xy[i + 1]
        cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))

    i_start = max(0, min(len(path_xy) - 2, last_s0_index - search_back))
    i_end = len(path_xy) - 2

    best_i, best_t, best_d2 = i_start, 0.0, float("inf")
    for i in range(i_start, i_end + 1):
        ax, ay = path_xy[i]
        bx, by = path_xy[i + 1]
        t, _, _, d2 = _project_point_to_segment(px, py, ax, ay, bx, by)
        if d2 < best_d2:
            best_d2, best_i, best_t = d2, i, t

    ax, ay = path_xy[best_i]
    bx, by = path_xy[best_i + 1]
    seg_len = math.hypot(bx - ax, by - ay)
    s_proj = cum[best_i] + best_t * seg_len
    s_lh = min(cum[-1], s_proj + max(0.0, lookahead_dist))

    j = 0
    while j < len(path_xy) - 1 and cum[j + 1] < s_lh:
        j += 1
    j = min(j, len(path_xy) - 2)

    j0x, j0y = path_xy[j]
    j1x, j1y = path_xy[j + 1]
    j_len = max(1e-12, math.hypot(j1x - j0x, j1y - j0y))
    alpha = clamp((s_lh - cum[j]) / j_len, 0.0, 1.0)
    lx = j0x + alpha * (j1x - j0x)
    ly = j0y + alpha * (j1y - j0y)

    return PathGuidance(
        s0_index=best_i,
        lookahead_xy=(lx, ly),
        seg_p0=path_xy[best_i],
        seg_p1=path_xy[best_i + 1],
        path_xy=path_xy,
    )


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------

def trajectories_sampling_from_path(
    u0: Tuple[float, float],
    x: Tuple[float, float, float],
    path_xy: List[Tuple[float, float]],
    *,
    last_s0_index: int = 0,
    lookahead_dist: float = 0.6,
    **kwargs,
) -> Tuple[List[List[Tuple[float, float]]], PathGuidance]:
    """Build :class:`PathGuidance` from a planned path, then sample trajectories."""
    guide = compute_path_guidance(x=x, path_xy=path_xy, last_s0_index=last_s0_index, lookahead_dist=lookahead_dist)
    trajs = trajectories_sampling(u0=u0, x=x, guide=guide, **kwargs)
    return trajs, guide


def trajectories_sampling(
    u0: Tuple[float, float],
    x: Tuple[float, float, float],
    guide: PathGuidance,
    *,
    N: int = 80,
    H: int = 20,
    dt: float = 0.1,
    v_max: float = 0.3,
    w_max: float = 0.8,
    v_min: float = 0.15,
    w_min: float = 0.15,
    p_stop: float = 0.1,
    a_v_max: float = 0.5,
    a_w_max: float = 1.0,
    k_yaw: float = 3.0,
    yaw_rotate_threshold: float = math.radians(40),
    noise_v: float = 0.05,
    noise_w: float = 0.15,
    warm_start_alpha: float = 0.0,
) -> List[List[Tuple[float, float]]]:
    """Velocity-sequence sampler around a path-lookahead heading."""
    x0, y0, th0 = x
    v0, w0 = u0

    if not guide.path_xy:
        zero_traj = [(0.0, 0.0)] * H
        return [zero_traj for _ in range(N)]

    lx, ly = guide.lookahead_xy
    path_heading = math.atan2(ly - y0, lx - x0)
    yaw_err = wrap_angle(path_heading - th0)

    w_target = clamp(k_yaw * yaw_err, -w_max, w_max)
    if abs(yaw_err) > yaw_rotate_threshold:
        v_target = 0.0
    else:
        v_target = v_max * max(0.2, 1.0 - abs(yaw_err) / yaw_rotate_threshold)

    # Warm-start: blend the path-heading target toward the previous action so
    # consecutive decisions stay close to each other, reducing jerk.
    if warm_start_alpha > 0.0:
        v_target = warm_start_alpha * v0 + (1.0 - warm_start_alpha) * v_target
        w_target = warm_start_alpha * w0 + (1.0 - warm_start_alpha) * w_target

    dv_max = a_v_max * dt
    dw_max = a_w_max * dt

    all_trajs: List[List[Tuple[float, float]]] = []

    for _ in range(N):
        rx, ry, rth = x0, y0, th0
        rv, rw = v0, w0
        traj: List[Tuple[float, float]] = []

        v_bias = random.uniform(-noise_v, noise_v)
        w_bias = random.uniform(-noise_w, noise_w)

        # One draw per trajectory: either the whole trajectory is a stop
        # (both v and w target zero) or both enforce the hardware minimum.
        # Using a single draw keeps the stop probability exactly p_stop.
        stop_traj = random.random() < p_stop
        v_stop_traj = stop_traj
        w_stop_traj = stop_traj

        for _ in range(H):
            # Move-mode: clamp command to [v_min, v_max] so hardware minimum is met.
            # Stop-mode: sample freely from the full range — allows v=0 naturally
            # without forcing it, so stop trajectories don't trivially outscore
            # move trajectories on effort and clearance.
            if v_stop_traj or v_target == 0.0:
                v_cmd = clamp(v_target + v_bias, -v_max, v_max)
            elif v_target > 0.0:
                v_cmd = clamp(v_target + v_bias, v_min, v_max)
            else:
                v_cmd = clamp(v_target + v_bias, -v_max, -v_min)

            if w_stop_traj or w_target == 0.0:
                w_cmd = clamp(w_target + w_bias, -w_max, w_max)
            elif w_target > 0.0:
                w_cmd = clamp(w_target + w_bias, w_min, w_max)
            else:
                w_cmd = clamp(w_target + w_bias, -w_max, -w_min)

            next_v = clamp(v_cmd, rv - dv_max, rv + dv_max)
            next_w = clamp(w_cmd, rw - dw_max, rw + dw_max)
            next_v = clamp(next_v, -v_max, v_max)
            next_w = clamp(next_w, -w_max, w_max)

            traj.append((next_v, next_w))

            rv, rw = next_v, next_w
            rth = wrap_angle(rth + rw * dt)
            rx += rv * math.cos(rth) * dt
            ry += rv * math.sin(rth) * dt

        all_trajs.append(traj)

    return all_trajs


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def visualize_trajectory_samples(
    x0: Tuple[float, float, float],
    trajectories: List[List[Tuple[float, float]]],
    *,
    dt: float = 0.1,
    ax=None,
) -> None:
    """Overlay sampled trajectories on an existing matplotlib axes."""
    if ax is None:
        return
    start_x, start_y, start_th = x0
    for traj in trajectories:
        xs, ys = [start_x], [start_y]
        th = start_th
        cx, cy = start_x, start_y
        for v, w in traj:
            th = wrap_angle(th + w * dt)
            cx += v * math.cos(th) * dt
            cy += v * math.sin(th) * dt
            xs.append(cx)
            ys.append(cy)
        ax.plot(xs, ys, alpha=0.25, linewidth=0.6)
