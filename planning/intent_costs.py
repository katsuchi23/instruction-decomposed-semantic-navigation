"""Shared instruction-conditioned penalty fields for planning and visualization."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from utils.config import get_param


def get_constraint_radius_m() -> float:
    return float(get_param(("intent_costs", "constraint_radius_m"), 1.0))


def get_preference_radius_m() -> float:
    return float(get_param(("intent_costs", "preference_radius_m"), 1.0))


def get_preference_min_dist_m() -> float:
    return float(get_param(("intent_costs", "preference_min_dist_m"), 0.3))


def get_constraint_weight() -> float:
    return float(get_param(("intent_costs", "w_constraint_path"), 3.0))


def get_preference_near_repel_weight() -> float:
    return float(get_param(("intent_costs", "w_preference_near_repel_path"), 1.0))


def get_preference_attract_weight() -> float:
    return float(get_param(("intent_costs", "w_preference_attract_path"), 10.0))


def constraint_penalty_from_distance(distance_m: float) -> float:
    constraint_radius_m = get_constraint_radius_m()
    if distance_m >= constraint_radius_m:
        return 0.0
    return get_constraint_weight() * (1.0 - distance_m / max(1e-6, constraint_radius_m))


def preference_penalty_from_distance(distance_m: float) -> float:
    preference_min_dist_m = get_preference_min_dist_m()
    preference_radius_m = get_preference_radius_m()
    if distance_m < preference_min_dist_m:
        return (
            get_preference_near_repel_weight()
            * (1.0 - distance_m / max(1e-6, preference_min_dist_m))
        )
    normalized_dist = (distance_m - preference_min_dist_m) / max(
        1e-6,
        (preference_radius_m - preference_min_dist_m),
    )
    normalized_dist = max(0.0, normalized_dist)
    if normalized_dist >= 1.0:
        return 0.0 
    return get_preference_attract_weight() * normalized_dist


def world_to_grid(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> Tuple[int, int]:
    ix = int((x - origin_x) / resolution)
    iy = int((y - origin_y) / resolution)
    return ix, iy


def grid_to_world(
    ix: int,
    iy: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> Tuple[float, float]:
    x = origin_x + (ix + 0.5) * resolution
    y = origin_y + (iy + 0.5) * resolution
    return x, y


def build_intent_penalty_grid(
    width: int,
    height: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
    constraint_xys: Tuple[Tuple[float, float], ...],
    preference_xys: Tuple[Tuple[float, float], ...],
) -> Optional[np.ndarray]:
    """Return the planner penalty grid induced by avoid / stay-close intents."""
    if not constraint_xys and not preference_xys:
        return None

    penalty = np.zeros((height, width), dtype=np.float64)

    for cx, cy in constraint_xys:
        ci, cj = world_to_grid(cx, cy, origin_x, origin_y, resolution)
        k = int(math.ceil(get_constraint_radius_m() / resolution))
        for ix in range(max(0, ci - k), min(width, ci + k + 1)):
            for iy in range(max(0, cj - k), min(height, cj + k + 1)):
                wx, wy = grid_to_world(ix, iy, origin_x, origin_y, resolution)
                d = math.hypot(wx - cx, wy - cy)
                penalty[iy, ix] += constraint_penalty_from_distance(d)

    for px, py in preference_xys:
        for ix in range(width):
            for iy in range(height):
                wx, wy = grid_to_world(ix, iy, origin_x, origin_y, resolution)
                d = math.hypot(wx - px, wy - py)
                penalty[iy, ix] += preference_penalty_from_distance(d)

    return penalty
