"""Occupancy-grid cost queries and frame conversions (map <-> odom <-> grid)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def map_to_odom(
    x_map: float,
    y_map: float,
    map_odom: dict,
) -> Tuple[float, float]:
    """Convert (x, y) from *map* frame to *odom* frame using ``T_map_odom``."""
    tx = float(map_odom["x"])
    ty = float(map_odom["y"])
    th = float(map_odom["yaw"])

    c = math.cos(th)
    s = math.sin(th)

    dx = x_map - tx
    dy = y_map - ty

    x_odom = c * dx + s * dy
    y_odom = -s * dx + c * dy
    return x_odom, y_odom


def odom_to_map(
    xo: float,
    yo: float,
    map_odom: dict,
) -> Tuple[float, float]:
    """Convert (x, y) from *odom* frame to *map* frame."""
    tx = float(map_odom["x"])
    ty = float(map_odom["y"])
    th = float(map_odom["yaw"])

    c = math.cos(th)
    s = math.sin(th)

    xm = tx + c * xo - s * yo
    ym = ty + s * xo + c * yo
    return xm, ym


def get_cost_at_pose(
    costmap: Optional[np.ndarray],
    pose_map: Tuple[float, float, float],
    *,
    map_odom: dict,
    origin_x: float,
    origin_y: float,
    resolution: float,
    out_of_bounds_cost: float = 100.0,
) -> float:
    """Query Nav2 OccupancyGrid cost at a robot pose given in the *map* frame.

    Conversion chain: map -> odom -> grid index.
    """
    if costmap is None:
        return out_of_bounds_cost

    x_map, y_map, _ = pose_map
    H, W = costmap.shape

    x_odom, y_odom = map_to_odom(x_map, y_map, map_odom)

    ix = int((x_odom - origin_x) / resolution)
    iy = int((y_odom - origin_y) / resolution)

    if ix < 0 or iy < 0 or ix >= W or iy >= H:
        return out_of_bounds_cost

    return float(costmap[iy, ix])


def pose_map_to_grid(
    pose_map: Tuple[float, float, float],
    *,
    map_odom: dict,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> Tuple[float, float]:
    """Convert a map-frame pose to fractional grid coordinates in the costmap frame."""
    x_map, y_map, _ = pose_map
    x_odom, y_odom = map_to_odom(x_map, y_map, map_odom)
    gx = (x_odom - origin_x) / resolution
    gy = (y_odom - origin_y) / resolution
    return gx, gy


def nearest_obstacle_distance_at_pose(
    costmap: Optional[np.ndarray],
    pose_map: Tuple[float, float, float],
    *,
    map_odom: dict,
    origin_x: float,
    origin_y: float,
    resolution: float,
    occupied_cost_thresh: float = 90.0,
) -> float:
    """Return the distance in metres from ``pose_map`` to the nearest occupied cell.

    Occupied cells are defined as those with occupancy/cost values greater than or
    equal to ``occupied_cost_thresh``. If no such cells exist, ``math.inf`` is
    returned.
    """
    if costmap is None:
        return math.inf

    occupied = np.argwhere(costmap >= float(occupied_cost_thresh))
    if occupied.size == 0:
        return math.inf

    gx, gy = pose_map_to_grid(
        pose_map,
        map_odom=map_odom,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution=resolution,
    )
    cell_centers = occupied[:, ::-1].astype(np.float64) + 0.5  # -> (ix, iy)
    deltas = cell_centers - np.array([gx, gy], dtype=np.float64)
    dists = np.hypot(deltas[:, 0], deltas[:, 1])
    if dists.size == 0:
        return math.inf
    return float(np.min(dists) * resolution)
