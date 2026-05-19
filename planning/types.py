"""Shared data types for the path planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PlannerGoalSpec:
    obj_xy: Tuple[float, float]
    r_min: float
    r_max: float
    phi0: float
    phi_tol: float
    x_episode0: Tuple[float, float, float]  # (x, y, yaw)
    constraint_xys: Tuple[Tuple[float, float], ...] = ()
    preference_xys: Tuple[Tuple[float, float], ...] = ()


@dataclass
class PlannerOutput:
    path_xy: List[Tuple[float, float]]
    stamp: float
    ok: bool = False
    goal_cells_count: int = 0


@dataclass
class PathGuidance:
    s0_index: int
    lookahead_xy: Tuple[float, float]
    seg_p0: Tuple[float, float]
    seg_p1: Tuple[float, float]
    path_xy: List[Tuple[float, float]]
