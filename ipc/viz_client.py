"""Viz publisher — direct ROS (replaces ZMQ IPC + semnav_rviz_node).

Same public API as before; callers need no changes.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple


def _node():
    from ros.ros_bridge import get_node
    return get_node()


def _rollout(
    robot_pose: Tuple[float, float, float],
    seq: List[Tuple[float, float]],
    dt: float,
) -> List[Tuple[float, float]]:
    cx, cy, th = robot_pose
    pts: List[Tuple[float, float]] = [(cx, cy)]
    for v, w in seq:
        th += w * dt
        if th > math.pi:
            th -= 2.0 * math.pi
        elif th < -math.pi:
            th += 2.0 * math.pi
        cx += v * math.cos(th) * dt
        cy += v * math.sin(th) * dt
        pts.append((cx, cy))
    return pts


def send_viz_data(
    robot_pose: Tuple[float, float, float],
    start_pose: Tuple[float, float, float],
    target_xy: Tuple[float, float],
    active_goal_xy: Tuple[float, float],
    path: List[Tuple[float, float]],
    all_seqs: List[List[Tuple[float, float]]],
    dt: float,
    r_min: float,
    r_max: float,
    lookahead_xy: Optional[Tuple[float, float]],
    mode: str,
    task_name: str,
    constraint_xys: tuple = (),
    preference_xys: tuple = (),
    max_traj_samples: int = 30,
) -> None:
    try:
        from planning.intent_costs import get_constraint_radius_m, get_preference_radius_m
        constraint_r = get_constraint_radius_m()
        preference_r = get_preference_radius_m()
    except Exception:
        constraint_r, preference_r = 1.0, 1.0

    try:
        seqs = all_seqs[:max_traj_samples] if len(all_seqs) > max_traj_samples else all_seqs
        traj_samples = [_rollout(robot_pose, seq, dt) for seq in seqs]

        data = {
            "robot": list(robot_pose),
            "start": list(start_pose),
            "target": list(target_xy),
            "goal": list(active_goal_xy),
            "lookahead": list(lookahead_xy) if lookahead_xy else None,
            "path": [list(p) for p in path],
            "traj_samples": [[list(p) for p in traj] for traj in traj_samples],
            "r_min": r_min,
            "r_max": r_max,
            "mode": mode,
            "task_name": task_name,
            "constraint_xys": [list(xy) for xy in constraint_xys],
            "preference_xys": [list(xy) for xy in preference_xys],
            "constraint_radius": constraint_r,
            "preference_radius": preference_r,
        }
        _node().publish_viz(data)
    except Exception:
        pass
