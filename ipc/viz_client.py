"""IPC publisher for sending navigation visualization data to the RViz bridge.

Uses a ZMQ PUB socket (bound on 5570) so the ROS node can subscribe at any time.
Messages are prefixed with "VIZ " and contain JSON-encoded visualization data.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import zmq

from utils.config import get_ipc_endpoint

_ctx = zmq.Context.instance()
_viz_pub: Optional[zmq.Socket] = None


def _ensure_viz_pub() -> zmq.Socket:
    global _viz_pub
    if _viz_pub is None:
        s = _ctx.socket(zmq.PUB)
        s.setsockopt(zmq.SNDHWM, 1)   # drop old messages if subscriber is slow
        s.bind(get_ipc_endpoint("viz_pub"))
        _viz_pub = s
    return _viz_pub


def _rollout(
    robot_pose: Tuple[float, float, float],
    seq: List[Tuple[float, float]],
    dt: float,
) -> List[Tuple[float, float]]:
    """Roll out a (v, w) velocity sequence from robot_pose into (x, y) positions."""
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
    max_traj_samples: int = 30,
) -> None:
    """Publish navigation viz data non-blocking — drop if no subscriber."""
    try:
        pub = _ensure_viz_pub()

        # Downsample trajectory samples and roll them out to (x, y) paths
        seqs = all_seqs[:max_traj_samples] if len(all_seqs) > max_traj_samples else all_seqs
        traj_samples = [_rollout(robot_pose, seq, dt) for seq in seqs]

        payload = json.dumps({
            "robot":   list(robot_pose),
            "start":   list(start_pose),
            "target":  list(target_xy),
            "goal":    list(active_goal_xy),
            "lookahead": list(lookahead_xy) if lookahead_xy else None,
            "path":    [list(p) for p in path],
            "traj_samples": [[list(p) for p in traj] for traj in traj_samples],
            "r_min":   r_min,
            "r_max":   r_max,
            "mode":    mode,
            "task_name": task_name,
        })
        pub.send_string(f"VIZ {payload}", flags=zmq.NOBLOCK)
    except Exception:
        pass
