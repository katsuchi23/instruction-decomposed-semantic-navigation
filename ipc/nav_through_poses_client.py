"""IPC clients for nav_through_pose mode.

ZMQ channels (all on localhost):
  5566 REQ/REP  nav_through_poses_req          — send NavigateThroughPoses goal
  5567 PUB/SUB  nav_cmd_vel_sub                — /cmd_vel telemetry stream
  5568 PUB/SUB  amcl_pose_sub                  — /amcl_pose stream
  5569 PUB/SUB  nav_through_poses_status_sub   — NavigateThroughPoses status
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import zmq

from utils.config import get_ipc_endpoint, get_ipc_timeout_ms

_ctx = zmq.Context.instance()

_nav_cmd_vel_sub: Optional[zmq.Socket] = None
_amcl_pose_sub: Optional[zmq.Socket] = None
_nav_through_poses_status_sub: Optional[zmq.Socket] = None


def _ensure_nav_cmd_vel_sub() -> zmq.Socket:
    global _nav_cmd_vel_sub
    if _nav_cmd_vel_sub is None:
        s = _ctx.socket(zmq.SUB)
        s.connect(get_ipc_endpoint("nav_cmd_vel_sub"))
        s.setsockopt_string(zmq.SUBSCRIBE, "CMD_VEL ")
        s.setsockopt(zmq.CONFLATE, 1)
        _nav_cmd_vel_sub = s
    return _nav_cmd_vel_sub


def _ensure_amcl_pose_sub() -> zmq.Socket:
    global _amcl_pose_sub
    if _amcl_pose_sub is None:
        s = _ctx.socket(zmq.SUB)
        s.connect(get_ipc_endpoint("amcl_pose_sub"))
        s.setsockopt_string(zmq.SUBSCRIBE, "AMCL_POSE ")
        s.setsockopt(zmq.CONFLATE, 1)
        _amcl_pose_sub = s
    return _amcl_pose_sub


def _ensure_nav_through_poses_status_sub() -> zmq.Socket:
    global _nav_through_poses_status_sub
    if _nav_through_poses_status_sub is None:
        s = _ctx.socket(zmq.SUB)
        s.connect(get_ipc_endpoint("nav_through_poses_status_sub"))
        s.setsockopt_string(zmq.SUBSCRIBE, "NAV_THROUGH_STATUS ")
        s.setsockopt(zmq.CONFLATE, 1)
        _nav_through_poses_status_sub = s
    return _nav_through_poses_status_sub


def send_nav_through_poses(
    poses: List[Tuple[float, float, float]],
    timeout_ms: Optional[int] = None,
) -> bool:
    """Send a list of (x, y, yaw) poses to NavigateThroughPoses. Returns True on ACK."""
    if timeout_ms is None:
        timeout_ms = get_ipc_timeout_ms("nav_through_poses_reply", 5000)
    sock = _ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(get_ipc_endpoint("nav_through_poses_req"))
    try:
        payload = json.dumps({
            "poses": [{"x": float(x), "y": float(y), "yaw": float(yaw)}
                      for x, y, yaw in poses]
        })
        sock.send_string(payload)
        if sock.poll(timeout=timeout_ms) == 0:
            return False
        return sock.recv_string().startswith("OK")
    except Exception:
        return False
    finally:
        sock.close()


def get_nav_cmd_vel(timeout_ms: int = 50) -> Optional[Dict[str, float]]:
    """Return latest /cmd_vel as {v, w} or None on timeout."""
    sub = _ensure_nav_cmd_vel_sub()
    if sub.poll(timeout=timeout_ms) == 0:
        return None
    try:
        raw = sub.recv_string(flags=zmq.NOBLOCK)
        return json.loads(raw[len("CMD_VEL "):])
    except Exception:
        return None


def get_amcl_pose(timeout_ms: int = 200) -> Optional[Dict[str, float]]:
    """Return latest /amcl_pose as {x, y, yaw} or None on timeout."""
    sub = _ensure_amcl_pose_sub()
    if sub.poll(timeout=timeout_ms) == 0:
        return None
    try:
        raw = sub.recv_string(flags=zmq.NOBLOCK)
        return json.loads(raw[len("AMCL_POSE "):])
    except Exception:
        return None


def get_nav_through_poses_status(timeout_ms: int = 50) -> Optional[Dict[str, Any]]:
    """Return latest NavigateThroughPoses status dict or None on timeout."""
    sub = _ensure_nav_through_poses_status_sub()
    if sub.poll(timeout=timeout_ms) == 0:
        return None
    try:
        raw = sub.recv_string(flags=zmq.NOBLOCK)
        return json.loads(raw[len("NAV_THROUGH_STATUS "):])
    except Exception:
        return None
