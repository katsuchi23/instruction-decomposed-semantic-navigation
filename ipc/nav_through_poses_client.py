"""Nav2 NavigateThroughPoses client — direct ROS (replaces ZMQ IPC).

Same public API as before; callers need no changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _node():
    from ros.ros_bridge import get_node
    return get_node()


def send_nav_through_poses(
    poses: List[Tuple[float, float, float]],
    timeout_ms: Optional[int] = None,
) -> bool:
    """Send a list of (x, y, yaw) poses to NavigateThroughPoses. Returns True on success."""
    return _node().send_nav_through_poses(poses)


def get_nav_cmd_vel(timeout_ms: int = 50) -> Optional[Dict[str, float]]:
    """Return latest /cmd_vel as {v, w} or None."""
    return _node().get_latest_cmd_vel()


def get_amcl_pose(timeout_ms: int = 200) -> Optional[Dict[str, float]]:
    """Return latest /amcl_pose as {x, y, yaw} or None."""
    return _node().get_amcl_pose()


def get_nav_through_poses_status(timeout_ms: int = 50) -> Optional[Dict[str, Any]]:
    """Return NavigateThroughPoses status dict."""
    return _node().get_nav_through_poses_status()
