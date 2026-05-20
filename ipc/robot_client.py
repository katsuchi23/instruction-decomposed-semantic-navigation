"""Robot pose and cmd_vel — direct ROS (replaces ZMQ IPC).

Same public API as before; callers (navigator.py etc.) need no changes.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

from utils.math_helpers import wrap_angle

# Diagnostic env var (kept for compat)
import os
_POSE_IPC_DIAG = os.getenv("SEMNAV_POSE_IPC_DIAG", "").strip().lower() not in {"", "0", "false", "no"}

# ---------------------------------------------------------------------------
# Pose-source selection state (unchanged logic from original)
# ---------------------------------------------------------------------------

_last_direct_pose_values: Optional[Tuple[float, float, float]] = None
_last_composed_pose_values: Optional[Tuple[float, float, float]] = None
_pose_select_last_warn_time = 0.0
_POSE_SELECT_WARN_PERIOD_SEC = 2.0
_POSE_SELECT_EPS_XY = 1e-3
_POSE_SELECT_EPS_YAW = 1e-3

_POSE_IPC_DIAG_LAST_RECV = None
_POSE_IPC_DIAG_LAST_SELECT = None

# Warn state for get_latest_robot_pose
_pose_warn_last_time = 0.0
_POSE_WARN_PERIOD_SEC = 2.0


def _node():
    from ros.ros_bridge import get_node
    return get_node()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reset_pose_selection_state() -> None:
    global _last_direct_pose_values, _last_composed_pose_values
    _last_direct_pose_values = None
    _last_composed_pose_values = None


def reconnect_pose_socket() -> None:
    """No-op — direct TF2 has no persistent socket to reconnect."""
    print("[INFO] reconnect_pose_socket: no-op (using direct TF2).")


def reconnect_nav_status_socket() -> None:
    """No-op — direct ROS has no ZMQ socket."""
    print("[INFO] reconnect_nav_status_socket: no-op (using direct ROS).")


def reset_ipc_channels() -> None:
    """Reset IPC-related state (no sockets to cycle in direct-ROS mode)."""
    global _POSE_IPC_DIAG_LAST_RECV, _POSE_IPC_DIAG_LAST_SELECT
    _POSE_IPC_DIAG_LAST_RECV = None
    _POSE_IPC_DIAG_LAST_SELECT = None
    reset_pose_selection_state()
    print("[INFO] IPC channels reset (direct ROS mode).")


def get_latest_robot_pose(timeout_ms: int = 2000) -> Optional[Dict[str, Any]]:
    """Return a TF bundle dict (same structure as the ZMQ version)."""
    global _pose_warn_last_time
    node = _node()

    map_frame = node.map_frame
    odom_frame = node.odom_frame
    base_frame = node.base_frame
    camera_frame = node.camera_frame

    tf_map_base = node.get_tf(map_frame, base_frame)
    tf_map_odom = node.get_tf(map_frame, odom_frame)
    tf_odom_base = node.get_tf(odom_frame, base_frame)
    tf_map_camera = node.get_tf(map_frame, camera_frame)

    if tf_map_base is None and tf_map_odom is None and tf_odom_base is None:
        now = time.time()
        if (now - _pose_warn_last_time) >= _POSE_WARN_PERIOD_SEC:
            print("[WARN] get_latest_robot_pose: no TF transforms available yet.")
            _pose_warn_last_time = now
        return None

    bundle: Dict[str, Any] = {
        "map_base": tf_map_base,
        "map_odom": tf_map_odom,
        "odom_base": tf_odom_base,
        "map_camera": tf_map_camera,
        "frames": {
            "map": map_frame,
            "odom": odom_frame,
            "base": base_frame,
        },
    }

    if _POSE_IPC_DIAG:
        global _POSE_IPC_DIAG_LAST_RECV
        def _sig(tf):
            if tf is None:
                return None
            return (
                round(float(tf["x"]), 4),
                round(float(tf["y"]), 4),
                round(float(tf["yaw"]), 4),
                int(tf.get("stamp_sec", 0)),
                int(tf.get("stamp_nanosec", 0)),
            )
        sig = (_sig(tf_map_base), _sig(tf_map_odom), _sig(tf_odom_base))
        if sig != _POSE_IPC_DIAG_LAST_RECV:
            print(f"[POSE_IPC_DIAG] recv map_base={sig[0]} map_odom={sig[1]} odom_base={sig[2]}")
            _POSE_IPC_DIAG_LAST_RECV = sig

    return bundle


def send_cmd_vel_via_ipc(linear_x: float, angular_z: float, timeout_ms: int = 500) -> bool:
    """Publish cmd_vel directly to /cmd_vel (always returns True)."""
    _node().publish_cmd_vel(linear_x, angular_z)
    return True


def send_nav_goal_via_ipc(x: float, y: float, yaw: float, timeout_ms: int = 2000) -> bool:
    """Send a single-pose nav goal via NavigateThroughPoses."""
    return _node().send_nav_through_poses([(x, y, yaw)])


def get_nav_goal_status(timeout_ms: int = 100) -> Optional[Dict[str, Any]]:
    """Return current NavigateThroughPoses status dict."""
    return _node().get_nav_through_poses_status()


def send_trajectory_via_ipc(poses, timeout_ms: int = 500) -> bool:
    """Not used in direct-ROS mode; kept for API compat."""
    return True


# ---------------------------------------------------------------------------
# Pose-source selection helpers (unchanged from original)
# ---------------------------------------------------------------------------

def select_map_base_pose(pose_bundle: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Return the most reliable map-frame base pose from a TF bundle."""
    global _last_direct_pose_values, _last_composed_pose_values, _pose_select_last_warn_time
    if pose_bundle is None:
        return None

    direct = pose_bundle.get("map_base")
    map_odom = pose_bundle.get("map_odom")
    odom_base = pose_bundle.get("odom_base")

    def _changed(prev: Optional[Tuple[float, float, float]], curr: Dict[str, float]) -> bool:
        if prev is None:
            return True
        dx = abs(float(curr["x"]) - prev[0])
        dy = abs(float(curr["y"]) - prev[1])
        dyaw = abs(wrap_angle(float(curr["yaw"]) - prev[2]))
        return dx > _POSE_SELECT_EPS_XY or dy > _POSE_SELECT_EPS_XY or dyaw > _POSE_SELECT_EPS_YAW

    direct_pose: Optional[Dict[str, float]] = None
    if direct is not None:
        direct_pose = {
            "x": float(direct["x"]),
            "y": float(direct["y"]),
            "yaw": float(direct["yaw"]),
            "stamp_sec": int(direct.get("stamp_sec", 0)),
            "stamp_nanosec": int(direct.get("stamp_nanosec", 0)),
            "source": "direct_map_base",
        }

    composed: Optional[Dict[str, float]] = None
    if map_odom is not None and odom_base is not None:
        th = float(map_odom["yaw"])
        c = math.cos(th)
        s = math.sin(th)
        ox = float(odom_base["x"])
        oy = float(odom_base["y"])
        composed = {
            "x": float(map_odom["x"]) + c * ox - s * oy,
            "y": float(map_odom["y"]) + s * ox + c * oy,
            "yaw": wrap_angle(float(map_odom["yaw"]) + float(odom_base["yaw"])),
            "stamp_sec": int(odom_base.get("stamp_sec", 0)),
            "stamp_nanosec": int(odom_base.get("stamp_nanosec", 0)),
            "source": "composed_map_odom_odom_base",
        }

    direct_changed = _changed(_last_direct_pose_values, direct_pose) if direct_pose is not None else False
    composed_changed = _changed(_last_composed_pose_values, composed) if composed is not None else False

    if direct_pose is not None:
        _last_direct_pose_values = (float(direct_pose["x"]), float(direct_pose["y"]), float(direct_pose["yaw"]))
    if composed is not None:
        _last_composed_pose_values = (float(composed["x"]), float(composed["y"]), float(composed["yaw"]))

    selected = None
    if direct_pose is not None and composed is not None:
        if direct_changed and not composed_changed:
            now = time.time()
            if (now - _pose_select_last_warn_time) >= _POSE_SELECT_WARN_PERIOD_SEC:
                print("[WARN] Pose selection: composed frozen, using direct_map_base.")
                _pose_select_last_warn_time = now
            selected = direct_pose
        elif composed_changed and not direct_changed:
            selected = composed
        else:
            d_stamp = (int(direct_pose.get("stamp_sec", 0)), int(direct_pose.get("stamp_nanosec", 0)))
            c_stamp = (int(composed.get("stamp_sec", 0)), int(composed.get("stamp_nanosec", 0)))
            selected = composed if c_stamp > d_stamp else direct_pose
    elif direct_pose is not None:
        selected = direct_pose
    elif composed is not None:
        selected = composed

    if _POSE_IPC_DIAG:
        global _POSE_IPC_DIAG_LAST_SELECT
        sig = None if selected is None else (
            round(float(selected["x"]), 4),
            round(float(selected["y"]), 4),
            round(float(selected["yaw"]), 4),
            selected.get("source"),
        )
        if sig != _POSE_IPC_DIAG_LAST_SELECT:
            print(f"[POSE_IPC_DIAG] select {sig}")
            _POSE_IPC_DIAG_LAST_SELECT = sig
    return selected


def wait_until_close_to_target(
    target_xy: Tuple[float, float],
    target_yaw: float,
    *,
    yaw_tol: float = 0.1,
    dist_tol: float = 0.8,
    weight_yaw: float = 0.8,
    weight_dist: float = 0.2,
    score_thresh: float = 0.8,
    timeout_s: float = 30.0,
    poll_hz: float = 10.0,
):
    """Block until the robot is close enough to the target or Nav2 reports SUCCEEDED."""
    wsum = float(weight_yaw) + float(weight_dist)
    wy = float(weight_yaw) / wsum
    wd = float(weight_dist) / wsum

    print(
        f"Starting wait_until_close_to_target: "
        f"target=({target_xy[0]:.3f}, {target_xy[1]:.3f}, {math.degrees(target_yaw):.1f} deg)"
    )

    deadline = time.time() + float(timeout_s)
    period = 1.0 / float(poll_hz)

    while time.time() < deadline:
        nav_status = get_nav_goal_status(timeout_ms=10)
        if nav_status is not None and nav_status.get("status") == "SUCCEEDED":
            pose_bundle = get_latest_robot_pose(timeout_ms=int(period * 1000))
            pose = select_map_base_pose(pose_bundle)
            if pose is not None:
                dx = pose["x"] - float(target_xy[0])
                dy = pose["y"] - float(target_xy[1])
                dist = math.hypot(dx, dy)
                dyaw = wrap_angle(pose["yaw"] - float(target_yaw))
                yaw_err = abs(dyaw)
                print(f"Nav2 goal SUCCEEDED! dist={dist:.3f}m, yaw_err={math.degrees(yaw_err):.1f} deg")
                pose["_dist_err"] = dist
                pose["_yaw_err"] = yaw_err
                pose["_score"] = 0.0
                pose["_nav2_success"] = True
                return pose

        pose_bundle = get_latest_robot_pose(timeout_ms=int(period * 1000))
        pose = select_map_base_pose(pose_bundle)
        if pose is None:
            time.sleep(period)
            continue

        dx = pose["x"] - float(target_xy[0])
        dy = pose["y"] - float(target_xy[1])
        dist = math.hypot(dx, dy)
        dyaw = wrap_angle(pose["yaw"] - float(target_yaw))
        yaw_err = abs(dyaw)

        yaw_n = min(yaw_err / float(yaw_tol), 1.0) if yaw_tol > 0 else 0.0
        dist_n = min(dist / float(dist_tol), 1.0) if dist_tol > 0 else 0.0
        score = wy * yaw_n + wd * dist_n

        if score <= float(score_thresh) and dist <= float(dist_tol) and yaw_err <= float(yaw_tol):
            print(f"Target reached! score={score:.3f}")
            pose["_dist_err"] = dist
            pose["_yaw_err"] = yaw_err
            pose["_score"] = score
            pose["_nav2_success"] = False
            return pose

        time.sleep(period)

    print(f"Timeout reached after {timeout_s}s")
    return None
