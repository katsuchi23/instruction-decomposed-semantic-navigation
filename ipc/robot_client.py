"""ZMQ IPC clients for robot pose, velocity commands, navigation goals, and trajectory."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zmq

from utils.config import get_ipc_endpoint
from utils.math_helpers import wrap_angle

_ctx = zmq.Context.instance()

# ---------------------------------------------------------------------------
# Singleton subscribers
# ---------------------------------------------------------------------------
_pose_sub = None
_nav_status_sub = None

# Singleton REQ socket for cmd_vel (called every control step at ~30Hz,
# so creating+destroying a socket every iteration is wasteful and can
# overwhelm the REP side).  If a send/recv cycle fails the socket is
# recreated automatically.
_cmd_vel_req = None
_cmd_vel_send_failures = 0

# Pose socket health diagnostics (5557)
_pose_poll_timeouts = 0
_pose_last_recv_time: Optional[float] = None
_pose_last_warn_time = 0.0
_POSE_WARN_PERIOD_SEC = 2.0

# Pose-source selection diagnostics/state
_last_direct_pose_values: Optional[Tuple[float, float, float]] = None
_last_composed_pose_values: Optional[Tuple[float, float, float]] = None
_pose_select_last_warn_time = 0.0
_POSE_SELECT_WARN_PERIOD_SEC = 2.0
_POSE_SELECT_EPS_XY = 1e-3
_POSE_SELECT_EPS_YAW = 1e-3
_POSE_IPC_DIAG = os.getenv("SEMNAV_POSE_IPC_DIAG", "").strip().lower() not in {"", "0", "false", "no"}
_POSE_IPC_DIAG_LAST_RECV = None
_POSE_IPC_DIAG_LAST_SELECT = None


def _ensure_pose_sub():
    global _pose_sub
    if _pose_sub is not None:
        return _pose_sub
    s = _ctx.socket(zmq.SUB)
    s.connect(get_ipc_endpoint("pose_sub"))
    s.setsockopt_string(zmq.SUBSCRIBE, "TF ")
    s.setsockopt(zmq.CONFLATE, 1)
    _pose_sub = s
    return _pose_sub


def reset_pose_selection_state() -> None:
    """Clear cached pose-source selection state.

    Call this before re-navigation attempts so that the selection heuristic
    (which compares current vs previous pose values) starts fresh and does not
    carry stale 'changed' / 'frozen' decisions from a prior navigation run.
    """
    global _last_direct_pose_values, _last_composed_pose_values
    _last_direct_pose_values = None
    _last_composed_pose_values = None


def reconnect_pose_socket() -> None:
    """Destroy and recreate the pose SUB socket.

    Call this after long pauses (e.g. semantic map update) where the socket
    may have gone stale.  The next ``get_latest_robot_pose()`` call will
    establish a fresh TCP connection and immediately receive the publisher's
    latest message.
    """
    global _pose_sub, _pose_poll_timeouts, _pose_last_recv_time
    if _pose_sub is not None:
        try:
            _pose_sub.setsockopt(zmq.LINGER, 0)
            _pose_sub.close()
        except Exception:
            pass
        _pose_sub = None
    _pose_poll_timeouts = 0
    _pose_last_recv_time = None
    print("[INFO] Pose SUB socket closed; will reconnect on next read.")


def reconnect_nav_status_socket() -> None:
    """Destroy and recreate the Nav2 status SUB socket."""
    global _nav_status_sub
    if _nav_status_sub is not None:
        try:
            _nav_status_sub.setsockopt(zmq.LINGER, 0)
            _nav_status_sub.close()
        except Exception:
            pass
        _nav_status_sub = None
    print("[INFO] Nav-status SUB socket closed; will reconnect on next read.")


def _ensure_nav_status_sub():
    global _nav_status_sub
    if _nav_status_sub is not None:
        return _nav_status_sub
    s = _ctx.socket(zmq.SUB)
    s.connect(get_ipc_endpoint("nav_status_sub"))
    s.setsockopt_string(zmq.SUBSCRIBE, "NAV_STATUS")
    s.setsockopt(zmq.CONFLATE, 1)
    _nav_status_sub = s
    return _nav_status_sub


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_latest_robot_pose(timeout_ms: int = 2000) -> Optional[Dict[str, Any]]:
    """Return a dict with ``map_base``, ``map_odom``, ``odom_base``, ``map_camera`` and ``frames``."""
    global _pose_poll_timeouts, _pose_last_recv_time, _pose_last_warn_time
    s = _ensure_pose_sub()
    now = time.time()
    if s.poll(timeout_ms) == 0:
        _pose_poll_timeouts += 1
        if (
            _pose_poll_timeouts == 1
            or (now - _pose_last_warn_time) >= _POSE_WARN_PERIOD_SEC
        ):
            silent_for = (
                timeout_ms / 1000.0
                if _pose_last_recv_time is None
                else max(0.0, now - _pose_last_recv_time)
            )
            print(
                f"[WARN] Pose socket timeout on {get_ipc_endpoint('pose_sub')}; "
                f"no TF bundle received for {silent_for:.2f}s "
                f"(timeout_ms={timeout_ms}, consecutive_timeouts={_pose_poll_timeouts})."
            )
            _pose_last_warn_time = now
        return None

    try:
        msg = s.recv_string()
    except Exception as exc:
        if (now - _pose_last_warn_time) >= _POSE_WARN_PERIOD_SEC:
            print(f"[WARN] Pose socket recv failed on {get_ipc_endpoint('pose_sub')}: {exc}")
            _pose_last_warn_time = now
        return None

    _pose_last_recv_time = now
    if _pose_poll_timeouts > 0:
        print(
            f"[INFO] Pose socket recovered on {get_ipc_endpoint('pose_sub')} "
            f"after {_pose_poll_timeouts} timeout(s)."
        )
        _pose_poll_timeouts = 0

    try:
        _, j = msg.split(" ", 1)
        data = json.loads(j)
    except Exception as exc:
        if (now - _pose_last_warn_time) >= _POSE_WARN_PERIOD_SEC:
            print(f"[WARN] Malformed TF bundle received on {get_ipc_endpoint('pose_sub')}: {exc}")
            _pose_last_warn_time = now
        return None

    tfs = data.get("tfs", {})

    def _extract(tf_key):
        tf = tfs.get(tf_key)
        if tf is None:
            return None
        return {
            "x": tf["x"],
            "y": tf["y"],
            "z": tf.get("z", 0.0),
            "yaw": tf["yaw"],
            "stamp_sec": tf["stamp_sec"],
            "stamp_nanosec": tf["stamp_nanosec"],
            "parent": tf["parent"],
            "child": tf["child"],
        }

    bundle = {
        "map_base": _extract("map_base"),
        "map_odom": _extract("map_odom"),
        "odom_base": _extract("odom_base"),
        "map_camera": _extract("map_camera"),
        "frames": {
            "map": data.get("map_frame"),
            "odom": data.get("odom_frame"),
            "base": data.get("base_frame"),
        },
    }
    if _POSE_IPC_DIAG:
        global _POSE_IPC_DIAG_LAST_RECV
        signature = tuple(
            None if bundle.get(key) is None else (
                round(float(bundle[key]["x"]), 4),
                round(float(bundle[key]["y"]), 4),
                round(float(bundle[key]["yaw"]), 4),
                int(bundle[key].get("stamp_sec", 0)),
                int(bundle[key].get("stamp_nanosec", 0)),
            )
            for key in ("map_base", "map_odom", "odom_base")
        )
        if signature != _POSE_IPC_DIAG_LAST_RECV:
            print(
                "[POSE_IPC_DIAG] recv "
                f"map_base={signature[0]} map_odom={signature[1]} odom_base={signature[2]}"
            )
            _POSE_IPC_DIAG_LAST_RECV = signature
    return bundle


def _ensure_cmd_vel_req():
    """Return (or create) a persistent REQ socket for cmd_vel."""
    global _cmd_vel_req
    if _cmd_vel_req is not None:
        return _cmd_vel_req
    s = _ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(get_ipc_endpoint("cmd_vel_req"))
    _cmd_vel_req = s
    return _cmd_vel_req


def _reset_cmd_vel_req():
    """Destroy and recreate the cmd_vel REQ socket (recovery after timeout)."""
    global _cmd_vel_req
    if _cmd_vel_req is not None:
        try:
            _cmd_vel_req.close()
        except Exception:
            pass
        _cmd_vel_req = None


def reset_ipc_channels() -> None:
    """Force all long-lived IPC sockets/state to reconnect fresh.

    Use this at the start of each experiment/task so runs don't inherit stale
    SUB/REQ state across ROS restarts or previous navigation attempts.
    """
    global _cmd_vel_send_failures, _POSE_IPC_DIAG_LAST_RECV, _POSE_IPC_DIAG_LAST_SELECT
    reconnect_pose_socket()
    reconnect_nav_status_socket()
    _reset_cmd_vel_req()
    _cmd_vel_send_failures = 0
    _POSE_IPC_DIAG_LAST_RECV = None
    _POSE_IPC_DIAG_LAST_SELECT = None
    reset_pose_selection_state()
    print("[INFO] IPC channels reset; pose/status/cmd_vel sockets will reconnect fresh.")


def send_cmd_vel_via_ipc(linear_x: float, angular_z: float, timeout_ms: int = 500) -> bool:
    """Send a velocity command ``(v, w)`` using a persistent REQ socket."""
    global _cmd_vel_send_failures
    sock = _ensure_cmd_vel_req()
    try:
        payload = json.dumps({"v": float(linear_x), "w": float(angular_z)})
        sock.send_string(payload)
        if sock.poll(timeout_ms) == 0:
            # Timeout — the REQ socket is now in a bad state (sent but no
            # reply).  Destroy it so the next call gets a fresh one.
            _cmd_vel_send_failures += 1
            if _cmd_vel_send_failures % 5 == 1:
                print(
                    f"[WARN] send_cmd_vel_via_ipc: timeout waiting for reply "
                    f"(consecutive_failures={_cmd_vel_send_failures})"
                )
            _reset_cmd_vel_req()
            return False
        resp = sock.recv_string()
        _cmd_vel_send_failures = 0
        return resp == "OK"
    except zmq.ZMQError as exc:
        _cmd_vel_send_failures += 1
        if _cmd_vel_send_failures % 5 == 1:
            print(f"[WARN] send_cmd_vel_via_ipc: ZMQ error: {exc} (failures={_cmd_vel_send_failures})")
        _reset_cmd_vel_req()
        return False


def send_nav_goal_via_ipc(x: float, y: float, yaw: float, timeout_ms: int = 2000) -> bool:
    """Send a 2-D navigation goal."""
    sock = _ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(get_ipc_endpoint("nav_goal_req"))
    try:
        sock.send_string(json.dumps({"x": float(x), "y": float(y), "yaw": float(yaw)}))
        if sock.poll(timeout_ms) == 0:
            return False
        resp = sock.recv_string()
        return resp == "OK"
    finally:
        sock.close()


def get_nav_goal_status(timeout_ms: int = 100) -> Optional[Dict[str, Any]]:
    """Return ``{status: "IDLE"|"ACTIVE"|"SUCCEEDED"|"FAILED"|"CANCELED"}``."""
    s = _ensure_nav_status_sub()
    if s.poll(timeout_ms) == 0:
        return None
    msg = s.recv_string()
    _, j = msg.split(" ", 1)
    return json.loads(j)


def send_trajectory_via_ipc(poses, timeout_ms: int = 500) -> bool:
    """Send a trajectory (list of pose dicts) on port 5563."""
    sock = _ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(get_ipc_endpoint("trajectory_req"))
    try:
        payload = json.dumps({"poses": poses})
        sock.send_string(payload)
        if sock.poll(timeout_ms) == 0:
            return False
        resp = sock.recv_string()
        return resp == "OK"
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def _extract_map_base_pose(pose_bundle) -> Optional[Dict[str, float]]:
    pose = select_map_base_pose(pose_bundle)
    if pose is None:
        return None
    return {
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "yaw": float(pose["yaw"]),
    }


def select_map_base_pose(pose_bundle: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Return the most reliable map-frame base pose from a TF bundle.

    Selection policy:
    1) If one source is changing while the other is frozen, pick the changing source.
    2) Otherwise pick newer timestamp.
    3) Tie-break to direct ``map_base``.
    """
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
        _last_direct_pose_values = (
            float(direct_pose["x"]),
            float(direct_pose["y"]),
            float(direct_pose["yaw"]),
        )
    if composed is not None:
        _last_composed_pose_values = (
            float(composed["x"]),
            float(composed["y"]),
            float(composed["yaw"]),
        )

    selected = None
    if direct_pose is not None and composed is not None:
        if direct_changed and not composed_changed:
            now = time.time()
            if (now - _pose_select_last_warn_time) >= _POSE_SELECT_WARN_PERIOD_SEC:
                print(
                    "[WARN] Pose selection fallback: composed map pose is frozen while "
                    "direct map_base is updating; using direct_map_base."
                )
                _pose_select_last_warn_time = now
            selected = direct_pose
        elif composed_changed and not direct_changed:
            selected = composed
        else:
            d_stamp = (
                int(direct_pose.get("stamp_sec", 0)),
                int(direct_pose.get("stamp_nanosec", 0)),
            )
            c_stamp = (
                int(composed.get("stamp_sec", 0)),
                int(composed.get("stamp_nanosec", 0)),
            )
            if c_stamp > d_stamp:
                selected = composed
            else:
                selected = direct_pose
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
    """Block until the robot is close enough to the target or Nav2 reports ``SUCCEEDED``."""
    wsum = float(weight_yaw) + float(weight_dist)
    wy = float(weight_yaw) / wsum
    wd = float(weight_dist) / wsum

    print(
        f"Starting wait_until_close_to_target: "
        f"target=({target_xy[0]:.3f}, {target_xy[1]:.3f}, {math.degrees(target_yaw):.1f} deg)",
        flush=True,
    )

    deadline = time.time() + float(timeout_s)
    period = 1.0 / float(poll_hz)

    while time.time() < deadline:
        nav_status = get_nav_goal_status(timeout_ms=10)
        if nav_status is not None and nav_status.get("status") == "SUCCEEDED":
            pose_bundle = get_latest_robot_pose(timeout_ms=int(period * 1000))
            pose = _extract_map_base_pose(pose_bundle)
            if pose is not None:
                dx = pose["x"] - float(target_xy[0])
                dy = pose["y"] - float(target_xy[1])
                dist = math.hypot(dx, dy)
                dyaw = wrap_angle(pose["yaw"] - float(target_yaw))
                yaw_err = abs(dyaw)
                print(f"Nav2 goal SUCCEEDED! dist={dist:.3f}m, yaw_err={math.degrees(yaw_err):.1f} deg", flush=True)
                pose["_dist_err"] = dist
                pose["_yaw_err"] = yaw_err
                pose["_score"] = 0.0
                pose["_nav2_success"] = True
                return pose

        pose_bundle = get_latest_robot_pose(timeout_ms=int(period * 1000))
        pose = _extract_map_base_pose(pose_bundle)
        if pose is None:
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
            print(f"Target reached! score={score:.3f}", flush=True)
            pose["_dist_err"] = dist
            pose["_yaw_err"] = yaw_err
            pose["_score"] = score
            pose["_nav2_success"] = False
            return pose

    print(f"Timeout reached after {timeout_s}s", flush=True)
    return None
