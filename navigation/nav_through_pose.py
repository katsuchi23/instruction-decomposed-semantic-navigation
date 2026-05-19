"""nav_through_pose mode.

Instead of sending cmd_vel directly, this mode runs the same trajectory
sampling and scoring as the normal mode, then extracts the endpoint of the
best trajectory and sends it as a single NavigateThroughPoses goal to Nav2.
Nav2 handles all velocity control; we handle the path decision via our cost
function.

Loop cadence: replans every H * dt seconds (the trajectory horizon), which
gives Nav2 enough time to reach each waypoint before the next one is issued.
"""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from algorithm.scoring import (
    categories_to_params,
    sample_and_select_action,
    satisfaction_violation,
    step_diff_drive,
)
from ipc.costmap_client import LocalCostmapIPCClient
from ipc.nav_through_poses_client import (
    get_amcl_pose,
    get_nav_cmd_vel,
    send_nav_through_poses,
)
from ipc.robot_client import get_latest_robot_pose, reset_pose_selection_state, select_map_base_pose
from parsing.intent_parser import TaskIntent
from utils.config import get_ipc_timeout_ms, get_param, get_runtime_value
from utils.math_helpers import wrap_angle


def _set_nav2_velocity_params(speed: str) -> None:
    """Update DWB FollowPath max velocities to match the speed category."""
    speed_cfg = get_param(("control", "behavior_mapping", "speed"), {})
    entry = speed_cfg.get(speed, speed_cfg.get("normal", {"v_max": 0.3, "w_max": 0.3}))
    v_max = float(entry.get("v_max", 0.3))
    w_max = float(entry.get("w_max", 0.3))

    setup_bash = get_runtime_value(("ros_integration", "setup_bash"), "")
    ws_bash = get_runtime_value(("ros_integration", "workspace_setup_bash"), "")
    source = ""
    if setup_bash:
        source += f"source {setup_bash} && "
    if ws_bash:
        source += f"source {ws_bash} && "

    cmds = " && ".join([
        f"ros2 param set /controller_server FollowPath.max_vel_x {v_max:.4f}",
        f"ros2 param set /controller_server FollowPath.max_vel_theta {w_max:.4f}",
        f"ros2 param set /controller_server FollowPath.max_speed_xy {v_max:.4f}",
    ])
    try:
        subprocess.run(["bash", "-c", f"{source}{cmds}"], timeout=10.0, capture_output=True)
    except Exception as exc:
        print(f"[WARN] nav_through_pose: could not set DWB velocity params: {exc}")

    print(f"[INFO] nav_through_pose: Nav2 DWB limits → max_vel_x={v_max:.2f}  max_vel_theta={w_max:.2f}")


def run_nav_through_pose_task(
    task: TaskIntent,
    task_idx: int,
    total_tasks: int,
    object_location: Tuple[float, float],
    constraint_xys: Tuple[Tuple[float, float], ...] = (),
    preference_xys: Tuple[Tuple[float, float], ...] = (),
    timeout_sec: float = 180.0,
    collision_cost_thresh: float = 90.0,
    collision_duration_sec: float = 5.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Execute one task via Nav2 NavigateThroughPoses.

    Uses the same trajectory sampling and cost function as the normal mode.
    The endpoint of the best-scored trajectory is sent as a NavigateThroughPoses
    goal every H * dt seconds.  Nav2 handles all velocity control.

    Returns ``(success, task_data)`` with the same structure as ``_run_single_task``.
    """
    params = categories_to_params(task)
    _set_nav2_velocity_params(task.behavior.speed)

    replan_interval = params.H * params.dt  # fallback timer: H * dt seconds
    approach_threshold_m = float(get_param(("nav_through_pose", "approach_threshold_m"), 0.4))

    _trajectory: List[Tuple[float, float]] = []
    _steps: List[Dict[str, Any]] = []
    task_start_time = time.time()

    def _task_data(success: bool, reason: Optional[str] = None) -> Dict[str, Any]:
        return {
            "success": success,
            "failure_reason": reason,
            "trajectory": list(_trajectory),
            "steps": list(_steps),
            "duration_sec": time.time() - task_start_time,
        }

    local_costmap = LocalCostmapIPCClient()
    nav_timeout_ms = get_ipc_timeout_ms("nav_through_poses_reply", 5000)

    # ---- initial pose ----
    pose_bundle = get_latest_robot_pose(timeout_ms=2000)
    if pose_bundle is None:
        local_costmap.close()
        return False, _task_data(False, "No initial robot pose over IPC")
    robot_pose_dict = select_map_base_pose(pose_bundle)
    if robot_pose_dict is None:
        local_costmap.close()
        return False, _task_data(False, "No usable map-base pose")

    robot_pose: Tuple[float, float, float] = (
        robot_pose_dict["x"], robot_pose_dict["y"], robot_pose_dict["yaw"]
    )
    x0 = robot_pose
    u_prev = (0.0, 0.0)

    print(f"\n{'='*60}")
    print(f"Task {task_idx + 1}/{total_tasks}  [nav_through_pose]")
    print(f"Target : {task.main.target.name} at ({object_location[0]:.2f}, {object_location[1]:.2f})")
    print(f"Speed  : {task.behavior.speed}   replan_interval={replan_interval:.1f}s")
    print(f"{'='*60}")

    last_send_time = 0.0
    current_goal_xy: Optional[Tuple[float, float]] = None
    prefetched_pose: Optional[Tuple[float, float, float]] = None
    costmap, meta = local_costmap.get(timeout_ms=1000)

    def _compute_next_waypoint(c, m) -> Optional[Tuple[float, float, float]]:
        """Run trajectory sampling with the already-fetched costmap (no extra IPC)."""
        nonlocal u_prev
        result = sample_and_select_action(
            x0=x0,
            x=robot_pose,
            obj_xy=object_location,
            params=params,
            costmap=c,
            u_prev=u_prev,
            rng_seed=None,
            pose_bundle=pose_bundle,
            meta=m,
            constraint_xys=constraint_xys,
            preference_xys=preference_xys,
        )
        u_prev = result.get("best_action", u_prev)
        if result.get("no_feasible_goal", False):
            return None
        target = robot_pose
        for v, w in result.get("best_trajectory", []):
            target = step_diff_drive(target, (v, w), params.dt)
        tx, ty, _ = target
        tyaw = math.atan2(ty - robot_pose[1], tx - robot_pose[0])
        return (tx, ty, tyaw)

    # ---- main loop ----
    while True:
        elapsed = time.time() - task_start_time

        # 1) Timeout
        if elapsed >= timeout_sec:
            print(f"[FAIL] Task {task_idx+1}/{total_tasks}: timed out after {elapsed:.1f}s.")
            local_costmap.close()
            return False, _task_data(False, f"timeout after {elapsed:.1f}s")

        # 2) Update robot pose + fetch costmap ONCE per tick (reused everywhere)
        pb = get_latest_robot_pose(timeout_ms=500)
        if pb is not None:
            rpd = select_map_base_pose(pb)
            if rpd is not None:
                robot_pose = (rpd["x"], rpd["y"], rpd["yaw"])
                pose_bundle = pb
        c, m = local_costmap.get(timeout_ms=100)
        if c is not None:
            costmap, meta = c, m

        # 3) Check termination and recovery
        _, r_current, alpha_error, e_phi = satisfaction_violation(
            x0=x0, x=robot_pose, obj_xy=object_location, params=params
        )
        r_ok = params.r_min <= r_current <= params.r_max
        phase_required = bool(getattr(params, "phase_required", True))
        phi_ok = (not phase_required) or (abs(e_phi) <= params.phi_tol)
        facing_ok = abs(alpha_error) <= params.alpha_max
        all_ok = r_ok and phi_ok and facing_ok

        if all_ok:
            print(f"[DONE] Task {task_idx+1}/{total_tasks}: goal reached.")
            local_costmap.close()
            return True, _task_data(True)

        if r_ok:
            # Robot is in the goal ring — send a NavigateThroughPoses pose at
            # the current position with target-facing yaw so Nav2 rotates in
            # place.  This avoids relying on cmd_vel_sender_node being active.
            rx, ry, _ = robot_pose
            tx_obj, ty_obj = object_location
            facing_yaw = math.atan2(ty_obj - ry, tx_obj - rx)
            send_nav_through_poses([(rx, ry, facing_yaw)], timeout_ms=500)
            time.sleep(0.05)
            continue

        # 4) Send prefetched waypoint when approaching, then immediately
        #    prefetch the next one using the already-fetched costmap (no extra IPC).
        dist_to_goal = (
            math.hypot(robot_pose[0] - current_goal_xy[0], robot_pose[1] - current_goal_xy[1])
            if current_goal_xy is not None else float("inf")
        )
        should_send = (
            current_goal_xy is None
            or dist_to_goal < approach_threshold_m
            or (time.time() - last_send_time) >= replan_interval
        )
        if should_send:
            pose_to_send = prefetched_pose if prefetched_pose is not None else _compute_next_waypoint(costmap, meta)
            if pose_to_send is not None:
                tx, ty, tyaw = pose_to_send
                send_nav_through_poses([(tx, ty, tyaw)], timeout_ms=500)
                current_goal_xy = (tx, ty)
                last_send_time = time.time()
                print(f"  [nav_through_pose] → ({tx:.3f}, {ty:.3f}, {math.degrees(tyaw):.1f}°)  dist_prev={dist_to_goal:.2f}m")
            # Prefetch NEXT waypoint using the same costmap — zero extra IPC
            prefetched_pose = _compute_next_waypoint(costmap, meta)

        # 5) Telemetry
        amcl = get_amcl_pose(timeout_ms=20)
        if amcl:
            robot_pose = (amcl["x"], amcl["y"], amcl.get("yaw", robot_pose[2]))
        cmd = get_nav_cmd_vel(timeout_ms=10) or {}
        x_rec, y_rec, yaw_rec = robot_pose
        _trajectory.append((x_rec, y_rec))
        _steps.append({
            "step": len(_steps),
            "timestamp": time.time(),
            "x": x_rec,
            "y": y_rec,
            "yaw_deg": math.degrees(yaw_rec),
            "v_cmd": cmd.get("v", 0.0),
            "w_cmd": cmd.get("w", 0.0),
            "heading_error_deg": math.degrees(alpha_error),
            "phase_error_deg": math.degrees(e_phi),
            "distance_to_target": math.hypot(
                object_location[0] - x_rec,
                object_location[1] - y_rec,
            ),
        })

        time.sleep(0.1)
