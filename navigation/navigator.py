"""Main navigation loop — orchestrates planning, scoring, and robot control.

Supports multi-task instructions: each task is executed sequentially."""

from __future__ import annotations

import csv
import datetime
import json
import math
import pprint
import random
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from algorithm.scoring import (
    ControlParams,
    categories_to_params,
    sample_and_select_action,
    satisfaction_violation,
)
from ipc.costmap_client import GlobalCostmapIPCClient, LocalCostmapIPCClient
from ipc.occupancy_grid import get_cost_at_pose
from ipc.robot_client import get_latest_robot_pose, reconnect_pose_socket, reset_pose_selection_state, select_map_base_pose, send_cmd_vel_via_ipc
from navigation.object_retrieval import (
    resolve_docs_path,
    retrieve_object_location,
    get_direction_min_distance_m,
)
from parsing.intent_parser import TaskIntent, ObjectRef
from utils.config import get_outputs_root, get_param, get_runtime_value, validate_runtime_prereqs
from ipc.viz_client import send_viz_data
from utils.path_visualization import plot_task_path
from utils.intent_cache import load_or_parse_instruction
from utils.math_helpers import wrap_angle

_LAST_INFLATION_CFG: Optional[Tuple[float, float]] = None


def _reset_action_selector_state() -> None:
    """Drop cached planner/sampling state so re-navigation starts fresh."""
    if hasattr(sample_and_select_action, "_planner"):
        del sample_and_select_action._planner
    if hasattr(sample_and_select_action, "_last_s0_index"):
        del sample_and_select_action._last_s0_index
    # Reset pose-source selection heuristic so re-navigation doesn't inherit
    # stale 'changed'/'frozen' decisions from the prior attempt.
    reset_pose_selection_state()


# ---------------------------------------------------------------------------
# Single-task execution loop
# ---------------------------------------------------------------------------

def _meta_stamp(meta: Optional[dict]) -> Optional[Tuple[int, int]]:
    if not meta:
        return None
    try:
        return (int(meta.get("stamp_sec", -1)), int(meta.get("stamp_nanosec", -1)))
    except Exception:
        return None


def _snapshot_costmap_stamps(timeout_ms: int = 300) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    with LocalCostmapIPCClient() as local_client, GlobalCostmapIPCClient() as global_client:
        _, lmeta = local_client.get(timeout_ms=timeout_ms)
        _, gmeta = global_client.get(timeout_ms=timeout_ms)
    return _meta_stamp(lmeta), _meta_stamp(gmeta)


def _wait_costmap_refresh(
    prev_local: Optional[Tuple[int, int]],
    prev_global: Optional[Tuple[int, int]],
    timeout_s: float = 5.0,
) -> bool:
    """Wait until both costmaps publish at least one fresh stamp after param update."""
    with LocalCostmapIPCClient() as local_client, GlobalCostmapIPCClient() as global_client:
        seen_local = prev_local is None
        seen_global = prev_global is None
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            _, lmeta = local_client.get(timeout_ms=250)
            _, gmeta = global_client.get(timeout_ms=250)
            lnow = _meta_stamp(lmeta)
            gnow = _meta_stamp(gmeta)
            if lnow is not None and lnow != prev_local:
                seen_local = True
            if gnow is not None and gnow != prev_global:
                seen_global = True
            if seen_local and seen_global:
                return True
            time.sleep(0.1)
    return False

def _inflation_by_caution(caution: str) -> Tuple[float, float]:
    key = (caution or "normal").strip().lower()
    if key == "low":
        return 0.30, 0.20   # local, global
    if key == "high":
        return 0.50, 0.40   # local, global
    return 0.40, 0.30       # medium/normal


def _set_costmap_inflation_by_caution(caution: str) -> bool:
    global _LAST_INFLATION_CFG
    if not bool(get_runtime_value(("feature_flags", "enable_ros_param_inflation_updates"), False)):
        return False

    local_r, global_r = _inflation_by_caution(caution)
    cfg = (local_r, global_r)
    if _LAST_INFLATION_CFG == cfg:
        return True

    prev_local, prev_global = _snapshot_costmap_stamps(timeout_ms=250)

    ros_setup_bash = str(get_runtime_value(("ros_integration", "setup_bash"), "/opt/ros/humble/setup.bash"))
    ws_setup_bash = str(get_runtime_value(("ros_integration", "workspace_setup_bash"), ""))
    local_param = str(
        get_runtime_value(
            ("ros_integration", "local_costmap_param"),
            "/local_costmap/local_costmap inflation_layer.inflation_radius",
        )
    )
    global_param = str(
        get_runtime_value(
            ("ros_integration", "global_costmap_param"),
            "/global_costmap/global_costmap inflation_layer.inflation_radius",
        )
    )
    source_parts = [f"source {ros_setup_bash}"]
    if ws_setup_bash:
        source_parts.append(f"source {ws_setup_bash}")
    cmd = (
        " && ".join(source_parts)
        + " && "
        + f"ros2 param set {local_param} {local_r:.3f} && "
        + f"ros2 param set {global_param} {global_r:.3f}"
    )
    try:
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        print(f"  [WARN] Failed to set inflation radius via ros2 param: {exc}")
        return False

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        err_msg = err[-1] if err else "unknown error"
        print(
            "  [WARN] ros2 param set failed for "
            f"local={local_r:.3f}, global={global_r:.3f}: {err_msg}"
        )
        return False

    _LAST_INFLATION_CFG = cfg
    refreshed = _wait_costmap_refresh(prev_local, prev_global, timeout_s=5.0)
    if not refreshed:
        print("  [WARN] Inflation params set, but fresh costmap stamps were not observed within timeout.")
    else:
        print("  [PARAM] Costmaps refreshed after inflation update.")
    print(
        "  [PARAM] inflation_layer.inflation_radius set from caution "
        f"'{caution}': local={local_r:.3f}, global={global_r:.3f}"
    )
    return True

def _describe_target(task: TaskIntent) -> str:
    """Human-readable description of the target for logging."""
    parts = [task.main.target.name]
    for ref in task.main.references:
        parts.append(f"{ref.type} {ref.name}")
    return " ".join(parts)


def _run_single_task(
    task: TaskIntent,
    task_idx: int,
    total_tasks: int,
    attempt_idx: int,
    object_location: Tuple[float, float],
    constraint_xys: Tuple[Tuple[float, float], ...] = (),
    preference_xys: Tuple[Tuple[float, float], ...] = (),
    timeout_sec: float = 180.0,
    collision_cost_thresh: float = 90.0,
    collision_duration_sec: float = 5.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Execute one navigation task until the termination conditions are met.

    Failure detection
    -----------------
    - **Timeout**: abort after *timeout_sec* wall-clock seconds.
    - **Collision**: abort if the local-costmap cost at the robot's pose exceeds
      *collision_cost_thresh* for *collision_duration_sec* consecutive seconds.
    Returns ``(success, task_data)`` where task_data contains collected telemetry.
    """

    params = categories_to_params(task)
    _set_costmap_inflation_by_caution(task.behavior.caution)

    print(f"\n{'='*60}")
    print(f"Task {task_idx + 1}/{total_tasks}")
    print(f"Target: {task.main.target.name} ({task.main.target.type})")
    if task.main.references:
        print(f"References: {[asdict(r) for r in task.main.references]}")
    if task.constraints:
        print(f"Constraints (avoid): {[asdict(c) for c in task.constraints]}")
    if task.preferences:
        print(f"Preferences (stay close): {[asdict(p) for p in task.preferences]}")
    print(f"Termination: dist={task.main.termination.distance_m}m, "
          f"phase_deg={task.main.termination.phase}, "
          f"strictness={task.main.termination.stop_strictness}")
    print(f"Behavior: {asdict(task.behavior)}")
    print(f"Object Location: x={object_location[0]:.2f}, y={object_location[1]:.2f}")
    if constraint_xys:
        print(f"Constraint locations (avoid): {constraint_xys}")
    if preference_xys:
        print(f"Preference locations (stay close): {preference_xys}")
    print(f"Control Parameters:")
    pprint.pprint(asdict(params), indent=2)
    print(f"{'='*60}")

    # ---- initial state ----
    local_costmap = LocalCostmapIPCClient()
    initial_pose_bundle = get_latest_robot_pose(timeout_ms=500)
    initial_map_base = select_map_base_pose(initial_pose_bundle)
    if initial_pose_bundle is None or initial_map_base is None:
        local_costmap.close()
        raise RuntimeError("No initial robot pose received over IPC.")

    x0 = (
        initial_map_base["x"],
        initial_map_base["y"],
        initial_map_base["yaw"],
    )

    # ---- recovery / dwell parameters ----
    is_recovering = False
    recovery_w = 0.0
    last_recovery_time = 0.0
    RECOV_DWELL_SEC = float(get_param(("navigator", "recovery_dwell_sec"), 2.0))
    DONE_DWELL_SEC  = float(get_param(("navigator", "done_dwell_sec"), 0.0))
    SPIN_W          = float(get_param(("navigator", "recovery_spin_w"), 0.2))

    near_goal_since: Optional[float] = None
    done_stable_since: Optional[float] = None

    step = 0
    u_prev = (0.0, 0.0)
    active_goal_xy = object_location
    active_phi0 = params.phi0

    # ---- smooth motion state (Options B / C) ----
    _sampling_cfg = get_param(("control", "sampling"), {})
    _exec_steps = int(_sampling_cfg.get("exec_steps", 3))
    _cmd_smooth_alpha = float(_sampling_cfg.get("cmd_smooth_alpha", 0.4))
    _traj_buffer: List[Tuple[float, float]] = []
    _v_smooth, _w_smooth = 0.0, 0.0

    _last_planner_result: Optional[Dict[str, Any]] = None  # cached for buffer-drain ticks

    # ---- telemetry collection ----
    _trajectory: List[Tuple[float, float]] = []
    _steps: List[Dict[str, Any]] = []
    _failure_reason: Optional[str] = None

    # ---- failure detection state ----
    task_start_time = time.time()
    # Collision detection: timestamp when high-cost streak started
    collision_streak_start: Optional[float] = None
    last_pose_stamp: Optional[Tuple[int, int]] = None
    pose_stale_since: Optional[float] = None
    pose_stale_warned = False
    POSE_STALE_TIMEOUT_SEC = float(get_param(("navigator", "pose_stale_timeout_sec"), 1.0))
    # Value-based stale detection: catch cases where timestamp updates but
    # the actual pose values are frozen (e.g. odometry source stopped).
    last_pose_values: Optional[Tuple[float, float, float]] = None
    pose_values_frozen_since: Optional[float] = None
    POSE_VALUES_FROZEN_TIMEOUT_SEC = float(get_param(("navigator", "pose_values_frozen_timeout_sec"), 1.5))
    # Ignore tiny localization jitter; detect meaningfully frozen motion feedback.
    POSE_VALUES_EPS = 2e-3  # metres / radians
    pose_missing_since: Optional[float] = None
    pose_missing_warned = False
    POSE_MISSING_WARN_SEC = float(get_param(("navigator", "pose_missing_warn_sec"), 1.0))
    # Diagnostic: detect when selected control pose stays constant even though
    # raw TF bundle entries keep changing.
    prev_direct_pose_values: Optional[Tuple[float, float, float]] = None
    prev_composed_pose_values: Optional[Tuple[float, float, float]] = None
    selected_pose_stuck_since: Optional[float] = None
    selected_pose_stuck_warned = False
    POSE_PIPELINE_WARN_SEC = float(get_param(("navigator", "pose_pipeline_warn_sec"), 1.0))
    POSE_PIPELINE_EPS = 1e-4

    def _pose_changed(
        prev_pose: Optional[Tuple[float, float, float]],
        curr_pose: Optional[Tuple[float, float, float]],
    ) -> bool:
        if curr_pose is None:
            return False
        if prev_pose is None:
            return True
        dx = abs(curr_pose[0] - prev_pose[0])
        dy = abs(curr_pose[1] - prev_pose[1])
        dyaw = abs(wrap_angle(curr_pose[2] - prev_pose[2]))
        return dx > POSE_PIPELINE_EPS or dy > POSE_PIPELINE_EPS or dyaw > POSE_PIPELINE_EPS
    def _build_task_data(success: bool, failure_reason: Optional[str] = None) -> Dict[str, Any]:
        return {
            "success": success,
            "failure_reason": failure_reason,
            "trajectory": list(_trajectory),
            "steps": list(_steps),
            "duration_sec": time.time() - task_start_time,
        }

    def _idle_result(v_cmd: float, w_cmd: float) -> dict:
        return {
            "best_cost": 0.0,
            "best_action": (v_cmd, w_cmd),
            "best_trajectory": [],
            "active_goal_xy": active_goal_xy,
            "final_result_cost": {
                "sigma": 0,
                "clear": 0,
                "effort": 0,
                "smooth": 0,
                "curv": 0,
            },
        }

    # ---- main loop ----
    while True:
        terminal_success_now = False
        t0 = time.time()

        pose_bundle = get_latest_robot_pose(timeout_ms=250)
        if pose_bundle is None:
            now = time.time()
            if pose_missing_since is None:
                pose_missing_since = now
            elif (now - pose_missing_since) >= POSE_MISSING_WARN_SEC and not pose_missing_warned:
                print(
                    "[WARN] Main navigation loop is not receiving new pose bundles "
                    f"for {now - pose_missing_since:.2f}s; holding robot."
                )
                pose_missing_warned = True
            send_cmd_vel_via_ipc(0.0, 0.0)
            continue

        if pose_missing_since is not None and pose_missing_warned:
            print(
                "[INFO] Pose bundle stream recovered after "
                f"{time.time() - pose_missing_since:.2f}s."
            )
        pose_missing_since = None
        pose_missing_warned = False

        robot_pose_dict = select_map_base_pose(pose_bundle)
        if robot_pose_dict is None:
            now = time.time()
            if pose_missing_since is None:
                pose_missing_since = now
            elif (now - pose_missing_since) >= POSE_MISSING_WARN_SEC and not pose_missing_warned:
                available = [k for k, v in pose_bundle.items() if k != "frames" and v is not None]
                print(
                    "[WARN] Pose bundle arrived but no usable map-base pose "
                    f"for {now - pose_missing_since:.2f}s; available_tf_entries={available}"
                )
                pose_missing_warned = True
            send_cmd_vel_via_ipc(0.0, 0.0)
            continue
        pose_stamp = _meta_stamp(robot_pose_dict)
        now = time.time()
        if pose_stamp is not None and pose_stamp == last_pose_stamp:
            if pose_stale_since is None:
                pose_stale_since = now
            if (now - pose_stale_since) >= POSE_STALE_TIMEOUT_SEC:
                if not pose_stale_warned:
                    print(
                        "[WARN] Robot pose TF is stale; holding still until fresh pose arrives. "
                        f"stale_for={now - pose_stale_since:.2f}s"
                    )
                    pose_stale_warned = True
                send_cmd_vel_via_ipc(0.0, 0.0)
                continue
        else:
            last_pose_stamp = pose_stamp
            pose_stale_since = None
            pose_stale_warned = False

        robot_pose = (robot_pose_dict["x"], robot_pose_dict["y"], robot_pose_dict["yaw"])
        pose_source = str(robot_pose_dict.get("source", "unknown"))

        direct_tf = pose_bundle.get("map_base")
        direct_pose = None
        if direct_tf is not None:
            direct_pose = (
                float(direct_tf["x"]),
                float(direct_tf["y"]),
                float(direct_tf["yaw"]),
            )
        map_odom_tf = pose_bundle.get("map_odom")
        odom_base_tf = pose_bundle.get("odom_base")
        composed_pose = None
        if map_odom_tf is not None and odom_base_tf is not None:
            th = float(map_odom_tf["yaw"])
            c = math.cos(th)
            s = math.sin(th)
            ox = float(odom_base_tf["x"])
            oy = float(odom_base_tf["y"])
            composed_pose = (
                float(map_odom_tf["x"]) + c * ox - s * oy,
                float(map_odom_tf["y"]) + s * ox + c * oy,
                wrap_angle(float(map_odom_tf["yaw"]) + float(odom_base_tf["yaw"])),
            )

        direct_changed = _pose_changed(prev_direct_pose_values, direct_pose)
        composed_changed = _pose_changed(prev_composed_pose_values, composed_pose)
        selected_changed = _pose_changed(last_pose_values, robot_pose)

        prev_direct_pose_values = direct_pose
        prev_composed_pose_values = composed_pose

        if not selected_changed and (direct_changed or composed_changed):
            if selected_pose_stuck_since is None:
                selected_pose_stuck_since = time.time()
            elif (
                (time.time() - selected_pose_stuck_since) >= POSE_PIPELINE_WARN_SEC
                and not selected_pose_stuck_warned
            ):
                print(
                    "[WARN] Pose pipeline mismatch: selected control pose is not updating "
                    "while raw TF bundle entries are changing. "
                    f"selected_src={pose_source}, direct_changed={direct_changed}, "
                    f"composed_changed={composed_changed}"
                )
                selected_pose_stuck_warned = True
        else:
            selected_pose_stuck_since = None
            selected_pose_stuck_warned = False

        # Value-based stale detection: timestamps may update while values stay frozen.
        if last_pose_values is not None:
            dx = abs(robot_pose[0] - last_pose_values[0])
            dy = abs(robot_pose[1] - last_pose_values[1])
            dyaw = abs(wrap_angle(robot_pose[2] - last_pose_values[2]))
            values_changed = (dx > POSE_VALUES_EPS or dy > POSE_VALUES_EPS or dyaw > POSE_VALUES_EPS)
        else:
            values_changed = True

        if values_changed:
            last_pose_values = robot_pose
            pose_values_frozen_since = None
        else:
            if abs(u_prev[0]) > 1e-3 or abs(u_prev[1]) > 1e-3:
                if pose_values_frozen_since is None:
                    pose_values_frozen_since = time.time()
                elif (time.time() - pose_values_frozen_since) >= POSE_VALUES_FROZEN_TIMEOUT_SEC:
                    print(
                        f"[WARN] Pose VALUES frozen for {time.time() - pose_values_frozen_since:.1f}s "
                        f"while cmd_vel is non-zero (v={u_prev[0]:.3f}, w={u_prev[1]:.3f}). "
                        f"Stopping robot and re-reading pose. source={pose_source}"
                    )
                    send_cmd_vel_via_ipc(0.0, 0.0)
                    u_prev = (0.0, 0.0)
                    pose_values_frozen_since = None
                    time.sleep(0.5)
                    continue

        eval_params = replace(params, phi0=active_phi0)
        bearing_angle = math.atan2(object_location[1] - robot_pose[1], object_location[0] - robot_pose[0])
        alpha_error = wrap_angle(bearing_angle - robot_pose[2])

        _, r_current, _, e_phi = satisfaction_violation(
            x0=x0, x=robot_pose, obj_xy=object_location, params=eval_params
        )

        r_ok = eval_params.r_min <= r_current <= eval_params.r_max
        phase_required = bool(getattr(eval_params, "phase_required", True))
        phi_ok = (not phase_required) or (abs(e_phi) <= eval_params.phi_tol)
        facing_ok = abs(alpha_error) <= eval_params.alpha_max
        all_ok = r_ok and phi_ok and facing_ok

        # ---- near-goal tracking ----
        if r_ok and (phi_ok or not phase_required):
            if near_goal_since is None:
                near_goal_since = time.time()
        else:
            near_goal_since = None

        # ---- recovery trigger ----
        if (
            not is_recovering
            and near_goal_since is not None
            and (time.time() - near_goal_since) >= RECOV_DWELL_SEC
            and (not facing_ok)
            and (time.time() - last_recovery_time > 5.0)
        ):
            print("[RECOVERY] Near goal but facing not aligned. Spinning to align.")
            is_recovering = True
            last_recovery_time = time.time()

        if is_recovering:
            if abs(alpha_error) <= math.radians(float(get_param(("navigator", "recovery_facing_tol_deg"), 3.0))):
                print("[RECOVERY] Facing aligned. Exiting recovery.")
                is_recovering = False
                done_stable_since = None
            else:
                recovery_w = SPIN_W if alpha_error > 0.0 else -SPIN_W

        costmap, meta = local_costmap.get(timeout_ms=1000)

        # ---- failure detection ----
        now = time.time()

        # 1) Timeout
        elapsed = now - task_start_time
        if elapsed >= timeout_sec:
            print(f"[FAIL] Task {task_idx+1}/{total_tasks}: timed out after {elapsed:.1f}s.")
            send_cmd_vel_via_ipc(0.0, 0.0)
            local_costmap.close()
            return False, _build_task_data(False, f"timeout after {elapsed:.1f}s")

        # 2) Collision / high-cost detection
        if costmap is not None and meta is not None and pose_bundle is not None:
            cell_cost = get_cost_at_pose(
                costmap=costmap,
                pose_map=robot_pose,
                map_odom=pose_bundle["map_odom"],
                origin_x=meta["origin_x"],
                origin_y=meta["origin_y"],
                resolution=meta["resolution"],
            )
            if cell_cost >= collision_cost_thresh:
                if collision_streak_start is None:
                    collision_streak_start = now
                elif (now - collision_streak_start) >= collision_duration_sec:
                    print(
                        f"[FAIL] Task {task_idx+1}/{total_tasks}: collision — "
                        f"costmap cost ≥ {collision_cost_thresh} for "
                        f"{collision_duration_sec}s (current={cell_cost:.0f})."
                    )
                    send_cmd_vel_via_ipc(0.0, 0.0)
                    local_costmap.close()
                    return False, _build_task_data(False, f"collision: costmap cost {cell_cost:.0f} >= {collision_cost_thresh} for {collision_duration_sec}s")
            else:
                collision_streak_start = None

        # If the task already satisfies terminal conditions, stop and dwell in
        # place instead of forcing another replan near the goal.
        if all_ok:
            is_recovering = False
            v_cmd, w_cmd = 0.0, 0.0
            result = _idle_result(v_cmd, w_cmd)
            if DONE_DWELL_SEC <= 0.0:
                print(f"[DONE] Task {task_idx+1}/{total_tasks} complete. Stopping.")
                terminal_success_now = True
            if done_stable_since is None:
                done_stable_since = time.time()
            elif (time.time() - done_stable_since) >= DONE_DWELL_SEC:
                print(f"[DONE] Task {task_idx+1}/{total_tasks} complete. Stopping.")
                terminal_success_now = True
        else:
            done_stable_since = None

        # ---- action selection ----
        if not all_ok and is_recovering:
            _traj_buffer.clear()  # drop buffer so recovery is immediate
            v_cmd, w_cmd = 0.0, recovery_w
            result = _idle_result(v_cmd, w_cmd)
        elif not all_ok and _traj_buffer:
            # Option C: drain buffered trajectory steps before replanning
            v_cmd, w_cmd = _traj_buffer.pop(0)
            result = _idle_result(v_cmd, w_cmd)
        elif not all_ok:
            result = sample_and_select_action(
                x0=x0,
                x=robot_pose,
                obj_xy=object_location,
                params=params,
                costmap=costmap,
                u_prev=u_prev,
                rng_seed=42,
                pose_bundle=pose_bundle,
                meta=meta,
                constraint_xys=constraint_xys,
                preference_xys=preference_xys,
                phi0_override=active_phi0,
            )

            # ---- no feasible goal → abort task ----
            if result.get("no_feasible_goal", False):
                planner_failure_reason = result.get(
                    "planner_failure_reason",
                    "No feasible goal — all ring-sector candidates blocked.",
                )
                if r_ok and (phi_ok or not phase_required):
                    print(
                        f"[WARN] Planner lost feasible ring-sector goals near the target; "
                        f"continuing terminal alignment instead. ({planner_failure_reason})"
                    )
                    is_recovering = not facing_ok
                    recovery_w = SPIN_W if alpha_error > 0.0 else -SPIN_W
                    v_cmd = 0.0
                    w_cmd = recovery_w if is_recovering else 0.0
                    result = _idle_result(v_cmd, w_cmd)
                else:
                    print(
                        f"[FAIL] Task {task_idx+1}/{total_tasks}: "
                        f"{planner_failure_reason}"
                    )
                    send_cmd_vel_via_ipc(0.0, 0.0)
                    local_costmap.close()
                    return False, _build_task_data(False, planner_failure_reason)

            if not result.get("no_feasible_goal", False):
                new_goal_xy = result.get("active_goal_xy", active_goal_xy)
                if new_goal_xy != active_goal_xy:
                    active_goal_xy = new_goal_xy
                    # Compute the actual phase of the new goal around the original object,
                    # so phi_error is measured relative to this reachable position.
                    _phi_new = math.atan2(object_location[1] - active_goal_xy[1],
                                          object_location[0] - active_goal_xy[0])
                    _phi_ref = math.atan2(object_location[1] - x0[1],
                                          object_location[0] - x0[0])
                    active_phi0 = wrap_angle(_phi_ref - _phi_new)
                v_cmd, w_cmd = result["best_action"]
                # Fill buffer with the next exec_steps-1 steps so the robot
                # keeps moving without replanning for the next few ticks.
                best_traj = result.get("best_trajectory", [])
                _traj_buffer = list(best_traj[1:_exec_steps])

        # ---- RViz visualization (IPC → ROS node → /semnav/* topics) ----
        if result.get("planner") is not None:
            _last_planner_result = result
        viz_result = _last_planner_result if _last_planner_result is not None else result
        _guide = viz_result.get("guide")
        send_viz_data(
            robot_pose=robot_pose,
            start_pose=x0,
            target_xy=object_location,
            active_goal_xy=active_goal_xy,
            path=viz_result.get("path") or [],
            all_seqs=viz_result.get("all_seqs") or [],
            dt=params.dt,
            r_min=params.r_min,
            r_max=params.r_max,
            lookahead_xy=_guide.lookahead_xy if _guide is not None else None,
            mode="RECOV" if is_recovering else ("IDLE" if all_ok else "NAV"),
            task_name=task.main.target.name,
            constraint_xys=constraint_xys,
            preference_xys=preference_xys,
        )

        # ---- telemetry collection ----
        _trajectory.append((robot_pose[0], robot_pose[1]))
        _steps.append({
            "step": step,
            "timestamp": t0,
            "x": robot_pose[0],
            "y": robot_pose[1],
            "yaw_deg": math.degrees(robot_pose[2]),
            "v_cmd": v_cmd,
            "w_cmd": w_cmd,
            "distance_to_target": r_current,
            "heading_error_deg": math.degrees(alpha_error),
            "phase_error_deg": math.degrees(e_phi),
            "r_ok": r_ok,
            "phi_ok": phi_ok,
            "facing_ok": facing_ok,
            "is_recovering": is_recovering,
        })
        # ---- logging ----
        print(
            "[CTRL] "
            f"step={step:04d} "
            f"pose=({robot_pose[0]:.3f},{robot_pose[1]:.3f},{math.degrees(robot_pose[2]):.1f}deg) "
            f"cmd=(v={v_cmd:.3f},w={w_cmd:.3f}) "
            f"dist={r_current:.3f} "
            f"mode={'RECOV' if is_recovering else ('IDLE' if all_ok else 'NAV')} "
            f"pose_src={pose_source}"
        )

        # Option A: output low-pass filter — smooths jerk at the motor level.
        # Bypass during recovery so spin commands are crisp.
        if not is_recovering:
            _v_smooth = (1.0 - _cmd_smooth_alpha) * _v_smooth + _cmd_smooth_alpha * v_cmd
            _w_smooth = (1.0 - _cmd_smooth_alpha) * _w_smooth + _cmd_smooth_alpha * w_cmd
        else:
            _v_smooth, _w_smooth = v_cmd, w_cmd  # reset to avoid filter windup
        v_send, w_send = _v_smooth, _w_smooth

        cmd_ok = send_cmd_vel_via_ipc(v_send, w_send)
        if not cmd_ok and (step % 5 == 0):
            print(
                "[WARN] cmd_vel IPC send did not receive OK; "
                f"latest_cmd=(v={v_send:.3f}, w={w_send:.3f})"
            )
        u_prev = (v_send, w_send)  # track what was actually sent
        step += 1

        if terminal_success_now:
            local_costmap.close()
            return True, _build_task_data(True)

        print(f"Timing: Total Loop={(time.time() - t0) * 1000.0:.3f}ms")

def _save_run(
    output_dir: Path,
    instruction: str,
    parsed: Any,
    task_results: List[Dict[str, Any]],
) -> None:
    """Write result.json, summary.txt, per-task CSVs and map PNGs to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_success = all(t["success"] for t in task_results)
    total_duration = sum(t["duration_sec"] for t in task_results)

    exp = {
        "instruction": instruction,
        "llm_output": asdict(parsed) if hasattr(parsed, "__dataclass_fields__") else parsed,
    }

    result = {
        "instruction": instruction,
        "overall_success": overall_success,
        "total_duration_sec": round(total_duration, 3),
        "timestamp": output_dir.name,
        "task_results": task_results,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Instruction : {instruction}\n")
        f.write(f"Success     : {overall_success}\n")
        f.write(f"Duration    : {total_duration:.1f}s\n")
        f.write(f"Timestamp   : {output_dir.name}\n\n")
        for t in task_results:
            idx = t.get("task_idx", 0)
            f.write(f"--- Task {idx} ---\n")
            f.write(f"  Target  : {t.get('target_name', '?')} at {t.get('target_location')}\n")
            f.write(f"  Success : {t['success']}\n")
            if t.get("failure_reason"):
                f.write(f"  Failure : {t['failure_reason']}\n")
            f.write(f"  Steps   : {len(t.get('steps', []))}\n")
            f.write(f"  Duration: {t['duration_sec']:.1f}s\n")
            steps = t.get("steps") or []
            if steps:
                last = steps[-1]
                def _fmt(val, spec):
                    try:
                        return format(float(val), spec)
                    except (TypeError, ValueError):
                        return "N/A"
                f.write(f"  Final dist: {_fmt(last.get('distance_to_target'), '.3f')}m\n")
                f.write(f"  Final heading err: {_fmt(last.get('heading_error_deg'), '.1f')}°\n")
                f.write(f"  Final phase err: {_fmt(last.get('phase_error_deg'), '.1f')}°\n")

    map_pgm_str = get_runtime_value(("paths", "viz_map_pgm"), "")
    map_pgm = Path(map_pgm_str) if map_pgm_str else None

    for t in task_results:
        idx = t.get("task_idx", 0)

        steps = t.get("steps") or []
        if steps:
            fieldnames = list(steps[0].keys())
            with (output_dir / f"telemetry_task{idx}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(steps)

        traj = t.get("trajectory") or []
        if traj:
            with (output_dir / f"trajectory_task{idx}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                writer.writerows(traj)

        task_result_for_viz = {**t, "task_idx": idx}
        try:
            plot_task_path(
                exp=exp,
                task_result=task_result_for_viz,
                out_path=output_dir / f"path_map_task{idx}.png",
                map_pgm=map_pgm,
            )
        except Exception as e:
            print(f"[WARN] Could not save path_map_task{idx}.png: {e}")

    print(f"[INFO] Run results saved to: {output_dir}")


def run_navigation(
    instruction: str,
    docs_path: Optional[str],
    timeout_sec: float = 180.0,
    collision_cost_thresh: float = 90.0,
    collision_duration_sec: float = 5.0,
) -> None:
    """Parse the instruction into tasks, then execute each sequentially."""

    validate_runtime_prereqs()
    print(
        "[INFO] Directional reference min distance: "
        f"{get_direction_min_distance_m():.3f} m"
    )

    # ---- resolve docs for object retrieval ----
    docs = resolve_docs_path(docs_path)
    if docs is None:
        print("No semantic docs.jsonl found. Target retrieval will fail.")
    else:
        print(f"Using semantic docs: {docs}")
    # ---- parse instruction (multi-task) ----
    parsed = load_or_parse_instruction(instruction)

    print(f"\nInstruction: {parsed.instruction}")
    print(f"Confidence : {parsed.confidence:.2f}")
    print(f"Tasks      : {len(parsed.tasks)}")

    if not parsed.tasks:
        print("[WARN] No tasks parsed from instruction. Nothing to do.")
        return


    # ---- set up output directory ----
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_outputs_root() / "runs" / run_ts
    output_dir.mkdir(parents=True, exist_ok=True)

    nav_through_pose_mode = bool(get_runtime_value(("feature_flags", "nav_through_pose"), False))
    if nav_through_pose_mode:
        from navigation.nav_through_pose import run_nav_through_pose_task
        print("[INFO] Navigation mode: nav_through_pose (trajectory sampling → Nav2 endpoint)")
    else:
        print("[INFO] Navigation mode: direct cmd_vel (trajectory sampling)")

    # ---- execute tasks sequentially ----
    collected_task_results: List[Dict[str, Any]] = []
    for idx, task in enumerate(parsed.tasks):
        desc = _describe_target(task)
        if docs is not None:
            print(f"\nRetrieving object: '{desc}'")
            try:
                object_location = retrieve_object_location(
                    target=task.main.target,
                    references=task.main.references,
                    docs_path=docs,
                )
                constraint_xys = tuple(
                    retrieve_object_location(c.target, c.references, docs)
                    for c in task.constraints
                )
                preference_xys = tuple(
                    retrieve_object_location(p.target, p.references, docs)
                    for p in task.preferences
                )
            except Exception as exc:
                print(f"[FAIL] Task {idx+1}/{len(parsed.tasks)} target retrieval failed: {exc}")
                return
        else:
            print(f"[FAIL] Task {idx+1}/{len(parsed.tasks)} target retrieval failed: no semantic docs available for '{desc}'.")
            return

        for c, loc in zip(task.constraints, constraint_xys):
            print(f"  Constraint (avoid): {c.target.name} at ({loc[0]:.2f}, {loc[1]:.2f})")
        for p, loc in zip(task.preferences, preference_xys):
            print(f"  Preference (stay close): {p.target.name} at ({loc[0]:.2f}, {loc[1]:.2f})")

        _reset_action_selector_state()
        if nav_through_pose_mode:
            success, task_data = run_nav_through_pose_task(
                task=task,
                task_idx=idx,
                total_tasks=len(parsed.tasks),
                object_location=tuple(object_location),
                constraint_xys=tuple(constraint_xys),
                preference_xys=tuple(preference_xys),
                timeout_sec=timeout_sec,
                collision_cost_thresh=collision_cost_thresh,
                collision_duration_sec=collision_duration_sec,
            )
        else:
            success, task_data = _run_single_task(
                task=task,
                task_idx=idx,
                total_tasks=len(parsed.tasks),
                attempt_idx=1,
                object_location=tuple(object_location),
                constraint_xys=tuple(constraint_xys),
                preference_xys=tuple(preference_xys),
                timeout_sec=timeout_sec,
                collision_cost_thresh=collision_cost_thresh,
                collision_duration_sec=collision_duration_sec,
            )
        task_data["task_idx"] = idx
        task_data["target_name"] = desc
        task_data["target_location"] = list(object_location)
        task_data["constraint_locations"] = [list(loc) for loc in constraint_xys]
        task_data["preference_locations"] = [list(loc) for loc in preference_xys]
        task_data["control_params"] = asdict(categories_to_params(task))
        collected_task_results.append(task_data)
        if not success:
            break

    print(f"\n{'='*60}")
    print("[DONE] All {0} task(s) completed.".format(len(parsed.tasks)))
    print(f"{'='*60}")

    _save_run(output_dir, instruction, parsed, collected_task_results)
