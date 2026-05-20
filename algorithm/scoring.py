"""Trajectory scoring, control-parameter mapping, and action selection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from algorithm.trajectory_sampling import trajectories_sampling_from_path
from ipc.occupancy_grid import get_cost_at_pose
from planning.path_planning import GlobalPlanner
from planning.intent_costs import constraint_penalty_from_distance, preference_penalty_from_distance
from utils.math_helpers import clamp, cm_to_m, wrap_angle
from utils.config import get_param
from parsing.intent_parser import TaskIntent


# ---------------------------------------------------------------------------
# Control parameters
# ---------------------------------------------------------------------------

@dataclass
class ControlParams:
    r_min: float
    r_max: float
    alpha_max: float
    phi0: float
    phi_tol: float
    phase_required: bool

    h_r: float
    h_alpha: float
    t_dwell: float

    v_max: float
    w_max: float
    a_v_max: float
    a_w_max: float

    w_u: float
    w_du: float
    w_curv: float
    w_path: float

    d_safe: float
    w_clear: float
    w_clear_v: float
    w_clear_w: float

    r_entry: float
    v_entry_max: float
    w_face_far: float
    w_face_near: float

    H: int
    dt: float
    N: int


def categories_to_params(task: TaskIntent) -> ControlParams:
    """Map a :class:`TaskIntent` to numeric :class:`ControlParams`.

    Fields are pulled from:
    - ``task.main.termination`` → distance, phase, stop_strictness, stop_policy
    - ``task.behavior``         → speed, caution
    """
    termination = task.main.termination
    behavior = task.behavior

    control_cfg = get_param(("control",), {})
    termination_cfg = control_cfg.get("termination", {})
    behavior_cfg = control_cfg.get("behavior_mapping", {})
    sampling_cfg = control_cfg.get("sampling", {})
    weights_cfg = control_cfg.get("weights", {})
    dynamics_cfg = control_cfg.get("dynamics", {})

    default_r_band_map = termination_cfg.get(
        "distance_band_m",
        {"loose": [0.50, 1.00], "normal": [0.40, 0.80], "strict": [0.35, 0.60]},
    )
    default_alpha_map = termination_cfg.get(
        "alpha_max_deg",
        {"loose": 15.0, "normal": 10.0, "strict": 5.0},
    )
    default_phi_tol_map = termination_cfg.get(
        "phi_tol_deg",
        {"loose": 15.0, "normal": 10.0, "strict": 5.0},
    )
    band_half_width_map = termination_cfg.get(
        "distance_band_half_width_m",
        {"loose": 0.10, "normal": 0.05, "strict": 0.03},
    )

    default_r_band = tuple(default_r_band_map[termination.stop_strictness])
    default_alpha_max = math.radians(float(default_alpha_map[termination.stop_strictness]))
    default_phi_tol = math.radians(float(default_phi_tol_map[termination.stop_strictness]))

    # ---- distance band (distance_m is already in metres) ----
    if termination.distance_m is not None and termination.distance_m > 0:
        r0 = clamp(termination.distance_m, 0.10, 3.00)
        dr = float(band_half_width_map[termination.stop_strictness])
        r_min = max(0.10, r0 - dr)
        r_max = r0 + dr
    else:
        r_min, r_max = default_r_band

    # ---- phase (single numeric angle in degrees; None when unspecified) ----
    phase_deg = getattr(termination, "phase", None)
    phase_required = bool(getattr(termination, "phase_explicit", False) and phase_deg is not None)
    if phase_deg is not None:
        phi0 = wrap_angle(math.radians(float(phase_deg)))
    else:
        phi0 = 0.0
    # If phase wasn't explicitly requested, keep 0 deg as a preference (phi0) but
    # disable hard phase rejection for goal satisfaction/planning feasibility.
    phi_tol = default_phi_tol if phase_required else math.pi

    # When phase is unspecified, relax alpha.
    alpha_max = math.radians(10.0) if phase_deg is None else default_alpha_max

    h_r = float(sampling_cfg.get("h_r", 0.05))
    h_alpha = math.radians(float(sampling_cfg.get("h_alpha_deg", 3.0)))
    t_dwell = float(
        sampling_cfg.get("dwell_sec_default", 0.5)
        if termination.stop_policy == "default"
        else sampling_cfg.get("dwell_sec_no_stop", 0.0)
    )

    speed_cfg = behavior_cfg.get("speed", {})
    speed_entry = speed_cfg[behavior.speed]
    v_max = float(speed_entry["v_max"])
    w_max = float(speed_entry["w_max"])

    a_v_max = float(dynamics_cfg.get("a_v_max", 0.5))
    a_w_max = float(dynamics_cfg.get("a_w_max", 0.5))

    w_u = float(
        behavior_cfg.get("w_u_fast", 0.5)
        if behavior.speed == "fast"
        else behavior_cfg.get("w_u_default", 1.0)
    )
    w_du = float(weights_cfg.get("w_du", 0.5))
    w_curv = float(weights_cfg.get("w_curv", 0.6))
    w_path = float(weights_cfg.get("w_path", 2.0))

    caution_cfg = behavior_cfg.get("caution", {})
    caution_entry = caution_cfg[behavior.caution]
    d_safe = float(caution_entry["d_safe"])
    w_clear = float(caution_entry["w_clear"])
    w_clear_v = float(caution_entry["w_clear_v"])
    w_clear_w = float(caution_entry["w_clear_w"])

    r_entry = r_max + float(behavior_cfg.get("r_entry_offset_m", 0.25))
    v_entry_cap = (
        float(behavior_cfg.get("v_entry_max_slow_mps", 0.25))
        if behavior.speed == "slow"
        else float(behavior_cfg.get("v_entry_max_default_mps", 0.35))
    )
    v_entry_max = min(v_max, v_entry_cap)

    # Face the target when a directional phase is specified
    if phase_deg is not None:
        w_face_far = float(behavior_cfg.get("w_face_far", 0.2))
        w_face_near = float(behavior_cfg.get("w_face_near", 1.0))
    else:
        w_face_far, w_face_near = 0.0, 0.0

    return ControlParams(
        r_min=r_min, r_max=r_max, alpha_max=alpha_max, phi0=phi0, phi_tol=phi_tol,
        phase_required=phase_required,
        h_r=h_r, h_alpha=h_alpha, t_dwell=t_dwell,
        v_max=v_max, w_max=w_max, a_v_max=a_v_max, a_w_max=a_w_max,
        w_u=w_u, w_du=w_du, w_curv=w_curv, w_path=w_path,
        d_safe=d_safe, w_clear=w_clear, w_clear_v=w_clear_v, w_clear_w=w_clear_w,
        r_entry=r_entry, v_entry_max=v_entry_max, w_face_far=w_face_far, w_face_near=w_face_near,
        H=int(sampling_cfg.get("horizon_steps", 30)),
        dt=float(sampling_cfg.get("dt_sec", 0.1)),
        N=int(sampling_cfg.get("num_samples", 100)),
    )


# ---------------------------------------------------------------------------
# Diff-drive dynamics
# ---------------------------------------------------------------------------

def step_diff_drive(
    x: Tuple[float, float, float],
    u: Tuple[float, float],
    dt: float,
) -> Tuple[float, float, float]:
    px, py, yaw = x
    v, w = u
    return (px + dt * v * math.cos(yaw), py + dt * v * math.sin(yaw), wrap_angle(yaw + dt * w))


# ---------------------------------------------------------------------------
# Satisfaction / violation metric
# ---------------------------------------------------------------------------

def satisfaction_violation(
    x0: Tuple[float, float, float],
    x: Tuple[float, float, float],
    obj_xy: Tuple[float, float],
    params: ControlParams,
) -> Tuple[float, float, float, float]:
    px, py, yaw = x
    ox, oy = obj_xy

    r = math.hypot(ox - px, oy - py)
    bearing = math.atan2(oy - py, ox - px)
    alpha = wrap_angle(bearing - yaw)

    phi = math.atan2(oy - py, ox - px)
    phi_ref = math.atan2(oy - x0[1], ox - x0[0])
    phi_rel = wrap_angle(phi_ref - phi)
    e_phi = wrap_angle(phi_rel - params.phi0)

    v_rad = 0.0
    if r < params.r_min:
        v_rad += params.r_min - r
    if r > params.r_max:
        v_rad += r - params.r_max

    v_phase = max(0.0, abs(e_phi) - params.phi_tol) if abs(e_phi) > params.phi_tol else 0.0
    v_face = max(0.0, abs(alpha) - params.alpha_max) if abs(alpha) > params.alpha_max else 0.0

    L = 0.5 * (params.r_min + params.r_max)
    sigma = v_rad + (L * v_face) + (L * v_phase)
    return sigma, r, alpha, e_phi


# ---------------------------------------------------------------------------
# Trajectory scorer
# ---------------------------------------------------------------------------

def score_trajectory(
    x0: Tuple[float, float, float],
    x: Tuple[float, float, float],
    u_seq: List[Tuple[float, float]],
    obj_xy: Tuple[float, float],
    params: ControlParams,
    u_prev: Tuple[float, float],
    guide: Optional[PathGuidance] = None,
    costmap: Optional[np.ndarray] = None,
    pose_bundle: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    constraint_xys: Tuple[Tuple[float, float], ...] = (),
    preference_xys: Tuple[Tuple[float, float], ...] = (),
) -> Tuple[float, Dict[str, Any]]:
    # Read scoring weights from config (fast dict lookup, not file I/O)
    _w = get_param(("control", "weights"), {})
    gamma          = float(_w.get("gamma",           0.95))
    w_sigma_run    = float(_w.get("w_sigma_run",     0.15))
    w_sigma_init   = float(_w.get("w_sigma_init",    0.75))
    w_sigma_term   = float(_w.get("w_sigma_term",    2.25))
    w_progress     = float(_w.get("w_progress",      1.25))
    w_eff_ang      = float(_w.get("w_effort_angular", 0.3))
    w_smo_ang      = float(_w.get("w_smooth_angular", 0.2))

    v_prev, w_prev = u_prev
    sigma0, _, _, _ = satisfaction_violation(x0, x, obj_xy, params)

    total = 0.0
    g = 1.0
    clear_sum = effort_sum = smooth_sum = curv_sum = 0.0
    constr_sum = pref_sum = 0.0
    path_sum = 0.0

    for v, w in u_seq:
        v = clamp(v, -0.5 * params.v_max, params.v_max)
        w = clamp(w, -params.w_max, params.w_max)

        x = step_diff_drive(x, (v, w), params.dt)
        px, py, yaw = x

        sigma, r, alpha, ephi = satisfaction_violation(x0=x0, x=x, obj_xy=obj_xy, params=params)

        if costmap is not None and pose_bundle is not None and meta is not None:
            cost = get_cost_at_pose(
                costmap=costmap,
                pose_map=(px, py, yaw),
                map_odom=pose_bundle["map_odom"],
                origin_x=meta["origin_x"],
                origin_y=meta["origin_y"],
                resolution=meta["resolution"],
            )
        else:
            cost = 0.0
        clear = cost / 100.0
        effort = params.w_u * (v * v + w_eff_ang * w * w)
        dv = (v - v_prev) / max(1e-6, params.dt)
        dw = (w - w_prev) / max(1e-6, params.dt)
        smooth = params.w_du * (dv * dv + w_smo_ang * dw * dw)
        curv = params.w_curv * (w * w)

        # -- path following cost --
        path_cost = 0.0
        if guide and guide.path_xy:
            min_d2 = float("inf")
            for i in range(len(guide.path_xy) - 1):
                p1 = guide.path_xy[i]
                p2 = guide.path_xy[i+1]
                _, _, _, d2 = _project_point_to_segment(px, py, p1[0], p1[1], p2[0], p2[1])
                if d2 < min_d2:
                    min_d2 = d2
            path_cost = params.w_path * min_d2

        # -- constraint penalty (repulsive) --
        constr_pen = 0.0
        for cx, cy in constraint_xys:
            d = math.hypot(px - cx, py - cy)
            constr_pen += constraint_penalty_from_distance(d)

        # -- preference penalty (attractive) --
        pref_pen = 0.0
        for pfx, pfy in preference_xys:
            d = math.hypot(px - pfx, py - pfy)
            pref_pen += preference_penalty_from_distance(d)

        near = (params.r_min - 0.05) <= r <= (params.r_max + 0.05)
        facing_ok = abs(alpha) <= params.alpha_max
        phase_big = abs(ephi) > (params.phi_tol + math.radians(5.0))
        relax = 0.25 if (near and facing_ok and phase_big) else 1.0

        effort *= relax
        smooth *= relax
        curv *= relax

        run = (w_sigma_run * (sigma * sigma)
               + params.w_clear * clear
               + effort + smooth + curv
               + constr_pen + pref_pen
               + path_cost)
        total += g * run

        clear_sum += g * (params.w_clear * clear)
        effort_sum += g * effort
        smooth_sum += g * smooth
        curv_sum += g * curv
        constr_sum += g * constr_pen
        pref_sum += g * pref_pen
        path_sum += g * path_cost

        v_prev, w_prev = v, w
        g *= gamma

    sigmaT, rT, alphaT, ephiT = satisfaction_violation(x0, x, obj_xy, params)
    progress = sigma0 - sigmaT
    total += (w_sigma_init * (sigma0 * sigma0)) + (w_sigma_term * (sigmaT * sigmaT)) - (w_progress * progress)

    info = {
        "cost": float(total),
        "sigma_start": float(sigma0),
        "sigma_end": float(sigmaT),
        "progress": float(progress),
        "sigma": float(total),
        "clear": float(clear_sum),
        "effort": float(effort_sum),
        "smooth": float(smooth_sum),
        "curv": float(curv_sum),
        "constr": float(constr_sum),
        "pref": float(pref_sum),
        "path": float(path_sum),
        "relax_used_end": float(
            0.25
            if (
                (params.r_min - 0.05) <= rT <= (params.r_max + 0.05)
                and abs(alphaT) <= params.alpha_max
                and abs(ephiT) > (params.phi_tol + math.radians(5.0))
            )
            else 1.0
        ),
    }
    return total, info


# ---------------------------------------------------------------------------
# Action selector (planning + sampling + scoring)
# ---------------------------------------------------------------------------

def sample_and_select_action(
    x0: Tuple[float, float, float],
    x: Tuple[float, float, float],
    obj_xy: Tuple[float, float],
    params: ControlParams,
    u_prev: Tuple[float, float] = (0.0, 0.0),
    rng_seed: Optional[int] = None,
    costmap: Optional[np.ndarray] = None,
    pose_bundle: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    constraint_xys: Tuple[Tuple[float, float], ...] = (),
    preference_xys: Tuple[Tuple[float, float], ...] = (),
    phi0_override: Optional[float] = None,
) -> Dict[str, Any]:
    if rng_seed is not None:
        random.seed(rng_seed)

    best_cost = float("inf")
    best_seq: List[Tuple[float, float]] = [(0.0, 0.0)] * params.H
    final_result: Dict[str, Any] = {}

    if not hasattr(sample_and_select_action, "_planner"):
        sample_and_select_action._planner = GlobalPlanner()
        sample_and_select_action._last_s0_index = 0

    planner = sample_and_select_action._planner

    path = planner.plan_path_ring_sector(
        x0, x, obj_xy, params.r_min, params.r_max, params.phi0, params.phi_tol,
        constraint_xys=constraint_xys, preference_xys=preference_xys,
    )

    # No feasible goal found — signal failure to the navigator.
    if path is None:
        return {
            "best_cost": float("inf"),
            "best_action": (0.0, 0.0),
            "best_trajectory": [(0.0, 0.0)] * params.H,
            "final_result_cost": {"sigma": 0, "clear": 0, "effort": 0, "smooth": 0, "curv": 0},
            "all_seqs": [],
            "path": [],
            "active_goal_xy": obj_xy,
            "planner": planner,
            "guide": None,
            "no_feasible_goal": True,
            "planner_failure_reason": (
                planner.last_failure_reason
                or "No feasible goal — all ring-sector candidates blocked."
            ),
        }

    # Use planner-selected goal (path endpoint) for local trajectory cost/recovery
    # so execution objective matches the actual chosen global goal.
    active_goal_xy = path[-1] if path else obj_xy
    eval_params = replace(params, phi0=phi0_override) if phi0_override is not None else params

    sampling_cfg = get_param(("control", "sampling"), {})
    v_min = float(sampling_cfg.get("v_min", 0.15))
    w_min = float(sampling_cfg.get("w_min", 0.15))
    p_stop = float(sampling_cfg.get("p_stop", 0.10))
    warm_start_alpha = float(sampling_cfg.get("warm_start_alpha", 0.0))

    lookahead_dist = float(sampling_cfg.get("lookahead_dist_m", 0.1))
    k_yaw = float(sampling_cfg.get("k_yaw", 3.0))
    noise_v = float(sampling_cfg.get("noise_v", 0.05))
    noise_w = float(sampling_cfg.get("noise_w", 0.15))
    yaw_rotate_threshold = math.radians(float(sampling_cfg.get("yaw_rotate_threshold_deg", 40.0)))

    seqs, guide = trajectories_sampling_from_path(
        u0=u_prev,
        x=x,
        path_xy=path if path else [],
        last_s0_index=sample_and_select_action._last_s0_index,
        lookahead_dist=lookahead_dist,
        N=params.N,
        H=params.H,
        dt=params.dt,
        v_max=params.v_max,
        w_max=params.w_max,
        v_min=v_min,
        w_min=w_min,
        p_stop=p_stop,
        a_v_max=params.a_v_max,
        a_w_max=params.a_w_max,
        warm_start_alpha=warm_start_alpha,
        k_yaw=k_yaw,
        noise_v=noise_v,
        noise_w=noise_w,
        yaw_rotate_threshold=yaw_rotate_threshold,
    )
    sample_and_select_action._last_s0_index = guide.s0_index

    for u_seq in seqs:
        cost, result = score_trajectory(
            # Score against the planner-selected reachable goal, not the raw object center.
            x0=x0, x=x, u_seq=u_seq, obj_xy=active_goal_xy, params=eval_params, u_prev=u_prev,
            costmap=costmap, pose_bundle=pose_bundle, meta=meta,
            constraint_xys=constraint_xys, preference_xys=preference_xys,
        )
        if cost < best_cost:
            best_cost = cost
            best_seq = u_seq
            final_result = result

    return {
        "best_cost": best_cost,
        "best_action": best_seq[0],
        "best_trajectory": best_seq,
        "final_result_cost": final_result,
        "all_seqs": seqs,
        "path": path,
        "active_goal_xy": active_goal_xy,
        "planner": planner,
        "guide": guide,
    }
