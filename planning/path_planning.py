"""Global path planner — multi-goal A* to a ring-sector goal set."""

from __future__ import annotations

import heapq
import math
import time
from typing import List, Optional, Tuple

import numpy as np

from ipc.costmap_client import GlobalCostmapIPCClient
from planning.intent_costs import (
    build_intent_penalty_grid,
    grid_to_world,
    world_to_grid,
)
from planning.types import PlannerGoalSpec, PlannerOutput
from utils.config import get_param
from utils.math_helpers import wrap_angle


def _planner_param(name: str, default: float | int) -> float | int:
    return get_param(("planner", name), default)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class GlobalPlanner:
    def __init__(self) -> None:
        self.costmap_client = GlobalCostmapIPCClient()
        self.grid: Optional[np.ndarray] = None
        self.meta: Optional[dict] = None
        self.last_goal_spec: Optional[PlannerGoalSpec] = None
        self.last_output = PlannerOutput(path_xy=[], stamp=0, ok=False)
        self.last_plan_time = 0.0
        self.sticky_goal: Optional[Tuple[int, int]] = None
        self.last_failure_reason = ""

    # ---- map refresh ----

    def update_map(self) -> bool:
        self.grid, self.meta = self.costmap_client.get(timeout_ms=2000)
        return self.grid is not None

    # ---- high-level step (timer + goal-change driven replanning) ----

    def PLANNER_STEP(
        self,
        goal_spec: PlannerGoalSpec,
        current_pose: Tuple[float, float, float],
    ) -> PlannerOutput:
        now = time.time()
        replan = False

        if (now - self.last_plan_time) > 1.0:
            replan = True
        if not self.last_output.ok:
            replan = True
        if self.last_goal_spec is None:
            replan = True
        elif goal_spec.obj_xy != self.last_goal_spec.obj_xy:
            replan = True
        elif goal_spec != self.last_goal_spec:
            replan = True

        if not replan:
            return self.last_output

        path = self.plan_path_ring_sector(
            x0=goal_spec.x_episode0,
            p=current_pose,
            o=goal_spec.obj_xy,
            r_min=goal_spec.r_min,
            r_max=goal_spec.r_max,
            phi0=goal_spec.phi0,
            phi_tol=goal_spec.phi_tol,
            constraint_xys=goal_spec.constraint_xys,
            preference_xys=goal_spec.preference_xys,
        )

        self.last_plan_time = now
        if self.last_goal_spec is not None and goal_spec != self.last_goal_spec:
            # New navigation target/constraints: unlock previous goal.
            self.sticky_goal = None
        self.last_goal_spec = goal_spec

        if path is not None:
            self.last_output = PlannerOutput(path_xy=path, stamp=now, ok=True)
        else:
            self.last_output = PlannerOutput(path_xy=[], stamp=now, ok=False)

        return self.last_output

    # ---- core A* to ring-sector goal set ----

    def plan_path_ring_sector(
        self,
        x0: Tuple[float, float, float],
        p: Tuple[float, float, float],
        o: Tuple[float, float],
        r_min: float,
        r_max: float,
        phi0: float,
        phi_tol: float,
        constraint_xys: Tuple[Tuple[float, float], ...] = (),
        preference_xys: Tuple[Tuple[float, float], ...] = (),
    ) -> Optional[List[Tuple[float, float]]]:
        self.last_failure_reason = ""
        self.update_map()

        if self.grid is None:
            print("No costmap available")
            self.last_failure_reason = "No global costmap available."
            return None

        width = self.meta["width"]
        height = self.meta["height"]
        resolution = self.meta["resolution"]
        origin_x = self.meta["origin_x"]
        origin_y = self.meta["origin_y"]

        ox, oy = o

        # Precompute constraint / preference penalty field
        penalty = build_intent_penalty_grid(
            width, height, origin_x, origin_y, resolution,
            constraint_xys, preference_xys,
        )

        phi_ref = math.atan2(oy - x0[1], ox - x0[0])
        start_idx = world_to_grid(p[0], p[1], origin_x, origin_y, resolution)
        phase_optional = phi_tol >= (math.pi - 1e-3)

        # --- helper: scan traversable goals inside a given distance band ---
        def _scan_band_goals(
            r_lo: float, r_hi: float,
        ) -> List[Tuple[Tuple[int, int], float, float, float]]:
            """Return [(grid_idx, |e_phi|, dist_to_robot, cell_cost), ...] for every
            traversable cell inside [r_lo, r_hi] around the object."""
            goals: List[Tuple[Tuple[int, int], float, float, float]] = []
            r_scan = r_hi + 2 * resolution
            k = int(math.ceil(r_scan / resolution))
            for ix in range(ox_idx - k, ox_idx + k + 1):
                for iy in range(oy_idx - k, oy_idx + k + 1):
                    if ix < 0 or ix >= width or iy < 0 or iy >= height:
                        continue
                    if self.grid[iy, ix] > int(_planner_param("hard_block_cost", 90)):
                        continue
                    xw, yw = grid_to_world(ix, iy, origin_x, origin_y, resolution)
                    r = math.hypot(xw - ox, yw - oy)
                    if r < r_lo or r > r_hi:
                        continue
                    phi = math.atan2(oy - yw, ox - xw)
                    phi_rel = wrap_angle(phi_ref - phi)
                    e_phi = wrap_angle(phi_rel - phi0)
                    d_robot = math.hypot(ix - start_idx[0], iy - start_idx[1]) * resolution
                    goals.append(((ix, iy), abs(e_phi), d_robot, float(self.grid[iy, ix])))
            return goals

        def _scan_nearest_free_goals(limit: int) -> List[Tuple[int, int]]:
            """Return nearest traversable cells to object, tie-broken by phase and
            then distance to the robot."""
            ranked: List[Tuple[Tuple[int, int], float, float, float, float]] = []
            for iy in range(height):
                for ix in range(width):
                    if self.grid[iy, ix] > int(_planner_param("hard_block_cost", 90)):
                        continue
                    xw, yw = grid_to_world(ix, iy, origin_x, origin_y, resolution)
                    r_obj = math.hypot(xw - ox, yw - oy)
                    phi = math.atan2(oy - yw, ox - xw)
                    phi_rel = wrap_angle(phi_ref - phi)
                    e_phi = abs(wrap_angle(phi_rel - phi0))
                    d_robot = math.hypot(ix - start_idx[0], iy - start_idx[1]) * resolution
                    ranked.append(((ix, iy), r_obj, float(self.grid[iy, ix]), e_phi, d_robot))
            ranked.sort(key=lambda item: (item[1], item[2], item[3], item[4]))
            return [g for g, *_ in ranked[: max(1, limit)]]

        def _rank_optional_phase_goals(
            goals: List[Tuple[Tuple[int, int], float, float, float]]
        ) -> List[Tuple[int, int]]:
            """For non-explicit phase instructions, prioritise low-cost / high-clearance
            regions first, then robot distance."""
            if not goals:
                return []
            max_dist = max(g[2] for g in goals) or 1e-6

            def _score(item):
                _, _, d_robot, cell_cost = item
                cost_norm = cell_cost / 100.0
                dist_norm = d_robot / max_dist
                return (
                    float(_planner_param("goal_w_cost_opt", 0.85)) * cost_norm
                    + float(_planner_param("goal_w_dist_opt", 0.15)) * dist_norm
                )

            ranked = sorted(goals, key=_score)
            return [g for g, *_ in ranked[: int(_planner_param("max_goal_retries", 80))]]

        ox_idx, oy_idx = world_to_grid(ox, oy, origin_x, origin_y, resolution)

        # 1) Try original distance band first.
        band_goals = _scan_band_goals(r_min, r_max)
        ordered_candidates: List[Tuple[int, int]] = []
        used_nearest_fallback = False

        # 2) If no traversable cells, relax distance band up to _MAX_DIST_RELAX.
        if not band_goals:
            max_dist_relax = float(_planner_param("max_dist_relax_m", 0.30))
            relaxed_r_min = max(0.05, r_min - max_dist_relax)
            relaxed_r_max = r_max + max_dist_relax
            band_goals = _scan_band_goals(relaxed_r_min, relaxed_r_max)
            if band_goals:
                print(
                    f"No goals in original band [{r_min:.2f}, {r_max:.2f}]; "
                    f"expanded to [{relaxed_r_min:.2f}, {relaxed_r_max:.2f}] "
                    f"(+{max_dist_relax:.2f}m)"
                )
            else:
                if phase_optional:
                    used_nearest_fallback = True
                    ordered_candidates = _scan_nearest_free_goals(int(_planner_param("max_goal_retries", 80)))
                    if ordered_candidates:
                        print(
                            "No traversable ring-sector goals after distance relaxation; "
                            "phase is optional, so falling back to nearest reachable cells "
                            "around the target."
                        )
                    else:
                        print(
                            f"[FAIL] No traversable goal cells even after expanding distance "
                            f"band by {max_dist_relax:.2f}m — no possible goal."
                        )
                        self.last_failure_reason = (
                            "No traversable free cells near target for optional-phase fallback."
                        )
                        return None
                else:
                    print(
                        f"[FAIL] No traversable goal cells even after expanding distance "
                        f"band by {max_dist_relax:.2f}m — no possible goal."
                    )
                    self.last_failure_reason = (
                        "No feasible goal cells in the requested ring sector, "
                        "even after distance-band relaxation."
                    )
                    return None

        if not used_nearest_fallback:
            if phase_optional:
                print(
                    "Phase not explicitly requested; ranking goal phases by "
                    "lowest-cost / highest-clearance cells inside distance band."
                )
                ordered_candidates = _rank_optional_phase_goals(band_goals)
            else:
                # 3) Rank goals with weighted score: 70% phase error, 30% distance.
                #    Normalise both components so they are comparable.
                max_phase = max(g[1] for g in band_goals) or 1e-6
                max_dist  = max(g[2] for g in band_goals) or 1e-6

                def _goal_score(item):
                    _, e_phi_abs, d_robot, _ = item
                    phase_norm = e_phi_abs / max_phase
                    dist_norm  = d_robot  / max_dist
                    return (
                        float(_planner_param("goal_w_phase", 0.70)) * phase_norm
                        + float(_planner_param("goal_w_dist", 0.30)) * dist_norm
                    )

                ranked_band_goals = sorted(band_goals, key=_goal_score)

                goals_in_tol = sum(1 for _, err, _, _ in ranked_band_goals if err <= phi_tol)
                if goals_in_tol == 0:
                    nearest_phase_err = min(g[1] for g in ranked_band_goals)
                    relaxed_phase_tol = max(
                        phi_tol,
                        nearest_phase_err * float(_planner_param("phase_relax_factor", 1.15)),
                    )
                    relaxed_candidates = [
                        (g, err, d, c) for g, err, d, c in ranked_band_goals if err <= relaxed_phase_tol
                    ]
                    if not relaxed_candidates:
                        relaxed_candidates = [ranked_band_goals[0]]
                    print(
                        "No goal cells satisfy requested phase tolerance; "
                        "falling back to closest phase error inside distance band "
                        f"(relaxed |e_phi| <= {relaxed_phase_tol:.3f} rad)"
                    )
                    ordered_candidates = [
                        g for g, *_ in relaxed_candidates[: int(_planner_param("max_goal_retries", 80))]
                    ]
                else:
                    ordered_candidates = [
                        g for g, *_ in ranked_band_goals[: int(_planner_param("max_goal_retries", 80))]
                    ]

            # Lock selected goal during navigation: once sticky goal exists and remains
            # valid in the current distance band, keep replanning to the same goal.
            band_goal_set = {g for g, *_ in band_goals}
            if self.sticky_goal is not None:
                if (
                    self.sticky_goal in band_goal_set
                    and self.grid[self.sticky_goal[1], self.sticky_goal[0]]
                    <= int(_planner_param("hard_block_cost", 90))
                ):
                    ordered_candidates = [self.sticky_goal]
                else:
                    self.sticky_goal = None

        if not (0 <= start_idx[0] < width and 0 <= start_idx[1] < height):
            print("Start pose is outside grid bounds")
            self.last_failure_reason = "Robot pose is outside the global costmap bounds."
            return None

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        def _search_candidates(candidates: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
            path_indices_local = None
            for goal in candidates:
                open_set: list = []
                h_start = heuristic(start_idx, goal)
                heapq.heappush(open_set, (h_start, 0, start_idx))

                came_from: dict = {}
                g_cost = {start_idx: 0}

                while open_set:
                    f, g, current = heapq.heappop(open_set)

                    if current == goal:
                        path_indices_local = []
                        while current in came_from:
                            path_indices_local.append(current)
                            current = came_from[current]
                        path_indices_local.append(start_idx)
                        path_indices_local.reverse()
                        self.sticky_goal = goal
                        break

                    if g > g_cost.get(current, float("inf")):
                        continue

                    for dx, dy in [
                        (-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1),
                    ]:
                        neighbor = (current[0] + dx, current[1] + dy)
                        if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                            continue
                        if self.grid[neighbor[1], neighbor[0]] > int(_planner_param("hard_block_cost", 90)):
                            continue

                        dist = math.hypot(dx, dy)
                        cell_cost = float(self.grid[neighbor[1], neighbor[0]])
                        w_costmap = float(_planner_param("costmap_weight", 3.0))
                        step_cost = dist * (1.0 + w_costmap * cell_cost / 100.0)
                        if penalty is not None:
                            step_cost += penalty[neighbor[1], neighbor[0]]
                        tentative_g = g + step_cost
                        if tentative_g < g_cost.get(neighbor, float("inf")):
                            came_from[neighbor] = current
                            g_cost[neighbor] = tentative_g
                            f_new = tentative_g + heuristic(neighbor, goal)
                            heapq.heappush(open_set, (f_new, tentative_g, neighbor))

                if path_indices_local is not None:
                    return path_indices_local
            return None

        path_indices = _search_candidates(ordered_candidates)
        if path_indices is None and phase_optional and not used_nearest_fallback:
            # For optional phase, if no feasible goals in the current ring, widen the
            # distance band and keep prioritising low-cost/high-clearance phases.
            max_dist_relax = float(_planner_param("max_dist_relax_m", 0.30))
            expanded_r_min = max(0.05, r_min - max_dist_relax)
            expanded_r_max = r_max + max_dist_relax
            expanded_band_goals = _scan_band_goals(expanded_r_min, expanded_r_max)
            if expanded_band_goals:
                print(
                    "No path found to low-cost phases in requested distance band; "
                    f"expanding band to [{expanded_r_min:.2f}, {expanded_r_max:.2f}] and retrying."
                )
                expanded_candidates = _rank_optional_phase_goals(expanded_band_goals)
                path_indices = _search_candidates(expanded_candidates)

        if path_indices is None and phase_optional:
            nearest_candidates = _scan_nearest_free_goals(int(_planner_param("max_goal_retries", 80)))
            if nearest_candidates:
                print(
                    "No path found after optional-phase distance-band expansion; "
                    "retrying with nearest reachable cells around target."
                )
                path_indices = _search_candidates(nearest_candidates)

        if path_indices is None:
            print("No path found (all distance-band phase candidates blocked)")
            if phase_optional:
                self.last_failure_reason = (
                    "No collision-free path found: tried low-cost phases in the requested "
                    "distance band, then expanded distance band, then nearest-goal fallback."
                )
            else:
                self.last_failure_reason = (
                    "Goal cells exist in the ring sector, but no collision-free path "
                    "could be found to any candidate."
                )
            return None

        self.last_failure_reason = ""
        return [
            grid_to_world(ix, iy, origin_x, origin_y, resolution)
            for ix, iy in path_indices
        ]

    # ---- visualisation ----

    def visualize(
        self,
        path,
        x0,
        p,
        o,
        r_min,
        r_max,
        phi0,
        phi_tol,
        ax=None,
        show_goal_zone: bool = True,
        show_legend: bool = True,
        cache_goal_zone: bool = True,
    ):
        if self.grid is None:
            return

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
            should_show = True
        else:
            should_show = False

        extent = [
            self.meta["origin_x"],
            self.meta["origin_x"] + self.meta["width"] * self.meta["resolution"],
            self.meta["origin_y"],
            self.meta["origin_y"] + self.meta["height"] * self.meta["resolution"],
        ]
        ax.imshow(self.grid, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=100)

        if path:
            px, py = zip(*path)
            ax.plot(px, py, "b-", linewidth=2, label="Path")
            ax.plot(path[-1][0], path[-1][1], "bx", markersize=10, markeredgewidth=2, label="Goal Reached")

        ax.plot(x0[0], x0[1], "go", label="Episode Start (x0)")
        ax.plot(p[0], p[1], "co", label="Current Pose (p)")
        ax.plot(o[0], o[1], "rx", markersize=12, markeredgewidth=3, label="Target Object")

        circle_min = plt.Circle(o, r_min, color="r", fill=False, linestyle="--", alpha=0.5, label="r_min")
        circle_max = plt.Circle(o, r_max, color="r", fill=False, linestyle="--", alpha=0.5, label="r_max")
        ax.add_patch(circle_min)
        ax.add_patch(circle_max)

        if show_goal_zone:
            res = self.meta["resolution"]
            key = (
                round(o[0], 3), round(o[1], 3),
                round(r_min, 3), round(r_max, 3),
                round(phi0, 4), round(phi_tol, 4),
                self.meta["width"], self.meta["height"],
                round(res, 6),
                round(self.meta["origin_x"], 4),
                round(self.meta["origin_y"], 4),
            )

            recompute = True
            if cache_goal_zone:
                if not hasattr(self, "_goal_zone_cache"):
                    self._goal_zone_cache = {"key": None, "xy": ([], [])}
                recompute = self._goal_zone_cache["key"] != key

            if recompute:
                gx_list, gy_list = [], []
                phi_ref = math.atan2(o[1] - x0[1], o[0] - x0[0])
                r_scan = r_max + res
                k = int(math.ceil(r_scan / res))
                ox_idx, oy_idx = world_to_grid(
                    o[0], o[1],
                    self.meta["origin_x"], self.meta["origin_y"], res,
                )

                for ix in range(ox_idx - k, ox_idx + k + 1):
                    for iy in range(oy_idx - k, oy_idx + k + 1):
                        if 0 <= ix < self.meta["width"] and 0 <= iy < self.meta["height"]:
                            xw, yw = grid_to_world(ix, iy, self.meta["origin_x"], self.meta["origin_y"], res)
                            r = math.hypot(xw - o[0], yw - o[1])
                            if r < r_min or r > r_max:
                                continue
                            phi = math.atan2(o[1] - yw, o[0] - xw)
                            phi_rel = wrap_angle(phi_ref - phi)
                            e_phi = wrap_angle(phi_rel - phi0)
                            if abs(e_phi) <= phi_tol:
                                gx_list.append(xw)
                                gy_list.append(yw)

                if cache_goal_zone:
                    self._goal_zone_cache["key"] = key
                    self._goal_zone_cache["xy"] = (gx_list, gy_list)
            else:
                gx_list, gy_list = self._goal_zone_cache["xy"]

            ax.scatter(gx_list, gy_list, s=1, c="g", alpha=0.3, label="Goal Zone")

        if show_legend:
            ax.legend(loc="upper right")
        ax.set_title("Plan Path Ring Sector")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True)
        ax.axis("equal")

        if should_show:
            plt.show()
