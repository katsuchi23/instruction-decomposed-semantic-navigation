# ROS2 Bridge Nodes

These nodes bridge between the semantic navigation stack (which runs as a plain Python
process) and your ROS2 environment.  All communication uses ZMQ sockets so the navigation
stack never depends on `rclpy` directly.

Copy the scripts you need into your ROS2 package, register them in `CMakeLists.txt` under
`install(PROGRAMS ...)`, and add the corresponding `Node(...)` entries to your launch file.

---

## Port map

| Port | Direction | IPC key in `ipc.yaml` | Node |
|------|-----------|----------------------|------|
| 5556 | REP (recv goal) | `nav_goal_req` | `nav2_sender_node.py` |
| 5557 | PUB (send pose) | `pose_sub` | `robot_pose_sender_node.py` |
| 5558 | PUB (send status) | `nav_status_sub` | `nav2_sender_node.py` |
| 5559 | REP (recv cmd_vel) | `cmd_vel_req` | `cmd_vel_sender_node.py` |
| 5562 | PUB (send scan) | `scan_sub` | `laserscan_sender_node.py` |
| 5563 | REP (recv trajectory) | `trajectory_req` | `trajectory_visualization_node.py` |
| 5564 | REP (send local costmap) | `local_costmap_req` | `occupancy_grid_sender_node.py` |
| 5565 | REP (send global costmap) | `global_costmap_req` | `global_costmap.py` |
| 5566 | REP (recv nav-through-poses goal) | `nav_through_poses_req` | `nav_through_poses_bridge.py` |
| 5567 | PUB (send /cmd_vel telemetry) | `nav_cmd_vel_sub` | `nav_through_poses_bridge.py` |
| 5568 | PUB (send /amcl_pose) | `amcl_pose_sub` | `nav_through_poses_bridge.py` |
| 5569 | PUB (send nav-through-poses status) | `nav_through_poses_status_sub` | `nav_through_poses_bridge.py` |
| 5570 | SUB (recv viz data) | `viz_pub` | `semnav_rviz_node.py` |

All ports are on `127.0.0.1` (localhost).  Update `ipc.yaml` if you run the navigation
stack on a different machine.

---

## Node descriptions

### `robot_pose_sender_node.py` — port 5557 PUB
Reads the robot's pose from the TF tree (`map → odom → base_link`) at 10 Hz and
publishes a JSON bundle on port 5557.  The navigation stack uses this as its primary
pose source.

**Required TF frames:** `map`, `odom`, `base_link`

---

### `nav2_sender_node.py` — ports 5556 REP + 5558 PUB
- Listens on port 5556 for `{"x": float, "y": float, "yaw": float}` goal requests and
  forwards them to Nav2's `navigate_to_pose` action.
- Continuously publishes the current goal status on port 5558.

**Required:** Nav2 `navigate_to_pose` action server running.

---

### `cmd_vel_sender_node.py` — port 5559 REP
Receives `{"v": float, "w": float}` velocity commands on port 5559 and publishes them
to `/cmd_vel` at 30 Hz.  Publishes zero-velocity if no command is received for 0.2 s
and stops publishing entirely after 5 s of silence (safety).

**Used in:** default trajectory-sampling mode (`nav_through_pose: false`).

---

### `occupancy_grid_sender_node.py` — port 5564 REP
Subscribes to `/local_costmap/costmap` and serves the latest grid on request via a
REP socket.  The navigation stack queries this for obstacle avoidance scoring.

---

### `global_costmap.py` — port 5565 REP
Same as above but for `/global_costmap/costmap`, used by the A\* global planner.

---

### `laserscan_sender_node.py` — port 5562 PUB
Subscribes to `/scan` and publishes the latest laser scan as JSON on port 5562.

---

### `trajectory_visualization_node.py` — port 5563 REP
Receives a trajectory (list of `{x, y, yaw}` poses) on port 5563 and publishes it as
a `nav_msgs/Path` on `/projected_path` for visualization in RViz.

---

### `nav_through_poses_bridge.py` — ports 5566 REP / 5567–5569 PUB
Used when `nav_through_pose: true` in `runtime.yaml`.

- **5566 REP** — receives `{"poses": [{x, y, yaw}, ...]}` and calls Nav2's
  `navigate_through_poses` action.
- **5567 PUB** — streams `/cmd_vel` as `CMD_VEL {json}`.
- **5568 PUB** — streams `/amcl_pose` as `AMCL_POSE {json}` (used for telemetry).
- **5569 PUB** — streams `NavigateThroughPoses` status as `NAV_THROUGH_STATUS {json}`.

**Required:** Nav2 `navigate_through_poses` action server running.

---

### `semnav_rviz_node.py` — port 5570 SUB
Receives navigation visualization data from the stack on port 5570 and publishes
RViz-compatible topics:

| Topic | Type | Content |
|-------|------|---------|
| `/semnav/path` | `nav_msgs/Path` | Global A\* planned path |
| `/semnav/traj_samples` | `visualization_msgs/MarkerArray` | Sampled trajectory rollouts |
| `/semnav/markers` | `visualization_msgs/MarkerArray` | Robot, start, target, goal ring, constraint/preference radii |

Add these topics in RViz under **Displays → Add → By topic**.

---

## Minimal setup (default mode)

The nodes required to run the stack in default `cmd_vel` mode:

```
robot_pose_sender_node.py       # pose
cmd_vel_sender_node.py          # velocity commands
occupancy_grid_sender_node.py   # local costmap
global_costmap.py               # global costmap
nav2_sender_node.py             # (optional) NavigateToPose fallback
semnav_rviz_node.py             # (optional) RViz visualization
```

For `nav_through_pose: true` mode, replace `cmd_vel_sender_node.py` with
`nav_through_poses_bridge.py`.
