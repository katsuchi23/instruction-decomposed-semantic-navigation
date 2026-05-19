# Configuration Reference

The repository publishes one canonical set of config files. There are no separate example copies; the committed `*.yaml` files are the intended public defaults and editing surface.

## `config/ipc.yaml`

- `endpoints.pose_sub`: ZMQ subscriber endpoint that publishes the TF / pose bundle consumed by the runtime.
- `endpoints.nav_status_sub`: ZMQ subscriber endpoint for navigation status updates.
- `endpoints.cmd_vel_req`: ZMQ request/reply endpoint used to send `(v, w)` commands.
- `endpoints.nav_goal_req`: Optional ZMQ request/reply endpoint for discrete goal commands.
- `endpoints.trajectory_req`: Optional ZMQ request/reply endpoint for trajectory visualization.
- `endpoints.image_req`: Optional RGB image request endpoint reserved for external tooling.
- `endpoints.depth_req`: Optional depth image request endpoint reserved for external tooling.
- `endpoints.scan_sub`: Laser scan subscriber endpoint.
- `endpoints.local_costmap_req`: Local costmap request/reply endpoint.
- `endpoints.global_costmap_req`: Global costmap request/reply endpoint.
- `timeouts_ms.*`: Socket-level timeouts in milliseconds. Increase when your bridge publishes slowly; decrease for more aggressive failure detection.

## `config/runtime.yaml`

- `scene`: Name of the active scene config under `config/scenes/`.
- `feature_flags.show_viz`: Enables or disables live matplotlib visualization.
- `feature_flags.enable_ros_param_inflation_updates`: Enables optional direct `ros2 param set` calls. Keep this `false` unless you intentionally want core runtime code to touch ROS.
- `paths.data_root`: Canonical dataset root directory.
- `paths.processed_data_root`: Root that contains prepared scene folders.
- `paths.outputs_root`: Root for generated outputs and caches.
- `paths.intent_cache_dir`: Location of cached LLM parses.
- `paths.docs_path`: Explicit fallback `docs.jsonl` path when not provided on the command line.
- `paths.viz_map_pgm`: Optional occupancy-map image used for visualization helpers.
- `assets.dovsg_root`: Expected location of the DovSG submodule used for Python-side imports.
- `assets.clip_checkpoint_dir`: Repository-owned CLIP checkpoint directory. The runtime expects `open_clip_pytorch_model.bin` inside this folder.

## `config/params.yaml`

### `object_retrieval`

- `direction_min_distance_m`: Compatibility knob for directional relations. Larger values make directional relations require more separation.
- `proximity_scale_m`: Larger values make the `near` score decay more slowly with distance.
- `reference_top_k`: Number of reference-object candidates retained for relational grounding.
- `isolation_scale_m`: Larger values make `alone` scores increase more gradually.
- `clip_weight`: Weight of the target CLIP score in the final retrieval fusion.
- `top_k`: Number of target candidates retained without relational references.

### `intent_costs`

- `constraint_radius_m`: Radius of the planner penalty field around avoid objects.
- `preference_radius_m`: Radius of the attraction field around preferred objects.
- `preference_min_dist_m`: Distance inside which preference becomes repulsive.
- `w_constraint_path`: Larger values make the planner avoid constraints more aggressively.
- `w_preference_attract_path`: Larger values make preferred-object attraction stronger.

### `planner`

- `max_goal_retries`: Maximum number of goal candidates tried during ring-sector planning.
- `phase_relax_factor`: Phase-tolerance relaxation multiplier when no exact directional goal is reachable.
- `goal_w_phase`: Weight of phase error in directional goal ranking.
- `goal_w_dist`: Weight of robot-goal distance in directional goal ranking.
- `goal_w_cost_opt`: Weight of clearance / cost in optional-phase goal ranking.
- `goal_w_dist_opt`: Weight of distance in optional-phase goal ranking.
- `max_dist_relax_m`: Maximum expansion of the target distance band when the nominal ring is blocked.
- `hard_block_cost`: Costmap cells above this value are treated as non-traversable.

### `control`

- `termination.distance_band_m`: Acceptable distance band around the grounded target for each strictness level.
- `termination.alpha_max_deg`: Allowed final heading error relative to the target bearing.
- `termination.phi_tol_deg`: Allowed final angular error relative to the requested phase.
- `sampling.horizon_steps`: Number of rollout steps sampled by the controller.
- `sampling.dt_sec`: Integration timestep for rollouts.
- `sampling.num_samples`: Number of sampled trajectories. Larger values increase runtime but improve search.
- `behavior_mapping.speed.*`: Maximum linear and angular velocities for each language speed label.
- `behavior_mapping.caution.*`: Safety distance and obstacle penalties for each caution label.

### `navigator`

- `timeout_sec`: Per-task timeout before the runtime aborts a task.
- `collision_cost_thresh`: Local-costmap value treated as collision.
- `collision_duration_sec`: How long the robot may remain in collision before the task aborts.
