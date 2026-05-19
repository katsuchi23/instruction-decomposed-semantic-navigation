# IPC Interface

The runtime depends only on these IPC endpoints. Any ROS, simulator, or real-robot bridge may be used as long as it satisfies this contract.

## `pose_sub`

- Socket: `SUB`
- Payload: UTF-8 string with topic prefix `TF ` followed by JSON
- Expected JSON keys:
  - `tfs.map_base`
  - `tfs.map_odom`
  - `tfs.odom_base`
  - `tfs.map_camera`
  - `map_frame`
  - `odom_frame`
  - `base_frame`
- Units:
  - position in metres
  - yaw in radians

## `nav_status_sub`

- Socket: `SUB`
- Payload: UTF-8 string with topic prefix `NAV_STATUS ` followed by JSON
- Expected JSON:
  - `status`: one of `IDLE`, `ACTIVE`, `SUCCEEDED`, `FAILED`, `CANCELED`

## `cmd_vel_req`

- Socket: `REQ/REP`
- Request JSON:
  - `v`: linear velocity in m/s
  - `w`: angular velocity in rad/s
- Reply:
  - plain string `OK`

## `nav_goal_req`

- Socket: `REQ/REP`
- Request JSON:
  - `x`, `y`: goal position in map frame
  - `yaw`: goal heading in radians
- Reply:
  - plain string `OK`

## `trajectory_req`

- Socket: `REQ/REP`
- Request JSON:
  - `poses`: list of pose dictionaries for visualization or inspection
- Reply:
  - plain string `OK`

## `image_req`

- Socket: `REQ/REP`
- Request:
  - plain string `GET`
- Reply:
  - JPEG bytes decoded into a BGR image

## `depth_req`

- Socket: `REQ/REP`
- Request:
  - plain string `GET`
- Reply:
  - raw `float32` depth buffer shaped as `(360, 640)`

## `scan_sub`

- Socket: `SUB`
- Payload:
  - JSON containing `ranges` and standard laser-scan metadata
- Units:
  - ranges in metres

## `local_costmap_req` and `global_costmap_req`

- Socket: `REQ/REP`
- Request:
  - plain string `GET`
- Reply multipart:
  1. JSON metadata with at least `ok`, `width`, `height`, `resolution`, `origin_x`, `origin_y`, `dtype`
  2. costmap grid bytes

## Timeout Behavior

The runtime uses the timeout values in `config/ipc.yaml`. If an endpoint times out, the runtime holds or aborts depending on which channel failed and for how long.
