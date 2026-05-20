# G1 Robot ROS 2 Stack

This document covers the ROS 2 packages required to run the Unitree G1 navigation pipeline with a Livox Mid360 LiDAR. The ROS workspace is the robotics-side counterpart to this repository — it provides all sensor drivers, localization, and Nav2 navigation that this semantic navigation stack sits on top of.

The ROS packages live in a separate repository: `https://github.com/katsuchi23/g1_ros_package`

---

## Build

```bash
git clone https://github.com/katsuchi23/g1_ros_package.git src
cd src
chmod +x build.sh
./build.sh
```

---

## Launch Order

Run each command in a separate terminal. Source the workspace before each:

```bash
source install/setup.bash
```

### 1. Livox ROS2 Driver (`livox_ros_driver2`)

Connects to the Livox Mid360 LiDAR hardware and publishes raw scan data as a Livox custom message type on `/livox/lidar` and `/livox/imu`.

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

### 2. Livox to PointCloud2 (`livox_to_pointcloud2`)

Converts the Livox custom message format into a standard `sensor_msgs/PointCloud2` message. The output cloud is stamped in the `body` frame and published on `/livox/lidar/pcd2`. This is required because FAST-LIO and most ROS tools expect standard PointCloud2, not the Livox-proprietary format.

```bash
ros2 launch livox_to_pointcloud2 livox_to_pointcloud2.launch.yml
```

### 3. PointCloud to LaserScan (`pointcloud_to_laserscan`)

Takes the 3D point cloud and produces a flat 2D `/scan` laser scan suitable for Nav2. It runs two steps internally:

- `pointcloud_convert.py` — applies the `body → base_link` rotation to the raw cloud so the 2D slice is taken in the correct horizontal plane.
- `pointcloud_to_laserscan_node` — slices the transformed cloud within a configurable height window (`min_height` / `max_height`) and publishes `/scan`.

```bash
ros2 launch pointcloud_to_laserscan sample_pointcloud_to_laserscan_launch.py
```

### 4. FAST-LIO Localization (`fast_lio_localization`)

Runs LiDAR-inertial odometry (FAST-LIO) in localization mode against a pre-built PCD map. The default map is `rrc2.pcd`. It publishes:

- `/Odometry` — raw `camera_init → body` odometry from FAST-LIO.
- `/localization` — fused `map → body` odometry after global localization.
- TF chain: `map → odom → camera_init → body` (see TF Layout section below).

```bash
ros2 launch fast_lio_localization localization.launch.py \
  map:=<path_to_ros_ws>/src/mapping/FAST_LIO_ROS2/PCD/rrc2.pcd
```

After launch, set the initial pose estimate in RViz using **2D Pose Estimate** to initialize global localization.

### 5. G1 Navigation (`g1_navigation`)

Launches the full Nav2 navigation stack (costmaps, planner, controller) configured for the G1 robot, along with the ZMQ IPC bridge nodes that connect to this semantic navigation stack.

```bash
ros2 launch g1_navigation g1_navigation.launch.py
```

Key nodes it starts:
- **Nav2 bringup** — AMCL-less localization (uses FAST-LIO pose), global and local costmaps, planner, controller.
- **`ros_ipc_bridge.py`** — bundles robot pose and sends it over ZMQ to this semantic nav stack.
- **`laserscan_sender_node.py`**, **`occupancy_grid_sender_node.py`**, **`global_costmap.py`** — pipe sensor and costmap data to the semantic nav stack via ZMQ.
- **`nav2_sender_node.py`** / **`nav_through_poses_bridge.py`** — receive goal commands from the semantic nav stack and relay them to Nav2.

### 6. G1 Controller (`g1_controller`)

Subscribes to `/cmd_vel` and translates velocity commands into Unitree SDK locomotion commands sent over the robot's network interface. This is the final actuator bridge between Nav2/semantic-nav and the physical robot.

```bash
ros2 launch g1_controller cmd_vel_to_g1.launch.py network_interface:=eno1
```

---

## TF Frame Layout

After all nodes are running, the full transform chain is:

```
map
 └── odom
      └── camera_init
           └── body
                └── base_link
```

### What each link does

| Link | Published by | Purpose |
|------|-------------|---------|
| `map → odom` | `transform_fusion.py` | Global localization offset. Updated by the FAST-LIO global localization node to align local odometry with the pre-built PCD map. |
| `odom → camera_init` | `transform_fusion.py` | **Orientation correction for the LiDAR mounting.** See below. |
| `camera_init → body` | FAST-LIO (`fastlio_mapping`) | Real-time LiDAR-inertial odometry in FAST-LIO's own coordinate frame. |
| `body → base_link` | `localization.launch.py` (static) | **Orientation correction for navigation.** See below. |

### Why `odom → camera_init`

FAST-LIO initializes its world frame (`camera_init`) at the sensor's pose at startup — with whatever orientation the LiDAR happens to be physically mounted in. On the G1 robot, the Mid360 is mounted at an angle (tilted in pitch and flipped in roll relative to the world). If `camera_init` is used directly as the odometry root, the entire map and all odometry would be rotated/tilted relative to the real-world gravity-aligned horizontal plane.

The `odom → camera_init` transform corrects for this physical mounting offset. It rotates `camera_init` so that the odometry frame (`odom`) is properly gravity-aligned and forward-pointing, making it consistent with the pre-built PCD map and with Nav2's expectations.

### Why `body → base_link`

FAST-LIO's `body` frame is the IMU/LiDAR body frame — its axes follow the sensor, not the robot's navigation convention. Nav2 and the costmap expect `base_link` to have the standard ROS convention: X forward, Y left, Z up, aligned with the robot's actual direction of travel.

The `body → base_link` transform corrects the sensor's physical tilt so that the laser scan slice and the robot footprint are in the correct horizontal plane relative to the real world environment. Without this, the 2D laser scan would be sliced at a tilted angle and the robot would appear rotated in the costmap.

---

## Fine-Tuning `odom_pitch` (and Other Orientation Angles)

The mounting orientation of the Mid360 on the G1 is not perfectly level — the robot's body and the LiDAR bracket introduce a pitch (and a 180° roll flip). The exact values need to be measured or empirically tuned per robot unit.

The two parameters you will most commonly need to adjust are `odom_roll` and `odom_pitch`. Both are defined **in one place** and automatically propagated to all relevant transforms.

### Config 1 — `odom → camera_init` transform

**File:** `src/mapping/FAST_LIO_LOCALIZATION2/config/mid360.yaml`

```yaml
publish:
    use_odom_transform: true
    odom_roll: -180.0   # degrees — 180° flip to invert the LiDAR's upside-down roll
    odom_pitch: 7.5     # degrees — tilt correction for the LiDAR mounting angle on G1
    odom_yaw: 0.0       # degrees — usually 0 unless the LiDAR is rotated horizontally
```

These values are read by `transform_fusion.py` to build the `odom → camera_init` rotation, and also by `localization.launch.py` to compute the inverse `body → base_link` static transform. Changing them here propagates to both TF links automatically.

**How to tune:** Run the full stack with RViz open. Publish the `/map` topic (PCD map) and the `/scan` topic together. Adjust `odom_pitch` until the laser scan ring aligns flatly with the walls and floor in the map. A wrong pitch will cause the scan to cut through walls diagonally or miss the floor entirely.

### Config 2 — `body → base_link` point cloud transform (for laser scan)

**File:** `src/mapping/pointcloud_to_laserscan/scripts/pointcloud_convert.py`

```python
roll  = np.radians(180.0)   # must match odom_roll magnitude
pitch = np.radians(7.5)     # must match odom_pitch
yaw   = 0.0
```

This script manually rotates the raw point cloud from `body` to `base_link` before slicing it into a 2D laser scan. It uses hardcoded values that **must be kept in sync** with the `mid360.yaml` values above. If you change `odom_pitch` in the config, update `pitch` here to match.

> **Summary of the sync rule:**
> - `mid360.yaml: odom_pitch` → sets `odom → camera_init` and `body → base_link` TF
> - `pointcloud_convert.py: pitch` → sets the point cloud rotation for `/scan` generation
> - Both must have the same value, otherwise the laser scan and the TF tree will disagree.
