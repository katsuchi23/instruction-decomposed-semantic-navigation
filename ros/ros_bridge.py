"""Singleton ROS node that replaces the ZMQ IPC bridge.

Provides direct access to TF2, costmaps, cmd_vel, viz, and Nav2 actions.
All access is thread-safe; the node runs on a background executor thread.
"""

from __future__ import annotations

import atexit
import math
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
    Twist,
    Vector3,
)
from nav2_msgs.action import NavigateThroughPoses
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_node: Optional["SemnavBridgeNode"] = None
_executor: Optional[MultiThreadedExecutor] = None
_spin_thread: Optional[threading.Thread] = None
_init_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_to_quat(yaw: float) -> Quaternion:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    q = Quaternion()
    q.x, q.y, q.z, q.w = 0.0, 0.0, sy, cy
    return q


def _color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def _point(x: float, y: float, z: float = 0.01) -> Point:
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


def _ring_points(cx: float, cy: float, radius: float, n: int = 64) -> List[Point]:
    return [
        _point(cx + radius * math.cos(a), cy + radius * math.sin(a))
        for a in (2 * math.pi * i / n for i in range(n + 1))
    ]


# ---------------------------------------------------------------------------
# Bridge node
# ---------------------------------------------------------------------------

class SemnavBridgeNode(Node):
    FRAME = "map"

    def __init__(self) -> None:
        super().__init__("semnav_bridge")

        # Frame names (match defaults in the sender nodes)
        self.map_frame = "map"
        self.odom_frame = "odom"
        self.base_frame = "base_link"
        self.camera_frame = "camera_link"

        # --- TF2 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Costmap subscriptions (RELIABLE + TRANSIENT_LOCAL to match Nav2) ---
        _costmap_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._costmap_lock = threading.Lock()
        self._local_grid: Optional[np.ndarray] = None
        self._local_meta: Optional[Dict] = None
        self._global_grid: Optional[np.ndarray] = None
        self._global_meta: Optional[Dict] = None

        self.create_subscription(
            OccupancyGrid, "/local_costmap/costmap", self._on_local_costmap, _costmap_qos
        )
        self.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", self._on_global_costmap, _costmap_qos
        )

        # --- cmd_vel publisher ---
        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # --- Viz publishers ---
        self._path_pub = self.create_publisher(Path, "/semnav/path", 10)
        self._traj_pub = self.create_publisher(MarkerArray, "/semnav/traj_samples", 10)
        self._marker_pub = self.create_publisher(MarkerArray, "/semnav/markers", 10)

        # --- Nav2 action client ---
        self._nav_action_client = ActionClient(self, NavigateThroughPoses, "navigate_through_poses")
        self._nav_lock = threading.Lock()
        self._nav_status = "IDLE"
        self._nav_status_code = int(GoalStatus.STATUS_UNKNOWN)
        self._nav_detail = "No goal sent yet."

        # --- /amcl_pose subscriber ---
        self._amcl_lock = threading.Lock()
        self._amcl_pose: Optional[Dict[str, float]] = None
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl_pose, 10
        )

        # --- /cmd_vel subscriber (for telemetry in nav_through_pose mode) ---
        self._cmd_vel_lock = threading.Lock()
        self._latest_cmd_vel: Optional[Dict[str, float]] = None
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)

        # --- warn counters ---
        self._tf_warn: Dict[str, int] = {}

        self.get_logger().info("SemnavBridgeNode ready (direct ROS, no IPC)")

    # ------------------------------------------------------------------ #
    # Costmap callbacks
    # ------------------------------------------------------------------ #

    def _costmap_from_msg(self, msg: OccupancyGrid, topic: str) -> Tuple[np.ndarray, Dict]:
        w, h = int(msg.info.width), int(msg.info.height)
        grid = np.asarray(msg.data, dtype=np.int16).reshape((h, w)).copy()
        meta = {
            "ok": True,
            "width": w,
            "height": h,
            "dtype": "int16",
            "order": "C",
            "resolution": float(msg.info.resolution),
            "origin_x": float(msg.info.origin.position.x),
            "origin_y": float(msg.info.origin.position.y),
            "frame_id": msg.header.frame_id,
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nanosec": int(msg.header.stamp.nanosec),
            "min": int(grid.min()),
            "max": int(grid.max()),
            "topic": topic,
        }
        return grid, meta

    def _on_local_costmap(self, msg: OccupancyGrid) -> None:
        grid, meta = self._costmap_from_msg(msg, "/local_costmap/costmap")
        with self._costmap_lock:
            self._local_grid = grid
            self._local_meta = meta

    def _on_global_costmap(self, msg: OccupancyGrid) -> None:
        grid, meta = self._costmap_from_msg(msg, "/global_costmap/costmap")
        with self._costmap_lock:
            self._global_grid = grid
            self._global_meta = meta

    def get_local_costmap(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        with self._costmap_lock:
            if self._local_grid is None:
                return None, None
            return self._local_grid.copy(), dict(self._local_meta)

    def get_global_costmap(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        with self._costmap_lock:
            if self._global_grid is None:
                return None, None
            return self._global_grid.copy(), dict(self._global_meta)

    # ------------------------------------------------------------------ #
    # TF2
    # ------------------------------------------------------------------ #

    def get_tf(self, parent: str, child: str) -> Optional[Dict]:
        try:
            tf = self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
            tr = tf.transform.translation
            q = tf.transform.rotation
            return {
                "parent": parent,
                "child": child,
                "x": float(tr.x),
                "y": float(tr.y),
                "z": float(tr.z),
                "yaw": float(_quat_to_yaw(q.x, q.y, q.z, q.w)),
                "stamp_sec": int(tf.header.stamp.sec),
                "stamp_nanosec": int(tf.header.stamp.nanosec),
            }
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            key = f"{parent}->{child}"
            cnt = self._tf_warn.get(key, 0) + 1
            self._tf_warn[key] = cnt
            if cnt % 20 == 1:
                self.get_logger().warn(f"TF lookup failed {key}: {type(exc).__name__}: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # cmd_vel
    # ------------------------------------------------------------------ #

    def publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(msg)

    def _on_cmd_vel(self, msg: Twist) -> None:
        with self._cmd_vel_lock:
            self._latest_cmd_vel = {"v": float(msg.linear.x), "w": float(msg.angular.z)}

    def get_latest_cmd_vel(self) -> Optional[Dict[str, float]]:
        with self._cmd_vel_lock:
            return dict(self._latest_cmd_vel) if self._latest_cmd_vel else None

    # ------------------------------------------------------------------ #
    # amcl_pose
    # ------------------------------------------------------------------ #

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        q = msg.pose.pose.orientation
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        with self._amcl_lock:
            self._amcl_pose = {
                "x": float(msg.pose.pose.position.x),
                "y": float(msg.pose.pose.position.y),
                "yaw": float(yaw),
            }

    def get_amcl_pose(self) -> Optional[Dict[str, float]]:
        with self._amcl_lock:
            return dict(self._amcl_pose) if self._amcl_pose else None

    # ------------------------------------------------------------------ #
    # Nav2 NavigateThroughPoses action
    # ------------------------------------------------------------------ #

    def send_nav_through_poses(self, poses: List[Tuple[float, float, float]]) -> bool:
        if not self._nav_action_client.server_is_ready():
            self.get_logger().warn("NavigateThroughPoses: action server not ready")
            return False

        goal = NavigateThroughPoses.Goal()
        for x, y, yaw in poses:
            ps = PoseStamped()
            ps.header.frame_id = self.FRAME
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation = _yaw_to_quat(float(yaw))
            goal.poses.append(ps)

        future = self._nav_action_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

        with self._nav_lock:
            self._nav_status = "ACTIVE"
            self._nav_status_code = int(GoalStatus.STATUS_ACCEPTED)
            self._nav_detail = f"Sending {len(poses)} poses."
        return True

    def _on_goal_response(self, future: Any) -> None:
        handle = future.result()
        if not handle.accepted:
            with self._nav_lock:
                self._nav_status = "FAILED"
                self._nav_status_code = int(GoalStatus.STATUS_UNKNOWN)
                self._nav_detail = "Goal rejected."
            return
        with self._nav_lock:
            self._nav_status = "ACTIVE"
            self._nav_status_code = int(GoalStatus.STATUS_EXECUTING)
            self._nav_detail = "Goal executing."
        handle.get_result_async().add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future: Any) -> None:
        status = future.result().status
        with self._nav_lock:
            self._nav_status_code = int(status)
            if status == GoalStatus.STATUS_SUCCEEDED:
                self._nav_status = "SUCCEEDED"
                self._nav_detail = "STATUS_SUCCEEDED."
            elif status == GoalStatus.STATUS_CANCELED:
                self._nav_status = "CANCELED"
                self._nav_detail = "STATUS_CANCELED."
            else:
                self._nav_status = "FAILED"
                self._nav_detail = f"Status code {status}."

    def get_nav_through_poses_status(self) -> Dict[str, Any]:
        with self._nav_lock:
            return {
                "status": self._nav_status,
                "status_code": self._nav_status_code,
                "detail": self._nav_detail,
            }

    # ------------------------------------------------------------------ #
    # Viz — ports semnav_rviz_node.py publishing logic directly
    # ------------------------------------------------------------------ #

    def publish_viz(self, data: Dict) -> None:
        try:
            now = self.get_clock().now().to_msg()
            self._pub_path(data, now)
            self._pub_traj_samples(data, now)
            self._pub_markers(data, now)
        except Exception as exc:
            self.get_logger().warn(f"publish_viz error: {exc}")

    def _pub_path(self, data: Dict, stamp: Any) -> None:
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = self.FRAME
        for x, y in (data.get("path") or []):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)

    def _pub_traj_samples(self, data: Dict, stamp: Any) -> None:
        ma = MarkerArray()
        del_m = Marker()
        del_m.action = Marker.DELETEALL
        ma.markers.append(del_m)
        for i, pts in enumerate(data.get("traj_samples") or []):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.FRAME
            m.ns = "traj_samples"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.01
            m.color = _color(0.96, 0.32, 0.12, 0.25)
            m.pose.orientation.w = 1.0
            for x, y in pts:
                m.points.append(_point(x, y))
            if len(m.points) > 1:
                ma.markers.append(m)
        self._traj_pub.publish(ma)

    def _pub_markers(self, data: Dict, stamp: Any) -> None:
        ma = MarkerArray()

        def _add(ns, mid, mtype, pos, color, scale=0.15, yaw=None, text=None):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.FRAME
            m.ns = ns
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = 0.05
            m.pose.orientation = _yaw_to_quat(yaw) if yaw is not None else Quaternion(w=1.0)
            m.scale.x = scale
            m.scale.y = scale * 0.5 if mtype == Marker.ARROW else scale
            m.scale.z = scale * 0.3 if mtype == Marker.ARROW else 0.05
            m.color = color
            if text:
                m.text = text
            ma.markers.append(m)

        robot = data.get("robot")
        start = data.get("start")
        target = data.get("target")
        goal = data.get("goal")
        lh = data.get("lookahead")
        r_min = float(data.get("r_min", 0.4))
        r_max = float(data.get("r_max", 0.8))
        mode = data.get("mode", "NAV")

        if robot:
            _add("robot", 0, Marker.ARROW, robot, _color(0.13, 0.55, 0.13), scale=0.25, yaw=robot[2])
        if start:
            _add("start", 1, Marker.SPHERE, start, _color(0.0, 0.39, 0.0), scale=0.12)
        if target:
            _add("target", 2, Marker.CYLINDER, target, _color(0.9, 0.1, 0.1), scale=0.18)
        if goal:
            _add("goal", 3, Marker.SPHERE, goal, _color(0.1, 0.1, 0.9), scale=0.14)
        if lh:
            _add("lookahead", 4, Marker.SPHERE, lh, _color(0.0, 0.85, 0.85), scale=0.10)

        if target:
            tx, ty = target[0], target[1]
            for ring_id, radius, color in [
                (5, r_max, _color(0.18, 0.80, 0.18, 0.8)),
                (6, r_min, _color(1.0, 0.60, 0.0, 0.8)),
            ]:
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = self.FRAME
                m.ns = "goal_ring"
                m.id = ring_id
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = 0.02
                m.color = color
                m.pose.orientation.w = 1.0
                m.points = _ring_points(tx, ty, radius)
                ma.markers.append(m)

        c_xys = data.get("constraint_xys") or []
        c_r = float(data.get("constraint_radius", 1.0))
        for i, (cx, cy) in enumerate(c_xys):
            disc = Marker()
            disc.header.stamp = stamp
            disc.header.frame_id = self.FRAME
            disc.ns = "constraint_disc"
            disc.id = i
            disc.type = Marker.CYLINDER
            disc.action = Marker.ADD
            disc.pose.position.x = float(cx)
            disc.pose.position.y = float(cy)
            disc.pose.orientation.w = 1.0
            disc.scale.x = c_r * 2.0
            disc.scale.y = c_r * 2.0
            disc.scale.z = 0.01
            disc.color = _color(0.9, 0.1, 0.1, 0.15)
            ma.markers.append(disc)
            ring = Marker()
            ring.header.stamp = stamp
            ring.header.frame_id = self.FRAME
            ring.ns = "constraint_ring"
            ring.id = i
            ring.type = Marker.LINE_STRIP
            ring.action = Marker.ADD
            ring.scale.x = 0.03
            ring.color = _color(0.9, 0.1, 0.1, 0.9)
            ring.pose.orientation.w = 1.0
            ring.points = _ring_points(cx, cy, c_r)
            ma.markers.append(ring)

        p_xys = data.get("preference_xys") or []
        p_r = float(data.get("preference_radius", 1.0))
        for i, (px, py) in enumerate(p_xys):
            disc = Marker()
            disc.header.stamp = stamp
            disc.header.frame_id = self.FRAME
            disc.ns = "preference_disc"
            disc.id = i
            disc.type = Marker.CYLINDER
            disc.action = Marker.ADD
            disc.pose.position.x = float(px)
            disc.pose.position.y = float(py)
            disc.pose.orientation.w = 1.0
            disc.scale.x = p_r * 2.0
            disc.scale.y = p_r * 2.0
            disc.scale.z = 0.01
            disc.color = _color(0.24, 0.52, 0.78, 0.12)
            ma.markers.append(disc)
            ring = Marker()
            ring.header.stamp = stamp
            ring.header.frame_id = self.FRAME
            ring.ns = "preference_ring"
            ring.id = i
            ring.type = Marker.LINE_STRIP
            ring.action = Marker.ADD
            ring.scale.x = 0.03
            ring.color = _color(0.24, 0.52, 0.78, 0.9)
            ring.pose.orientation.w = 1.0
            ring.points = _ring_points(px, py, p_r)
            ma.markers.append(ring)

        if robot:
            mode_color = {
                "RECOV": _color(1.0, 0.5, 0.0),
                "IDLE": _color(0.5, 0.5, 0.5),
            }.get(mode, _color(0.2, 0.8, 0.2))
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.FRAME
            m.ns = "mode_label"
            m.id = 10
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = float(robot[0])
            m.pose.position.y = float(robot[1])
            m.pose.position.z = 0.4
            m.pose.orientation.w = 1.0
            m.scale.z = 0.18
            m.color = mode_color
            m.text = mode
            ma.markers.append(m)

        self._marker_pub.publish(ma)


# ---------------------------------------------------------------------------
# Public API — get (or lazily create) the singleton node
# ---------------------------------------------------------------------------

def get_node() -> SemnavBridgeNode:
    """Return the singleton bridge node, starting it if necessary."""
    global _node, _executor, _spin_thread
    with _init_lock:
        if _node is not None:
            return _node

        if not rclpy.ok():
            rclpy.init()

        _node = SemnavBridgeNode()
        _executor = MultiThreadedExecutor()
        _executor.add_node(_node)

        _spin_thread = threading.Thread(target=_executor.spin, daemon=True, name="semnav_ros_spin")
        _spin_thread.start()

        atexit.register(_shutdown)

        return _node


def _shutdown() -> None:
    """Cleanly stop the executor and shut down rclpy on process exit."""
    global _node, _executor, _spin_thread
    if _executor is not None:
        try:
            _executor.shutdown(wait=False)
        except Exception:
            pass
    if _node is not None:
        try:
            _node.destroy_node()
        except Exception:
            pass
    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass
