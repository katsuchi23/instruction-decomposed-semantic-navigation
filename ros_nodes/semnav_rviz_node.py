#!/usr/bin/env python3
"""Receives navigation visualization data from the semantic navigation stack via ZMQ
and republishes it as RViz-compatible ROS topics.

Subscribes to: tcp://127.0.0.1:5570 (ZMQ SUB, prefix "VIZ ")

Publishes:
  /semnav/path              nav_msgs/Path              — global A* planned path
  /semnav/traj_samples      visualization_msgs/MarkerArray — sampled trajectory rollouts
  /semnav/markers           visualization_msgs/MarkerArray — robot, start, target, goal,
                                                             lookahead, goal ring
"""

import json
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3
import zmq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaw_to_quat(yaw: float):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.x, q.y, q.z, q.w = 0.0, 0.0, sy, cy
    return q


def _color(r, g, b, a=1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def _point(x, y, z=0.01) -> Point:
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


def _ring_points(cx, cy, radius, n=64) -> list:
    return [_point(cx + radius * math.cos(a), cy + radius * math.sin(a))
            for a in (2 * math.pi * i / n for i in range(n + 1))]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SemnavRvizNode(Node):
    FRAME = "map"

    def __init__(self):
        super().__init__("semnav_rviz_node")

        self._path_pub   = self.create_publisher(Path,        "/semnav/path",         10)
        self._traj_pub   = self.create_publisher(MarkerArray, "/semnav/traj_samples",  10)
        self._marker_pub = self.create_publisher(MarkerArray, "/semnav/markers",       10)

        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.SUB)
        self._sock.connect("tcp://127.0.0.1:5570")
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "VIZ ")
        self._sock.setsockopt(zmq.CONFLATE, 1)

        self.create_timer(0.05, self._poll)   # 20 Hz poll
        self.get_logger().info("semnav_rviz_node ready — subscribing to tcp://127.0.0.1:5570")

    # ---- IPC poll --------------------------------------------------------

    def _poll(self):
        if self._sock.poll(timeout=0) == 0:
            return
        try:
            raw = self._sock.recv_string(flags=zmq.NOBLOCK)
            data = json.loads(raw[len("VIZ "):])
            now = self.get_clock().now().to_msg()
            self._publish_path(data, now)
            self._publish_traj_samples(data, now)
            self._publish_markers(data, now)
        except Exception as exc:
            self.get_logger().warn(f"semnav_rviz_node poll error: {exc}")

    # ---- /semnav/path ----------------------------------------------------

    def _publish_path(self, data, stamp):
        path_pts = data.get("path") or []
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = self.FRAME
        for x, y in path_pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)

    # ---- /semnav/traj_samples --------------------------------------------

    def _publish_traj_samples(self, data, stamp):
        seqs = data.get("traj_samples") or []
        ma = MarkerArray()
        # Delete stale markers first
        del_m = Marker()
        del_m.action = Marker.DELETEALL
        ma.markers.append(del_m)
        for i, pts in enumerate(seqs):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.FRAME
            m.ns = "traj_samples"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.01
            m.color = _color(0.96, 0.32, 0.12, 0.25)  # orange, semi-transparent
            m.pose.orientation.w = 1.0
            for x, y in pts:
                m.points.append(_point(x, y))
            if len(m.points) > 1:
                ma.markers.append(m)
        self._traj_pub.publish(ma)

    # ---- /semnav/markers -------------------------------------------------

    def _publish_markers(self, data, stamp):
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
            if yaw is not None:
                m.pose.orientation = _yaw_to_quat(yaw)
            else:
                m.pose.orientation.w = 1.0
            m.scale.x = scale
            m.scale.y = scale * 0.5 if mtype == Marker.ARROW else scale
            m.scale.z = scale * 0.3 if mtype == Marker.ARROW else 0.05
            m.color = color
            if text:
                m.text = text
            ma.markers.append(m)

        robot = data.get("robot")        # [x, y, yaw]
        start = data.get("start")        # [x, y, yaw]
        target = data.get("target")      # [x, y]
        goal   = data.get("goal")        # [x, y]
        lh     = data.get("lookahead")   # [x, y] or None
        r_min  = float(data.get("r_min", 0.4))
        r_max  = float(data.get("r_max", 0.8))
        mode   = data.get("mode", "NAV")
        task_name = data.get("task_name", "")

        # Robot — green arrow
        if robot:
            _add("robot", 0, Marker.ARROW, robot,
                 _color(0.13, 0.55, 0.13), scale=0.25, yaw=robot[2])

        # Start — dark green sphere
        if start:
            _add("start", 1, Marker.SPHERE, start,
                 _color(0.0, 0.39, 0.0), scale=0.12)

        # Target object — red cylinder
        if target:
            _add("target", 2, Marker.CYLINDER, target,
                 _color(0.9, 0.1, 0.1), scale=0.18)

        # Active goal — blue sphere
        if goal:
            _add("goal", 3, Marker.SPHERE, goal,
                 _color(0.1, 0.1, 0.9), scale=0.14)

        # Lookahead — cyan sphere
        if lh:
            _add("lookahead", 4, Marker.SPHERE, lh,
                 _color(0.0, 0.85, 0.85), scale=0.10)

        # Goal ring outer — green LINE_STRIP
        if target:
            tx, ty = target[0], target[1]
            for ring_id, radius, color in [
                (5, r_max, _color(0.18, 0.80, 0.18, 0.8)),
                (6, r_min, _color(1.0,  0.60, 0.0,  0.8)),
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

        # Constraint objects — red filled circle + red ring
        c_xys = data.get("constraint_xys") or []
        c_r = float(data.get("constraint_radius", 1.0))
        for i, (cx, cy) in enumerate(c_xys):
            # Filled disc (semi-transparent red)
            disc = Marker()
            disc.header.stamp = stamp
            disc.header.frame_id = self.FRAME
            disc.ns = "constraint_disc"
            disc.id = i
            disc.type = Marker.CYLINDER
            disc.action = Marker.ADD
            disc.pose.position.x = float(cx)
            disc.pose.position.y = float(cy)
            disc.pose.position.z = 0.0
            disc.pose.orientation.w = 1.0
            disc.scale.x = c_r * 2.0
            disc.scale.y = c_r * 2.0
            disc.scale.z = 0.01
            disc.color = _color(0.9, 0.1, 0.1, 0.15)
            ma.markers.append(disc)
            # Border ring
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

        # Preference objects — teal filled circle + teal ring
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
            disc.pose.position.z = 0.0
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

        # Mode text label near robot
        if robot:
            mode_color = {"RECOV": _color(1.0, 0.5, 0.0), "IDLE": _color(0.5, 0.5, 0.5)}.get(
                mode, _color(0.2, 0.8, 0.2))
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

def main():
    rclpy.init()
    node = SemnavRvizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
