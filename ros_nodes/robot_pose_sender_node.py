#!/usr/bin/env python3
import json
import math
import os
import zmq

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def quat_to_yaw(x, y, z, w) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class RobotPoseSender(Node):
    def __init__(self):
        super().__init__("robot_pose_sender")

        # Frames
        self.map_frame = self.declare_parameter("map_frame", "map").value
        self.odom_frame = self.declare_parameter("odom_frame", "odom").value
        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        self.camera_frame = self.declare_parameter("camera_frame", "camera_link").value

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ZMQ PUB
        ctx = zmq.Context.instance()
        self.pub = ctx.socket(zmq.PUB)
        self.pub.bind("tcp://127.0.0.1:5557")

        self.get_logger().info(
            f"RobotPoseSender: streaming TF at 10 Hz: "
            f"{self.map_frame}->{self.base_frame}, {self.map_frame}->{self.odom_frame}, "
            f"{self.odom_frame}->{self.base_frame}, {self.map_frame}->{self.camera_frame}"
        )
        self.get_logger().info("RobotPoseSender: ZMQ PUB on tcp://127.0.0.1:5557")

        self.create_timer(0.1, self.publish_pose)  # 10 Hz

        self._warn_count = 0
        self._diag_enabled = os.getenv("SEMNAV_POSE_IPC_DIAG", "").strip().lower() not in {"", "0", "false", "no"}
        self._diag_last_signature = None
        self._diag_tick = 0

    def _lookup(self, parent: str, child: str):
        try:
            return self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self._warn_count += 1
            if self._warn_count % 20 == 1:
                self.get_logger().warn(
                    f"TF lookup failed for {parent}->{child}: {type(e).__name__}: {e}"
                )
            return None

    def _tf_to_dict(self, tf_msg, parent: str, child: str):
        tr = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        return {
            "parent": parent,
            "child": child,
            "x": float(tr.x),
            "y": float(tr.y),
            "z": float(tr.z),
            "yaw": float(yaw),
            "stamp_sec": int(tf_msg.header.stamp.sec),
            "stamp_nanosec": int(tf_msg.header.stamp.nanosec),
        }

    def publish_pose(self):
        self._diag_tick += 1
        # Lookup all three transforms independently
        tf_map_base = self._lookup(self.map_frame, self.base_frame)
        tf_map_odom = self._lookup(self.map_frame, self.odom_frame)
        tf_odom_base = self._lookup(self.odom_frame, self.base_frame)
        tf_map_camera = self._lookup(self.map_frame, self.camera_frame)

        # Build payload with whatever is available this tick
        payload = {
            "type": "TF_BUNDLE",
            "map_frame": self.map_frame,
            "odom_frame": self.odom_frame,
            "base_frame": self.base_frame,
            "tfs": {},
        }

        if tf_map_base is not None:
            payload["tfs"]["map_base"] = self._tf_to_dict(tf_map_base, self.map_frame, self.base_frame)
        if tf_map_odom is not None:
            payload["tfs"]["map_odom"] = self._tf_to_dict(tf_map_odom, self.map_frame, self.odom_frame)
        if tf_odom_base is not None:
            payload["tfs"]["odom_base"] = self._tf_to_dict(tf_odom_base, self.odom_frame, self.base_frame)
        if tf_map_camera is not None:
            payload["tfs"]["map_camera"] = self._tf_to_dict(tf_map_camera, self.map_frame, self.camera_frame)

        # If nothing available, do nothing
        if not payload["tfs"]:
            return

        if self._diag_enabled:
            def _sig(tf_msg):
                if tf_msg is None:
                    return None
                tr = tf_msg.transform.translation
                q = tf_msg.transform.rotation
                return (
                    round(float(tr.x), 4),
                    round(float(tr.y), 4),
                    round(float(quat_to_yaw(q.x, q.y, q.z, q.w)), 4),
                    int(tf_msg.header.stamp.sec),
                    int(tf_msg.header.stamp.nanosec),
                )

            diag_signature = (
                _sig(tf_map_base),
                _sig(tf_map_odom),
                _sig(tf_odom_base),
            )
            if diag_signature != self._diag_last_signature or (self._diag_tick % 20 == 0):
                self.get_logger().info(
                    "POSE_IPC_DIAG publish "
                    f"map_base={diag_signature[0]} "
                    f"map_odom={diag_signature[1]} "
                    f"odom_base={diag_signature[2]}"
                )
                self._diag_last_signature = diag_signature

        try:
            self.pub.send_string("TF " + json.dumps(payload))
        except Exception as e:
            self.get_logger().error(f"publish_pose error: {e}")


def main():
    rclpy.init()
    node = RobotPoseSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
