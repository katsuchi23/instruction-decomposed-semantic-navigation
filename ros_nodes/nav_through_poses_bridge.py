#!/usr/bin/env python3
"""ROS2 bridge for nav_through_pose mode.

ZMQ endpoints:
  5566 REP  — receive NavigateThroughPoses goal, reply OK/ERR
  5567 PUB  — stream /cmd_vel as "CMD_VEL {json}"
  5568 PUB  — stream /amcl_pose as "AMCL_POSE {json}"
  5569 PUB  — stream NavigateThroughPoses status as "NAV_THROUGH_STATUS {json}"
"""

import math

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateThroughPoses
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import zmq
import json


class NavThroughPosesBridge(Node):
    def __init__(self):
        super().__init__("nav_through_poses_bridge")

        self._action_client = ActionClient(self, NavigateThroughPoses, "navigate_through_poses")

        ctx = zmq.Context.instance()

        self._goal_sock = ctx.socket(zmq.REP)
        self._goal_sock.bind("tcp://127.0.0.1:5566")

        self._cmd_vel_pub = ctx.socket(zmq.PUB)
        self._cmd_vel_pub.bind("tcp://127.0.0.1:5567")

        self._amcl_pub = ctx.socket(zmq.PUB)
        self._amcl_pub.bind("tcp://127.0.0.1:5568")

        self._status_pub = ctx.socket(zmq.PUB)
        self._status_pub.bind("tcp://127.0.0.1:5569")

        self._nav_status = "IDLE"
        self._nav_status_code = int(GoalStatus.STATUS_UNKNOWN)
        self._nav_detail = "No goal sent yet."

        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl_pose, 10
        )

        self.create_timer(0.01, self._poll_goal_requests)
        self.create_timer(0.1, self._publish_status)

        self.get_logger().info("NavThroughPosesBridge: waiting for navigate_through_poses...")
        self._action_client.wait_for_server()
        self.get_logger().info("NavThroughPosesBridge: ready.")

    # ---- ROS subscribers -------------------------------------------------------

    def _on_cmd_vel(self, msg: Twist) -> None:
        try:
            payload = json.dumps({"v": msg.linear.x, "w": msg.angular.z})
            self._cmd_vel_pub.send_string(f"CMD_VEL {payload}", flags=zmq.NOBLOCK)
        except Exception:
            pass

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        try:
            q = msg.pose.pose.orientation
            yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
            payload = json.dumps({
                "x": msg.pose.pose.position.x,
                "y": msg.pose.pose.position.y,
                "yaw": yaw,
            })
            self._amcl_pub.send_string(f"AMCL_POSE {payload}", flags=zmq.NOBLOCK)
        except Exception:
            pass

    # ---- ZMQ REP: receive goal -------------------------------------------------

    def _poll_goal_requests(self) -> None:
        try:
            if self._goal_sock.poll(timeout=0) == 0:
                return
            raw = self._goal_sock.recv_string()
            data = json.loads(raw)
            poses_data = data.get("poses", [])
            if not poses_data:
                self._goal_sock.send_string("ERR: empty pose list")
                return

            goal = NavigateThroughPoses.Goal()
            for p in poses_data:
                ps = PoseStamped()
                ps.header.frame_id = "map"
                ps.header.stamp = self.get_clock().now().to_msg()
                ps.pose.position.x = float(p["x"])
                ps.pose.position.y = float(p["y"])
                q = R.from_euler("z", float(p["yaw"])).as_quat()
                ps.pose.orientation.x = float(q[0])
                ps.pose.orientation.y = float(q[1])
                ps.pose.orientation.z = float(q[2])
                ps.pose.orientation.w = float(q[3])
                goal.poses.append(ps)

            self._nav_status = "ACTIVE"
            self._nav_status_code = int(GoalStatus.STATUS_ACCEPTED)
            self._nav_detail = f"Sending {len(poses_data)} poses to navigate_through_poses."

            future = self._action_client.send_goal_async(goal)
            future.add_done_callback(self._on_goal_response)

            self._goal_sock.send_string("OK")
        except Exception as exc:
            self.get_logger().error(f"_poll_goal_requests: {exc}")
            try:
                self._goal_sock.send_string(f"ERR: {exc}")
            except Exception:
                pass

    def _on_goal_response(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("NavigateThroughPoses: goal rejected.")
            self._nav_status = "FAILED"
            self._nav_status_code = int(GoalStatus.STATUS_UNKNOWN)
            self._nav_detail = "Goal rejected by navigate_through_poses action server."
            return
        self.get_logger().info("NavigateThroughPoses: goal accepted.")
        self._nav_status = "ACTIVE"
        self._nav_status_code = int(GoalStatus.STATUS_EXECUTING)
        self._nav_detail = "Goal executing."
        handle.get_result_async().add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        status = future.result().status
        self._nav_status_code = int(status)
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._nav_status = "SUCCEEDED"
            self._nav_detail = "navigate_through_poses STATUS_SUCCEEDED."
            self.get_logger().info("NavigateThroughPoses: succeeded.")
        elif status == GoalStatus.STATUS_CANCELED:
            self._nav_status = "CANCELED"
            self._nav_detail = "navigate_through_poses STATUS_CANCELED."
        else:
            self._nav_status = "FAILED"
            self._nav_detail = f"navigate_through_poses status code {status}."
            self.get_logger().warn(f"NavigateThroughPoses: failed with status {status}.")

    # ---- ZMQ PUB: status -------------------------------------------------------

    def _publish_status(self) -> None:
        try:
            payload = json.dumps({
                "status": self._nav_status,
                "status_code": self._nav_status_code,
                "detail": self._nav_detail,
            })
            self._status_pub.send_string(f"NAV_THROUGH_STATUS {payload}", flags=zmq.NOBLOCK)
        except Exception:
            pass


def main():
    rclpy.init()
    node = NavThroughPosesBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
