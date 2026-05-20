#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from scipy.spatial.transform import Rotation as R
import zmq
import json

class Nav2Sender(Node):
    def __init__(self):
        super().__init__('nav2_sender')
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        ctx = zmq.Context.instance()
        
        # REP socket for receiving goal requests
        self.sock = ctx.socket(zmq.REP)
        self.sock.bind("tcp://127.0.0.1:5556")
        
        # PUB socket for publishing goal status
        self.status_sock = ctx.socket(zmq.PUB)
        self.status_sock.bind("tcp://127.0.0.1:5558")

        self.goal_status = "IDLE"  # IDLE, ACTIVE, SUCCEEDED, FAILED, CANCELED
        self.current_goal_handle = None
        self.goal_status_code = int(GoalStatus.STATUS_UNKNOWN)
        self.goal_status_detail = "No goal sent yet."

        self.get_logger().info("Nav2Sender: ZMQ REP on tcp://127.0.0.1:5556")
        self.get_logger().info("Nav2Sender: ZMQ PUB (status) on tcp://127.0.0.1:5558")
        self.get_logger().info("Nav2Sender: waiting for navigate_to_pose...")

        self.client.wait_for_server()

        self.create_timer(0.01, self.poll_requests)
        self.create_timer(0.1, self.publish_status)  # Publish status at 10Hz

    def poll_requests(self):
        try:
            if self.sock.poll(timeout=0) == 0:
                return
            raw = self.sock.recv_string()
            data = json.loads(raw)
            x, y, yaw = float(data["x"]), float(data["y"]), float(data["yaw"])

            goal = NavigateToPose.Goal()
            goal.pose = PoseStamped()
            goal.pose.header.frame_id = "map"
            goal.pose.header.stamp = self.get_clock().now().to_msg()
            goal.pose.pose.position.x = x
            goal.pose.pose.position.y = y

            q = R.from_euler('z', yaw).as_quat()  # x,y,z,w
            goal.pose.pose.orientation.x = float(q[0])
            goal.pose.pose.orientation.y = float(q[1])
            goal.pose.pose.orientation.z = float(q[2])
            goal.pose.pose.orientation.w = float(q[3])

            # Send goal and track status
            self.goal_status = "ACTIVE"
            self.goal_status_code = int(GoalStatus.STATUS_ACCEPTED)
            self.goal_status_detail = (
                f"Goal sent to navigate_to_pose: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} rad"
            )
            send_goal_future = self.client.send_goal_async(goal)
            send_goal_future.add_done_callback(self.goal_response_callback)
            
            self.sock.send_string("OK")
        except Exception as e:
            self.get_logger().error(f"poll_requests error: {e}")
            self.sock.send_string("ERR")
    
    def goal_response_callback(self, future):
        """Callback when goal is accepted/rejected."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            self.goal_status = "FAILED"
            self.goal_status_code = int(GoalStatus.STATUS_UNKNOWN)
            self.goal_status_detail = "Goal rejected by NavigateToPose action server."
            return
        
        self.get_logger().info('Goal accepted')
        self.current_goal_handle = goal_handle
        self.goal_status = "ACTIVE"
        self.goal_status_code = int(GoalStatus.STATUS_ACCEPTED)
        self.goal_status_detail = "Goal accepted by NavigateToPose action server."
        
        # Get result future
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        """Callback when goal completes."""
        result = future.result()
        status = result.status
        self.goal_status_code = int(status)
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('✓ Goal succeeded!')
            self.goal_status = "SUCCEEDED"
            self.goal_status_detail = "NavigateToPose action reported STATUS_SUCCEEDED."
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn('Goal aborted')
            self.goal_status = "FAILED"
            self.goal_status_detail = "NavigateToPose action reported STATUS_ABORTED."
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn('Goal canceled')
            self.goal_status = "CANCELED"
            self.goal_status_detail = "NavigateToPose action reported STATUS_CANCELED."
        else:
            self.get_logger().warn(f'Goal failed with status: {status}')
            self.goal_status = "FAILED"
            self.goal_status_detail = f"NavigateToPose action reported status code {status}."
    
    def publish_status(self):
        """Publish current goal status via ZMQ."""
        try:
            status_msg = json.dumps({
                "status": self.goal_status,
                "status_code": int(self.goal_status_code),
                "detail": self.goal_status_detail,
            })
            self.status_sock.send_string(f"NAV_STATUS {status_msg}")
        except Exception as e:
            self.get_logger().error(f"Error publishing status: {e}")

def main():
    rclpy.init()
    node = Nav2Sender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
