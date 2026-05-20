#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import zmq
import json
import math

class TrajectoryVisualizationNode(Node):
    def __init__(self):
        super().__init__('trajectory_visualization_node')
        
        # Publisher for the trajectory path
        self.path_pub = self.create_publisher(Path, '/projected_path', 10)
        
        ctx = zmq.Context.instance()
        # REP socket for receiving trajectory requests
        self.sock = ctx.socket(zmq.REP)
        self.sock.bind("tcp://127.0.0.1:5563")
        
        self.get_logger().info("TrajectoryVisualizationNode: ZMQ REP on tcp://127.0.0.1:5563 publishing to /projected_path")
        
        # Check explicitly for messages frequently
        self.create_timer(0.01, self.poll_requests)

    def poll_requests(self):
        try:
            # Check for incoming messages non-blocking
            if self.sock.poll(timeout=0) == 0:
                return
            
            raw = self.sock.recv_string()
            data = json.loads(raw)
            poses_list = data.get("poses", [])
            
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = "base_link"
            
            for p in poses_list:
                ps = PoseStamped()
                ps.header = path_msg.header
                ps.pose.position.x = float(p.get("x", 0.0))
                ps.pose.position.y = float(p.get("y", 0.0))
                ps.pose.position.z = 0.0
                
                yaw = float(p.get("yaw", 0.0))
                # Yaw to Quaternion conversion 
                # (q_w, q_x, q_y, q_z) = (cos(yaw/2), 0, 0, sin(yaw/2))
                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                
                ps.pose.orientation.x = 0.0
                ps.pose.orientation.y = 0.0
                ps.pose.orientation.z = sy
                ps.pose.orientation.w = cy
                
                path_msg.poses.append(ps)
            
            self.path_pub.publish(path_msg)
            
            # Simple acknowledgment
            self.sock.send_string("OK")
            
        except Exception as e:
            self.get_logger().error(f"poll_requests error: {e}")
            # If processing failed but we received the request, we should probably still reply
            # to prevent the client from hanging, or the client should have a timeout.
            # In this simple implementation, if an exception occurs before send_string, 
            # the client will timeout.
            # Try to send error if possible:
            try:
                self.sock.send_string("ERROR")
            except:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
