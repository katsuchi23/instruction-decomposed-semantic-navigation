#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import zmq
import json
import numpy as np

class LaserScanSender(Node):
    def __init__(self):
        super().__init__('laserscan_sender')
        
        # Subscribe to /scan
        # QoS 10 is standard best effort/reliable mapping
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Setup ZMQ PUB socket
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        # Bind to port 5562
        self.sock.bind("tcp://*:5562")
        
        self.get_logger().info("LaserScan IPC Sender running on tcp://*:5562")

    def scan_callback(self, msg):
        try:
            # We construct a dictionary. 
            # Note: Msg.ranges might contain 'inf' or 'nan'.
            # Python's json.dumps produces NaN/Infinity. 
            # Python's json.loads accepts them. 
            # This is fine for python-to-python IPC.
            
            data = {
                "header_stamp_sec": msg.header.stamp.sec,
                "header_stamp_nanosec": msg.header.stamp.nanosec,
                "frame_id": msg.header.frame_id,
                "angle_min": msg.angle_min,
                "angle_max": msg.angle_max,
                "angle_increment": msg.angle_increment,
                "range_min": msg.range_min,
                "range_max": msg.range_max,
                "ranges": list(msg.ranges), # Convert array.array/tuple to list
                "intensities": list(msg.intensities)
            }
            
            # Serialize
            json_str = json.dumps(data)
            
            # Publish
            self.sock.send_string(json_str)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing scan: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LaserScanSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
