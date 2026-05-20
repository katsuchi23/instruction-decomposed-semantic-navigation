#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import zmq
import json

class CmdVelSender(Node):
    def __init__(self):
        super().__init__('cmd_vel_sender')

        cmd_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel').value

        # Publisher for direct velocity control
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_topic, 10)

        self._ctx = zmq.Context.instance()
        self._endpoint = "tcp://127.0.0.1:5559"
        self.sock = None
        self._bind_socket()

        # Store the current velocity command
        self.current_twist = Twist()
        self.last_req_time = self.get_clock().now()
        
        self.get_logger().info(
            f"CmdVelSender: ZMQ REP on tcp://127.0.0.1:5559 receiving updates. "
            f"Publishing {cmd_topic} at 30Hz."
        )

        # Timer 1: Poll for ZMQ requests frequently (e.g., 100 Hz) to be responsive
        self.create_timer(0.01, self.poll_requests)
        
        # Timer 2: Publish the stored command at a fixed rate (30 Hz)
        self.create_timer(1.0/30.0, self.publish_loop)

    def _bind_socket(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = self._ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(self._endpoint)

    def _reset_socket(self, reason: str):
        self.get_logger().warn(f"Resetting ZMQ REP socket on {self._endpoint}: {reason}")
        self._bind_socket()

    def poll_requests(self):
        try:
            # Check for incoming messages non-blocking
            if self.sock.poll(timeout=0) == 0:
                return

            raw = self.sock.recv_string(flags=zmq.NOBLOCK)
            data = json.loads(raw)
            
            linear_x = float(data.get("v", 0.0))
            angular_z = float(data.get("w", 0.0))
            
            # Update the stored twist message
            self.current_twist.linear.x = linear_x
            self.current_twist.angular.z = angular_z
            self.last_req_time = self.get_clock().now()

            # Send acknowledgment back to the client
            self.sock.send_string("OK", flags=zmq.DONTWAIT)

        except zmq.Again:
            return
        except zmq.ZMQError as e:
            self.get_logger().error(f"poll_requests ZMQ error: {e}")
            self._reset_socket(str(e))
        except Exception as e:
            self.get_logger().error(f"poll_requests error: {e}")
            # Try to send error if possible, but if recv failed might be bad state
            # If socket is in REQ/REP cadence, we MUST send a reply if we received a request.
            # If recv failed, we probably didn't get the request fully or Parse error.
            # If parse error, we still have the socket ready to send reply.
            try:
                self.sock.send_string("ERR", flags=zmq.DONTWAIT)
            except Exception as send_exc:
                self._reset_socket(f"ERR reply failed: {send_exc}")

    def publish_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_req_time).nanoseconds / 1e9

        # Safety: if no new value in > 5.0s, dormant (stop publishing)
        if dt > 5.0:
            return

        # Safety: if no new value in > 0.2s, publish 0
        if dt > 0.2:
            self.cmd_vel_pub.publish(Twist()) # zero velocity
        else:
            # Publish the currently stored twist message
            self.cmd_vel_pub.publish(self.current_twist)

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
