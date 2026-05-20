#!/usr/bin/env python3
import json
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
import zmq


class GlobalCostmapRelay(Node):
    """
    Subscribes to /global_costmap/costmap (nav_msgs/OccupancyGrid) and serves latest grid via ZMQ REP.

    ZMQ protocol:
      Client sends: "GET"
      Server replies multipart:
        frame 0: JSON metadata
        frame 1: raw grid bytes (row-major, int16)
    """

    def __init__(self):
        super().__init__("global_costmap_relay")

        # Parameters
        self.declare_parameter("grid_topic", "/global_costmap/costmap")
        self.declare_parameter("zmq_port", 5565)

        self.grid_topic = self.get_parameter("grid_topic").value
        self.zmq_port = int(self.get_parameter("zmq_port").value)

        # Match Nav2 publisher QoS: RELIABLE + TRANSIENT_LOCAL
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub = self.create_subscription(
            OccupancyGrid,
            self.grid_topic,
            self.grid_callback,
            qos
        )

        self.latest_grid: Optional[np.ndarray] = None
        self.latest_info: Optional[dict] = None
        self.lock = threading.Lock()
        self.recv_count = 0

        # ZMQ server
        self.zmq_ctx = zmq.Context()
        self.zmq_socket = self.zmq_ctx.socket(zmq.REP)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_socket.bind(f"tcp://*:{self.zmq_port}")

        self.zmq_thread = threading.Thread(target=self.zmq_loop, daemon=True)
        self.zmq_thread.start()

        self.create_timer(1.0, self._status_timer)

        self.get_logger().info(f"Subscribed to: {self.grid_topic}")
        self.get_logger().info(f"ZMQ REP server on tcp://*:{self.zmq_port}")

    def _status_timer(self):
        if self.recv_count == 0:
            self.get_logger().warn(
                f"No messages received yet on {self.grid_topic}. "
                f"If Nav2 is running, this is almost always QoS or wrong topic."
            )

    def grid_callback(self, msg: OccupancyGrid):
        w = int(msg.info.width)
        h = int(msg.info.height)

        # Convert to int16 for safety and consistent binary transport
        data = np.asarray(msg.data, dtype=np.int16)
        if data.size != w * h:
            self.get_logger().warn(f"Grid size mismatch: data={data.size} vs w*h={w*h}")
            return

        grid = data.reshape((h, w))  # grid[y][x], row-major
        self.recv_count += 1

        with self.lock:
            self.latest_grid = grid.copy(order="C")
            self.latest_info = {
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
                "topic": self.grid_topic,
            }

    def zmq_loop(self):
        while rclpy.ok():
            try:
                if not self.zmq_socket.poll(100):
                    continue

                req = self.zmq_socket.recv_string()

                if req == "GET":
                    with self.lock:
                        if self.latest_grid is None or self.latest_info is None:
                            meta = {"ok": False, "error": "no_grid_yet", "topic": self.grid_topic}
                            self.zmq_socket.send_multipart([json.dumps(meta).encode("utf-8")])
                            continue

                        meta_bytes = json.dumps(self.latest_info).encode("utf-8")
                        payload = self.latest_grid.tobytes(order="C")

                    self.zmq_socket.send_multipart([meta_bytes, payload])

                elif req == "PING":
                    self.zmq_socket.send_multipart([b"PONG"])

                else:
                    meta = {"ok": False, "error": "unknown_command"}
                    self.zmq_socket.send_multipart([json.dumps(meta).encode("utf-8")])

            except Exception as e:
                try:
                    self.get_logger().error(f"ZMQ loop error: {e}")
                except Exception:
                    pass
                time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalCostmapRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
