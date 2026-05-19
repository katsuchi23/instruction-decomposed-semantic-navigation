"""ZMQ IPC clients for local and global costmaps."""

from __future__ import annotations

import json

import numpy as np
import zmq

from utils.config import get_ipc_endpoint


class LocalCostmapIPCClient:
    """Request/reply client for the local costmap on port 5564."""

    def __init__(self, endpoint: str | None = None) -> None:
        self._ctx = zmq.Context.instance()
        self._endpoint = endpoint or get_ipc_endpoint("local_costmap_req")
        self._sock = self._create_socket()

    def _create_socket(self):
        s = self._ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(self._endpoint)
        return s

    def _reset_socket(self):
        """Destroy and recreate the REQ socket (recovery after timeout)."""
        try:
            self._sock.close()
        except Exception:
            pass
        self._sock = self._create_socket()

    def get(self, timeout_ms: int = 1000):
        try:
            self._sock.send_string("GET")
            if self._sock.poll(timeout_ms) == 0:
                self._reset_socket()
                return None, None
            frames = self._sock.recv_multipart()
        except zmq.ZMQError:
            self._reset_socket()
            return None, None

        if not frames:
            return None, None

        meta = json.loads(frames[0].decode("utf-8"))
        if not meta.get("ok", False) or len(frames) < 2:
            return None, meta

        w = int(meta["width"])
        h = int(meta["height"])
        dtype = np.int16 if meta.get("dtype", "int16") == "int16" else np.int8

        grid = np.frombuffer(frames[1], dtype=dtype)
        if grid.size != w * h:
            return None, meta

        grid = grid.reshape((h, w))
        return grid, meta

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class GlobalCostmapIPCClient:
    """Request/reply client for the global costmap on port 5565."""

    def __init__(self, endpoint: str | None = None) -> None:
        self._ctx = zmq.Context.instance()
        self._endpoint = endpoint or get_ipc_endpoint("global_costmap_req")
        self._sock = self._create_socket()

    def _create_socket(self):
        s = self._ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(self._endpoint)
        return s

    def _reset_socket(self):
        """Destroy and recreate the REQ socket (recovery after timeout)."""
        try:
            self._sock.close()
        except Exception:
            pass
        self._sock = self._create_socket()

    def get(self, timeout_ms: int = 1000):
        try:
            self._sock.send_string("GET")
            if self._sock.poll(timeout_ms) == 0:
                self._reset_socket()
                return None, None
            frames = self._sock.recv_multipart()
        except zmq.ZMQError:
            self._reset_socket()
            return None, None

        if not frames:
            return None, None

        meta = json.loads(frames[0].decode("utf-8"))
        if not meta.get("ok", False) or len(frames) < 2:
            return None, meta

        w = int(meta["width"])
        h = int(meta["height"])
        dtype = np.int16 if meta.get("dtype", "int16") == "int16" else np.int8

        grid = np.frombuffer(frames[1], dtype=dtype)
        if grid.size != w * h:
            return None, meta

        grid = grid.reshape((h, w))
        return grid, meta

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
