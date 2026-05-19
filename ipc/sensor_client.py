"""ZMQ IPC clients for camera images, depth, lidar, and segmented-image publishing."""

from __future__ import annotations

import json
from typing import Optional

import cv2
import numpy as np
import zmq

from utils.config import get_ipc_endpoint

_ctx = zmq.Context.instance()

# ---------------------------------------------------------------------------
# Singleton subscribers
# ---------------------------------------------------------------------------
_scan_sub = None


def _ensure_scan_sub():
    global _scan_sub
    if _scan_sub is not None:
        return _scan_sub
    s = _ctx.socket(zmq.SUB)
    s.connect(get_ipc_endpoint("scan_sub"))
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.setsockopt(zmq.CONFLATE, 1)
    _scan_sub = s
    return _scan_sub


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_image_via_ipc(timeout_ms: int = 2000) -> Optional[np.ndarray]:
    """Retrieve a BGR image from port 5555."""
    sock = _ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(get_ipc_endpoint("image_req"))
    try:
        sock.send_string("GET")
        if sock.poll(timeout_ms) == 0:
            return None
        jpg = sock.recv()
        if not jpg:
            return None
        arr = np.frombuffer(jpg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return bgr
    finally:
        sock.close()


def get_depth_image_array_via_ipc(timeout_ms: int = 2000) -> Optional[np.ndarray]:
    """Retrieve a float32 depth array (360x640) from port 5561."""
    sock = _ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(get_ipc_endpoint("depth_req"))
    try:
        sock.send_string("GET")
        if sock.poll(timeout_ms) == 0:
            return None
        buf = sock.recv()
        if not buf:
            return None
        arr = np.frombuffer(buf, dtype=np.float32)
        arr = arr.reshape((360, 640))
        return arr
    finally:
        sock.close()


def get_laser_scan_via_ipc(timeout_ms: int = 100):
    """Return a dict with ``ranges`` (np.array) and scan metadata, or ``None``."""
    s = _ensure_scan_sub()
    if s.poll(timeout_ms) == 0:
        return None
    msg = s.recv_string()
    data = json.loads(msg)
    if "ranges" in data:
        data["ranges"] = np.array(data["ranges"], dtype=np.float32)
    return data
