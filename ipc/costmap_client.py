"""Costmap clients — direct ROS (replaces ZMQ IPC).

Same public API as before; callers need no changes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _node():
    from ros.ros_bridge import get_node
    return get_node()


class LocalCostmapIPCClient:
    """Drop-in replacement — reads the latest cached costmap from the ROS bridge."""

    def get(self, timeout_ms: int = 1000) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        return _node().get_local_costmap()

    def close(self) -> None:
        pass  # nothing to close

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        pass


class GlobalCostmapIPCClient:
    """Drop-in replacement — reads the latest cached costmap from the ROS bridge."""

    def get(self, timeout_ms: int = 1000) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        return _node().get_global_costmap()

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        pass
