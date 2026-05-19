"""Shared math utilities — single canonical definitions."""

from __future__ import annotations

import math


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp *x* to the interval [lo, hi]."""
    return max(lo, min(hi, x))


def wrap_angle(a: float) -> float:
    """Wrap angle *a* to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def cm_to_m(cm: float) -> float:
    """Convert centimetres to metres."""
    return cm / 100.0
