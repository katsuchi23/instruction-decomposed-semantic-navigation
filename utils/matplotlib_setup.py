"""Helpers for selecting an interactive matplotlib backend at runtime."""

from __future__ import annotations

import os
import sys
from typing import Dict


_NON_GUI_BACKENDS = {
    "agg",
    "pdf",
    "ps",
    "svg",
    "template",
    "cairo",
}
_LAST_BACKEND_ATTEMPTS: Dict[str, str] = {}


def backend_supports_windows() -> bool:
    try:
        import matplotlib
    except Exception:
        return False

    backend = str(matplotlib.get_backend() or "").strip().lower()
    if not backend:
        return False
    if backend in _NON_GUI_BACKENDS:
        return False
    if "inline" in backend:
        return False
    if backend.startswith("module://matplotlib_inline"):
        return False
    return True


def _can_create_figure() -> bool:
    if not backend_supports_windows():
        return False
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.close(fig)
        return True
    except Exception:
        return False


def ensure_interactive_backend() -> bool:
    """Prefer a GUI backend when a display is available.

    The DovSG conda environment defaults to ``agg``, which suppresses the live
    costmap window. Override that only when a display is present and pyplot has
    not been imported yet.
    """
    global _LAST_BACKEND_ATTEMPTS
    _LAST_BACKEND_ATTEMPTS = {}

    if not os.environ.get("DISPLAY"):
        _LAST_BACKEND_ATTEMPTS["DISPLAY"] = "DISPLAY is unset"
        return False

    import matplotlib

    if "matplotlib.pyplot" in sys.modules:
        try:
            import matplotlib.pyplot as plt

            if not _can_create_figure():
                for candidate in ("TkAgg", "QtAgg", "Qt5Agg"):
                    try:
                        plt.switch_backend(candidate)
                        if _can_create_figure():
                            return True
                    except Exception as exc:
                        _LAST_BACKEND_ATTEMPTS[f"switch:{candidate}"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
        except Exception as exc:
            _LAST_BACKEND_ATTEMPTS["pyplot_preimported"] = (
                f"pyplot preimported but switch attempt failed: {type(exc).__name__}: {exc}"
            )

        ok = _can_create_figure()
        if not ok:
            _LAST_BACKEND_ATTEMPTS["pyplot_preimported"] = (
                "matplotlib.pyplot already imported and figure creation failed"
            )
        return ok

    for candidate in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            matplotlib.use(candidate, force=True)
            import matplotlib.pyplot as plt  # noqa: F401
            if str(plt.get_backend()).lower() == candidate.lower() and _can_create_figure():
                return True
            _LAST_BACKEND_ATTEMPTS[candidate] = (
                f"selected backend={plt.get_backend()} but figure creation failed"
            )
        except Exception as exc:
            _LAST_BACKEND_ATTEMPTS[candidate] = f"{type(exc).__name__}: {exc}"
        sys.modules.pop("matplotlib.pyplot", None)
    ok = _can_create_figure()
    if not ok and "fallback" not in _LAST_BACKEND_ATTEMPTS:
        _LAST_BACKEND_ATTEMPTS["fallback"] = "final _can_create_figure() returned False"
    return ok


def show_live_figure(fig) -> bool:
    """Force a matplotlib figure window to become visible."""
    if not backend_supports_windows():
        return False
    shown = False
    try:
        fig.show()
        shown = True
    except Exception:
        pass
    try:
        manager = getattr(fig.canvas, "manager", None)
        window = getattr(manager, "window", None)
        if window is not None:
            for method_name in ("deiconify", "lift", "focus_force"):
                method = getattr(window, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        shown = True
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt

        plt.show(block=False)
        plt.pause(0.05)
        shown = True
    except Exception:
        pass
    return shown


def safe_pause(interval: float = 0.001) -> bool:
    if not backend_supports_windows():
        return False
    try:
        import matplotlib.pyplot as plt

        plt.pause(interval)
        return True
    except Exception:
        return False


def collect_backend_diagnostics(*, try_enable_gui: bool = False) -> Dict[str, object]:
    if try_enable_gui:
        ensure_interactive_backend()

    diag: Dict[str, object] = {
        "display": os.environ.get("DISPLAY"),
        "xauthority": os.environ.get("XAUTHORITY"),
        "mplbackend_env": os.environ.get("MPLBACKEND"),
        "pyplot_preimported": "matplotlib.pyplot" in sys.modules,
        "backend": None,
        "supports_windows": False,
        "can_create_figure": False,
        "backend_attempt_errors": {},
    }
    try:
        import matplotlib

        diag["backend"] = str(matplotlib.get_backend())
        diag["supports_windows"] = backend_supports_windows()
        diag["can_create_figure"] = _can_create_figure()
        if _LAST_BACKEND_ATTEMPTS:
            diag["backend_attempt_errors"] = dict(_LAST_BACKEND_ATTEMPTS)
    except Exception as exc:
        diag["error"] = str(exc)
    return diag
