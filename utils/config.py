"""Central configuration loading, path resolution, and runtime validation."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SUBMODULES_DIR = PROJECT_ROOT / "submodules"
DOVSG_ROOT = SUBMODULES_DIR / "DovSG"

_DEFAULT_CONFIG: dict[str, Any] = {
    "ipc": {
        "endpoints": {
            "pose_sub": "tcp://127.0.0.1:5557",
            "nav_status_sub": "tcp://127.0.0.1:5558",
            "cmd_vel_req": "tcp://127.0.0.1:5559",
            "nav_goal_req": "tcp://127.0.0.1:5556",
            "trajectory_req": "tcp://127.0.0.1:5563",
            "image_req": "tcp://127.0.0.1:5555",
            "depth_req": "tcp://127.0.0.1:5561",
            "scan_sub": "tcp://127.0.0.1:5562",
            "local_costmap_req": "tcp://127.0.0.1:5564",
            "global_costmap_req": "tcp://127.0.0.1:5565",
        },
        "timeouts_ms": {
            "pose_poll": 2000,
            "cmd_vel_reply": 500,
            "image_reply": 2000,
            "depth_reply": 2000,
            "scan_poll": 100,
            "costmap_reply": 1000,
            "nav_goal_reply": 2000,
            "trajectory_reply": 500,
        },
    },
    "runtime": {
        "scene": "testing_ground",
        "feature_flags": {
            "show_viz": True,
            "enable_ros_param_inflation_updates": False,
        },
        "paths": {
            "data_root": "data",
            "processed_data_root": "data/processed",
            "outputs_root": "outputs",
            "intent_cache_dir": "outputs/intent_cache",
            "docs_path": "",
            "viz_map_pgm": "",
        },
        "assets": {
            "dovsg_root": "submodules/DovSG",
            "clip_checkpoint_dir": "models/clip/CLIP-ViT-H-14-laion2B-s32B-b79K",
        },
        "ros_integration": {
            "setup_bash": "/opt/ros/humble/setup.bash",
            "workspace_setup_bash": "",
            "local_costmap_param": "/local_costmap/local_costmap inflation_layer.inflation_radius",
            "global_costmap_param": "/global_costmap/global_costmap inflation_layer.inflation_radius",
        },
    },
    "params": {
        "object_retrieval": {
            "direction_min_distance_m": 0.30,
            "proximity_scale_m": 1.0,
            "reference_top_k": 8,
            "isolation_scale_m": 0.5,
            "alone_clip_max_gap": 0.12,
            "alone_max_candidates": 12,
            "alone_clearance_gate_m": 0.20,
            "target_top_k_with_references": 8,
            "clip_gate_start": 0.27,
            "clip_gate_min": 0.01,
            "clip_gate_step": 0.01,
            "reference_max_dist_m": 0.50,
            "reference_score_gain": 1.50,
            "clip_weight": 0.30,
            "top_k": 5,
        },
        "intent_costs": {
            "constraint_radius_m": 1.0,
            "preference_radius_m": 1.0,
            "preference_min_dist_m": 0.3,
            "w_constraint_path": 3.0,
            "w_preference_near_repel_path": 1.0,
            "w_preference_attract_path": 10.0,
        },
        "control": {
            "termination": {
                "distance_band_m": {
                    "loose": [0.50, 1.00],
                    "normal": [0.40, 0.80],
                    "strict": [0.35, 0.60],
                },
                "alpha_max_deg": {
                    "loose": 15.0,
                    "normal": 10.0,
                    "strict": 5.0,
                },
                "phi_tol_deg": {
                    "loose": 15.0,
                    "normal": 10.0,
                    "strict": 5.0,
                },
                "distance_band_half_width_m": {
                    "loose": 0.10,
                    "normal": 0.05,
                    "strict": 0.03,
                },
            },
            "sampling": {
                "h_r": 0.05,
                "h_alpha_deg": 3.0,
                "dwell_sec_default": 0.5,
                "dwell_sec_no_stop": 0.0,
                "horizon_steps": 30,
                "dt_sec": 0.1,
                "num_samples": 100,
                "v_min": 0.15,
                "w_min": 0.15,
                "p_stop": 0.10,
            },
            "dynamics": {
                "a_v_max": 0.5,
                "a_w_max": 0.5,
            },
            "weights": {
                "w_du": 0.5,
                "w_curv": 0.6,
                "w_path": 2.0,
            },
            "behavior_mapping": {
                "speed": {
                    "slow": {"v_max": 0.2, "w_max": 0.2},
                    "normal": {"v_max": 0.3, "w_max": 0.3},
                    "fast": {"v_max": 0.4, "w_max": 0.4},
                },
                "caution": {
                    "low": {"d_safe": 0.20, "w_clear": 0.6, "w_clear_v": 0.08, "w_clear_w": 0.03},
                    "normal": {"d_safe": 0.30, "w_clear": 4.0, "w_clear_v": 0.3, "w_clear_w": 0.2},
                    "high": {"d_safe": 0.40, "w_clear": 12.0, "w_clear_v": 0.8, "w_clear_w": 0.35},
                },
                "w_u_fast": 0.5,
                "w_u_default": 1.0,
                "r_entry_offset_m": 0.25,
                "v_entry_max_slow_mps": 0.25,
                "v_entry_max_default_mps": 0.35,
                "w_face_far": 0.2,
                "w_face_near": 1.0,
            },
        },
        "navigator": {
            "timeout_sec": 180.0,
            "collision_cost_thresh": 90.0,
            "collision_duration_sec": 5.0,
            "post_goal_max_retries": 3,
            "pose_stale_timeout_sec": 1.0,
            "pose_values_frozen_timeout_sec": 1.5,
            "pose_missing_warn_sec": 1.0,
            "pose_pipeline_warn_sec": 1.0,
        },
    },
    "scene": {},
}

_CONFIG: dict[str, Any] = copy.deepcopy(_DEFAULT_CONFIG)


def _deep_merge(dst: dict[str, Any], src: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")
    return data


def _resolve_path_value(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith("tcp://") or value.startswith("udp://") or value.startswith("http://") or value.startswith("https://"):
        return value
    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def _resolve_runtime_paths(config: dict[str, Any]) -> None:
    runtime = config.setdefault("runtime", {})
    for section_name in ("paths", "assets", "ros_integration"):
        section = runtime.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for key, value in list(section.items()):
            if key.endswith("_param"):
                continue
            section[key] = _resolve_path_value(value)

    scene = config.get("scene", {})
    if isinstance(scene, dict):
        for key, value in list(scene.items()):
            if key.endswith("_path") or key.endswith("_dir") or key in {"viz_map_pgm"}:
                scene[key] = _resolve_path_value(value)


def load_repo_config(
    *,
    ipc_path: str | None = None,
    params_path: str | None = None,
    runtime_path: str | None = None,
    scene_path: str | None = None,
) -> dict[str, Any]:
    """Load repository configuration and store it globally."""
    global _CONFIG, DOVSG_ROOT

    config = copy.deepcopy(_DEFAULT_CONFIG)

    default_ipc = Path(ipc_path) if ipc_path else CONFIG_DIR / "ipc.yaml"
    default_params = Path(params_path) if params_path else CONFIG_DIR / "params.yaml"
    default_runtime = Path(runtime_path) if runtime_path else CONFIG_DIR / "runtime.yaml"

    _deep_merge(config, {"ipc": _read_yaml(default_ipc)})
    _deep_merge(config, {"params": _read_yaml(default_params)})
    _deep_merge(config, {"runtime": _read_yaml(default_runtime)})

    runtime_scene = get_nested(config, ("runtime", "scene"), default="") or ""
    if scene_path:
        scene_file = Path(scene_path)
    elif runtime_scene:
        scene_file = CONFIG_DIR / "scenes" / f"{runtime_scene}.yaml"
    else:
        scene_file = CONFIG_DIR / "scenes" / "default.yaml"
    _deep_merge(config, {"scene": _read_yaml(scene_file)})

    _resolve_runtime_paths(config)
    _CONFIG = config

    configured_dovsg_root = get_nested(_CONFIG, ("runtime", "assets", "dovsg_root"), default=str(DOVSG_ROOT))
    DOVSG_ROOT = Path(configured_dovsg_root)
    return _CONFIG


def get_repo_config() -> dict[str, Any]:
    return _CONFIG


def get_nested(data: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def get_param(path: Sequence[str] | str, default: Any = None) -> Any:
    keys = tuple(path.split(".")) if isinstance(path, str) else tuple(path)
    return get_nested(_CONFIG.get("params", {}), keys, default)


def get_runtime_value(path: Sequence[str] | str, default: Any = None) -> Any:
    keys = tuple(path.split(".")) if isinstance(path, str) else tuple(path)
    return get_nested(_CONFIG.get("runtime", {}), keys, default)


def get_scene_value(path: Sequence[str] | str, default: Any = None) -> Any:
    keys = tuple(path.split(".")) if isinstance(path, str) else tuple(path)
    return get_nested(_CONFIG.get("scene", {}), keys, default)


def get_ipc_endpoint(name: str) -> str:
    return str(get_nested(_CONFIG, ("ipc", "endpoints", name), _DEFAULT_CONFIG["ipc"]["endpoints"][name]))


def get_ipc_timeout_ms(name: str, default: int) -> int:
    return int(get_nested(_CONFIG, ("ipc", "timeouts_ms", name), default))


def get_cache_dir() -> Path:
    return Path(get_runtime_value(("paths", "intent_cache_dir"), PROJECT_ROOT / "outputs" / "intent_cache"))


def get_outputs_root() -> Path:
    return Path(get_runtime_value(("paths", "outputs_root"), PROJECT_ROOT / "outputs"))


def get_docs_path_from_config() -> Path | None:
    explicit = get_runtime_value(("paths", "docs_path"), "")
    if explicit:
        return Path(str(explicit))
    scene_path = get_scene_value(("docs_path",), "")
    if scene_path:
        return Path(str(scene_path))
    return None


def get_clip_checkpoint_file() -> Path:
    clip_dir = Path(str(get_runtime_value(("assets", "clip_checkpoint_dir"), "")))
    return clip_dir / "open_clip_pytorch_model.bin"


def ensure_dovsg_python_path() -> None:
    """Add the configured DovSG root to ``sys.path`` so its packages are importable."""
    p = str(DOVSG_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def validate_runtime_prereqs(*, require_dovsg_assets: bool = True) -> None:
    """Raise ``RuntimeError`` if required runtime assets are missing."""
    if not require_dovsg_assets:
        return

    if not DOVSG_ROOT.exists():
        raise RuntimeError(f"DovSG submodule not found: {DOVSG_ROOT}")

    clip_checkpoint = get_clip_checkpoint_file()
    if not clip_checkpoint.exists():
        raise RuntimeError(f"Missing CLIP checkpoint: {clip_checkpoint}")
