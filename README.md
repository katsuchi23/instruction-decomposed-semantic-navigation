# Instruction-Decomposed Semantic Navigation

Research code release for **Language-Conditioned Autonomous Navigation through Instruction Decomposition**.

This repository exposes the semantic-navigation pipeline as a **single-device**, **single-entrypoint** runtime. The navigation stack reads all robot state through IPC endpoints, grounds language against DovSG semantic memory (`docs.jsonl`), and outputs commands back through IPC. The external ROS/simulator/real-robot environment is expected to provide the IPC bridge nodes documented in `ros_nodes/`.

---

## Introduction

Natural-language navigation instructions often contain more than a single destination. In addition to a main target, users may specify reference objects, path preferences, avoidance constraints, and behavior modifiers such as speed or caution. This project studies how those instruction components can be decomposed into structured intent fields and then mapped into an interpretable navigation pipeline.

The central idea of this repository is to treat language as a modular control interface rather than a monolithic command. Instead of relying on a fully end-to-end policy, the system separates instruction parsing, semantic grounding, global planning, and local trajectory scoring. That design makes it easier to inspect how each language component affects robot behavior and makes the system easier to tune, debug, and evaluate in a research setting.

## Demo

> *"go to the red cube near the blue sphere while avoiding the red cube"*

![Navigation demo](assets/demo.gif)

---

## What This Repository Contains

- **Core pipeline** — language parsing, semantic grounding (CLIP + DovSG), A\* global planning, and local trajectory scoring with a cost function that handles goal satisfaction, obstacle clearance, path following, constraints, and preferences.
- **Two navigation modes** — direct `cmd_vel` trajectory-sampling mode and a Nav2-delegated `nav_through_pose` mode.
- **Single public entrypoint** — `main.py`.
- **Config-driven everything** — all algorithm parameters are in `config/params.yaml`; IPC addresses in `config/ipc.yaml`; runtime flags in `config/runtime.yaml`. No tuning values are hardcoded.
- **ROS bridge nodes** — ready-to-use ROS2 Python scripts in `ros_nodes/` that implement the IPC bridge between this stack and any ROS2 environment.
- **RViz visualization** — the navigation state (path, trajectory samples, goal ring, constraint/preference radii) is published as standard RViz topics via `ros_nodes/semnav_rviz_node.py`.
- **Automatic result saving** — every run saves `result.json`, `summary.txt`, per-task telemetry CSVs, trajectory CSVs, and a path-map PNG to `outputs/runs/<timestamp>/`.
- **DovSG submodule** — `submodules/DovSG`.
- **Dataset placeholders and preparation scripts**.
- **Docs** — ROS environment setup, system overview, IPC interface contract, dataset format, experiment protocol, reproducibility guide, and full config reference (see Documentation section below).

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ros_env.md](docs/ros_env.md) | **Start here for hardware setup.** Full ROS 2 launch order (LiDAR driver → localization → Nav2 → robot controller), TF frame layout, and `odom_pitch` fine-tuning guide. |
| [docs/system_overview.md](docs/system_overview.md) | High-level architecture: how instruction parsing, semantic grounding, planning, and control fit together. |
| [docs/ipc_interface.md](docs/ipc_interface.md) | IPC contract — all ZMQ endpoint addresses, message schemas, and direction. Required reading if you are writing your own ROS bridge. |
| [docs/dataset_format.md](docs/dataset_format.md) | Expected directory layout for `data/`, file naming conventions, and field descriptions for `docs.jsonl`. |
| [docs/experiment_protocol.md](docs/experiment_protocol.md) | Step-by-step protocol for running and logging experiments, including script invocation order. |
| [docs/reproducibility.md](docs/reproducibility.md) | Post-processing and analysis scripts for reproducing reported results. |
| [config/config_reference.md](config/config_reference.md) | Full reference for every field in `config/params.yaml`, `config/ipc.yaml`, and `config/runtime.yaml`. |
| [ros_nodes/README.md](ros_nodes/README.md) | Per-node descriptions, ZMQ port map, and minimum node sets for each navigation mode. |
| [data/README.md](data/README.md) | Notes on the `data/` placeholder structure and what is expected vs. excluded from version control. |

---

## Tested Environment

- Ubuntu 22.04
- Python 3.10
- ROS 2 Humble (external robotics side)
- DovSG checked out as a submodule at the commit recorded in this repository

The core runtime is IPC-driven rather than ROS-topic-driven. If your robotics stack can provide the documented IPC endpoints, the navigation code can stay unchanged.

---

## Clone With Submodules

```bash
git clone --recurse-submodules https://github.com/katsuchi23/instruction-decomposed-semantic-navigation
cd instruction-decomposed-semantic-navigation
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

---

## Installation

There are two ways to set up the Python dependencies for this stack:

### Option A — Conda environment (recommended for isolation)

Creates a dedicated `semnav` environment. This is the original design: the navigation stack runs in its own env and communicates with ROS via ZMQ IPC, so neither side imports the other's packages directly.

```bash
conda env create -f environment.yml
conda activate semnav
```

### Option B — Install directly on your base environment

If you prefer not to manage a separate conda env (e.g., you want to run everything in the same environment as ROS), install the dependencies directly:

```bash
pip install -r requirements.txt
```

> **Note:** `environment.yml` and the IPC bridge nodes in `ros_nodes/` are still available and functional. You can run the stack through IPC (Option A) or straight from your base/ROS environment (Option B) — the core navigation code works the same either way. If using Option B, ensure that all packages in `requirements.txt` are compatible with any ROS packages already installed in your environment.

---

## Required Model Checkpoints

### CLIP checkpoint (required by this repository)

The semantic-grounding runtime requires an OpenCLIP `ViT-H-14` checkpoint. The model directory is configured in `config/runtime.yaml` under `assets.clip_checkpoint_dir`.

Download via `git lfs` (recommended — preserves large-file pointers):

```bash
# 1. Install git-lfs if not already installed
git lfs install

# 2. Clone the model repository from HuggingFace
git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
    models/clip/CLIP-ViT-H-14-laion2B-s32B-b79K
```

After cloning, the runtime expects this file to exist:

```text
models/clip/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
```

If `git lfs` is not available you can also download individual files directly from:

```
https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

### DovSG checkpoints

For the full DovSG model setup, follow the upstream repository instructions:

- `https://github.com/BJHYZJ/DovSG`

---

## Dataset Layout

Keep the dataset directory name exactly as `data/`.

```text
data/
├── raw/
│   └── <scene_name>/
│       └── bag/
├── processed/
│   └── <scene_name>/
│       ├── rgb/
│       ├── depth/
│       ├── point/
│       ├── mask/
│       ├── poses/
│       ├── calibration/
│       └── memory/
│           └── .../data_json/docs.jsonl
└── sample/
```

See [docs/dataset_format.md](docs/dataset_format.md) for the exact expected layout.

---

## Preparing `docs.jsonl` From DovSG Outputs

Convert `instance_objects.pkl` into `docs.jsonl` with:

```bash
python scripts/prepare_docs_from_dovsg.py \
  --instance-pkl data/processed/<scene_name>/memory/.../instance_objects.pkl \
  --output-dir data/processed/<scene_name>/memory/.../data_json
```

If you need to prepare a scene from a ROS 2 bag first:

```bash
python scripts/prepare_dataset_from_bag.py \
  --bag_dir data/raw/<scene_name>/bag \
  --out_dir data/processed/<scene_name>
```

---

## Configuration

All tuning knobs live in the config files. **Do not hardcode values in scripts.**

| File | Controls |
|------|----------|
| `config/ipc.yaml` | ZMQ endpoint addresses and timeouts |
| `config/params.yaml` | All algorithm parameters (retrieval, planner, control, scoring, navigator) |
| `config/runtime.yaml` | Runtime flags (navigation mode, visualization, file paths) |
| `config/scenes/<scene>.yaml` | Per-scene overrides |

### Key `config/params.yaml` sections

- **`object_retrieval`** — CLIP gate, reference scoring, top-K candidates
- **`intent_costs`** — constraint/preference radii and cost weights
- **`planner`** — A\* goal candidate search and relaxation
- **`control.sampling`** — trajectory horizon, noise, lookahead distance, hardware velocity limits (`v_min`, `w_min`), smoothing (`cmd_smooth_alpha`, `warm_start_alpha`), replanning frequency (`exec_steps`)
- **`control.weights`** — scoring cost coefficients (sigma, effort, smoothness, path, clearance, discount factor)
- **`control.behavior_mapping`** — speed and caution categories mapped to max velocities and clearance parameters
- **`navigator`** — timeout, collision thresholds, recovery behavior

### Key `config/runtime.yaml` flags

| Flag | Default | Effect |
|------|---------|--------|
| `feature_flags.nav_through_pose` | `false` | Switch between direct cmd_vel mode and Nav2 NavigateThroughPoses mode |
| `feature_flags.enable_ros_param_inflation_updates` | `true` | Update Nav2 costmap inflation radius when caution changes |
| `paths.docs_path` | — | Path to the `docs_merged.jsonl` semantic memory file |
| `paths.viz_map_pgm` | — | Path to the map PGM used for the saved path-map PNG |

---

## Navigation Modes

### Default mode — direct `cmd_vel` (`nav_through_pose: false`)

The stack runs a trajectory sampler every tick: samples N velocity sequences, scores each with the full cost function (goal, clearance, path, constraints, preferences, smoothness), and sends the best `cmd_vel` directly via IPC. Suitable when you want full control over local velocity decisions.

```bash
python main.py --instruction "go to the red cube"
```

### Nav2-delegated mode — `nav_through_pose: true`

The stack still runs the same trajectory sampler and cost function, but instead of sending `cmd_vel` it extracts the endpoint of the best trajectory and sends it to Nav2 `NavigateThroughPoses`. Nav2 handles all velocity control; this stack handles the path decision. Useful when Nav2's DWB is preferred for low-level control (e.g., humanoid robots that need smooth motion).

Enable by setting in `config/runtime.yaml`:

```yaml
feature_flags:
  nav_through_pose: true
```

In this mode `nav_through_poses_bridge.py` must be running (see ROS Bridge Nodes below).

---

## ROS Environment Setup

For the full ROS 2 side of the stack (LiDAR driver, localization, Nav2, robot controller), see:

- **[docs/ros_env.md](docs/ros_env.md)** — launch order, package descriptions, TF frame layout, and `odom_pitch` fine-tuning guide.

The ROS packages live in a separate repository: `https://github.com/katsuchi23/g1_ros_package`

## ROS Bridge Nodes

The `ros_nodes/` directory contains all ROS2 Python scripts that bridge this stack to a ROS2 environment. Copy the scripts you need into your ROS2 package.

See [`ros_nodes/README.md`](ros_nodes/README.md) for the full port map, per-node description, and minimum required node sets for each navigation mode.

### Quick port reference

| Port | Node | Purpose |
|------|------|---------|
| 5557 | `robot_pose_sender_node.py` | TF → pose bundle |
| 5556 / 5558 | `nav2_sender_node.py` | NavigateToPose goal + status |
| 5559 | `cmd_vel_sender_node.py` | Receive cmd_vel → publish /cmd_vel |
| 5562 | `laserscan_sender_node.py` | /scan → laser data |
| 5563 | `trajectory_visualization_node.py` | Trajectory → /projected_path |
| 5564 | `occupancy_grid_sender_node.py` | Local costmap |
| 5565 | `global_costmap.py` | Global costmap |
| 5566–5569 | `nav_through_poses_bridge.py` | NavigateThroughPoses + telemetry |
| 5570 | `semnav_rviz_node.py` | Navigation viz → RViz |

### RViz visualization

Start the launch file (which includes `semnav_rviz_node.py`) and add these topics in RViz:

| Topic | Type |
|-------|------|
| `/semnav/path` | `nav_msgs/Path` |
| `/semnav/traj_samples` | `visualization_msgs/MarkerArray` |
| `/semnav/markers` | `visualization_msgs/MarkerArray` |

The markers include: robot pose arrow, start position, target object, active goal, lookahead point, goal ring (inner/outer), constraint radii (red), and preference radii (blue).

---

## Running The System

```bash
conda activate semnav
python main.py --instruction "go to the red cube and stay close to the blue sphere"
```

The only command-line argument is the natural-language instruction. All other settings (memory path, IPC addresses, timeouts, navigation mode, visualization) are read from the config files.

---

## Output Structure

Every run automatically saves results to `outputs/runs/<YYYYMMDD_HHMMSS>/`:

```text
outputs/runs/<timestamp>/
├── result.json          # Full run data — tasks, trajectory, telemetry, success/fail
├── summary.txt          # Human-readable: success, duration, final errors per task
├── telemetry_task0.csv  # Per-step: pose, cmd_vel, distance, heading/phase errors
├── trajectory_task0.csv # Robot x, y path
└── path_map_task0.png   # Map overlay with trajectory and goal ring
```

The map PNG uses `viz_map_pgm` from `config/runtime.yaml` as the background.

---

## Reproducing Experiments

```bash
python scripts/postprocess_docs.py --input data/processed/<scene>/memory/.../data_json/docs.jsonl
python scripts/analyze_experiments.py --result-dir outputs/runs/
```

See [docs/reproducibility.md](docs/reproducibility.md) for the full protocol.

---

## Limitations

- Requires a ROS 2 environment providing sensor data and velocity control. The `ros_nodes/` IPC bridge is one way to connect this; direct integration on the base env is also supported.
- Single-device runtime only.
- Dataset is not shipped with this repository.
- DovSG preprocessing requires the upstream DovSG setup for full memory generation.

---

## References

Related upstream dependencies:

- DovSG: `https://github.com/BJHYZJ/DovSG`
- LAION OpenCLIP ViT-H-14: `https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
