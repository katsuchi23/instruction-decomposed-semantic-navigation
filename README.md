# Instruction-Decomposed Semantic Navigation

Research code release for **Language-Conditioned Autonomous Navigation through Instruction Decomposition**.

This repository exposes the semantic-navigation pipeline as a **single-device**, **single-entrypoint** runtime. The navigation stack reads all robot state through IPC endpoints, grounds language against DovSG semantic memory (`docs.jsonl`), and outputs commands back through IPC. The external ROS/simulator/real-robot environment is expected to already provide the IPC bridge.

## Introduction

Natural-language navigation instructions often contain more than a single destination. In addition to a main target, users may specify reference objects, path preferences, avoidance constraints, and behavior modifiers such as speed or caution. This project studies how those instruction components can be decomposed into structured intent fields and then mapped into an interpretable navigation pipeline.

The central idea of this repository is to treat language as a modular control interface rather than a monolithic command. Instead of relying on a fully end-to-end policy, the system separates instruction parsing, semantic grounding, global planning, and local trajectory scoring. That design makes it easier to inspect how each language component affects robot behavior and makes the system easier to tune, debug, and evaluate in a research setting.

## What This Repository Contains

- Core language parsing, semantic grounding, planning, and control logic.
- A single public runtime entrypoint: `main.py`.
- Config-driven IPC endpoints and algorithm parameters.
- DovSG as a git submodule under `submodules/DovSG`.
- Dataset placeholders and preparation scripts.
- Docs for dataset layout, IPC payloads, and reproducibility.

## Tested Environment

- Ubuntu 22.04
- Python 3.10
- ROS 2 Humble on the external robotics side
- DovSG checked out as a submodule at the commit recorded in this repository

The core runtime is intentionally IPC-driven rather than ROS-topic-driven. If your robotics stack can provide the documented IPC endpoints, the navigation code can stay unchanged.

## Clone With Submodules

```bash
git clone --recurse-submodules https://github.com/katsuchi23/instruction-decomposed-semantic-navigation 
cd instruction-decomposed-semantic-navigation
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Installation

`environment.yml` is the recommended path because DovSG and its transitive dependencies are easier to manage in a controlled environment.

```bash
conda env create -f environment.yml
conda activate semnav
```

If you prefer `pip`, install the repository dependencies first and then follow the DovSG setup instructions from its upstream project:

```bash
pip install -r requirements.txt
```

## Required Model Checkpoints

For the full DovSG model setup, including any additional checkpoints required by DovSG itself, please refer to the upstream repository:

- DovSG repository: `https://github.com/BJHYZJ/DovSG`

For this repository specifically, the CLIP checkpoint must still be present because the semantic-grounding runtime uses it directly.

Required CLIP checkpoint directory (you can change it in `config/runtime.yaml`):

```text
instruction-decomposed-semantic-navigation/models/clip/CLIP-ViT-H-14-laion2B-s32B-b79K/
```

Create that directory locally if it does not exist yet, then place the downloaded CLIP checkpoint contents there.

The runtime expects this file inside that directory:

```text
instruction-decomposed-semantic-navigation/models/clip/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
```

CLIP download link:

- OpenCLIP `ViT-H-14` LAION2B weights: `https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K`

Usage summary:

- `main.py`: requires the CLIP checkpoint above
- `scripts/prepare_docs_from_dovsg.py`: does not require CLIP if `instance_objects.pkl` already exists
- DovSG memory generation: follow the DovSG repository instructions for all required models inside the DovSG submodule environment

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

If you distribute prepared scenes separately, for example via Google Drive, the extracted directories should land under `data/processed/<scene_name>/...` without renaming.

See [docs/dataset_format.md](docs/dataset_format.md) and [data/README.md](data/README.md) for the exact expected layout.

## Preparing `docs.jsonl` From DovSG Outputs

This release assumes the reader already has DovSG-generated scene outputs. At minimum, the supported public workflow assumes you already have `instance_objects.pkl`.

Convert it into `docs.jsonl` with:

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

## IPC Configuration

The runtime reads IPC endpoint addresses from `config/ipc.yaml`. The IPC contract is documented in [docs/ipc_interface.md](docs/ipc_interface.md).

This repository does **not** require ROS topic names in `main.py`. Topic names are part of the external bridge, not the core runtime.

## Running The System

Edit the config files as needed. `main.py` always reads its runtime settings from these files:

- `config/ipc.yaml`
- `config/params.yaml`
- `config/runtime.yaml`
- `config/scenes/<scene>.yaml`

Then run:

```bash
python main.py --instruction "go to the red cube and stay close to the blue sphere"
```

The only command-line argument is the instruction itself. If you want to change:

- semantic memory path
- IPC endpoint addresses
- visualization behavior
- timeout values
- collision thresholds
- scene-specific settings

edit the config files above instead of passing runtime overrides on the command line.

## Reproducing Experiments

The scripts under `scripts/` are preparation or analysis helpers. They are not runtime entrypoints.

Examples:

```bash
python scripts/postprocess_docs.py --input data/processed/<scene>/memory/.../data_json/docs.jsonl
python scripts/analyze_experiments.py --result-dir outputs/experiments/<scene>
```

## Limitations

- The repository assumes an external IPC bridge already exists.
- The public runtime is single-device only.
- The repository does not ship the real dataset.
- DovSG preprocessing is only partially wrapped here; upstream DovSG setup is still required for full memory generation.

## References

If you use this repository in academic work, cite the project report or resulting paper associated with:

- **Language-Conditioned Autonomous Navigation through Instruction Decomposition**

Related upstream dependency:

- DovSG repository: `https://github.com/BJHYZJ/DovSG`

Related CLIP model source:

- LAION OpenCLIP ViT-H-14: `https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
