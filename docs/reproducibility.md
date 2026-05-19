# Reproducibility

## Public Reproduction Boundary

This repository assumes an external robotics environment already exposes the documented IPC endpoints. The publishable boundary of the code release is:

- language parsing
- semantic grounding from `docs.jsonl`
- planning and local trajectory scoring
- navigation logic
- IPC-based environment interaction

## Minimal Reproduction Flow

1. Clone with submodules.
2. Install the environment from `environment.yml`.
3. Place the required CLIP checkpoint in the documented directory.
4. Place prepared semantic data under `data/processed/<scene_name>/...`.
5. Configure `config/ipc.yaml` to match your existing IPC bridge.
6. Configure `config/runtime.yaml`, `config/params.yaml`, and the scene config.
7. Run `python main.py --instruction "..."`

## Scene Preparation Flow

If you are reproducing the full semantic-memory pipeline:

1. Record or obtain a ROS 2 bag.
2. Export a DovSG-style scene directory with `scripts/prepare_dataset_from_bag.py`.
3. Run DovSG to obtain `instance_objects.pkl`.
4. Convert to `docs.jsonl` with `scripts/prepare_docs_from_dovsg.py`.
5. Optionally merge near-duplicates with `scripts/postprocess_docs.py`.

## Expected External Components

- A robotics stack that publishes the IPC endpoints defined in `config/ipc.yaml`.
- DovSG submodule and the CLIP checkpoint used by this repository.
- OpenAI API access for instruction parsing.
