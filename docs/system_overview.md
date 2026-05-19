# System Overview

This repository maps natural-language navigation instructions to motion commands by combining:

1. LLM-based instruction parsing
2. Semantic grounding against DovSG `docs.jsonl`
3. Global planning over occupancy/cost maps
4. Local sampled trajectory scoring
5. IPC-based interaction with an external robotics environment

## Main Components

- `parsing/intent_parser.py`: converts natural language into structured tasks
- `navigation/object_retrieval.py`: resolves target and reference objects from `docs.jsonl`
- `planning/path_planning.py`: computes the reachable global path
- `algorithm/trajectory_sampling.py`: samples candidate local trajectories
- `algorithm/scoring.py`: scores those trajectories against the task intent
- `navigation/navigator.py`: orchestrates task execution and recovery
- `ipc/`: receives environment state and sends command outputs

## Public Runtime Boundary

This repository does not require direct ROS topic access. It expects the environment to expose the required data through the IPC contract documented in [ipc_interface.md](ipc_interface.md).

## Semantic Data Flow

Prepared DovSG outputs are converted into `docs.jsonl`. At runtime the system:

1. parses the instruction
2. retrieves the target object and any references from `docs.jsonl`
3. plans toward the grounded target
4. executes the resulting motion commands through IPC

## Configuration

All public tuning knobs are concentrated in:

- `config/ipc.yaml`
- `config/runtime.yaml`
- `config/params.yaml`

These files are intended to be the main tuning surface for other researchers.
