# Dataset Format

The repository does not ship the real dataset. It expects external data to be placed under `data/`.

## Canonical Layout

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
│           └── <memory_suffix>/
│               └── step_0/
│                   └── data_json/
│                       ├── docs.jsonl
│                       └── docs_merged.jsonl
└── sample/
```

## What `main.py` Actually Needs

For normal navigation runtime, the minimum required semantic artifact is:

- `docs.jsonl` or `docs_merged.jsonl`

The runtime does not require the full raw bag once semantic memory has already been prepared.

## What Preprocessing Needs

If you start from a ROS 2 bag:

- place the bag under `data/raw/<scene_name>/bag/`
- run `scripts/prepare_dataset_from_bag.py`
- run DovSG externally to generate `instance_objects.pkl`
- run `scripts/prepare_docs_from_dovsg.py`

## External Dataset Distribution

If you distribute prepared scenes externally, for example via Google Drive:

- extract them into `data/processed/<scene_name>/`
- preserve directory names exactly
- do not rename `memory/`, `step_0/`, or `data_json/`

If you need to override the default docs path, set it in `config/runtime.yaml` or pass `--docs-path` to `main.py`.
