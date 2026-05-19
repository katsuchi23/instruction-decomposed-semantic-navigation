# Experiment Protocol

The public runtime entrypoint is `main.py`. Scripts under `scripts/` are auxiliary helpers for preprocessing or post-analysis.

## Runtime

Use `main.py` for single-instruction evaluation:

```bash
python main.py --instruction "go to the red cube"
```

## Preparation Scripts

- `scripts/prepare_dataset_from_bag.py`: export a ROS 2 bag into a DovSG-style scene folder.
- `scripts/prepare_docs_from_dovsg.py`: convert `instance_objects.pkl` into `docs.jsonl`.
- `scripts/postprocess_docs.py`: merge near-duplicate semantic objects in `docs.jsonl`.

## Analysis Scripts

The analysis scripts in `scripts/` remain available for internal study and paper figure regeneration. They are not part of the main runtime path and may require result directories arranged under `outputs/`.
