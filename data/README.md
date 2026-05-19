# Data Directory

This repository keeps `data/` in version control only as a placeholder and format contract.

## What Goes Here

- `data/raw/`: raw ROS 2 bags or equivalent capture assets
- `data/processed/`: prepared scene folders and DovSG outputs
- `data/sample/`: optional tiny public example assets

## What Does Not Go Into Git

- real project datasets
- full bags
- large point clouds
- generated semantic memory

Those artifacts should be distributed externally and extracted into this directory while preserving the documented layout.
