#!/usr/bin/env python3
"""Convert DovSG instance outputs into docs.jsonl for semantic grounding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import DOVSG_ROOT, ensure_dovsg_python_path
from utils.preprocess_docs_merge_near_duplicates import postprocess_docs_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a DovSG instance_objects.pkl file into docs.jsonl and "
            "optionally merge near-duplicate semantic objects."
        )
    )
    parser.add_argument(
        "--instance-pkl",
        required=True,
        help="Path to DovSG instance_objects.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where docs.jsonl will be written.",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Skip near-duplicate docs postprocessing.",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for duplicate merging.",
    )
    parser.add_argument(
        "--dist-threshold-m",
        type=float,
        default=0.10,
        help="Centroid distance threshold for duplicate merging.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    instance_path = Path(args.instance_pkl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not DOVSG_ROOT.exists():
        raise FileNotFoundError(
            f"DovSG submodule is missing at {DOVSG_ROOT}. Clone with --recurse-submodules first."
        )
    if not instance_path.exists():
        raise FileNotFoundError(f"instance_objects.pkl not found: {instance_path}")

    ensure_dovsg_python_path()
    from semantic_field_navigation.utils.instance_object_to_json import convert_pickle_to_jsonl

    output_dir.mkdir(parents=True, exist_ok=True)
    convert_pickle_to_jsonl(instance_path, output_dir)
    docs_path = output_dir / "docs.jsonl"

    if args.skip_postprocess:
        print(f"docs.jsonl written: {docs_path}")
        return

    merged_path, stats = postprocess_docs_jsonl(
        docs_path=docs_path,
        sim_threshold=float(args.sim_threshold),
        dist_threshold_m=float(args.dist_threshold_m),
        rebuild_tfidf=True,
    )
    print(
        "postprocess complete: "
        f"input={int(stats.get('input_records', 0))} "
        f"output={int(stats.get('output_records', 0))} "
        f"merged_clusters={int(stats.get('merged_clusters', 0))}"
    )
    print(f"merged docs: {merged_path}")


if __name__ == "__main__":
    main()
