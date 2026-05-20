#!/usr/bin/env python3
"""Single public entrypoint for instruction-decomposed semantic navigation."""

from __future__ import annotations

import argparse
import os

from utils.config import (
    get_docs_path_from_config,
    get_param,
    load_repo_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one natural-language semantic navigation instruction using "
            "IPC endpoints exposed by an external robotics environment."
        )
    )
    parser.add_argument(
        "--instruction",
        "-i",
        required=True,
        help="Natural-language instruction to execute.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # This workstation is CPU-only. Keep the main entrypoint on the CPU path
    # unless the user explicitly overrides it in their environment.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from navigation.navigator import run_navigation
    from navigation.object_retrieval import configure_object_retrieval_params

    load_repo_config()

    configure_object_retrieval_params()

    docs_path = str(get_docs_path_from_config() or "")

    timeout_sec = float(get_param(("navigator", "timeout_sec"), 180.0))
    collision_cost_thresh = float(get_param(("navigator", "collision_cost_thresh"), 90.0))
    collision_duration_sec = float(get_param(("navigator", "collision_duration_sec"), 5.0))

    run_navigation(
        instruction=args.instruction,
        docs_path=docs_path or None,
        timeout_sec=timeout_sec,
        collision_cost_thresh=collision_cost_thresh,
        collision_duration_sec=collision_duration_sec,
    )


if __name__ == "__main__":
    main()
