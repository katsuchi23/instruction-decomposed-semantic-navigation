"""Intent-caching layer — avoids redundant LLM calls for repeated instructions."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from parsing.intent_parser import (
    Behavior,
    ConstraintItem,
    MainTarget,
    ObjectRef,
    ParsedInstruction,
    PreferenceItem,
    TaskIntent,
    Termination,
    _extract_termination_phase_deg,
    normalize_relation_type,
    parse_instruction,
)
from utils.config import get_cache_dir


def get_cached_intent_path(instruction: str, cache_dir: str | os.PathLike[str] | None = None) -> str:
    cache_path = Path(cache_dir) if cache_dir is not None else get_cache_dir()
    cache_path.mkdir(parents=True, exist_ok=True)
    instruction_hash = hashlib.md5(instruction.encode("utf-8")).hexdigest()[:16]
    return str(cache_path / f"intent_{instruction_hash}.json")


# ---------------------------------------------------------------------------
# Reconstruct frozen dataclasses from plain dicts (JSON round-trip)
# ---------------------------------------------------------------------------

def _dict_to_object_ref(d: dict) -> ObjectRef:
    return ObjectRef(name=d["name"], type=normalize_relation_type(d.get("type", "near")))


def _dict_to_termination(d: dict) -> Termination:
    # Backward compatibility:
    # - old caches: phase is categorical string (near/front/back/left/right)
    # - newer caches: phase is numeric degrees
    # - transitional caches: phase_deg may exist
    phase_raw = d.get("phase", None)
    phase_deg_raw = d.get("phase_deg", None)

    phase_map = {"front": 0.0, "left": 90.0, "right": -90.0, "back": 180.0, "near": None}

    phase: float | None
    phase = None
    if isinstance(phase_raw, str):
        key = phase_raw.strip().lower()
        if key in phase_map:
            phase = phase_map[key]
        else:
            try:
                phase = float(key)
            except ValueError:
                phase = None
    else:
        try:
            phase = float(phase_raw) if phase_raw is not None else None
        except (TypeError, ValueError):
            phase = None

    if phase_deg_raw is not None:
        try:
            phase = float(phase_deg_raw)
        except (TypeError, ValueError):
            pass

    phase_explicit = d.get("phase_explicit")
    if phase_explicit is None:
        phase_explicit = (phase is not None)
    if phase is None:
        phase_explicit = False
    return Termination(
        distance_m=d.get("distance_m", 0.5),
        phase=phase,
        phase_explicit=bool(phase_explicit),
        stop_strictness=d.get("stop_strictness", "normal"),
        stop_policy=d.get("stop_policy", "default"),
    )


def _dict_to_constraint(d: dict) -> ConstraintItem:
    return ConstraintItem(
        target=_dict_to_object_ref(d["target"]),
        references=tuple(_dict_to_object_ref(r) for r in d.get("references", [])),
    )


def _dict_to_preference(d: dict) -> PreferenceItem:
    return PreferenceItem(
        target=_dict_to_object_ref(d["target"]),
        references=tuple(_dict_to_object_ref(r) for r in d.get("references", [])),
    )


def _dict_to_behavior(d: dict) -> Behavior:
    return Behavior(
        speed=d.get("speed", "normal"),
        caution=d.get("caution", "normal"),
    )


def _repair_cached_termination(term: Termination, instruction: str) -> Termination:
    if term.phase is not None:
        return term
    inferred = _extract_termination_phase_deg(instruction or "")
    if inferred is None:
        return term
    return Termination(
        distance_m=term.distance_m,
        phase=inferred,
        phase_explicit=True,
        stop_strictness=term.stop_strictness,
        stop_policy=term.stop_policy,
    )


def _dict_to_task(d: dict, instruction: str) -> TaskIntent:
    m = d["main"]
    term = _dict_to_termination(m.get("termination", {}))
    term = _repair_cached_termination(term, instruction)
    target_name = m["target"]["name"]
    return TaskIntent(
        main=MainTarget(
            target=ObjectRef(name=target_name, type="near"),
            references=tuple(_dict_to_object_ref(r) for r in m.get("references", [])),
            termination=term,
        ),
        constraints=tuple(_dict_to_constraint(c) for c in d.get("constraints", [])),
        preferences=tuple(_dict_to_preference(p) for p in d.get("preferences", [])),
        behavior=_dict_to_behavior(d.get("behavior", {})),
    )


def _dict_to_parsed_instruction(d: dict) -> ParsedInstruction:
    instruction = d.get("instruction", "")
    return ParsedInstruction(
        instruction=instruction,
        tasks=tuple(_dict_to_task(t, instruction) for t in d.get("tasks", [])),
        confidence=d.get("confidence", 0.0),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_or_parse_instruction(
    instruction: str,
    cache_dir: str | os.PathLike[str] | None = None,
) -> ParsedInstruction:
    """Return a :class:`ParsedInstruction`, using a local JSON cache when possible."""
    cache_path = get_cached_intent_path(instruction, cache_dir)

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            return _dict_to_parsed_instruction(cached_data)

    result = parse_instruction(instruction)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    return result
