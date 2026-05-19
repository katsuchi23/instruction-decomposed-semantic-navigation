"""LLM-powered instruction parser — converts natural-language instructions into
structured multi-task navigation intents with constraints and preferences."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore[assignment]

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


# ============================================================================
# Dataclasses returned to callers
# ============================================================================

@dataclass(frozen=True)
class ObjectRef:
    """An object or landmark with a spatial relation type."""
    name: str
    type: str = "near"  # near | alone


@dataclass(frozen=True)
class Termination:
    """How the robot should stop relative to the main target."""
    distance_m: float = 0.5
    phase: Optional[float] = None  # explicit approach angle in degrees; None if unspecified
    phase_explicit: bool = False   # True only when user explicitly requested phase/angle
    stop_strictness: str = "normal"  # loose | normal | strict
    stop_policy: str = "default"     # default | no_stop


@dataclass(frozen=True)
class ConstraintItem:
    """An object / region the robot must *avoid*."""
    target: ObjectRef
    references: tuple = ()  # tuple[ObjectRef, ...] for frozen compatibility


@dataclass(frozen=True)
class PreferenceItem:
    """An object / region the robot should *stay close to*."""
    target: ObjectRef
    references: tuple = ()


@dataclass(frozen=True)
class Behavior:
    """Motion-style parameters for a task."""
    speed: str = "normal"    # slow | normal | fast
    caution: str = "normal"  # low | normal | high


@dataclass(frozen=True)
class MainTarget:
    """Primary navigation objective for one task."""
    target: ObjectRef
    references: tuple = ()  # tuple[ObjectRef, ...]
    termination: Termination = Termination()


@dataclass(frozen=True)
class TaskIntent:
    """A single navigation task extracted from the instruction."""
    main: MainTarget
    constraints: tuple = ()   # tuple[ConstraintItem, ...]
    preferences: tuple = ()   # tuple[PreferenceItem, ...]
    behavior: Behavior = Behavior()


@dataclass(frozen=True)
class ParsedInstruction:
    """Complete parse result: one or more sequential tasks + confidence."""
    instruction: str
    tasks: tuple = ()   # tuple[TaskIntent, ...]
    confidence: float = 0.0


# ============================================================================
# Pydantic schemas (for OpenAI structured outputs)
# ============================================================================

class ObjectRefSchema(BaseModel):
    name: str = Field(description="Object or landmark name from the instruction. For 'alone' type this can be empty string.")
    type: Literal["near", "alone"] = Field(
        default="near",
        description=(
            "Spatial relation describing where the MAIN TARGET is relative to THIS reference object. "
            "Use 'near' when the relation is unspecified or just 'near'/'next to'/'close to'. "
            "Use 'alone' when the target is described as standing by itself / isolated / the only one."
        ),
    )


class TerminationSchema(BaseModel):
    distance_m: float = Field(
        default=0.5,
        description="Distance in metres from the main target.",
    )
    phase: Literal["front", "back", "left", "right", "near"] = Field(
        default="near",
        description="Final robot position relative to the main target.",
    )
    phase_deg: Optional[float] = Field(
        default=None,
        description=(
            "Optional explicit approach angle in degrees for termination phase "
            "(e.g. 45, -90, 180). Use this when the user gives a numeric angle."
        ),
    )
    phase_explicit: bool = Field(
        default=False,
        description=(
            "True only if the user explicitly requested a final phase/angle "
            "(e.g. front/back/left/right or explicit angle). "
            "False when phase is only an implicit default."
        ),
    )
    stop_strictness: Literal["loose", "normal", "strict"] = Field(
        default="normal",
        description="Stopping precision / tolerance.",
    )
    stop_policy: Literal["default", "no_stop"] = Field(
        default="default",
        description="'default' to stop at goal, 'no_stop' to keep moving.",
    )


class ConstraintItemSchema(BaseModel):
    target: ObjectRefSchema = Field(
        description="Object or region to avoid.",
    )
    references: List[ObjectRefSchema] = Field(
        default_factory=list,
        description="Extra context to disambiguate which instance.",
    )


class PreferenceItemSchema(BaseModel):
    target: ObjectRefSchema = Field(
        description="Object or region to stay close to.",
    )
    references: List[ObjectRefSchema] = Field(
        default_factory=list,
        description="Extra context to disambiguate which instance.",
    )


class BehaviorSchema(BaseModel):
    speed: Literal["slow", "normal", "fast"] = Field(
        default="normal",
        description="Speed of execution.",
    )
    caution: Literal["low", "normal", "high"] = Field(
        default="normal",
        description="Caution level / clearance preference.",
    )


class MainTargetSchema(BaseModel):
    target: ObjectRefSchema = Field(
        description="Primary object / region to navigate to.",
    )
    references: List[ObjectRefSchema] = Field(
        default_factory=list,
        description=(
            "Reference landmarks used to disambiguate which instance of the target is meant. "
            "Use type='near' for normal landmark disambiguation and type='alone' for isolation."
        ),
    )
    termination: TerminationSchema = Field(
        default_factory=TerminationSchema,
        description="How the robot should stop relative to the target.",
    )


class TaskIntentSchema(BaseModel):
    main: MainTargetSchema
    constraints: List[ConstraintItemSchema] = Field(
        default_factory=list,
        description="Objects / regions to avoid.",
    )
    preferences: List[PreferenceItemSchema] = Field(
        default_factory=list,
        description="Objects / regions to stay close to.",
    )
    behavior: BehaviorSchema = Field(
        default_factory=BehaviorSchema,
        description="Motion-style parameters.",
    )


class ParsedInstructionSchema(BaseModel):
    """Top-level schema sent to the LLM as ``response_format``."""
    instruction: str = Field(
        description="The original instruction, echoed back.",
    )
    tasks: List[TaskIntentSchema] = Field(
        description=(
            "Ordered list of navigation tasks decomposed from the instruction. "
            "Split on 'then', 'after that', 'next', 'and then'."
        ),
    )
    confidence: float = Field(
        default=0.9,
        description="Confidence in the parse (0-1).",
    )


# ============================================================================
# Conversion: Pydantic model → frozen dataclasses
# ============================================================================

_VALID_RELATION_TYPES = {
    "near", "alone"
}


def normalize_relation_type(value: str) -> str:
    key = (value or "near").strip().lower().replace("-", "_").replace(" ", "_")
    if key in _VALID_RELATION_TYPES:
        return key
    return "near"


def _to_object_ref(s: ObjectRefSchema) -> ObjectRef:
    return ObjectRef(name=s.name, type=normalize_relation_type(s.type))


_TERM_DIR_TO_DEG = {
    "front": 0.0,
    "left": 90.0,
    "right": -90.0,
    "back": 180.0,
}

_SIGNED_NUMBER_RE = r"(?:(negative|positive)\s+)?(-?\d+(?:\.\d+)?)"

_ANGLE_TOKEN_RE = re.compile(
    rf"{_SIGNED_NUMBER_RE}\s*(°|deg(?:ree)?s?|rad(?:ian)?s?)\b",
    re.IGNORECASE,
)

_ANGLE_BARE_RE = re.compile(
    rf"\bangle\s+{_SIGNED_NUMBER_RE}\b",
    re.IGNORECASE,
)

_TERM_DIR_AFTER_STOP_RE = re.compile(
    r"\b(?:stop|stopping|approach|arrive|end|finish|position|stand)\b[^.;,\n]{0,100}\b(front|back|left|right)\b",
    re.IGNORECASE,
)

_TERM_DIR_FROM_RE = re.compile(
    r"\bfrom\s+(?:the\s+)?(front|back|left|right)\b(?!\s+of\b)",
    re.IGNORECASE,
)


def _has_explicit_termination_phase_cue(instruction: str) -> bool:
    text = (instruction or "").strip().lower()
    if not text:
        return False
    return bool(
        _ANGLE_TOKEN_RE.search(text)
        or _ANGLE_BARE_RE.search(text)
        or _TERM_DIR_AFTER_STOP_RE.search(text)
        or _TERM_DIR_FROM_RE.search(text)
    )


def _normalize_phase_deg(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not (-3600.0 <= x <= 3600.0):
        return None
    # Canonicalize into [-180, 180).
    x = ((x + 180.0) % 360.0) - 180.0
    return x


def _parse_signed_number(sign_token: Optional[str], magnitude_token: Optional[str]) -> Optional[float]:
    if magnitude_token is None:
        return None
    try:
        value = float(magnitude_token)
    except (TypeError, ValueError):
        return None
    sign = (sign_token or "").strip().lower()
    if sign == "negative":
        value = -value
    return value


def _extract_termination_phase_deg(instruction: str) -> Optional[float]:
    text = (instruction or "").strip().lower()
    if not text:
        return None

    # Explicit numeric angles with units.
    for m in _ANGLE_TOKEN_RE.finditer(text):
        val = _parse_signed_number(m.group(1), m.group(2))
        if val is None:
            continue
        unit = m.group(3).lower()
        deg = math.degrees(val) if unit.startswith("rad") else val
        context = text[max(0, m.start() - 48): min(len(text), m.end() + 48)]
        if ("angle" in context) or re.search(r"\b(stop|approach|arrive|end|finish|position|from|at)\b", context):
            return _normalize_phase_deg(deg)

    # Fallback: "angle 45" (without deg/rad token).
    m = _ANGLE_BARE_RE.search(text)
    if m:
        val = _parse_signed_number(m.group(1), m.group(2))
        if val is not None:
            return _normalize_phase_deg(val)

    # Directional termination cues, separate from reference-direction usage.
    m = _TERM_DIR_AFTER_STOP_RE.search(text)
    if m:
        return _TERM_DIR_TO_DEG.get(m.group(1).lower())
    m = _TERM_DIR_FROM_RE.search(text)
    if m:
        return _TERM_DIR_TO_DEG.get(m.group(1).lower())
    return None


def _phase_label_to_deg(label: str) -> Optional[float]:
    key = (label or "").strip().lower()
    if key == "near":
        return None
    return _TERM_DIR_TO_DEG.get(key)


def _repair_phase_fields(term: Termination, instruction: str) -> Termination:
    # If parser marks phase explicit but we still have no usable numeric phase,
    # clear the explicit flag unless a strong angle cue exists.
    if term.phase_explicit and term.phase is None:
        if not _has_explicit_termination_phase_cue(instruction):
            return Termination(
                distance_m=term.distance_m,
                phase=term.phase,
                phase_explicit=False,
                stop_strictness=term.stop_strictness,
                stop_policy=term.stop_policy,
            )
    return term


def _to_termination(s: TerminationSchema, instruction: str) -> Termination:
    inst = instruction or ""
    has_term_cue = _has_explicit_termination_phase_cue(inst)

    # Guardrail: never trust schema phase fields unless explicit termination
    # cues appear in the user text. This prevents reference-direction words
    # (e.g. "right of red cube") from leaking into termination phase.
    phase_from_schema = _normalize_phase_deg(s.phase_deg) if has_term_cue else None
    if phase_from_schema is None and has_term_cue:
        phase_from_schema = _phase_label_to_deg(s.phase)

    phase_from_text = _extract_termination_phase_deg(inst)
    phase = phase_from_text if phase_from_text is not None else phase_from_schema
    phase_explicit = bool(phase is not None)

    return Termination(
        distance_m=s.distance_m,
        phase=phase,
        phase_explicit=phase_explicit,
        stop_strictness=s.stop_strictness,
        stop_policy=s.stop_policy,
    )


def _to_constraint(s: ConstraintItemSchema) -> ConstraintItem:
    return ConstraintItem(
        target=_to_object_ref(s.target),
        references=tuple(_to_object_ref(r) for r in s.references),
    )


def _to_preference(s: PreferenceItemSchema) -> PreferenceItem:
    return PreferenceItem(
        target=_to_object_ref(s.target),
        references=tuple(_to_object_ref(r) for r in s.references),
    )


def _to_behavior(s: BehaviorSchema) -> Behavior:
    return Behavior(
        speed=s.speed,
        caution=s.caution,
    )


def _to_task(s: TaskIntentSchema, instruction: str) -> TaskIntent:
    term = _to_termination(s.main.termination, instruction)
    term = _repair_phase_fields(term, instruction)
    return TaskIntent(
        main=MainTarget(
            # Main target relation type is always canonical "near".
            # Spatial disambiguation belongs in main.references[*].type.
            target=ObjectRef(name=s.main.target.name, type="near"),
            references=tuple(_to_object_ref(r) for r in s.main.references),
            termination=term,
        ),
        constraints=tuple(_to_constraint(c) for c in s.constraints),
        preferences=tuple(_to_preference(p) for p in s.preferences),
        behavior=_to_behavior(s.behavior),
    )


def _schema_to_parsed_instruction(
    s: ParsedInstructionSchema,
) -> ParsedInstruction:
    return ParsedInstruction(
        instruction=s.instruction,
        tasks=tuple(_to_task(t, s.instruction) for t in s.tasks),
        confidence=s.confidence,
    )


# ============================================================================
# Parser entry-point
# ============================================================================

_SYSTEM_PROMPT = """\
You are a robotic navigation intent parser.

Decompose the user's instruction into one or more sequential navigation tasks.
Split on markers like "then", "after that", "next", "and then".

For each task output:
- main: the primary target with name, spatial type, references for disambiguation,
  and termination (distance_m in metres, phase as "front"/"back"/"left"/"right"/"near",
  optional numeric phase_deg in degrees, stop_strictness, stop_policy). Do NOT include
  facing or yaw tolerance.
- constraints: objects to AVOID (implicit). Leave empty if none mentioned.
- preferences: objects to STAY CLOSE TO (implicit). Leave empty if none mentioned.
- behavior: speed and caution only.

Reference objects and their type field:
The reference `type` describes WHERE THE TARGET IS relative to the reference landmark.
- "near"  — target is close to the reference (default when relation is unspecified).
- "alone" — the target is described as standing by itself, isolated, or "the only one".
  When type is "alone", the reference name can be empty string "".

Examples of correct reference parsing:
- "the cup near the keyboard"
  → target: {name: "cup"}, references: [{name: "keyboard", type: "near"}]
- "the cup next to the keyboard"
  → target: {name: "cup"}, references: [{name: "keyboard", type: "near"}]
- "the ball standing by itself" or "the isolated ball" or "the lone ball"
  → target: {name: "ball"}, references: [{name: "", type: "alone"}]
- "the ball left alone" or "the ball that was left alone"
  → target: {name: "ball"}, references: [{name: "", type: "alone"}]
- "the sphere to the left of the cube, the one standing alone"
  → target: {name: "sphere"}, references: [{name: "cube", type: "near"}, {name: "", type: "alone"}]
- "go to the cylinder in front of the box, the one by itself"
  → target: {name: "cylinder"}, references: [{name: "box", type: "near"}, {name: "", type: "alone"}]
- "the ball at the back of the room"
  → target: {name: "ball"}, references: [{name: "room", type: "near"}]

IMPORTANT:
- Only two reference relation types are allowed: "near" and "alone".
- If users mention reference directions like left/right/front/back/top/bottom,
  ignore directional semantics and map those references to type="near".
- "left alone" in natural language means the object is by itself / isolated.
  Parse it as type: "alone".

Examples:
- "go to the blue sphere to the right of the red cube"
  → references include {name:"red cube", type:"near"}, termination.phase="near", phase_explicit=false.
- "go to the blue sphere and stop at the left side"
  → termination.phase="left", phase_explicit=true (reference types unchanged).
- "go to the blue sphere and stop at angle -45 degrees"
  → termination.phase="near", termination.phase_deg=-45, phase_explicit=true.

Rules:
- Termination always refers to the main target.
- Every object must have "name" and "type".
- The target's own type is ALWAYS "near". NEVER set the target type to "alone",
  and NEVER use directional relation types on references.
- When the target is described as "standing alone", "by itself", "isolated", etc.,
  you MUST add a reference with {name: "", type: "alone"} to the references list.
  Do NOT put "alone" on the target's type field.
  CORRECT:   target: {name: "yellow cylinder", type: "near"}, references: [{name: "", type: "alone"}]
  INCORRECT: target: {name: "yellow cylinder", type: "alone"}, references: []
- References default to type "near" unless the user clearly means isolation ("alone").
- If distance is not specified, default to 0.5 m.
- If phase is not specified, default to "near".
- Set termination.phase_explicit=true only when the user explicitly requests a phase/angle.
- If phase is not explicitly requested, keep phase="near" and phase_explicit=false.
- Speed and caution default to "normal" when unspecified.
- Provide a confidence value between 0 and 1.
"""


def parse_instruction(
    instruction: str,
    model: str = "gpt-4o",
) -> ParsedInstruction:
    """Use OpenAI structured outputs to parse *instruction* into a
    :class:`ParsedInstruction` (multi-task with constraints & preferences)."""
    client = OpenAI()

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        response_format=ParsedInstructionSchema,
    )

    parsed = completion.choices[0].message.parsed
    return _schema_to_parsed_instruction(parsed)
