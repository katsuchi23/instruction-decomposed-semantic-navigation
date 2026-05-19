"""Semantic object retrieval — CLIP-based with spatial-reference disambiguation.

When multiple instances of an object exist in the scene (e.g. two red cubes),
reference landmarks with relations (near / alone) are used to pick the
correct one.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.config import (
    DOVSG_ROOT,
    PROJECT_ROOT,
    ensure_dovsg_python_path,
    get_clip_checkpoint_file,
    get_docs_path_from_config,
    get_param,
)
from parsing.intent_parser import ObjectRef, normalize_relation_type


# ============================================================================
# Docs-path resolution
# ============================================================================

def resolve_docs_path(cli_path: Optional[str] = None) -> Optional[Path]:
    """Search known locations for ``docs.jsonl``.  Returns the first match."""
    candidates: List[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))

    env_docs = os.getenv("SEMNAV_DOCS_PATH")
    if env_docs:
        candidates.append(Path(env_docs))

    configured_docs = get_docs_path_from_config()
    if configured_docs is not None:
        candidates.append(configured_docs)

    candidates.extend([
        PROJECT_ROOT / "data/testing_ground/memory/30_0.1_0.02_True_0.2_0.5/step_0/data_json/docs.jsonl",
        PROJECT_ROOT / "data_example/testing_ground/memory/30_0.1_0.02_True_0.2_0.5/step_0/data_json/docs.jsonl",
        DOVSG_ROOT / "data_example" / "testing_ground" / "memory" / "30_0.1_0.02_True_0.2_0.5" / "step_0" / "data_json" / "docs.jsonl",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# ============================================================================
# Internal data structures
# ============================================================================

@dataclass
class _SceneObject:
    """One object loaded from docs.jsonl."""
    index: int
    class_id: str
    class_name: str
    xy: Tuple[float, float]
    z: Optional[float]
    clip_feat: Any  # torch.Tensor (1024-D)
    record: Dict[str, Any]


@dataclass(frozen=True)
class RetrievedObjectMatch:
    """Resolved scene object plus debug metadata."""
    index: int
    class_id: str
    class_name: str
    xy: Tuple[float, float]
    z: Optional[float]


# ============================================================================
# Helpers
# ============================================================================

def _extract_position_xy(record: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for key in ("global_position", "pos", "position", "xy"):
        value = record.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
        if isinstance(value, dict) and "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
    xyz = record.get("xyz")
    if isinstance(xyz, (list, tuple)) and len(xyz) >= 2:
        return float(xyz[0]), float(xyz[1])
    return None


def _extract_position_xyz(record: Dict[str, Any]) -> Optional[Tuple[float, float, Optional[float]]]:
    for key in ("global_position", "pos", "position", "xy"):
        value = record.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            z = float(value[2]) if len(value) >= 3 else None
            return float(value[0]), float(value[1]), z
        if isinstance(value, dict) and "x" in value and "y" in value:
            z_raw = value.get("z")
            z = float(z_raw) if z_raw is not None else None
            return float(value["x"]), float(value["y"]), z
    xyz = record.get("xyz")
    if isinstance(xyz, (list, tuple)) and len(xyz) >= 2:
        z = float(xyz[2]) if len(xyz) >= 3 else None
        return float(xyz[0]), float(xyz[1]), z
    return None


def _extract_class_id(record: Dict[str, Any], fallback_index: int) -> str:
    raw_fields = record.get("raw_fields") if isinstance(record, dict) else None
    class_id = record.get("class_id")
    if class_id is None and isinstance(raw_fields, dict):
        class_id = raw_fields.get("class_id")
    if class_id is not None and str(class_id).strip():
        return str(class_id)

    class_name = str(
        record.get("class_name") or record.get("label") or record.get("name") or "object"
    ).strip()
    obj_index = record.get("obj_index")
    if obj_index is not None:
        return f"{class_name}_{obj_index}"
    return f"{class_name}_{fallback_index}"


def _load_scene_objects(docs_path: Path) -> List[_SceneObject]:
    """Load every object in ``docs.jsonl`` that has a CLIP feature and XY."""
    import torch

    objects: List[_SceneObject] = []
    with docs_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            pos_xyz = _extract_position_xyz(rec)
            rf = rec.get("raw_fields") if isinstance(rec, dict) else None
            clip_ft = rf.get("clip_ft") if isinstance(rf, dict) else None
            if pos_xyz is None or clip_ft is None:
                continue
            x, y, z = pos_xyz

            objects.append(_SceneObject(
                index=idx,
                class_id=_extract_class_id(rec, idx),
                class_name=str(
                    rec.get("class_name") or rec.get("label") or rec.get("name") or ""
                ).lower(),
                xy=(x, y),
                z=z,
                clip_feat=torch.tensor(clip_ft, device="cpu"),
                record=rec,
            ))
    return objects


def _clip_similarities(query: str, objects: List[_SceneObject], myclip: Any):
    """Return a 1-D tensor of cosine similarities between *query* and each object."""
    import torch
    import torch.nn.functional as F

    if not objects:
        return torch.tensor([])

    with torch.no_grad():
        query_feat = myclip.get_text_feature([query]).squeeze(0)
        feats = torch.stack([o.clip_feat for o in objects])
        sims = F.cosine_similarity(query_feat.unsqueeze(0), feats, dim=-1)
    return sims


def _scene_object_to_match(obj: _SceneObject) -> RetrievedObjectMatch:
    return RetrievedObjectMatch(
        index=obj.index,
        class_id=obj.class_id,
        class_name=obj.class_name,
        xy=obj.xy,
        z=obj.z,
    )


# ============================================================================
# Spatial relation scoring
# ============================================================================

# Default length-scale for the proximity Gaussian (metres).
_PROXIMITY_SCALE = 1.0
_DIRECTION_MIN_DIST_M = 0.30  # compatibility knob (directional refs are disabled)

# Number of top reference candidates (by CLIP similarity) to consider.
# Using a fixed shortlist prevents per-target "reference drift", where each
# target candidate picks a different arbitrary reference object.
_REF_TOP_K = 8

# Length-scale for isolation scoring (metres).
# Controls how quickly isolation score rises with distance from neighbours.
_ISOLATION_SCALE = 0.5
_ALONE_CLIP_MAX_GAP = 0.12
_ALONE_MAX_CANDIDATES = 12
_ALONE_CLEARANCE_GATE_M = 0.20
_TARGET_TOP_K_WITH_REFERENCES = 8
_CLIP_GATE_START = 0.27
_CLIP_GATE_MIN = 0.20
_CLIP_GATE_STEP = 0.01
_REFERENCE_MAX_DIST_M = 0.50
_REFERENCE_SCORE_GAIN = 1.50


def set_direction_min_distance_m(distance_m: float) -> None:
    """Compatibility setter; directional reference scoring is disabled."""
    global _DIRECTION_MIN_DIST_M
    if distance_m < 0.0:
        raise ValueError("direction_min_distance_m must be >= 0")
    _DIRECTION_MIN_DIST_M = float(distance_m)


def get_direction_min_distance_m() -> float:
    """Compatibility getter; directional reference scoring is disabled."""
    return float(_DIRECTION_MIN_DIST_M)


def configure_object_retrieval_params() -> None:
    """Load all retrieval parameters from config into module-level globals."""
    global _DIRECTION_MIN_DIST_M, _PROXIMITY_SCALE, _REF_TOP_K, _ISOLATION_SCALE
    global _ALONE_CLIP_MAX_GAP, _ALONE_MAX_CANDIDATES, _ALONE_CLEARANCE_GATE_M
    global _TARGET_TOP_K_WITH_REFERENCES, _CLIP_GATE_START, _CLIP_GATE_MIN
    global _CLIP_GATE_STEP, _REFERENCE_MAX_DIST_M, _REFERENCE_SCORE_GAIN
    p = ("object_retrieval",)
    _DIRECTION_MIN_DIST_M      = float(get_param(p + ("direction_min_distance_m",),   _DIRECTION_MIN_DIST_M))
    _PROXIMITY_SCALE           = float(get_param(p + ("proximity_scale_m",),          _PROXIMITY_SCALE))
    _REF_TOP_K                 = int(  get_param(p + ("reference_top_k",),            _REF_TOP_K))
    _ISOLATION_SCALE           = float(get_param(p + ("isolation_scale_m",),          _ISOLATION_SCALE))
    _ALONE_CLIP_MAX_GAP        = float(get_param(p + ("alone_clip_max_gap",),         _ALONE_CLIP_MAX_GAP))
    _ALONE_MAX_CANDIDATES      = int(  get_param(p + ("alone_max_candidates",),       _ALONE_MAX_CANDIDATES))
    _ALONE_CLEARANCE_GATE_M    = float(get_param(p + ("alone_clearance_gate_m",),     _ALONE_CLEARANCE_GATE_M))
    _TARGET_TOP_K_WITH_REFERENCES = int(get_param(p + ("target_top_k_with_references",), _TARGET_TOP_K_WITH_REFERENCES))
    _CLIP_GATE_START           = float(get_param(p + ("clip_gate_start",),            _CLIP_GATE_START))
    _CLIP_GATE_MIN             = float(get_param(p + ("clip_gate_min",),              _CLIP_GATE_MIN))
    _CLIP_GATE_STEP            = float(get_param(p + ("clip_gate_step",),             _CLIP_GATE_STEP))
    _REFERENCE_MAX_DIST_M      = float(get_param(p + ("reference_max_dist_m",),       _REFERENCE_MAX_DIST_M))
    _REFERENCE_SCORE_GAIN      = float(get_param(p + ("reference_score_gain",),       _REFERENCE_SCORE_GAIN))


def _spatial_relation_score(
    target_xy: Tuple[float, float],
    ref_xy: Tuple[float, float],
    relation_type: str,
    target_z: Optional[float] = None,
    ref_z: Optional[float] = None,
    *,
    enforce_direction_min_dist: bool = True,
) -> float:
    """Score how well *target_xy* satisfies ``relation_type`` relative to *ref_xy*.

    Only ``near`` is supported here. Any non-``alone`` relation is treated
    as ``near`` to keep retrieval behavior deterministic.
    """
    _ = (target_z, ref_z, enforce_direction_min_dist)  # compatibility params
    dx = target_xy[0] - ref_xy[0]
    dy = target_xy[1] - ref_xy[1]
    dist = math.sqrt(dx * dx + dy * dy)

    relation = normalize_relation_type(relation_type)
    proximity = math.exp(-dist / _PROXIMITY_SCALE)
    if relation == "alone":
        return 0.0
    if dist > _REFERENCE_MAX_DIST_M:
        return 0.0
    return proximity


def _isolation_score(
    cand_idx: int,
    objects: List[_SceneObject],
) -> float:
    """Score how isolated candidate *cand_idx* is from all other scene objects.

    Returns a value in [0, 1].  Higher means more isolated.
    Uses ``1 - exp(-min_dist / _ISOLATION_SCALE)`` so that objects with no
    close neighbours score near 1.0.
    """
    cand = objects[cand_idx]
    min_dist = float("inf")
    for m, obj_m in enumerate(objects):
        if m == cand_idx:
            continue
        dx = cand.xy[0] - obj_m.xy[0]
        dy = cand.xy[1] - obj_m.xy[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d < min_dist:
            min_dist = d
    if min_dist == float("inf"):
        return 1.0  # only object in the scene
    if min_dist < _ALONE_CLEARANCE_GATE_M:
        # "alone" requires a hard local-clearance gate.
        return 0.0
    return 1.0 - math.exp(-min_dist / _ISOLATION_SCALE)


# ============================================================================
# Reference-aware retrieval (CLIP + spatial)
# ============================================================================

def _retrieve_with_references(
    target: ObjectRef,
    references: Tuple[ObjectRef, ...],
    docs_path: Path,
    top_k: int = 5,
    clip_weight: float = 0.3,
) -> Optional[RetrievedObjectMatch]:
    """CLIP retrieval with spatial-reference disambiguation.

    Algorithm
    ---------
    1. Load all scene objects (CLIP features + XY positions).
    2. Rank objects by CLIP similarity to ``target.name`` → take top-K candidates.
       When any reference is ``"alone"``, candidate search is widened to all objects.
    3. If there are no references, return the top-1 position immediately.
    4. For each reference in *references*:
       a. If ``ref.type == "alone"``: no CLIP similarity needed.
       b. Else: compute CLIP similarity of **all** scene objects to ``ref.name``,
          keep a fixed top reference-candidate pool.
    5. For each target candidate *i*:
       a. For each reference *j*:
          - If ``alone``: compute isolation score for candidate *i*.
          - Else: find the scene object *m* in the fixed reference pool that maximises
            ``ref_score = clip_sim(ref_j, obj_m) × spatial(candidate_i, obj_m, ref_j.type)``
       b. ``reference_total_i`` = geometric mean of per-reference best scores.
    6. ``final_score_i = clip_weight × target_clip_sim_i + (1 - clip_weight) × (REFERENCE_SCORE_GAIN × reference_total_i)``
       Raw (un-normalised) CLIP cosine similarities are used so that tiny
       differences between close candidates don't get amplified.
    7. Return position of the candidate with highest ``final_score``.
    """
    try:
        ensure_dovsg_python_path()
        from dovsg.utils import utils as dovsg_utils
        import dovsg.perception.models.myclip as myclip_module
    except Exception as _e:
        import traceback
        print(f"[ERROR] DovSG/CLIP import failed — retrieval unavailable: {_e}")
        traceback.print_exc()
        return None

    clip_checkpoint = get_clip_checkpoint_file().resolve()
    if not clip_checkpoint.exists():
        raise FileNotFoundError(f"CLIP checkpoint not found: {clip_checkpoint}")

    # DovSG hardcodes the checkpoint path as a module-level variable.
    # Override it here so this repository owns the runtime CLIP checkpoint path.
    dovsg_utils.clip_checkpoint_path = str(clip_checkpoint)
    myclip_module.clip_checkpoint_path = str(clip_checkpoint)
    MyClip = myclip_module.MyClip
    if getattr(MyClip, "_instance", None) is not None:
        MyClip._instance = None

    objects = _load_scene_objects(docs_path)
    if not objects:
        return None

    myclip = MyClip(device="cpu")

    # ── Step 2: target CLIP similarities ──────────────────────────────
    target_sims = _clip_similarities(target.name, objects, myclip)
    if target_sims.numel() == 0:
        return None

    # Adaptive CLIP gate schedule:
    # try stricter semantic filtering first, then relax incrementally.
    gate_values: List[float] = []
    g = _CLIP_GATE_START
    while g >= (_CLIP_GATE_MIN - 1e-9):
        gate_values.append(round(g, 4))
        g -= _CLIP_GATE_STEP

    import torch

    norm_ref_types = [normalize_relation_type(ref.type) for ref in references]
    has_alone = any(t == "alone" for t in norm_ref_types)
    for clip_gate in gate_values:
        target_gate_mask = target_sims >= clip_gate
        if int(target_gate_mask.sum().item()) <= 0:
            continue

        gated_indices = torch.nonzero(target_gate_mask, as_tuple=False).flatten()
        gated_order = torch.argsort(target_sims[gated_indices], descending=True)
        sorted_indices = gated_indices[gated_order]
        sorted_vals = target_sims[sorted_indices]

        if has_alone:
            clip_max = float(sorted_vals[0].item())
            clip_rel_gate = clip_max - _ALONE_CLIP_MAX_GAP
            clip_mask = sorted_vals >= clip_rel_gate
            clip_indices = sorted_indices[clip_mask]
            clip_vals = sorted_vals[clip_mask]

            if clip_indices.numel() > _ALONE_MAX_CANDIDATES:
                clip_indices = clip_indices[:_ALONE_MAX_CANDIDATES]
                clip_vals = clip_vals[:_ALONE_MAX_CANDIDATES]

            clearance_keep: List[int] = []
            clearance_vals: List[float] = []
            for idx, val in zip(clip_indices.tolist(), clip_vals.tolist()):
                if _isolation_score(int(idx), objects) > 0.0:
                    clearance_keep.append(int(idx))
                    clearance_vals.append(float(val))

            if not clearance_keep:
                print(
                    "  [Retrieval][WARN] Alone gate failed at "
                    f"clip>={clip_gate:.2f}: kept={int(clip_indices.numel())}/{len(objects)}, "
                    f"none satisfied clearance >= {_ALONE_CLEARANCE_GATE_M:.2f}m."
                )
                continue

            topk_indices = torch.tensor(clearance_keep, dtype=sorted_indices.dtype)
            topk_vals = torch.tensor(clearance_vals, dtype=sorted_vals.dtype)
            print(
                "  [Retrieval] Alone gate: "
                f"clip>={clip_gate:.2f}, clip max={clip_max:.4f}, "
                f"rel_gate={clip_rel_gate:.4f}, clearance kept={int(topk_indices.numel())}"
            )
        else:
            has_any_reference = len(references) > 0
            k_plain = min(
                int(sorted_indices.numel()),
                max(top_k, _TARGET_TOP_K_WITH_REFERENCES) if has_any_reference else top_k,
            )
            topk_indices = sorted_indices[:k_plain]
            topk_vals = sorted_vals[:k_plain]

        k = int(topk_indices.numel())
        if k <= 0:
            continue

        # ── No references: return top-1 at the first gate with candidates ─
        if not references:
            best_idx = topk_indices[0].item()
            best_obj = objects[best_idx]
            print(f"  [Retrieval] No references (clip>={clip_gate:.2f}). Top-1 CLIP match: "
                  f"'{best_obj.class_name}' class_id={best_obj.class_id} "
                  f"at ({best_obj.xy[0]:.3f}, {best_obj.xy[1]:.3f}) "
                  f"sim={topk_vals[0].item():.4f}")
            return _scene_object_to_match(best_obj)

        # ── Reference CLIP similarities with same gate ───────────────────
        ref_sims_list = []  # one similarity tensor per reference (None for 'alone')
        ref_candidate_pool_list: List[Optional[List[Tuple[int, float]]]] = []
        for ref in references:
            relation = normalize_relation_type(ref.type)
            if relation == "alone":
                ref_sims_list.append(None)
                ref_candidate_pool_list.append(None)
                continue

            sims = _clip_similarities(ref.name, objects, myclip)
            ref_sims_list.append(sims)
            if sims.numel() == 0:
                ref_candidate_pool_list.append([])
                continue

            ref_gate_mask = sims >= clip_gate
            if int(ref_gate_mask.sum().item()) <= 0:
                ref_candidate_pool_list.append([])
                continue
            ref_gated_indices = torch.nonzero(ref_gate_mask, as_tuple=False).flatten()
            ref_order = torch.argsort(sims[ref_gated_indices], descending=True)
            ref_sorted_indices = ref_gated_indices[ref_order]
            k_ref = min(_REF_TOP_K, int(ref_sorted_indices.numel()))
            idxs = ref_sorted_indices[:k_ref]
            vals = sims[idxs]
            ref_pool: List[Tuple[int, float]] = []
            for raw_sim, idx in zip(vals.tolist(), idxs.tolist()):
                ref_clip_w = max(0.0, min(1.0, 0.5 * (float(raw_sim) + 1.0)))
                ref_pool.append((int(idx), ref_clip_w))
            ref_candidate_pool_list.append(ref_pool)

        # ── Score candidates and require all references to be satisfiable ─
        best_final_score = -1.0
        best_candidate = None
        scoring_details: List[str] = []

        for rank in range(k):
            cand_idx = topk_indices[rank].item()
            cand = objects[cand_idx]
            cand_clip_raw = topk_vals[rank].item()

            ref_scores: List[float] = []
            ref_details: List[str] = []
            relation_failed = False
            for ref_j, (ref, ref_sims, ref_pool) in enumerate(
                zip(references, ref_sims_list, ref_candidate_pool_list)
            ):
                relation = normalize_relation_type(ref.type)
                if relation == "alone":
                    iso_score = _isolation_score(cand_idx, objects)
                    if iso_score <= 0.0:
                        relation_failed = True
                    ref_scores.append(iso_score)
                    ref_details.append(f"ref[{ref_j}] (alone): isolation={iso_score:.4f}")
                    continue

                assert ref_sims is not None
                assert ref_pool is not None
                best_ref_score = 0.0
                best_ref_match = ""
                for m, ref_clip in ref_pool:
                    if m == cand_idx:
                        continue
                    obj_m = objects[m]
                    strict_spatial = _spatial_relation_score(
                        cand.xy,
                        obj_m.xy,
                        relation,
                        target_z=cand.z,
                        ref_z=obj_m.z,
                        enforce_direction_min_dist=True,
                    )
                    strict_combined = ref_clip * strict_spatial
                    if strict_combined > best_ref_score:
                        best_ref_score = strict_combined
                        best_ref_match = (
                            f"{obj_m.class_name}[{obj_m.class_id}]@({obj_m.xy[0]:.2f},{obj_m.xy[1]:.2f})"
                        )
                if best_ref_score <= 0.0:
                    relation_failed = True
                ref_scores.append(best_ref_score)
                ref_details.append(
                    f"ref[{ref_j}] '{ref.name}'({relation}): "
                    f"score={best_ref_score:.4f} via {best_ref_match}"
                )

            if relation_failed:
                detail = (
                    f"  candidate '{cand.class_name}' "
                    f"pos=({cand.xy[0]:.3f},{cand.xy[1]:.3f}) "
                    f"clip={cand_clip_raw:.4f} INVALID(ref-unsatisfied) | " + " | ".join(ref_details)
                )
                scoring_details.append(detail)
                continue

            log_sum = sum(math.log(max(s, 1e-12)) for s in ref_scores)
            ref_total = math.exp(log_sum / len(ref_scores)) if ref_scores else 0.0
            boosted_ref_total = _REFERENCE_SCORE_GAIN * ref_total
            final_score = clip_weight * cand_clip_raw + (1.0 - clip_weight) * boosted_ref_total

            detail = (
                f"  candidate '{cand.class_name}' "
                f"pos=({cand.xy[0]:.3f},{cand.xy[1]:.3f}) "
                f"clip={cand_clip_raw:.4f} ref_total={ref_total:.4f} "
                f"ref_boosted={boosted_ref_total:.4f} "
                f"final={final_score:.4f} | " + " | ".join(ref_details)
            )
            scoring_details.append(detail)

            if final_score > best_final_score:
                best_final_score = final_score
                best_candidate = cand

        print(f"  [Retrieval] Reference-aware scoring (clip>={clip_gate:.2f}):")
        for d in scoring_details:
            print(d)

        if best_candidate is not None:
            print(f"  [Retrieval] Selected: '{best_candidate.class_name}' "
                  f"class_id={best_candidate.class_id} "
                  f"at ({best_candidate.xy[0]:.3f}, {best_candidate.xy[1]:.3f}) "
                  f"score={best_final_score:.4f}")
            return _scene_object_to_match(best_candidate)

        print(
            "  [Retrieval][WARN] No candidate satisfied all reference constraints at "
            f"clip>={clip_gate:.2f}; lowering clip gate."
        )

    print(
        "  [Retrieval][WARN] Exhausted adaptive clip gates "
        f"({_CLIP_GATE_START:.2f} -> {_CLIP_GATE_MIN:.2f}) with no valid match."
    )
    return None


# ============================================================================
# Fallback: keyword overlap (no CLIP)
# ============================================================================

def _retrieve_fallback(target_name: str, docs_path: Path) -> Optional[Tuple[float, float]]:
    """Keyword-overlap heuristic when CLIP is unavailable."""
    tokens = {t for t in target_name.lower().replace(",", " ").split() if len(t) > 2}
    best_score = -1
    best_xy: Optional[Tuple[float, float]] = None

    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            xy = _extract_position_xy(rec)
            if xy is None:
                continue

            label = str(
                rec.get("class_name") or rec.get("label") or rec.get("name") or rec.get("text") or ""
            ).lower()
            label_tokens = {t for t in label.replace(",", " ").split() if len(t) > 2}
            overlap = len(tokens & label_tokens)

            if overlap > best_score:
                best_score = overlap
                best_xy = xy

    return best_xy


# ============================================================================
# Public API
# ============================================================================

def retrieve_object_location(
    target: ObjectRef,
    references: Tuple[ObjectRef, ...],
    docs_path: Path,
    *,
    allow_fallback: bool = False,
) -> Tuple[float, float]:
    """Return *(x, y)* of the best-matching object for *target*, disambiguated
    by *references* and their spatial relations.

    Retrieval is fail-safe: if semantic grounding cannot find a valid match,
    raise :class:`LookupError` instead of returning a synthetic coordinate.
    """
    return retrieve_object_match(
        target,
        references,
        docs_path,
        allow_fallback=allow_fallback,
    ).xy


def retrieve_object_match(
    target: ObjectRef,
    references: Tuple[ObjectRef, ...],
    docs_path: Path,
    *,
    allow_fallback: bool = False,
) -> RetrievedObjectMatch:
    """Return the best semantic match with debug metadata."""
    top_k = int(get_param(("object_retrieval", "top_k"), 5))
    clip_weight = float(get_param(("object_retrieval", "clip_weight"), 0.3))
    match = _retrieve_with_references(target, references, docs_path, top_k=top_k, clip_weight=clip_weight)
    if match is not None:
        return match

    has_alone_ref = any(normalize_relation_type(r.type) == "alone" for r in references)
    if has_alone_ref:
        raise LookupError(
            "No semantic match passed 'alone' gates for target "
            f"'{target.name}' (CLIP gate first, then clearance >= {_ALONE_CLEARANCE_GATE_M:.2f}m) "
            f"in {docs_path}"
        )

    if allow_fallback:
        fb = _retrieve_fallback(target.name, docs_path)
        if fb is not None:
            return RetrievedObjectMatch(
                index=-1,
                class_id="fallback_keyword_overlap",
                class_name=target.name.lower(),
                xy=fb,
                z=None,
            )

    raise LookupError(
        f"No semantic match found for target '{target.name}' in {docs_path}"
    )
