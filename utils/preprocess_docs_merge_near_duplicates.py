#!/usr/bin/env python3
"""Merge near-duplicate semantic objects in docs.jsonl.

Rule:
- cosine similarity between CLIP embeddings >= sim_threshold
- distance between object centroids < dist_threshold_m

When two objects satisfy both, they are treated as the same object and merged
via connected components (union-find). Cluster centroids and CLIP features are
averaged.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ObjView:
    idx: int
    x: float
    y: float
    z: Optional[float]
    class_name: str
    clip_unit: Optional[np.ndarray]


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _extract_xyz(rec: dict) -> Optional[Tuple[float, float, Optional[float]]]:
    for key in ("global_position", "pos", "position", "xy"):
        v = rec.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            z = float(v[2]) if len(v) >= 3 else None
            return float(v[0]), float(v[1]), z
        if isinstance(v, dict) and "x" in v and "y" in v:
            z = float(v["z"]) if v.get("z") is not None else None
            return float(v["x"]), float(v["y"]), z
    xyz = rec.get("xyz")
    if isinstance(xyz, (list, tuple)) and len(xyz) >= 2:
        z = float(xyz[2]) if len(xyz) >= 3 else None
        return float(xyz[0]), float(xyz[1]), z
    return None


def _extract_clip(rec: dict) -> Optional[np.ndarray]:
    rf = rec.get("raw_fields") if isinstance(rec, dict) else None
    clip = None
    if isinstance(rf, dict):
        clip = rf.get("clip_ft")
    if clip is None:
        clip = rec.get("clip_ft")
    if not isinstance(clip, (list, tuple)) or len(clip) == 0:
        return None
    try:
        arr = np.asarray(clip, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return None
    return arr / norm


def _extract_vector(rec: dict, field_name: str) -> Optional[np.ndarray]:
    rf = rec.get("raw_fields") if isinstance(rec, dict) else None
    vec = None
    if isinstance(rf, dict):
        vec = rf.get(field_name)
    if vec is None:
        vec = rec.get(field_name)
    if not isinstance(vec, (list, tuple)) or len(vec) == 0:
        return None
    try:
        arr = np.asarray(vec, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    return arr


def _extract_points(rec: dict) -> Optional[np.ndarray]:
    pts = rec.get("points")
    if pts is None and isinstance(rec.get("raw_fields"), dict):
        pts = rec["raw_fields"].get("points")
    if not isinstance(pts, (list, tuple)) or len(pts) == 0:
        return None
    try:
        arr = np.asarray(pts, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def _rebuild_text(rec: dict) -> str:
    class_name = str(rec.get("class_name") or "unknown")
    class_id = str(rec.get("class_id") or class_name)
    pos = _extract_xyz(rec) or (0.0, 0.0, 0.0)
    points = _extract_points(rec)
    clip_ft = _extract_vector(rec, "clip_ft")
    text_ft = _extract_vector(rec, "text_ft")

    text = f"Object: {class_name}\n"
    text += f"ID: {class_id}\n"
    if pos[2] is None:
        text += f"Global Position (XYZ): ({pos[0]:.3f}, {pos[1]:.3f}, 0.000)\n"
    else:
        text += f"Global Position (XYZ): ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n"
    if points is not None:
        text += f"Has {len(points)} 3D point cloud points\n"
    if clip_ft is not None:
        text += f"Has CLIP embedding: {len(clip_ft)}D vector\n"
    if text_ft is not None:
        text += f"Has Text embedding: {len(text_ft)}D vector\n"
    text += f"Keywords: {class_name}\n"
    text += f"Type: {class_name}\n"
    if pos[2] is None:
        text += f"Located at: ({pos[0]:.3f}, {pos[1]:.3f}, 0.000)"
    else:
        text += f"Located at: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
    return text


def _load_records(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _build_views(records: Sequence[dict]) -> List[Optional[ObjView]]:
    views: List[Optional[ObjView]] = []
    for i, rec in enumerate(records):
        pos = _extract_xyz(rec)
        if pos is None:
            views.append(None)
            continue
        x, y, z = pos
        name = str(rec.get("class_name") or rec.get("label") or rec.get("name") or "").strip().lower()
        clip = _extract_clip(rec)
        views.append(ObjView(idx=i, x=x, y=y, z=z, class_name=name, clip_unit=clip))
    return views


def _distance(view_a: ObjView, view_b: ObjView) -> float:
    dx = view_a.x - view_b.x
    dy = view_a.y - view_b.y
    if view_a.z is not None and view_b.z is not None:
        dz = view_a.z - view_b.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    return math.sqrt(dx * dx + dy * dy)


def _merge_cluster(records: Sequence[dict], idxs: List[int], cluster_id: int) -> dict:
    merged = copy.deepcopy(records[idxs[0]])
    cluster_recs = [records[i] for i in idxs]

    class_names = [str(r.get("class_name") or "").strip() for r in cluster_recs if r.get("class_name")]
    chosen_class = Counter(class_names).most_common(1)[0][0] if class_names else str(merged.get("class_name") or "")

    xyzs = [_extract_xyz(r) for r in cluster_recs]
    xyzs = [p for p in xyzs if p is not None]
    if xyzs:
        xs = [p[0] for p in xyzs]
        ys = [p[1] for p in xyzs]
        zs = [p[2] for p in xyzs if p[2] is not None]
        mean_pos: List[float] = [float(np.mean(xs)), float(np.mean(ys))]
        if zs:
            mean_pos.append(float(np.mean(zs)))
        merged["global_position"] = mean_pos
        rf = merged.setdefault("raw_fields", {})
        rf["global_position"] = mean_pos

    points_vecs = [_extract_points(r) for r in cluster_recs]
    points_vecs = [pts for pts in points_vecs if pts is not None and len(pts) > 0]
    if points_vecs:
        merged_points = np.concatenate(points_vecs, axis=0)
        merged_points_list = merged_points.astype(float).tolist()
        merged["points"] = merged_points_list
        rf = merged.setdefault("raw_fields", {})
        rf["points"] = merged_points_list

    for field_name in ("clip_ft", "text_ft"):
        vecs = []
        for r in cluster_recs:
            arr = _extract_vector(r, field_name)
            if arr is not None:
                vecs.append(arr)
        if vecs:
            mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
            rf = merged.setdefault("raw_fields", {})
            rf[field_name] = mean_vec.astype(float).tolist()

    if chosen_class:
        merged["class_name"] = chosen_class
        rf = merged.setdefault("raw_fields", {})
        rf["class_name"] = chosen_class
        merged["class_id"] = f"{chosen_class}_{cluster_id}"
        rf["class_id"] = merged["class_id"]

    merged["merge_meta"] = {
        "merged_count": len(idxs),
        "source_indices": idxs,
        "cluster_id": cluster_id,
    }
    merged["text"] = _rebuild_text(merged)
    return merged


def merge_docs(
    records: Sequence[dict],
    sim_threshold: float,
    dist_threshold_m: float,
    verbose: bool = False,
) -> Tuple[List[dict], dict]:
    n = len(records)
    views = _build_views(records)
    uf = UnionFind(n)

    eligible = [v for v in views if v is not None and v.clip_unit is not None]
    if verbose:
        print(f"[INFO] total={n}, eligible_for_merge={len(eligible)}")

    cell_size = max(dist_threshold_m, 1e-6)
    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    pair_checks = 0
    union_hits = 0
    for view in eligible:
        cx = int(math.floor(view.x / cell_size))
        cy = int(math.floor(view.y / cell_size))

        candidates: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(grid.get((cx + dx, cy + dy), []))

        for j in candidates:
            other = views[j]
            if other is None or other.clip_unit is None:
                continue
            if _distance(view, other) >= dist_threshold_m:
                continue
            pair_checks += 1
            sim = float(np.dot(view.clip_unit, other.clip_unit))
            if sim >= sim_threshold:
                if uf.union(view.idx, j):
                    union_hits += 1

        grid[(cx, cy)].append(view.idx)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    clusters = sorted(groups.values(), key=lambda g: g[0])
    out_records: List[dict] = []
    merged_clusters = 0
    removed_total = 0
    for cluster_id, idxs in enumerate(clusters, 1):
        if len(idxs) == 1:
            out_records.append(copy.deepcopy(records[idxs[0]]))
            continue
        out_records.append(_merge_cluster(records, idxs, cluster_id))
        merged_clusters += 1
        removed_total += (len(idxs) - 1)

    stats = {
        "input_records": n,
        "output_records": len(out_records),
        "eligible_records": len(eligible),
        "pair_checks_within_distance": pair_checks,
        "union_hits": union_hits,
        "merged_clusters": merged_clusters,
        "removed_records": removed_total,
        "sim_threshold": sim_threshold,
        "dist_threshold_m": dist_threshold_m,
    }
    return out_records, stats


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def postprocess_docs_jsonl(
    docs_path: Path,
    *,
    output_path: Optional[Path] = None,
    sim_threshold: float = 0.8,
    dist_threshold_m: float = 0.10,
    rebuild_tfidf: bool = False,
) -> Tuple[Path, dict]:
    docs_path = Path(docs_path).expanduser().resolve()
    if not docs_path.exists():
        raise FileNotFoundError(f"Input not found: {docs_path}")

    out_path = Path(output_path).expanduser().resolve() if output_path else docs_path
    records = _load_records(docs_path)
    merged, stats = merge_docs(
        records=records,
        sim_threshold=float(sim_threshold),
        dist_threshold_m=float(dist_threshold_m),
        verbose=False,
    )
    _write_jsonl(out_path, merged)

    if rebuild_tfidf:
        try:
            from utils.config import ensure_dovsg_python_path

            ensure_dovsg_python_path()
            from semantic_field_navigation.utils.instance_object_to_json import create_tfidf_artifacts

            if merged:
                create_tfidf_artifacts(merged, out_path.parent)
            else:
                for artifact in ("tfidf_vectorizer.pkl", "tfidf_matrix.npz"):
                    artifact_path = out_path.parent / artifact
                    if artifact_path.exists():
                        artifact_path.unlink()
        except Exception as exc:
            print(f"[WARN] Failed to rebuild TF-IDF artifacts after docs merge: {exc}")

    return out_path, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge near-duplicate semantic objects in docs.jsonl when "
            "cosine(clip_ft)>=sim-threshold and centroid distance<dist-threshold-m."
        )
    )
    parser.add_argument("--input", required=True, help="Input docs.jsonl path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output docs.jsonl path (default: <input_stem>_merged.jsonl).",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.8,
        help="Semantic cosine threshold (default: 0.8).",
    )
    parser.add_argument(
        "--dist-threshold-m",
        type=float,
        default=0.10,
        help="Distance threshold in metres (default: 0.10).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file in place.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress.",
    )
    parser.add_argument(
        "--rebuild-tfidf",
        action="store_true",
        help="Rebuild tfidf_vectorizer.pkl and tfidf_matrix.npz next to the output docs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    if args.in_place:
        final_out_path = in_path
        tmp_out = in_path.with_suffix(in_path.suffix + ".tmp")
    else:
        if args.output:
            final_out_path = Path(args.output).expanduser().resolve()
        else:
            final_out_path = in_path.with_name(f"{in_path.stem}_merged{in_path.suffix}")
        tmp_out = final_out_path

    written_path, stats = postprocess_docs_jsonl(
        docs_path=in_path,
        output_path=tmp_out,
        sim_threshold=float(args.sim_threshold),
        dist_threshold_m=float(args.dist_threshold_m),
        rebuild_tfidf=bool(args.rebuild_tfidf),
    )
    if args.in_place:
        written_path.replace(final_out_path)
        written_path = final_out_path

    print("[DONE] docs preprocessing complete")
    print(f"  input:  {in_path}")
    print(f"  output: {written_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
