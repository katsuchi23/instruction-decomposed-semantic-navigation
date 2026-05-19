#!/usr/bin/env python3
"""Export a DovSG-style RGB-D scene directory from a ROS 2 bag."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Export rgb/, depth/, poses/, calibration/, timestamp/, metadata.json, "
            "and calib*.txt for DovSG from an RGB-D rosbag."
        )
    )
    ap.add_argument(
        "--bag_dir",
        type=Path,
        required=True,
        help="ROS 2 bag directory.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output scene directory.",
    )
    ap.add_argument("--rgb_topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--depth_topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    ap.add_argument("--camera_info_topic", default="/camera/camera/aligned_depth_to_color/camera_info")
    ap.add_argument("--localization_topic", default="/localization")
    ap.add_argument("--tf_static_topic", default="/tf_static")
    ap.add_argument("--map_frame", default="map")
    ap.add_argument("--body_frame", default="body")
    ap.add_argument("--base_link_frame", default="base_link")
    ap.add_argument("--frame_stride", type=int, default=1, help="Keep every Nth RGB frame.")
    ap.add_argument("--max_frames", type=int, default=0, help="0 means no limit.")
    ap.add_argument("--max_pair_ms", type=float, default=50.0)
    ap.add_argument("--jpg_quality", type=int, default=95)
    ap.add_argument("--min_depth", type=float, default=0.0)
    ap.add_argument("--max_depth", type=float, default=5.0)
    ap.add_argument("--depth_scale", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--camera_tx", type=float, default=0.056744402376091595)
    ap.add_argument("--camera_ty", type=float, default=0.0175)
    ap.add_argument("--camera_tz", type=float, default=0.01598012825293377)
    ap.add_argument("--camera_roll", type=float, default=0.0)
    ap.add_argument("--camera_pitch", type=float, default=1.2)
    ap.add_argument("--camera_yaw", type=float, default=0.0)
    return ap.parse_args()


def header_stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def quat_slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quat(q0 + u * (q1 - q0))
    theta_0 = math.acos(dot)
    sin_0 = math.sin(theta_0)
    theta = theta_0 * u
    s0 = math.sin(theta_0 - theta) / sin_0
    s1 = math.sin(theta) / sin_0
    return normalize_quat(s0 * q0 + s1 * q1)


def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def T_from_tr_quat(tx, ty, tz, qx, qy, qz, qw) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_R(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T


class PoseTimeline:
    def __init__(self) -> None:
        self.t = []
        self.trans = []
        self.quat = []

    def add(self, t_ns: int, trans, quat) -> None:
        self.t.append(int(t_ns))
        self.trans.append(np.array(trans, dtype=np.float64))
        self.quat.append(normalize_quat(np.array(quat, dtype=np.float64)))

    def sort(self) -> None:
        if not self.t:
            return
        idx = np.argsort(self.t)
        self.t = [self.t[i] for i in idx]
        self.trans = [self.trans[i] for i in idx]
        self.quat = [self.quat[i] for i in idx]

    def lookup(self, t_ns: int) -> np.ndarray | None:
        if not self.t:
            return None
        if t_ns <= self.t[0]:
            tr = self.trans[0]
            q = self.quat[0]
            return T_from_tr_quat(tr[0], tr[1], tr[2], q[0], q[1], q[2], q[3])
        if t_ns >= self.t[-1]:
            tr = self.trans[-1]
            q = self.quat[-1]
            return T_from_tr_quat(tr[0], tr[1], tr[2], q[0], q[1], q[2], q[3])

        lo, hi = 0, len(self.t) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.t[mid] <= t_ns:
                lo = mid
            else:
                hi = mid

        t0, t1 = self.t[lo], self.t[hi]
        u = 0.0 if t1 == t0 else (t_ns - t0) / float(t1 - t0)
        tr = self.trans[lo] + u * (self.trans[hi] - self.trans[lo])
        q = quat_slerp(self.quat[lo], self.quat[hi], u)
        return T_from_tr_quat(tr[0], tr[1], tr[2], q[0], q[1], q[2], q[3])


def nearest_by_time(items, t_ns: int):
    if not items:
        return None
    if t_ns <= items[0][0]:
        return items[0]
    if t_ns >= items[-1][0]:
        return items[-1]
    lo, hi = 0, len(items) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if items[mid][0] <= t_ns:
            lo = mid
        else:
            hi = mid
    if abs(items[lo][0] - t_ns) <= abs(items[hi][0] - t_ns):
        return items[lo]
    return items[hi]


def decode_rgb(msg) -> np.ndarray:
    enc = (msg.encoding or "").lower()
    if "rgb8" in enc:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if "bgr8" in enc:
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    raise RuntimeError(f"Unsupported RGB encoding: {msg.encoding}")


def decode_depth_m(msg) -> np.ndarray:
    enc = (msg.encoding or "").lower()
    if "16uc1" in enc:
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return depth.astype(np.float32) * 0.001
    if "32fc1" in enc:
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    raise RuntimeError(f"Unsupported depth encoding: {msg.encoding}")


def write_matrix(path: Path, M: np.ndarray) -> None:
    np.savetxt(path, M, fmt="%.18e")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def estimate_fps(frame_times_ns: list[int]) -> int:
    if len(frame_times_ns) < 2:
        return 30
    diffs = np.diff(np.asarray(frame_times_ns, dtype=np.int64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 30
    fps = 1e9 / float(np.median(diffs))
    return max(1, int(round(fps)))


def detect_storage_id(bag_dir: Path) -> str:
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        return "sqlite3"

    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("storage_identifier:"):
            continue
        value = line.split(":", 1)[1].strip()
        return value or "sqlite3"
    return "sqlite3"


def make_converter_options(storage_id: str):
    storage_id = (storage_id or "").strip().lower()
    if storage_id == "mcap":
        return rosbag2_py.ConverterOptions(
            input_serialization_format="",
            output_serialization_format="",
        )
    return rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )


def main() -> None:
    args = parse_args()

    bag_dir = args.bag_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not bag_dir.exists():
        raise FileNotFoundError(f"Bag directory not found: {bag_dir}")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {out_dir}. Use --overwrite.")

    subdirs = {
        "rgb": out_dir / "rgb",
        "depth": out_dir / "depth",
        "poses": out_dir / "poses",
        "calibration": out_dir / "calibration",
        "timestamp": out_dir / "timestamp",
    }
    if args.overwrite:
        for path in subdirs.values():
            if path.exists():
                shutil.rmtree(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    storage_id = detect_storage_id(bag_dir)
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        make_converter_options(storage_id),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    needed = [
        args.rgb_topic,
        args.depth_topic,
        args.camera_info_topic,
        args.localization_topic,
        args.tf_static_topic,
    ]
    missing = [topic for topic in needed if topic not in type_map]
    if missing:
        raise RuntimeError(f"Missing topics in bag: {missing}")
    msg_types = {topic: get_message(type_map[topic]) for topic in needed}

    rgb_list = []
    depth_list = []
    pose_timeline = PoseTimeline()
    camera_info = None
    T_body_base = None
    skipped_empty = {
        args.rgb_topic: 0,
        args.depth_topic: 0,
        args.camera_info_topic: 0,
        args.localization_topic: 0,
        args.tf_static_topic: 0,
    }

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in skipped_empty and len(data) == 0:
            skipped_empty[topic] += 1
            continue
        if topic == args.rgb_topic:
            msg = deserialize_message(data, msg_types[topic])
            rgb_list.append((header_stamp_ns(msg), msg))
        elif topic == args.depth_topic:
            msg = deserialize_message(data, msg_types[topic])
            depth_list.append((header_stamp_ns(msg), msg))
        elif topic == args.camera_info_topic and camera_info is None:
            camera_info = deserialize_message(data, msg_types[topic])
        elif topic == args.localization_topic:
            msg = deserialize_message(data, msg_types[topic])
            pose_timeline.add(
                header_stamp_ns(msg),
                (
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ),
                (
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ),
            )
        elif topic == args.tf_static_topic and T_body_base is None:
            msg = deserialize_message(data, msg_types[topic])
            for tr in msg.transforms:
                if tr.header.frame_id.strip("/") == args.body_frame and tr.child_frame_id.strip("/") == args.base_link_frame:
                    T_body_base = T_from_tr_quat(
                        tr.transform.translation.x,
                        tr.transform.translation.y,
                        tr.transform.translation.z,
                        tr.transform.rotation.x,
                        tr.transform.rotation.y,
                        tr.transform.rotation.z,
                        tr.transform.rotation.w,
                    )
                    break

    if camera_info is None:
        raise RuntimeError("No CameraInfo found in bag")
    if not rgb_list or not depth_list:
        raise RuntimeError("Missing RGB or depth messages")
    if not pose_timeline.t:
        raise RuntimeError("Missing localization poses")
    if T_body_base is None:
        raise RuntimeError(f"Missing static transform {args.body_frame}->{args.base_link_frame} in {args.tf_static_topic}")

    rgb_list.sort(key=lambda x: x[0])
    depth_list.sort(key=lambda x: x[0])
    pose_timeline.sort()

    K = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)
    q_base_cam = quaternion_from_euler(args.camera_roll, args.camera_pitch, args.camera_yaw)
    T_base_cam = T_from_tr_quat(
        args.camera_tx,
        args.camera_ty,
        args.camera_tz,
        q_base_cam[0],
        q_base_cam[1],
        q_base_cam[2],
        q_base_cam[3],
    )

    max_pair_ns = int(args.max_pair_ms * 1e6)
    kept = 0
    seen_rgb = 0
    dropped_no_depth = 0
    dropped_no_pose = 0
    dropped_decode = 0
    frame_times_ns = []

    for t_rgb, rgb_msg in rgb_list:
        seen_rgb += 1
        if (seen_rgb - 1) % max(1, args.frame_stride) != 0:
            continue

        depth_pair = nearest_by_time(depth_list, t_rgb)
        if depth_pair is None:
            dropped_no_depth += 1
            continue
        t_depth, depth_msg = depth_pair
        dt_ns = abs(t_depth - t_rgb)
        if dt_ns > max_pair_ns:
            dropped_no_depth += 1
            continue

        T_map_body = pose_timeline.lookup(t_rgb)
        if T_map_body is None:
            dropped_no_pose += 1
            continue

        try:
            rgb_bgr = decode_rgb(rgb_msg)
            depth_m = decode_depth_m(depth_msg) * float(args.depth_scale)
        except Exception:
            dropped_decode += 1
            continue

        if rgb_bgr.shape[:2] != depth_m.shape[:2]:
            depth_m = cv2.resize(depth_m, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        T_map_cam = T_map_body @ T_body_base @ T_base_cam
        stem = f"{kept:06d}"

        cv2.imwrite(
            str(subdirs["rgb"] / f"{stem}.jpg"),
            rgb_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality],
        )
        np.save(subdirs["depth"] / f"{stem}.npy", depth_m.astype(np.float32, copy=False))
        write_matrix(subdirs["poses"] / f"{stem}.txt", T_map_cam)
        write_matrix(subdirs["calibration"] / f"{stem}.txt", K)
        (subdirs["timestamp"] / f"{stem}.txt").write_text(
            "\n".join(
                [
                    f"t_rgb_ns={t_rgb}",
                    f"t_depth_ns={t_depth}",
                    f"dt_ns={dt_ns}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        kept += 1
        frame_times_ns.append(t_rgb)
        if args.max_frames > 0 and kept >= args.max_frames:
            break

    if kept == 0:
        raise RuntimeError("No frames exported. Check pairing threshold and topics.")

    metadata = {
        "w": int(camera_info.width),
        "h": int(camera_info.height),
        "dw": int(camera_info.width),
        "dh": int(camera_info.height),
        "fps": estimate_fps(frame_times_ns),
        "K": K.tolist(),
        "depth_scale": float(args.depth_scale),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "cameraType": 1,
        "dist_coef": list(camera_info.d[:5]) if getattr(camera_info, "d", None) else [0.0] * 5,
        "length": kept,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    calib_line = f"{K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f}\n"
    (out_dir / "calib.txt").write_text(calib_line, encoding="utf-8")
    (out_dir / "calib_droidslam.txt").write_text(calib_line, encoding="utf-8")

    summary = {
        "bag_dir": str(bag_dir),
        "out_dir": str(out_dir),
        "storage_id": storage_id,
        "frames_exported": kept,
        "rgb_messages": len(rgb_list),
        "depth_messages": len(depth_list),
        "localization_messages": len(pose_timeline.t),
        "dropped_no_depth": dropped_no_depth,
        "dropped_no_pose": dropped_no_pose,
        "dropped_decode": dropped_decode,
        "skipped_empty_messages": skipped_empty,
        "max_pair_ms": args.max_pair_ms,
        "camera_pitch": args.camera_pitch,
        "camera_translation_xyz": [args.camera_tx, args.camera_ty, args.camera_tz],
        "pose_frame": "T_map_camera_link",
        "point_frame": "camera_link",
        "camera_info_frame": camera_info.header.frame_id,
        "localization_child_frame": args.body_frame,
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
