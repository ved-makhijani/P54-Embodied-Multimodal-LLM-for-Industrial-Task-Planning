"""
Generate a YOLOv8 dataset from the real PyBullet workspace.

RGB images come from simulation_backend.vision.Camera. Labels are generated
only from PyBullet segmentation masks and the ObjectRegistry, never from
ground-truth planner fallback data.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import pybullet as p
import yaml


WORKSPACE_CLASSES = [
    "red block",
    "blue block",
    "green block",
    "yellow block",
    "left tray",
    "right tray",
    "workstation",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(WORKSPACE_CLASSES)}

BASE_POSES = {
    "red block": (0.45, -0.20, 0.05),
    "blue block": (0.25, 0.35, 0.05),
    "green block": (0.35, 0.12, 0.05),
    "yellow block": (0.25, -0.35, 0.05),
    "left tray": (0.65, 0.45, 0.01),
    "right tray": (0.65, -0.45, 0.01),
    "workstation": (0.80, 0.00, 0.00),
}

JITTER = {
    "red block": (0.04, 0.04),
    "blue block": (0.04, 0.04),
    "green block": (0.04, 0.035),
    "yellow block": (0.04, 0.04),
    "left tray": (0.02, 0.02),
    "right tray": (0.02, 0.02),
    "workstation": (0.015, 0.015),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="fine_tuning/yolov8n_p54_finetuning/datasets/p54_workspace",
        help="Dataset directory to create.",
    )
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--val", type=int, default=60)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_output_dir(path: Path, overwrite: bool) -> None:
    path = path.resolve()
    root = ROOT.resolve()
    if root not in path.parents:
        raise ValueError(f"Refusing to write dataset outside repo: {path}")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite")
        shutil.rmtree(path)
    for split in ("train", "val"):
        (path / "images" / split).mkdir(parents=True, exist_ok=True)
        (path / "labels" / split).mkdir(parents=True, exist_ok=True)


def _seg_mask(seg, body_id: int):
    valid = seg >= 0
    decoded = seg & ((1 << 24) - 1)
    return valid & ((seg == body_id) | (decoded == body_id))


def _bbox_from_mask(mask, width: int, height: int, min_area: int) -> str | None:
    ys, xs = mask.nonzero()
    if len(xs) < min_area:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    box_w = max(1, x2 - x1 + 1)
    box_h = max(1, y2 - y1 + 1)

    x_c = (x1 + x2 + 1) / 2.0 / width
    y_c = (y1 + y2 + 1) / 2.0 / height
    w = box_w / width
    h = box_h / height
    return f"{x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def _random_pose(label: str, rng: random.Random) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    x, y, z = BASE_POSES[label]
    jx, jy = JITTER[label]
    pos = (
        round(x + rng.uniform(-jx, jx), 4),
        round(y + rng.uniform(-jy, jy), 4),
        z,
    )
    yaw = rng.uniform(-3.14159, 3.14159)
    orn = p.getQuaternionFromEuler((0.0, 0.0, yaw))
    return pos, orn


def _random_camera(rng: random.Random, default_view: bool) -> dict:
    if default_view:
        return {
            "position": [0.70, 0.0, 1.80],
            "target": [0.55, 0.0, 0.02],
            "up_vector": [0.0, 1.0, 0.0],
            "fov": 55,
            "aspect_ratio": 1.333,
            "near_val": 0.1,
            "far_val": 10.0,
            "width": 640,
            "height": 480,
        }

    return {
        "position": [
            round(rng.uniform(0.58, 0.82), 4),
            round(rng.uniform(-0.08, 0.08), 4),
            round(rng.uniform(1.65, 2.05), 4),
        ],
        "target": [
            round(rng.uniform(0.50, 0.60), 4),
            round(rng.uniform(-0.04, 0.04), 4),
            0.02,
        ],
        "up_vector": [0.0, 1.0, 0.0],
        "fov": round(rng.uniform(50, 62), 3),
        "aspect_ratio": 1.333,
        "near_val": 0.1,
        "far_val": 10.0,
        "width": 640,
        "height": 480,
    }


def _apply_scene(sim, rng: random.Random, sample_idx: int) -> None:
    default_pose = sample_idx % 5 == 0
    for label in WORKSPACE_CLASSES:
        entry = sim.registry.get_by_label(label)
        if entry is None:
            raise RuntimeError(f"Workspace object not found in registry: {label}")

        if default_pose:
            pos = BASE_POSES[label]
            orn = (0.0, 0.0, 0.0, 1.0)
        else:
            pos, orn = _random_pose(label, rng)

        p.resetBasePositionAndOrientation(
            entry.body_id,
            pos,
            orn,
            physicsClientId=sim.client,
        )
        p.resetBaseVelocity(
            entry.body_id,
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0),
            physicsClientId=sim.client,
        )
        sim.registry.update_position(entry.body_id, tuple(pos))

    sim.camera.reconfigure(_random_camera(rng, default_view=default_pose))
    for _ in range(3):
        p.stepSimulation(physicsClientId=sim.client)


def _write_yaml(dataset_dir: Path) -> None:
    relative_dir = dataset_dir.resolve().relative_to(ROOT.resolve())
    payload = {
        "path": relative_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(WORKSPACE_CLASSES)},
        "nc": len(WORKSPACE_CLASSES),
    }
    with (dataset_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _generate_split(sim, dataset_dir: Path, split: str, count: int, rng: random.Random, start_idx: int, min_area: int) -> int:
    written = 0
    attempts = 0
    max_attempts = count * 10

    while written < count and attempts < max_attempts:
        sample_idx = start_idx + attempts
        attempts += 1
        _apply_scene(sim, rng, sample_idx)
        frame = sim.camera.capture()
        labels = []

        for class_name in WORKSPACE_CLASSES:
            entry = sim.registry.get_by_label(class_name)
            mask = _seg_mask(frame.seg, entry.body_id)
            bbox = _bbox_from_mask(mask, frame.width, frame.height, min_area)
            if bbox is None:
                break
            labels.append(f"{CLASS_TO_ID[class_name]} {bbox}")
        else:
            stem = f"p54_{split}_{written:05d}"
            cv2.imwrite(str(dataset_dir / "images" / split / f"{stem}.jpg"), frame.bgr)
            (dataset_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(labels) + "\n",
                encoding="utf-8",
            )
            written += 1

    if written != count:
        raise RuntimeError(
            f"Only generated {written}/{count} {split} samples after {attempts} attempts."
        )
    return attempts


def main() -> int:
    args = parse_args()
    dataset_dir = (ROOT / args.output).resolve()
    rng = random.Random(args.seed)

    _safe_output_dir(dataset_dir, args.overwrite)

    import os
    os.environ["SIMULATION_MODE"] = "DIRECT"
    os.environ["ROBOT_MODEL"] = "mock"
    os.environ.pop("VISION_DETECTOR", None)

    from simulation_backend.simulation import Simulation

    sim = Simulation(mode="DIRECT")
    try:
        train_attempts = _generate_split(
            sim, dataset_dir, "train", args.train, rng, 0, args.min_area
        )
        val_attempts = _generate_split(
            sim, dataset_dir, "val", args.val, rng, train_attempts, args.min_area
        )
        _write_yaml(dataset_dir)
    finally:
        sim.disconnect()

    print("Synthetic YOLO dataset generated")
    print(f"Dataset dir : {dataset_dir}")
    print(f"Classes     : {WORKSPACE_CLASSES}")
    print(f"Train       : {args.train} images")
    print(f"Validation  : {args.val} images")
    print(f"Data YAML   : {dataset_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
