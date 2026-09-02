"""
Verify raw YOLODetector output on the real  PyBullet scene.

This script intentionally calls YOLODetector.detect(frame) directly and does
not count Simulation.get_live_scene() ground-truth fallback as success.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pybullet as p

REQUIRED_LABELS = [
    "red block",
    "blue block",
    "green block",
    "yellow block",
    "left tray",
    "right tray",
    "workstation",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="fine_tuning/yolov8n_p54_finetuning/yolov8n_p54_workspace_final/weights/best.pt",
    )
    parser.add_argument("--confidence", default=os.getenv("YOLO_CONFIDENCE", "0.25"))
    parser.add_argument("--iou", default=os.getenv("YOLO_IOU", "0.45"))
    parser.add_argument("--device", default=os.getenv("YOLO_DEVICE", "cpu"))
    parser.add_argument("--imgsz", default=os.getenv("YOLO_IMGSZ", "640"))
    parser.add_argument("--frames", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["SIMULATION_MODE"] = "DIRECT"
    os.environ["ROBOT_MODEL"] = "mock"
    os.environ["VISION_DETECTOR"] = "yolo"
    os.environ["YOLO_WEIGHTS"] = args.weights
    os.environ["YOLO_CONFIDENCE"] = str(args.confidence)
    os.environ["YOLO_IOU"] = str(args.iou)
    os.environ["YOLO_DEVICE"] = str(args.device)
    os.environ["YOLO_IMGSZ"] = str(args.imgsz)

    from simulation_backend.simulation import Simulation

    sim = Simulation(mode="DIRECT")
    try:
        detector = sim.detector
        if detector is None or detector.name != "yolo":
            raise RuntimeError("VISION_DETECTOR did not create YOLODetector.")

        detections_by_label = {}
        for _ in range(max(1, args.frames)):
            p.stepSimulation(physicsClientId=sim.client)
            frame = sim.camera.capture()
            for det in detector.detect(frame):
                current = detections_by_label.get(det.label)
                if current is None or det.confidence > current.confidence:
                    detections_by_label[det.label] = det

        print("YOLO workspace verification")
        print(f"Repo              : {_repo_root()}")
        print(f"Weights           : {args.weights}")
        print(f"Detector source   : {detector.name}")
        print(f"Frames evaluated  : {max(1, args.frames)}")
        print()
        print("Raw YOLODetector.detect(frame) detections:")

        if not detections_by_label:
            print("  NONE")
        else:
            for label in REQUIRED_LABELS:
                det = detections_by_label.get(label)
                if det is None:
                    continue
                print(
                    f"  {det.label:<14} "
                    f"source={det.source:<5} "
                    f"conf={det.confidence:.3f} "
                    f"pos={tuple(round(float(v), 4) for v in det.position_3d)} "
                    f"bbox={det.bounding_box_2d}"
                )

            extra_labels = sorted(set(detections_by_label) - set(REQUIRED_LABELS))
            for label in extra_labels:
                det = detections_by_label[label]
                print(
                    f"  {det.label:<14} "
                    f"source={det.source:<5} "
                    f"conf={det.confidence:.3f} "
                    f"pos={tuple(round(float(v), 4) for v in det.position_3d)} "
                    f"bbox={det.bounding_box_2d}"
                )

        detected = [label for label in REQUIRED_LABELS if label in detections_by_label]
        missing = [label for label in REQUIRED_LABELS if label not in detections_by_label]
        non_yolo = [
            label for label, det in detections_by_label.items()
            if label in REQUIRED_LABELS and det.source != "yolo"
        ]

        print()
        print(f"Detected required : {detected}")
        print(f"Missing required  : {missing}")
        if non_yolo:
            print(f"Wrong source      : {non_yolo}")

        if missing or non_yolo:
            print("RESULT: FAIL - raw YOLO did not detect every required workspace object.")
            return 2

        print("RESULT: PASS - raw YOLO detected every required workspace object.")
        return 0
    finally:
        sim.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
