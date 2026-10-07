"""
simulation_backend/vision/detection_implementation/yolo_detector.py
--------------------------------------------------------------------
YOLOv8 object detector for the simulation vision pipeline.

Config is read from environment variables (set in .env):
    YOLO_WEIGHTS      : weights filename inside detection_models/. Default: yolov8n.pt
    YOLO_CONFIDENCE   : min detection confidence (0.0-1.0). Default: 0.25
    YOLO_IOU          : NMS IoU threshold. Default: 0.45
    YOLO_DEVICE       : cpu | cuda | mps | auto. Default: cpu
    YOLO_IMGSZ        : inference image size in pixels. Default: 640

Weights are stored in and loaded from:
    simulation_backend/vision/detection_models/<YOLO_WEIGHTS>

On first run the file is downloaded automatically by ultralytics and
then copied into detection_models/ so every subsequent run is local
with no network access needed.

How detection works:
    1. warmup()  — loads the model once, copies weights into
                   detection_models/ if they were downloaded.

    2. detect()  — for each bounding box YOLO returns:
                     a. Reads frame.seg at the bbox centre pixel to get
                        the PyBullet body_id (no COCO class lookup needed).
                     b. Resolves body_id -> label via ObjectRegistry.
                     c. Samples depth at bbox centre -> Z in metres.
                     d. Uses registry cached (x, y) + depth Z as 3D position.
                     e. Returns Detection with YOLO's real confidence score.

    Why seg mask instead of class names?
        YOLO is trained on COCO — not on simulation blocks. The seg mask
        gives pixel-perfect body_ids from PyBullet's own renderer. YOLO
        provides bounding boxes + confidence; the seg mask provides identity.

Required:
    pip install ultralytics

Usage:
    Set VISION_DETECTOR=yolo in .env.
"""

import logging
import os
import shutil
from pathlib import Path

import numpy as np

from simulation_backend.simulation_environment.scene_builder import Detection
from simulation_backend.vision.camera import CameraFrame
from simulation_backend.vision.detection_base import DetectorBase
from simulation_backend.simulation_environment.object_registry import ObjectRegistry

logger = logging.getLogger(__name__)

# Canonical weights folder: simulation_backend/vision/detection_models/
_MODELS_DIR = Path(__file__).resolve().parent.parent / "detection_models"


class YOLODetector(DetectorBase):
    """
    YOLOv8-backed object detector.

    All config is read from environment variables at init time.
    No scene_config.yaml keys are used.

    Weights resolution order in warmup():
        1. simulation_backend/vision/detection_models/<YOLO_WEIGHTS>  (local, preferred)
        2. Current working directory / <YOLO_WEIGHTS>
        3. Ultralytics auto-download -> then copied into detection_models/

    After the first successful run the weights file will always be in
    detection_models/ so no network access is needed afterwards.
    """

    name = "yolo"

    def __init__(self, registry: ObjectRegistry, config: dict = None) -> None:
        super().__init__(registry, config)
        self._model          = None
        self._conf_threshold = float(os.getenv("YOLO_CONFIDENCE",  "0.25"))
        self._iou_threshold  = float(os.getenv("YOLO_IOU",         "0.45"))
        self._device         = str(os.getenv("YOLO_DEVICE",        "cpu"))
        self._imgsz          = int(os.getenv("YOLO_IMGSZ",         "640"))
        self._weights_name   = str(os.getenv("YOLO_WEIGHTS",       "yolov8n.pt"))
        self._weights_path   = self._resolve_weights_path(self._weights_name)

        logger.info(
            f"[yolo] weights={self._weights_path}  conf={self._conf_threshold}  "
            f"iou={self._iou_threshold}  device={self._device}  imgsz={self._imgsz}"
        )

    # ── warmup ────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Load YOLOv8 weights and ensure they are saved in detection_models/.

        If the weights file is not in detection_models/, ultralytics downloads
        it automatically on YOLO() instantiation. After loading we check where
        the model saved its checkpoint and copy it into detection_models/ so
        the next run finds it locally at priority 1.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is not installed.  Run: pip install ultralytics"
            )

        logger.info(f"[yolo] Loading: {self._weights_path}")
        self._model = YOLO(str(self._weights_path))

        # Copy into detection_models/ if the file was downloaded elsewhere
        local_target = _MODELS_DIR / self._weights_name
        if not local_target.exists():
            cached = None
            for attr in ("ckpt_path", "pt_path", "model_path"):
                val = getattr(self._model, attr, None)
                if val and Path(str(val)).exists():
                    cached = Path(str(val))
                    break

            if cached and cached.exists():
                try:
                    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cached, local_target)
                    logger.info(
                        f"[yolo] Weights copied to detection_models/: {local_target}"
                    )
                    self._weights_path = local_target
                except Exception as e:
                    logger.warning(
                        f"[yolo] Could not copy weights to detection_models/: {e}"
                    )
            else:
                logger.warning(
                    "[yolo] Could not locate downloaded weights to copy. "
                    "Weights remain in ultralytics cache."
                )

        # JIT warm-up — one dummy frame before the first real call
        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._model(dummy, verbose=False, device=self._device)

        logger.info(f"[yolo] Ready — {self._weights_name} on {self._device}")
        self._warmed_up = True

    # ── detect ────────────────────────────────────────────────────────────────

    def detect(self, frame: CameraFrame) -> list[Detection]:
        """
        Run YOLOv8 inference on one camera frame.

        For each bounding box returned by YOLO:
            1. Centre pixel (cx, cy) of the box.
            2. frame.seg[cy, cx] -> PyBullet body_id at that pixel.
            3. registry.get_by_id(body_id) -> ObjectEntry (label + position).
            4. _depth_at_bbox_centre() -> metric Z in metres.
            5. Registry (x, y) + depth Z -> 3D world position.
            6. Deduplicate by body_id, keeping highest confidence per object.

        Returns:
            list[Detection] — empty if YOLO finds nothing above threshold.
        """
        self._ensure_warmed_up()

        results = self._model(
            frame.bgr,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            logger.debug("[yolo] No boxes returned for this frame.")
            return []

        h, w = frame.bgr.shape[:2]
        detections: dict[int, Detection] = {}

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            bbox       = {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}
            cx, cy     = (x1 + x2) // 2, (y1 + y2) // 2
            confidence = float(box.conf[0])
            body_id    = int(frame.seg[cy, cx])

            if body_id < 0:
                logger.debug(f"[yolo] Box centre ({cx},{cy}) is background, skipping.")
                continue

            entry = self._registry.get_by_id(body_id)
            if entry is None:
                logger.debug(f"[yolo] body_id={body_id} not in registry, skipping.")
                continue

            z = self._depth_at_bbox_centre(frame, bbox)
            x = float(entry.position[0])
            y = float(entry.position[1])

            if body_id in detections and confidence <= detections[body_id].confidence:
                continue

            detections[body_id] = Detection(
                body_id=body_id,
                label=entry.label,
                position_3d=(round(x, 4), round(y, 4), round(float(z), 4)),
                bounding_box_2d=bbox,
                confidence=round(confidence, 3),
                source=self.name,
            )

            logger.debug(
                f"[yolo] '{entry.label}'  body_id={body_id}  "
                f"conf={confidence:.2f}  bbox=({x1},{y1},{x2},{y2})  z={z:.3f}m"
            )

        found = list(detections.values())
        logger.info(f"[yolo] {len(found)} detected: {[d.label for d in found]}")
        return found

    # ── draw_detections ───────────────────────────────────────────────────────

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Yellow bounding boxes with [YOLO] prefix — distinct from ground truth green."""
        import cv2
        canvas = frame.copy()
        for det in detections:
            bb = det.bounding_box_2d
            if bb is None:
                continue
            x1, y1 = int(bb["x_min"]), int(bb["y_min"])
            x2, y2 = int(bb["x_max"]), int(bb["y_max"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 215, 255), 2)
            label_text = f"[YOLO] {det.label}  {det.confidence:.2f}"
            (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), (0, 215, 255), -1)
            cv2.putText(canvas, label_text, (x1 + 2, y1 - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return canvas

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_weights_path(weights_name: str) -> Path:
        """
        Priority:
            1. simulation_backend/vision/detection_models/<weights_name>
            2. Current working directory / <weights_name>
            3. weights_name as-is — ultralytics will download on YOLO()
        """
        local = _MODELS_DIR / weights_name
        if local.exists():
            logger.info(f"[yolo] Using local weights: {local}")
            return local

        cwd_path = Path.cwd() / weights_name
        if cwd_path.exists():
            logger.info(f"[yolo] Using cwd weights: {cwd_path}")
            return cwd_path

        logger.info(
            f"[yolo] {weights_name} not in detection_models/. "
            f"Will download on warmup and save to detection_models/."
        )
        return Path(weights_name)
