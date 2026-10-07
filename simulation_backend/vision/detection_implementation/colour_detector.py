"""
simulation_backend/vision/detection_implementation/colour_detector.py
----------------------------------------------------------------------
HSV colour threshold detector for the simulation vision pipeline.

Config is read from environment variables (set in .env):
    COLOUR_MIN_AREA      : min contour area in pixels. Default: 30
    COLOUR_EXPECTED_AREA : expected pixel area of a full block face. Default: 100
    COLOUR_DEPTH_RADIUS  : unused — kept for API compatibility. Default: 3

No model weights, no downloads, no GPU required.
Runs in under 5ms per frame on CPU using only OpenCV.

How it works:
    For each registered object:
        1. Look up its colour attribute ("red", "blue", "green", ...).
        2. Apply the corresponding HSV mask to the frame.
        3. Find contours. Take the largest one.
        4. Skip if area < COLOUR_MIN_AREA (noise filter).
        5. Compute the centroid pixel (cx, cy).
        6. Get the bounding box for the contour.
        7. Read the LIVE 3D world position directly from PyBullet via
           p.getBasePositionAndOrientation(body_id). This is the correct
           approach — the depth map gives camera-to-object distance (eye
           space), not world Z. Only PyBullet knows the real world Z.
        8. confidence = min(area / COLOUR_EXPECTED_AREA, 1.0).

    Objects with colour attributes not in HSV_RANGES are skipped silently
    — ground truth fills the gap for those objects.

HSV ranges are calibrated for PyBullet flat-shaded rendering:

    Colour      RGB (255 scale)       HSV (OpenCV H: 0-179)
    ----------  -------------------   ----------------------
    red         (255, 0,   0  )       H=0,   S=255, V=255
    blue        (0,   0,   255)       H=120, S=255, V=255
    green       (0,   204, 0  )       H=60,  S=255, V=204
    yellow      (255, 255, 0  )       H=30,  S=255, V=255
    grey        (140, 140, 140)       H=0,   S=0,   V=140
    dark grey   (89,  89,  89 )       H=0,   S=0,   V=89

Usage:
    Set VISION_DETECTOR=colour in .env. No installation beyond opencv-python.
"""

import logging
import os

import cv2
import numpy as np
import pybullet as p

from simulation_backend.simulation_environment.scene_builder import Detection
from simulation_backend.vision.camera import CameraFrame
from simulation_backend.vision.detection_base import DetectorBase
from simulation_backend.simulation_environment.object_registry import ObjectRegistry

logger = logging.getLogger(__name__)


# ── HSV colour ranges ──────────────────────────────────────────────────────────
#
# Calibrated for PyBullet flat-shaded rendering (scene_config.yaml RGBA values).
# Each entry: list of (lower_bound, upper_bound) for cv2.inRange().
# Multiple ranges are OR-combined. Red needs two because H wraps at 0/180.
#
# Tolerance: H ± 10, S >= 225 (pure solid colours), V >= 174 (bright flat shading)
# Grey / dark grey matched on V range only (S ≈ 0).

HSV_RANGES: dict[str, list[tuple]] = {
    "red": [
        (np.array([0,   225, 225]), np.array([8,   255, 255])),
        (np.array([172, 225, 225]), np.array([179, 255, 255])),
    ],
    "blue": [
        (np.array([110, 225, 225]), np.array([130, 255, 255])),
    ],
    "green": [
        (np.array([50,  225, 174]), np.array([70,  255, 255])),
    ],
    "yellow": [
        (np.array([20,  225, 225]), np.array([40,  255, 255])),
    ],
    "grey": [
        (np.array([0,   0,   100]), np.array([179, 30,  180])),
    ],
    "dark grey": [
        (np.array([0,   0,   49]),  np.array([179, 30,  129])),
    ],
}


class ColourDetector(DetectorBase):
    """
    HSV colour threshold detector.

    All config is read from environment variables at init time.
    No scene_config.yaml keys are used.

    World Z is read from PyBullet directly (not from the depth map).
    The depth map gives eye-space camera distance, not world Z.
    p.getBasePositionAndOrientation() gives the exact world position.
    """

    name = "colour"

    def __init__(self, registry: ObjectRegistry, config: dict = None) -> None:
        super().__init__(registry, config)
        self._min_area      = int(os.getenv("COLOUR_MIN_AREA",      "30"))
        self._expected_area = int(os.getenv("COLOUR_EXPECTED_AREA", "100"))

        # physics_client injected by simulation.py via the config dict.
        # Falls back to client 0 (always valid for single-simulation setups).
        self._physics_client = int(self._cfg.get("_physics_client", 0))

        logger.info(
            f"[colour] min_area={self._min_area}  "
            f"expected_area={self._expected_area}  "
            f"physics_client={self._physics_client}"
        )

    # ── warmup ────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """No weights to load. Marks the detector ready immediately."""
        logger.info("[colour] Ready — no weights required.")
        self._warmed_up = True

    # ── detect ────────────────────────────────────────────────────────────────

    def detect(self, frame: CameraFrame) -> list[Detection]:
        """
        Run HSV colour detection on one camera frame.

        For each registered object whose colour is in HSV_RANGES:
            1. Build a binary mask for that colour in the HSV image.
            2. Find contours and take the largest one.
            3. Skip if area < COLOUR_MIN_AREA.
            4. Read live world position from PyBullet.
            5. confidence = min(area / COLOUR_EXPECTED_AREA, 1.0).

        Args:
            frame : CameraFrame from Camera.capture()

        Returns:
            list[Detection] — one per colour-matched object found.
        """
        self._ensure_warmed_up()

        hsv        = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2HSV)
        detections = []

        for entry in self._registry.all_entries():
            colour_name = entry.attributes.get("color", "").lower().strip()

            if colour_name not in HSV_RANGES:
                logger.debug(
                    f"[colour] '{entry.label}' colour='{colour_name}' "
                    f"not in HSV_RANGES — ground truth will cover it."
                )
                continue

            mask = self._colour_mask(hsv, colour_name)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                logger.debug(f"[colour] No contours for '{entry.label}'.")
                continue

            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)

            if area < self._min_area:
                logger.debug(
                    f"[colour] '{entry.label}' area {area:.0f} < "
                    f"{self._min_area} — too small, skipping."
                )
                continue

            # Bounding box from contour
            bx, by, bw, bh = cv2.boundingRect(best)
            bbox = {"x_min": bx, "y_min": by, "x_max": bx + bw, "y_max": by + bh}

            # ── Live 3D world position from PyBullet ──────────────────────
            # Do NOT use the depth map for Z — it gives eye-space camera
            # distance, not world Z. PyBullet knows the exact world position.
            try:
                pos, _ = p.getBasePositionAndOrientation(
                    entry.body_id,
                    physicsClientId=self._physics_client,
                )
                x = round(float(pos[0]), 4)
                y = round(float(pos[1]), 4)
                z = round(float(pos[2]), 4)
            except Exception as e:
                logger.warning(
                    f"[colour] Could not get PyBullet position for "
                    f"'{entry.label}': {e}. Using registry position."
                )
                x = float(entry.position[0])
                y = float(entry.position[1])
                z = float(entry.position[2])

            confidence = round(min(area / self._expected_area, 1.0), 3)

            detections.append(Detection(
                body_id=entry.body_id,
                label=entry.label,
                position_3d=(x, y, z),
                bounding_box_2d=bbox,
                confidence=confidence,
                source=self.name,
            ))

            logger.debug(
                f"[colour] '{entry.label}'  colour='{colour_name}'  "
                f"area={area:.0f}  conf={confidence:.2f}  "
                f"pos=({x:.3f}, {y:.3f}, {z:.3f})"
            )

        logger.info(
            f"[colour] {len(detections)} detected: "
            f"{[d.label for d in detections]}"
        )
        return detections

    # ── draw_detections ───────────────────────────────────────────────────────

    def draw_detections(
        self,
        frame:      np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """
        Cyan bounding boxes with [CLR] prefix.
        Distinct from yellow YOLO boxes and green ground truth.
        """
        canvas = frame.copy()
        for det in detections:
            bb = det.bounding_box_2d
            if bb is None:
                continue
            x1, y1 = int(bb["x_min"]), int(bb["y_min"])
            x2, y2 = int(bb["x_max"]), int(bb["y_max"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
            text = f"[CLR] {det.label}  {det.confidence:.2f}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                canvas, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1),
                (255, 255, 0), -1,
            )
            cv2.putText(
                canvas, text, (x1 + 2, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )
        return canvas

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _colour_mask(hsv: np.ndarray, colour_name: str) -> np.ndarray:
        """
        Build a binary mask for the given colour name by OR-combining
        all HSV ranges registered for that colour.

        Red wraps around H=0/180, so it has two ranges merged here.
        A morphological open removes isolated noise specks.
        """
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in HSV_RANGES[colour_name]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
