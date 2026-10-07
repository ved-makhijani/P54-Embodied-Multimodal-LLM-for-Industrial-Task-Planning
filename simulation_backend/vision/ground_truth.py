"""
ground_truth.py
---------------
Last-resort fallback scene source. Reads object positions directly
from PyBullet's physics state via getBasePositionAndOrientation().

No camera, no detection, no noise — exact positions from the physics
engine. Used when both the primary detector and segmentation fail to
find an object.

Usage:
    from simulation_backend.vision.ground_truth import GroundTruth

    gt = GroundTruth(physics_client, registry)
    detections = gt.get_all()
"""

import logging
import pybullet as p

from simulation_backend.simulation_environment.object_registry import ObjectRegistry
from simulation_backend.simulation_environment.scene_builder import Detection

logger = logging.getLogger(__name__)


class GroundTruth:
    """
    Reads exact object positions from PyBullet physics state.
    Always returns positions for every registered object.
    Confidence is always 1.0 — positions are exact, not estimated.
    Source is always "ground_truth".
    """

    def __init__(self, physics_client: int, registry: ObjectRegistry):
        self._client   = physics_client
        self._registry = registry

    def get_all(self) -> list[Detection]:
        """
        Return a Detection for every registered object using exact
        PyBullet physics positions.

        Returns:
            List of Detection objects, one per registered object.
        """
        detections = []

        for entry in self._registry.all_entries():
            try:
                pos, _ = p.getBasePositionAndOrientation(
                    entry.body_id,
                    physicsClientId=self._client,
                )
                detections.append(Detection(
                    body_id=entry.body_id,
                    label=entry.label,
                    position_3d=(
                        round(float(pos[0]), 4),
                        round(float(pos[1]), 4),
                        round(float(pos[2]), 4),
                    ),
                    bounding_box_2d=None,
                    confidence=1.0,
                    source="ground_truth",
                ))
            except Exception as e:
                logger.warning(
                    f"Could not get ground truth for '{entry.label}' "
                    f"(body_id={entry.body_id}): {e}"
                )

        logger.debug(f"Ground truth: {len(detections)} objects resolved.")
        return detections

    def get_by_label(self, label: str) -> Detection | None:
        """
        Return ground truth Detection for a single object by label.
        Returns None if the object is not in the registry.
        """
        entry = self._registry.get_by_label(label)
        if entry is None:
            return None

        try:
            pos, _ = p.getBasePositionAndOrientation(
                entry.body_id,
                physicsClientId=self._client,
            )
            return Detection(
                body_id=entry.body_id,
                label=entry.label,
                position_3d=(
                    round(float(pos[0]), 4),
                    round(float(pos[1]), 4),
                    round(float(pos[2]), 4),
                ),
                bounding_box_2d=None,
                confidence=1.0,
                source="ground_truth",
            )
        except Exception as e:
            logger.warning(f"Ground truth failed for '{label}': {e}")
            return None
