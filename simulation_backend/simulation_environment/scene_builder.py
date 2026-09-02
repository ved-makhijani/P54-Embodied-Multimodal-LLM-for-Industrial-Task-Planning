"""
scene_builder.py
----------------
Assembles the rich scene dictionary from vision detections and the
object registry.

Takes raw detection output (from whichever detector ran) and builds
the structured scene format used by the pipeline. Adds spatial
relationships, graspability flags, timestamps, and workspace status.
Also provides to_planner_format() which converts the rich scene into
the simple format task_planner/planner.py currently expects.

The rich scene format matches scene_abstraction.json:
{
    "scene_timestamp": "...",
    "workspace_status": "clear",
    "detected_objects": [
        {
            "object_id":   "obj_001",
            "label":       "red block",
            "attributes":  {"color": "red", "shape": "cube"},
            "position": {
                "coordinates_3d": {"x": 0.45, "y": -0.20, "z": 0.05},
                "bounding_box_2d": {"x_min": 120, "y_min": 340,
                                    "x_max": 180, "y_max": 400}
            },
            "spatial_relationships": [
                {"relation": "on_top_of", "target_id": "obj_003"}
            ],
            "graspable":   true,
            "source":      "detector"   # or "segmentation" or "ground_truth"
        }
    ]
}

Usage:
    from simulation_backend.simulation_environment.scene_builder import SceneBuilder

    builder = SceneBuilder(registry)
    scene   = builder.build(detections, seg_detections, gt_detections)
    simple  = builder.to_planner_format(scene)
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from .object_registry import ObjectRegistry

logger = logging.getLogger(__name__)

# Vertical overlap threshold for on_top_of spatial relation (metres)
_ON_TOP_OF_Z_THRESHOLD   = 0.03
# XY overlap threshold for on_top_of spatial relation (metres)
_ON_TOP_OF_XY_THRESHOLD  = 0.08
# Proximity threshold for near_to spatial relation (metres)
_NEAR_TO_THRESHOLD       = 0.20


@dataclass
class Detection:
    """
    Standardised detection output from any detector or fallback source.
    All detectors and ground_truth.py return lists of this type.

    Fields:
        body_id         : PyBullet body ID
        label           : Human-readable object label
        position_3d     : (x, y, z) world coordinates in metres
        bounding_box_2d : Pixel bounding box dict or None
        confidence      : Detection confidence 0.0-1.0 (1.0 for ground truth)
        source          : "detector" | "segmentation" | "ground_truth"
    """
    body_id:         int
    label:           str
    position_3d:     tuple[float, float, float]
    bounding_box_2d: Optional[dict]   = None
    confidence:      float            = 1.0
    source:          str              = "detector"


class SceneBuilder:
    """
    Assembles the rich scene dict from detection results.

    Priority of sources per object:
        1. Primary detector result (VISION_DETECTOR)
        2. Segmentation mask result (SEGMENTATION_MASK=true)
        3. Ground truth fallback (always last resort)

    If an object expected in the registry is not found by any source,
    it is logged as a warning and excluded from the scene.
    """

    def __init__(self, registry: ObjectRegistry):
        self._registry = registry

    def build(
        self,
        detector_results:     list[Detection],
        segmentation_results: Optional[list[Detection]] = None,
        ground_truth_results: Optional[list[Detection]] = None,
    ) -> dict:
        """
        Build the rich scene dictionary.

        Args:
            detector_results:     Output from the primary VISION_DETECTOR
            segmentation_results: Output from segmentation.py if enabled
            ground_truth_results: Output from ground_truth.py as last resort

        Returns:
            Rich scene dict matching scene_abstraction.json format
        """
        detector_map     = {d.body_id: d for d in (detector_results or [])}
        seg_map          = {d.body_id: d for d in (segmentation_results or [])}
        gt_map           = {d.body_id: d for d in (ground_truth_results or [])}

        detected_objects = []

        for entry in self._registry.all_entries():
            detection = self._resolve_detection(
                entry.body_id, detector_map, seg_map, gt_map
            )

            if detection is None:
                logger.warning(
                    f"Object '{entry.label}' (id={entry.body_id}) "
                    f"not found by any source — excluded from scene."
                )
                continue

            self._registry.update_position(entry.body_id, detection.position_3d)

            x, y, z = detection.position_3d
            obj_dict = {
                "object_id":   self._find_config_id(entry.label),
                "label":       entry.label,
                "attributes":  entry.attributes,
                "position": {
                    "coordinates_3d": {
                        "x": round(x, 4),
                        "y": round(y, 4),
                        "z": round(z, 4),
                    },
                    "bounding_box_2d": detection.bounding_box_2d,
                },
                "spatial_relationships": [],
                "graspable":   entry.graspable,
                "confidence":  round(detection.confidence, 3),
                "source":      detection.source,
            }
            detected_objects.append(obj_dict)

        self._compute_spatial_relationships(detected_objects)

        scene = {
            "scene_timestamp":  datetime.now(timezone.utc).isoformat(),
            "workspace_status": "clear" if detected_objects else "empty",
            "detected_objects": detected_objects,
        }

        logger.debug(
            f"Scene built: {len(detected_objects)} objects, "
            f"timestamp={scene['scene_timestamp']}"
        )
        return scene

    def to_planner_format(self, rich_scene: dict) -> dict:
        """
        Convert the rich scene dict to the simple format planner.py expects.

        Includes the full (x, y, z) 3D position so that the real robot arm
        receives correct Z values for approach height calculations.
        Without Z the arm targets the wrong height — blocks are at z=0.05
        above the table, not z=0.0.

        Args:
            rich_scene: Output of build()

        Returns:
            {"objects": [{"label": str, "position": [x, y, z]}]}
        """
        objects = []
        for obj in rich_scene.get("detected_objects", []):
            coords = obj["position"]["coordinates_3d"]
            objects.append({
                "label":    obj["label"],
                "position": [coords["x"], coords["y"], coords["z"]],
            })
        return {"objects": objects}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _resolve_detection(
        self,
        body_id:      int,
        detector_map: dict,
        seg_map:      dict,
        gt_map:       dict,
    ) -> Optional[Detection]:
        """
        Return the best available detection for a body_id.
        Priority: primary detector -> segmentation -> ground truth.
        """
        if body_id in detector_map:
            return detector_map[body_id]
        if body_id in seg_map:
            d = seg_map[body_id]
            d.source = "segmentation"
            logger.debug(
                f"body_id={body_id} not in detector — using segmentation fallback"
            )
            return d
        if body_id in gt_map:
            d = gt_map[body_id]
            d.source = "ground_truth"
            logger.warning(
                f"body_id={body_id} not in detector or segmentation — "
                f"using ground truth fallback"
            )
            return d
        return None

    def _compute_spatial_relationships(self, objects: list[dict]) -> None:
        """
        Compute pairwise spatial relationships and add them to each object.

        Currently computed:
            on_top_of : object A is directly above object B
            near_to   : object A is within proximity threshold of object B
        """
        for i, obj_a in enumerate(objects):
            a_coords = obj_a["position"]["coordinates_3d"]
            ax, ay, az = a_coords["x"], a_coords["y"], a_coords["z"]

            for j, obj_b in enumerate(objects):
                if i == j:
                    continue

                b_coords = obj_b["position"]["coordinates_3d"]
                bx, by, bz = b_coords["x"], b_coords["y"], b_coords["z"]

                dx = abs(ax - bx)
                dy = abs(ay - by)
                dz = az - bz

                if (dz > _ON_TOP_OF_Z_THRESHOLD
                        and dx < _ON_TOP_OF_XY_THRESHOLD
                        and dy < _ON_TOP_OF_XY_THRESHOLD):
                    obj_a["spatial_relationships"].append({
                        "relation":  "on_top_of",
                        "target_id": obj_b["object_id"],
                    })

                xy_dist = (dx ** 2 + dy ** 2) ** 0.5
                if xy_dist < _NEAR_TO_THRESHOLD:
                    obj_a["spatial_relationships"].append({
                        "relation":  "near_to",
                        "target_id": obj_b["object_id"],
                    })

    def _find_config_id(self, label: str) -> str:
        """
        Return a stable object_id string for a given label.
        Falls back to a slugified label if registry has no config ID.
        """
        entry = self._registry.get_by_label(label)
        if entry and hasattr(entry, "config_id"):
            return entry.config_id
        return label.lower().replace(" ", "_")
