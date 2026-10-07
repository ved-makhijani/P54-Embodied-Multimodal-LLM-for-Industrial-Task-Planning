"""
object_registry.py
------------------
Maps PyBullet integer object IDs to human-readable labels and attributes.

PyBullet identifies every loaded body by an integer ID returned from
loadURDF() or createMultiBody(). This registry is the single source of
truth that translates those IDs back into the labels and properties the
rest of the pipeline uses (e.g. "red block", graspable=True).

The registry is populated at simulation startup by object_loader.py and
read by the vision module to resolve segmentation mask pixel values into
scene object data.

Usage:
    from simulation_backend.simulation_environment.object_registry import ObjectRegistry

    registry = ObjectRegistry()
    registry.register(body_id=3, label="red block", ...)
    obj = registry.get_by_id(3)
    obj = registry.get_by_label("red block")
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Object entry ───────────────────────────────────────────────────────────────

@dataclass
class ObjectEntry:
    """
    Represents a single registered simulation object.

    Fields:
        body_id     : PyBullet integer body ID returned from loadURDF()
        label       : Human-readable name used throughout the pipeline
        color       : RGBA colour [r, g, b, a] each 0.0-1.0
        position    : Initial (x, y, z) world position in metres
        graspable   : Whether the robot can attempt to grasp this object
        mass_kg     : Object mass in kg. 0.0 means static/fixed.
        attributes  : Additional properties (color name, shape, etc.)
    """
    body_id:    int
    label:      str
    color:      list[float]
    position:   tuple[float, float, float]
    graspable:  bool
    mass_kg:    float
    attributes: dict        = field(default_factory=dict)


# ── Registry ───────────────────────────────────────────────────────────────────

class ObjectRegistry:
    """
    Bidirectional map between PyBullet body IDs and object labels.

    Populated once at startup by object_loader.py.
    Read by vision/segmentation.py and vision/ground_truth.py at runtime.
    """

    def __init__(self):
        self._by_id:    dict[int, ObjectEntry]  = {}
        self._by_label: dict[str, ObjectEntry]  = {}

    def register(
        self,
        body_id:    int,
        label:      str,
        color:      list[float],
        position:   tuple[float, float, float],
        graspable:  bool,
        mass_kg:    float,
        attributes: dict = None,
    ) -> ObjectEntry:
        """
        Register a newly loaded PyBullet object.

        Called by object_loader.py immediately after loadURDF() returns
        the body_id.

        Args:
            body_id:    Integer ID returned by PyBullet loadURDF()
            label:      Human-readable name e.g. "red block"
            color:      RGBA list e.g. [1.0, 0.0, 0.0, 1.0]
            position:   World position (x, y, z) in metres
            graspable:  Whether the robot can grasp this object
            mass_kg:    Mass in kg. 0.0 = static.
            attributes: Optional extra properties dict

        Returns:
            The created ObjectEntry
        """
        entry = ObjectEntry(
            body_id=body_id,
            label=label,
            color=color,
            position=position,
            graspable=graspable,
            mass_kg=mass_kg,
            attributes=attributes or {},
        )
        self._by_id[body_id]        = entry
        self._by_label[label.lower()] = entry
        logger.debug(f"Registered: body_id={body_id} label='{label}' graspable={graspable}")
        return entry

    def get_by_id(self, body_id: int) -> Optional[ObjectEntry]:
        """Return the ObjectEntry for a given PyBullet body ID, or None."""
        return self._by_id.get(body_id)

    def get_by_label(self, label: str) -> Optional[ObjectEntry]:
        """
        Return the ObjectEntry for a given label (case-insensitive).
        Supports partial matching — "red" will match "red block".
        """
        label_lower = label.lower()
        # Exact match first
        if label_lower in self._by_label:
            return self._by_label[label_lower]
        # Partial match fallback
        for key, entry in self._by_label.items():
            if label_lower in key or key in label_lower:
                return entry
        return None

    def all_ids(self) -> list[int]:
        """Return all registered PyBullet body IDs."""
        return list(self._by_id.keys())

    def all_entries(self) -> list[ObjectEntry]:
        """Return all registered ObjectEntry objects."""
        return list(self._by_id.values())

    def graspable_entries(self) -> list[ObjectEntry]:
        """Return only entries where graspable=True."""
        return [e for e in self._by_id.values() if e.graspable]

    def update_position(self, body_id: int, position: tuple[float, float, float]) -> None:
        """
        Update the cached position of an object.
        Called after the robot places an object at a new location.
        """
        if body_id in self._by_id:
            self._by_id[body_id].position = position

    def clear(self) -> None:
        """Clear all registered objects. Called on simulation reset."""
        self._by_id.clear()
        self._by_label.clear()
        logger.debug("Object registry cleared.")

    def __len__(self) -> int:
        return len(self._by_id)

    def __repr__(self) -> str:
        entries = [f"  {e.body_id}: '{e.label}'" for e in self._by_id.values()]
        return "ObjectRegistry(\n" + "\n".join(entries) + "\n)"
