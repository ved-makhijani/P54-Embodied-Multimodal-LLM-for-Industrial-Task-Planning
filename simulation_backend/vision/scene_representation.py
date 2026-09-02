"""
simulation_backend/vision/scene_representation.py
---------------------------------------------------
Production scene adapter for the main pipeline.

The public contract is intentionally small:

    get_current_scene() -> {"objects": [{"label": str, "position": [x, y, z]}]}

The data comes from the live simulation vision stack.  The Simulation class owns
camera capture, YOLO/colour detector selection, ground-truth fallback, and
conversion through SceneBuilder.
"""

from __future__ import annotations

import os
from typing import Any


def _normalise_position(position: Any) -> list[float]:
    """Return a JSON-compatible [x, y, z] position list."""
    if isinstance(position, dict):
        return [
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
        ]

    if isinstance(position, (list, tuple)):
        if len(position) < 2:
            return [0.0, 0.0, 0.0]
        z = position[2] if len(position) > 2 else 0.0
        return [float(position[0]), float(position[1]), float(z)]

    return [0.0, 0.0, 0.0]


def _normalise_planner_scene(scene: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Keep only the planner-facing scene contract and make it literal JSON data.
    """
    objects = []
    for obj in scene.get("objects", []):
        label = obj.get("label") or obj.get("name")
        if not label:
            continue
        objects.append({
            "label": str(label),
            "position": _normalise_position(obj.get("position", [0.0, 0.0, 0.0])),
        })
    return {"objects": objects}


def _validate_all_workspace_objects(scene: dict[str, Any], sim: Any) -> None:
    """Fail fast if the live scene omits an object registered in the workspace."""
    registry = getattr(sim, "registry", None)
    if registry is None:
        return

    expected = [entry.label for entry in registry.all_entries()]
    found = {obj["label"] for obj in scene.get("objects", [])}
    missing = [label for label in expected if label not in found]

    if missing:
        raise RuntimeError(
            "Live vision scene is missing workspace object(s): "
            + ", ".join(missing)
        )


def get_current_scene(verbose: bool = False, sim: Any | None = None) -> dict[str, Any]:
    """
    Capture the current workspace through the real simulation vision pipeline.

    Args:
        verbose: Print the detector summary emitted by Simulation.get_live_scene().
        sim: Optional existing Simulation instance. If omitted, a temporary
             headless Simulation is created and disconnected before returning.

    Returns:
        Planner-compatible, JSON-serialisable scene dict:
        {"objects": [{"label": "red block", "position": [x, y, z]}, ...]}
    """
    os.environ.setdefault("VISION_DETECTOR", "yolo")

    owns_sim = sim is None
    if sim is None:
        from simulation_backend.simulation import Simulation
        sim = Simulation()

    try:
        scene = _normalise_planner_scene(sim.get_live_scene(verbose=verbose))
        _validate_all_workspace_objects(scene, sim)
        return scene
    finally:
        if owns_sim:
            sim.disconnect()


def get_planner_scene(verbose: bool = False) -> dict[str, Any]:
    """Backward-compatible alias used by older imports."""
    return get_current_scene(verbose=verbose)


def build_scene_representation(vision_output: dict[str, Any]) -> dict[str, Any]:
    """
    Convert older vision-output dictionaries into the planner scene format.

    This keeps historical tests/imports working while production code uses
    get_current_scene().
    """
    if "objects" in vision_output:
        return _normalise_planner_scene(vision_output)

    objects = []
    for obj in vision_output.get("detected_objects", []):
        label = obj.get("label") or obj.get("name")
        if not label:
            continue
        position = obj.get("position", {})
        coords = position.get("coordinates_3d", position) if isinstance(position, dict) else position
        objects.append({"label": str(label), "position": _normalise_position(coords)})
    return {"objects": objects}


def calculate_center(bbox: dict[str, float] | list[float] | tuple[float, ...]) -> tuple[float, float]:
    """Return the centre point of a 2D bounding box."""
    if isinstance(bbox, dict):
        x1 = bbox.get("x_min", bbox.get("xmin", bbox.get("x1", 0.0)))
        y1 = bbox.get("y_min", bbox.get("ymin", bbox.get("y1", 0.0)))
        x2 = bbox.get("x_max", bbox.get("xmax", bbox.get("x2", 0.0)))
        y2 = bbox.get("y_max", bbox.get("ymax", bbox.get("y2", 0.0)))
    else:
        x1, y1, x2, y2 = bbox[:4]
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)
