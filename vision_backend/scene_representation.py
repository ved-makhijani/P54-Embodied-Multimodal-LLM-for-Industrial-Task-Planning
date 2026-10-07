"""
vision_backend/scene_representation.py
---------------------------------------
JSON-based scene provider for the pipeline's Vision Lookup stage.

Reads a static scene JSON file from disk and converts it into the
planner-compatible scene dict that main.py and task_planner/planner.py expect.

Public interface:
    get_current_scene() -> dict

Input:
    JSON file at vision_backend/scene_representation.json (default)
    or path set via VISION_SCENE_FILE environment variable.

    Expected JSON format:
        {
            "objects": [
                {"label": "red block", "position": [150.0, 110.0]},
                ...
            ]
        }

Output:
    {
        "objects": [
            {"label": "red block", "position": [150.0, 110.0]},
            ...
        ]
    }

Usage:
    from vision_backend.scene_representation import get_current_scene
    scene = get_current_scene()
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_SCENE_FILE = Path(__file__).resolve().parent / "scene_representation.json"


def get_current_scene() -> dict:
    """
    Read the static scene JSON file and return a planner-compatible scene dict.

    Returns:
        {"objects": [{"label": str, "position": [x, y]}, ...]}

    Raises:
        FileNotFoundError : Scene file does not exist.
        ValueError        : File is not valid JSON or missing "objects" key.
    """
    scene_file = Path(os.getenv("VISION_SCENE_FILE", str(_DEFAULT_SCENE_FILE)))

    if not scene_file.exists():
        raise FileNotFoundError(
            f"Scene file not found: {scene_file}. "
            f"Set VISION_SCENE_FILE in .env or ensure vision_backend/scene_representation.json exists."
        )

    try:
        with open(scene_file, encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Scene file is not valid JSON: {scene_file} — {e}") from e

    if "objects" not in raw:
        raise ValueError(f"Scene file missing required 'objects' key: {scene_file}")

    scene = _convert(raw)
    logger.info(f"[scene_representation] Loaded {len(scene['objects'])} objects from {scene_file}")
    return scene


def _convert(raw: dict) -> dict:
    """
    Normalise raw JSON into planner-compatible scene format.
    Strips Z if position has 3 elements.
    """
    objects = []
    for obj in raw["objects"]:
        label = obj["label"]
        position = list(obj["position"])[:2]
        objects.append({"label": label, "position": position})
    return {"objects": objects}