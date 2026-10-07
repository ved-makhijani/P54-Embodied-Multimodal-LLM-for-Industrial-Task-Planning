from typing import Dict, Any


def object_exists(scene: Dict[str, Any], object_name: str) -> bool:
    """
    Check whether an object exists in the planner scene.
    Planner scene format:
    {
        "objects": [
            {"label": "red block", "position": [150, 110]}
        ]
    }
    """
    object_name = object_name.lower()

    for obj in scene.get("objects", []):
        label = obj.get("label", "").lower()

        if object_name in label or label in object_name:
            return True

    return False


def validate_object_exists(scene: Dict[str, Any], object_name: str) -> None:
    """
    Stop execution if the object does not exist.
    """
    if not object_exists(scene, object_name):
        available_objects = [obj.get("label") for obj in scene.get("objects", [])]

        raise ValueError(
            f"Object '{object_name}' not found in scene. "
            f"Available objects: {available_objects}"
        )