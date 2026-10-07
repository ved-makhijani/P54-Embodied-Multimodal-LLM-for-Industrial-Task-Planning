import json
from typing import List, Dict, Any


def create_detection(label: str, confidence: float, x: int, y: int, width: int, height: int) -> Dict[str, Any]:
    """
    Create one detected object entry.

    Bounding box format:
    [x, y, width, height]
    """
    return {
        "label": label,
        "confidence": round(confidence, 2),
        "bounding_box": [x, y, width, height]
    }


def create_scene_output(image_id: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create the full vision output structure.
    """
    return {
        "image_id": image_id,
        "objects": detections
    }


def save_json(data: Dict[str, Any], filename: str) -> None:
    """
    Save output to a JSON file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    detections = [
        create_detection("red block", 0.95, 120, 80, 60, 60),
        create_detection("blue tray", 0.91, 300, 200, 100, 70),
        create_detection("green block", 0.88, 220, 100, 55, 55),
    ]

    scene_output = create_scene_output("scene_01", detections)

    print(json.dumps(scene_output, indent=2))
    save_json(scene_output, "scene_output.json")
    print("\nSaved to scene_output.json")


if __name__ == "__main__":
    main()