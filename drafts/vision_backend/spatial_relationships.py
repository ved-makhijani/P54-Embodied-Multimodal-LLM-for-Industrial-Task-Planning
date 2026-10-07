import json
import math
from typing import Dict, Any, List


def load_json(filename: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filename: str) -> None:
    """
    Save JSON data to a file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def euclidean_distance(point1: List[int], point2: List[int]) -> float:
    """
    Compute Euclidean distance between two center points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def compute_relationships(scene_data: Dict[str, Any], near_threshold: float = 160.0) -> List[str]:
    """
    Compute pairwise spatial relationships between objects.
    """
    relationships = []
    scene = scene_data.get("scene", {})
    labels = list(scene.keys())

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            obj1 = labels[i]
            obj2 = labels[j]

            center1 = scene[obj1]["center"]
            center2 = scene[obj2]["center"]

            x1, y1 = center1
            x2, y2 = center2

            # Left / Right
            if x1 < x2:
                relationships.append(f"{obj1} is left of {obj2}")
                relationships.append(f"{obj2} is right of {obj1}")
            elif x1 > x2:
                relationships.append(f"{obj1} is right of {obj2}")
                relationships.append(f"{obj2} is left of {obj1}")

            # Near
            distance = euclidean_distance(center1, center2)
            if distance <= near_threshold:
                relationships.append(f"{obj1} is near {obj2}")
                relationships.append(f"{obj2} is near {obj1}")

    return relationships


def build_spatial_output(scene_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build final structured spatial relationship output.
    """
    return {
        "image_id": scene_data.get("image_id", "unknown"),
        "relationships": compute_relationships(scene_data)
    }


def get_spatial_relationships(filename: str = "scene_representation.json") -> List[str]:
    """
    Load scene data and return spatial relationships.
    This allows other modules (planner, validation) to use this directly.
    """
    scene_data = load_json(filename)
    return compute_relationships(scene_data)


def main() -> None:
    scene_data = load_json("scene_representation.json")
    spatial_output = build_spatial_output(scene_data)

    print(json.dumps(spatial_output, indent=2))
    save_json(spatial_output, "spatial_relationships.json")
    print("\nSaved to spatial_relationships.json")


if __name__ == "__main__":
    main()
