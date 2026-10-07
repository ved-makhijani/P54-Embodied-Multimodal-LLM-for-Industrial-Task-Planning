import json
import os
from datetime import datetime
from typing import Dict, Any, List


LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "task_log.json")


def log_invalid_action(
    instruction: str,
    object_name: str,
    available_objects: List[str],
    reason: str
) -> None:
    log_entry = {
        "task_id": "invalid_action_001",
        "instruction": instruction,
        "model": "vision_validation",
        "created_at": datetime.now().isoformat(),
        "status": "failed",
        "stages": {
            "vision_lookup": {
                "stage": "vision_lookup",
                "status": "failed",
                "payload": {
                    "object_target": object_name,
                    "available_objects": available_objects
                },
                "error": reason
            }
        }
    }

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def handle_invalid_action(
    scene: Dict[str, Any],
    object_name: str,
    instruction: str
) -> bool:
    available_objects = [obj.get("label") for obj in scene.get("objects", [])]

    if object_name not in available_objects:
        reason = f"Object '{object_name}' not found in scene"
        print("ERROR:", reason)

        log_invalid_action(
            instruction=instruction,
            object_name=object_name,
            available_objects=available_objects,
            reason=reason
        )

        return False

    print("Action valid:", object_name)
    return True