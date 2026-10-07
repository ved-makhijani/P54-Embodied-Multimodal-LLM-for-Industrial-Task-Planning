"""
fake_ros_robot.py
-----------------
Simulates the ROS2 UR5 robot locally without needing a real robot or ROS network.
Logs every command to the terminal exactly as the real ros_robot.py would publish it.

Set in .env:
    ROBOT_BACKEND=fake_ros
"""
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class FakeROSRobot:
    """
    Drop-in replacement for ROSRobot that prints ROS2 commands
    instead of publishing them. Used for sandbox testing.
    """

    def __init__(self):
        self._held_object: Optional[str] = None
        self._position = (0.0, 0.0, 0.3)
        self._scene: dict = {}
        logger.info("[FakeROSRobot] Initialised — ROS2 commands will be logged only")
        print("\n  [FakeROSRobot] Sandbox mode — simulating UR5 via ROS2 logs\n")

    def load_scene(self, scene: dict) -> None:
        self._scene = {
            obj["label"]: obj["position"]
            for obj in scene.get("objects", [])
        }
        logger.info(f"[FakeROSRobot] Scene loaded: {len(self._scene)} objects")

    def locate(self, object_name: str) -> dict:
        name_lower = object_name.lower()
        for label, pos in self._scene.items():
            if name_lower in label.lower():
                print(f"  [ROS2] → /p54/target_pose  LOCATE '{label}' at {pos}")
                return {"label": label, "position": pos}
        raise ValueError(f"Object '{object_name}' not found in scene.")

    def move_to(self, position: tuple) -> bool:
        x, y = float(position[0]), float(position[1])
        z = 0.3
        print(f"  [ROS2] → /p54/target_pose  MOVE to ({x:.3f}, {y:.3f}, {z:.3f})")
        time.sleep(0.1)  # simulate ack delay
        self._position = (x, y, z)
        return True

    def pick(self, object_name: str) -> bool:
        if self._held_object:
            raise ValueError(f"Already holding '{self._held_object}'. Place it first.")
        print(f"  [ROS2] → /p54/gripper_command  CLOSE (pick '{object_name}')")
        time.sleep(0.05)
        self._held_object = object_name
        return True

    def place(self, object_name: str) -> bool:
        print(f"  [ROS2] → /p54/gripper_command  OPEN (place '{object_name}')")
        time.sleep(0.05)
        self._held_object = None
        return True

    def reset(self) -> None:
        print(f"  [ROS2] → /p54/target_pose  HOME (0.0, 0.0, 0.5)")
        self._held_object = None
        self._position = (0.0, 0.0, 0.3)

    def get_state(self) -> dict:
        return {
            "position":    self._position,
            "held_object": self._held_object,
            "ros_active":  False,
            "sandbox":     True,
        }