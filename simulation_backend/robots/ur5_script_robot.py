"""
ur5_script_robot.py
-------------------
Controls UR5 by sending URScript commands over a TCP socket.
Gripper: OnRobot RG2 via URScript rg_grip() function.
No ROS2, no RTDE library — works over plain Ethernet.

Why URScript over socket:
    The UR5 controller box listens on port 30002 for URScript programs.
    We send movement and gripper commands as plain text strings.
    If the robot is unreachable, commands are logged only — pipeline still completes.

Set in .env:
    ROBOT_BACKEND=urscript
    UR5_ROBOT_IP=<robot IP from teach pendant Settings → Network>
    UR5_MOVE_SPEED=0.1         (m/s, default 0.1 — slow and safe)
    UR5_MOVE_ACCEL=0.5         (m/s², default 0.5)
    UR5_TCP_HEIGHT=0.15        (approach height above table in metres)
    UR5_GRASP_HEIGHT=0.01      (height to descend to for grasp)
    UR5_GRIPPER_WIDTH_OPEN=1000  (RG2 open width in tenths of mm, 1000 = 100mm)
    UR5_GRIPPER_WIDTH_CLOSE=0    (RG2 close width in tenths of mm, 0 = fully closed)
    UR5_GRIPPER_FORCE=40         (RG2 grip force in Newtons)
"""

import os
import socket
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config from environment ────────────────────────────────────────────────────
UR5_IP               = os.getenv("UR5_ROBOT_IP",          "192.168.1.100")
UR5_PORT             = int(os.getenv("UR5_PORT",           "30002"))
MOVE_SPEED           = float(os.getenv("UR5_MOVE_SPEED",   "0.1"))
MOVE_ACCEL           = float(os.getenv("UR5_MOVE_ACCEL",   "0.5"))
TCP_HEIGHT           = float(os.getenv("UR5_TCP_HEIGHT",   "0.15"))
GRASP_HEIGHT         = float(os.getenv("UR5_GRASP_HEIGHT", "0.01"))
GRIPPER_WIDTH_OPEN   = int(os.getenv("UR5_GRIPPER_WIDTH_OPEN",  "1000"))
GRIPPER_WIDTH_CLOSE  = int(os.getenv("UR5_GRIPPER_WIDTH_CLOSE", "0"))
GRIPPER_FORCE        = int(os.getenv("UR5_GRIPPER_FORCE",       "40"))
GRIPPER_DELAY        = float(os.getenv("UR5_GRIPPER_DELAY",     "1.5"))
MOVE_TIMEOUT         = float(os.getenv("UR5_MOVE_TIMEOUT",      "10.0"))

# ── Robot orientation (tool pointing straight down) ───────────────────────────
TOOL_ORIENTATION = [0.0, 3.14159, 0.0]


class UR5ScriptRobot:
    """
    UR5 robot driver using URScript over TCP socket.
    Gripper: OnRobot RG2 via rg_grip() URScript function.
    Falls back to dry-run logging if the robot is unreachable.

    Interface matches MockRobot exactly — swap in main.py with one env var.
    """

    def __init__(self):
        self._ip:    str           = UR5_IP
        self._held:  Optional[str] = None
        self._pos:   tuple         = (0.0, 0.0, TCP_HEIGHT)
        self._scene: dict          = {}
        self._connected: bool      = self._check_connection()

        if self._connected:
            logger.info(f"[UR5-Script] Connected to {self._ip}:{UR5_PORT}")
            print(f"\n  [UR5-Script] Connected to robot at {self._ip}\n")
        else:
            logger.warning(
                f"[UR5-Script] Cannot reach {self._ip}:{UR5_PORT} "
                f"— running in dry-run mode (commands logged only)"
            )
            print(
                f"\n  [UR5-Script] Robot at {self._ip} unreachable "
                f"— dry-run mode\n"
            )

    # ── Connection ─────────────────────────────────────────────────────────────

    def _check_connection(self) -> bool:
        """Check if the UR5 controller is reachable."""
        try:
            with socket.create_connection(
                (self._ip, UR5_PORT), timeout=2.0
            ):
                return True
        except (OSError, ConnectionRefusedError, TimeoutError):
            return False

    def _send(self, script: str) -> bool:
        """
        Send a URScript line to the robot.
        Returns True on success, False on failure.
        """
        log_line = script.strip()[:80]
        if not self._connected:
            logger.info(f"[UR5-Script][DRY-RUN] {log_line}")
            time.sleep(0.05)
            return False

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(MOVE_TIMEOUT)
                s.connect((self._ip, UR5_PORT))
                payload = (script.strip() + "\n").encode("utf-8")
                s.sendall(payload)
                logger.info(f"[UR5-Script] SENT: {log_line}")
            return True
        except Exception as e:
            logger.warning(f"[UR5-Script] Send failed ({e}) — dry-run fallback")
            return False

    # ── Scene ──────────────────────────────────────────────────────────────────

    def load_scene(self, scene: dict) -> None:
        """Load the current scene from the vision module."""
        self._scene = {}
        for obj in scene.get("objects", []):
            label = obj.get("label") or obj.get("name", "")
            pos   = obj.get("position", (0.0, 0.0))
            if isinstance(pos, (list, tuple)):
                self._scene[label] = (float(pos[0]), float(pos[1]))
        logger.info(f"[UR5-Script] Scene loaded: {len(self._scene)} objects")

    # ── Robot commands ─────────────────────────────────────────────────────────

    def locate(self, object_name: str):
        """Return CommandResult confirming object exists in scene."""
        from simulation_backend.mock_robot import CommandResult
        name_lower = object_name.lower()
        for label, pos in self._scene.items():
            if name_lower in label.lower():
                logger.info(f"[UR5-Script] Located '{label}' at {pos}")
                print(f"  [UR5] LOCATE  '{label}' → ({pos[0]:.3f}, {pos[1]:.3f})")
                return CommandResult(
                    success=True,
                    command="locate",
                    message=f"Found '{label}' at {pos}"
                )
        return CommandResult(
            success=False,
            command="locate",
            message=f"Object '{object_name}' not found in scene"
        )

    def move_to(self, x: float, y: float):
        """Move TCP to (x, y) at approach height."""
        from simulation_backend.mock_robot import CommandResult
        x, y = float(x), float(y)
        rx, ry, rz = TOOL_ORIENTATION

        approach_pose = (
            f"p[{x:.4f},{y:.4f},{TCP_HEIGHT:.4f},"
            f"{rx:.4f},{ry:.4f},{rz:.4f}]"
        )
        script = (
            f"movel({approach_pose}, "
            f"a={MOVE_ACCEL}, v={MOVE_SPEED})"
        )
        print(f"  [UR5] MOVE    ({x:.3f}, {y:.3f}, {TCP_HEIGHT:.3f})")
        self._send(script)
        time.sleep(1.0)
        self._pos = (x, y, TCP_HEIGHT)
        return CommandResult(
            success=True,
            command="move",
            message=f"Moved to ({x:.3f}, {y:.3f})"
        )

    def move_to_object(self, object_name: str):
        """Move to a named object's position."""
        from simulation_backend.mock_robot import CommandResult
        name_lower = object_name.lower()
        for label, pos in self._scene.items():
            if name_lower in label.lower():
                return self.move_to(pos[0], pos[1])
        return CommandResult(
            success=False,
            command="move",
            message=f"Object '{object_name}' not found"
        )

    def pick(self, object_name: str):
        """Descend, close RG2 gripper, ascend."""
        from simulation_backend.mock_robot import CommandResult
        if self._held:
            return CommandResult(
                success=False,
                command="pick",
                message=f"Already holding '{self._held}'"
            )

        print(f"  [UR5] PICK    '{object_name}'")
        self._descend()
        self._gripper_close()
        self._ascend()

        self._held = object_name
        logger.info(f"[UR5-Script] Picked '{object_name}'")
        return CommandResult(
            success=True,
            command="pick",
            message=f"Picked '{object_name}'"
        )

    def place(self, object_name: str):
        """Descend, open RG2 gripper, ascend."""
        from simulation_backend.mock_robot import CommandResult
        print(f"  [UR5] PLACE   '{self._held}' at '{object_name}'")
        self._descend()
        self._gripper_open()
        self._ascend()

        self._held = None
        logger.info(f"[UR5-Script] Placed at '{object_name}'")
        return CommandResult(
            success=True,
            command="place",
            message=f"Placed at '{object_name}'"
        )

    def reset(self) -> None:
        """Move robot to home position."""
        rx, ry, rz = TOOL_ORIENTATION
        home_pose = (
            f"p[0.300,0.000,0.400,"
            f"{rx:.4f},{ry:.4f},{rz:.4f}]"
        )
        script = f"movel({home_pose}, a={MOVE_ACCEL}, v={MOVE_SPEED})"
        print("  [UR5] HOME    (0.300, 0.000, 0.400)")
        self._send(script)
        self._held = None
        self._pos  = (0.3, 0.0, 0.4)

    def stop(self) -> None:
        """Emergency stop."""
        self._send("stopj(2.0)")
        logger.warning("[UR5-Script] Emergency stop sent")

    def get_state(self) -> dict:
        return {
            "ip":          self._ip,
            "connected":   self._connected,
            "position":    self._pos,
            "held_object": self._held,
        }

    # ── Motion helpers ─────────────────────────────────────────────────────────

    def _descend(self) -> None:
        """Descend to grasp height at current (x, y)."""
        x, y, _ = self._pos
        rx, ry, rz = TOOL_ORIENTATION
        grasp_pose = (
            f"p[{x:.4f},{y:.4f},{GRASP_HEIGHT:.4f},"
            f"{rx:.4f},{ry:.4f},{rz:.4f}]"
        )
        script = (
            f"movel({grasp_pose}, "
            f"a={MOVE_ACCEL}, v={MOVE_SPEED * 0.5})"
        )
        print(f"  [UR5] DESCEND ({x:.3f}, {y:.3f}, {GRASP_HEIGHT:.3f})")
        self._send(script)
        time.sleep(0.8)

    def _ascend(self) -> None:
        """Return to approach height."""
        x, y, _ = self._pos
        rx, ry, rz = TOOL_ORIENTATION
        retract_pose = (
            f"p[{x:.4f},{y:.4f},{TCP_HEIGHT:.4f},"
            f"{rx:.4f},{ry:.4f},{rz:.4f}]"
        )
        script = (
            f"movel({retract_pose}, "
            f"a={MOVE_ACCEL}, v={MOVE_SPEED * 0.5})"
        )
        print(f"  [UR5] ASCEND  ({x:.3f}, {y:.3f}, {TCP_HEIGHT:.3f})")
        self._send(script)
        time.sleep(0.8)

    # ── OnRobot RG2 gripper ───────────────────────────────────────────────────

    def _gripper_close(self) -> None:
        """
        Close OnRobot RG2 gripper using rg_grip() URScript function.
        rg_grip(width, force, depth_compensation, wait_for_grip, blocking)
            width: target width in tenths of mm (0 = fully closed)
            force: grip force in Newtons
        Requires OnRobot RG2 URCap installed on teach pendant.
        """
        script = (
            f"def rg2_close():\n"
            f"  rg_grip({GRIPPER_WIDTH_CLOSE}, {GRIPPER_FORCE}, 0, False, False)\n"
            f"end\n"
            f"rg2_close()"
        )
        print(f"  [UR5] RG2 CLOSE (width={GRIPPER_WIDTH_CLOSE}, force={GRIPPER_FORCE}N)")
        self._send(script)
        time.sleep(GRIPPER_DELAY)

    def _gripper_open(self) -> None:
        """
        Open OnRobot RG2 gripper using rg_grip() URScript function.
            width: target width in tenths of mm (1000 = 100mm open)
        """
        script = (
            f"def rg2_open():\n"
            f"  rg_grip({GRIPPER_WIDTH_OPEN}, {GRIPPER_FORCE}, 0, False, False)\n"
            f"end\n"
            f"rg2_open()"
        )
        print(f"  [UR5] RG2 OPEN  (width={GRIPPER_WIDTH_OPEN}, force={GRIPPER_FORCE}N)")
        self._send(script)
        time.sleep(GRIPPER_DELAY)