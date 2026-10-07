"""
gripper_base.py
---------------
Abstract base class for all gripper controllers in the simulation pipeline.

Every gripper — Franka Hand, Robotiq 85, custom parallel-jaw, or future
models — must inherit from GripperBase and implement the three abstract
methods: open(), close(), and get_state().

Design goals:
    1. One interface.  RobotBase subclasses call self._gripper.open() and
       self._gripper.close() inside pick() and place() without knowing which
       gripper model is attached. Swap grippers in scene_config.yaml without
       touching any robot control code.

    2. One return type.  All commands return GripperResult, a lightweight
       dataclass mirroring CommandResult from mock_robot.py. This keeps the
       gripper layer consistent with the rest of the pipeline's error-reporting
       convention.

    3. Force/width feedback.  get_state() returns a GripperState dataclass
       so higher-level code can detect grasp success (object actually held vs.
       gripper closed on air) by checking finger_width and contact_force.

    4. PyBullet-backed.  Real subclasses receive the parent robot's body_id
       and the PyBullet client so they can drive gripper joints via
       setJointMotorControl2(). The joint indices for the specific gripper
       links are stored at init time via _resolve_finger_joints().

Hierarchy:
    GripperBase       (this file — abstract)
        └─ FrankaHand     (robots/gripper/franka_hand.py)
        └─ Robotiq85      (robots/gripper/robotiq85.py)  [future]

Usage (inside a RobotBase subclass):
    from simulation_backend.robots.gripper.gripper_base import GripperBase

    class FrankaPanda(RobotBase):
        def __init__(self, physics_client, body_id, config=None):
            super().__init__(physics_client, body_id, config)
            self._gripper = FrankaHand(physics_client, body_id, config)

        def pick(self, object_name):
            ...
            self._gripper.open()
            # descend to object
            self._gripper.close()
            state = self._gripper.get_state()
            success = state.contact_force > self._cfg.get("grasp_force_threshold", 0.1)
            ...

Usage (implementing a new gripper):
    from simulation_backend.robots.gripper.gripper_base import GripperBase, GripperResult

    class MyGripper(GripperBase):
        model_name = "my_gripper"

        def open(self):
            ...
            return GripperResult(success=True, action="open", width_m=0.08)

        def close(self, target_width_m=0.0, max_force_n=20.0):
            ...
            return GripperResult(success=True, action="close", width_m=actual_width)

        def get_state(self):
            ...
            return GripperState(is_open=..., finger_width_m=..., contact_force_n=...)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Gripper result dataclass ───────────────────────────────────────────────────

@dataclass
class GripperResult:
    """
    Return value from every GripperBase command method.

    Fields:
        success     : True if the command completed without error.
        action      : Name of the command ("open", "close", "get_state").
        width_m     : Actual finger separation in metres after the command.
                      0.0 when fully closed; max_width when fully open.
        force_n     : Measured contact force in Newtons (0.0 if no contact).
        latency_ms  : Wall-clock time taken to execute the command.
        message     : Human-readable description of the outcome.
    """
    success:    bool
    action:     str
    width_m:    float = 0.0
    force_n:    float = 0.0
    latency_ms: float = 0.0
    message:    str   = ""

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return (f"{status} [{self.action}] {self.message} "
                f"width={self.width_m:.4f}m force={self.force_n:.2f}N "
                f"({self.latency_ms:.0f}ms)")


# ── Gripper state dataclass ────────────────────────────────────────────────────

@dataclass
class GripperState:
    """
    Point-in-time snapshot of the gripper's physical state.

    Returned by GripperBase.get_state() and used by RobotBase subclasses
    to detect whether a grasp was successful (object in hand vs. air grasp).

    Fields:
        is_open         : True if the gripper is at or near its maximum width.
        finger_width_m  : Current separation between finger tips in metres.
        contact_force_n : Sum of contact normal forces from all fingertip
                          contacts in Newtons. > 0.0 indicates an object is held.
        max_width_m     : Maximum finger separation for this gripper model.
        min_width_m     : Minimum finger separation (fully closed).
    """
    is_open:          bool
    finger_width_m:   float
    contact_force_n:  float
    max_width_m:      float = 0.08   # 8 cm — typical parallel jaw gripper
    min_width_m:      float = 0.0

    @property
    def is_closed(self) -> bool:
        """True if the gripper is at or near its minimum width."""
        return not self.is_open

    @property
    def has_contact(self) -> bool:
        """True if contact force indicates an object is being grasped."""
        return self.contact_force_n > 0.0

    @property
    def grasp_successful(self) -> bool:
        """
        Heuristic: True if the gripper is partially open (not air-grasping)
        and has detected contact force. Subclasses can override this property
        with model-specific logic.
        """
        not_air_grasp = self.finger_width_m > self.min_width_m
        return self.is_closed and not_air_grasp and self.has_contact


# ── Abstract base class ────────────────────────────────────────────────────────

class GripperBase(ABC):
    """
    Abstract base class for all gripper controllers.

    Subclasses must:
        1. Set a class-level `model_name` string (e.g. "franka_hand").
        2. Implement open() → GripperResult.
        3. Implement close(target_width_m, max_force_n) → GripperResult.
        4. Implement get_state() → GripperState.
        5. Call super().__init__(physics_client, robot_body_id, config).

    Subclasses may:
        - Override _resolve_finger_joints() to return the correct joint
          indices for their specific URDF structure.
        - Override max_width and min_width class attributes.
        - Add model-specific config keys consumed from self._cfg.

    Key implementation note:
        Gripper joints in PyBullet are driven with POSITION_CONTROL.
        Each finger is a separate revolute or prismatic joint.
        The joint indices depend entirely on the URDF link ordering —
        always call p.getJointInfo() to discover them rather than
        hard-coding indices.
    """

    #: Human-readable model name — override in every subclass.
    model_name: str = "base_gripper"

    #: Maximum finger separation in metres — override per model.
    max_width:  float = 0.08   # 8 cm  (Franka Hand default)

    #: Minimum finger separation in metres (fully closed) — override per model.
    min_width:  float = 0.0

    def __init__(
        self,
        physics_client: int,
        robot_body_id:  int,
        config:         dict = None,
    ) -> None:
        """
        Args:
            physics_client : PyBullet client ID from p.connect().
            robot_body_id  : Body ID of the parent robot arm. Gripper joints
                             are part of the same URDF body.
            config         : Optional gripper config dict (from scene_config.yaml
                             or the robot's config section).
        """
        self._client        = physics_client
        self._robot_body_id = robot_body_id
        self._cfg           = config or {}

        # Resolved by _resolve_finger_joints() — populated in subclass __init__
        self._finger_joint_indices: list[int] = []

        logger.info(f"[{self.model_name}] Gripper controller initialised "
                    f"(robot_body_id={robot_body_id}).")

    # ── Abstract commands (must implement) ────────────────────────────────────

    @abstractmethod
    def open(self) -> GripperResult:
        """
        Fully open the gripper to its maximum width.

        Implementation guide:
            1. Set each finger joint to its open position using POSITION_CONTROL:
                   for joint_idx in self._finger_joint_indices:
                       p.setJointMotorControl2(
                           self._robot_body_id, joint_idx,
                           p.POSITION_CONTROL,
                           targetPosition=self._open_position(joint_idx),
                           force=self._cfg.get("open_force_n", 20.0),
                           physicsClientId=self._client,
                       )
            2. Step the simulation until joints reach target (or timeout).
            3. Read the final finger width via get_state().
            4. Return GripperResult(success=True, action="open", width_m=...).

        Returns:
            GripperResult describing the outcome.
        """
        ...

    @abstractmethod
    def close(
        self,
        target_width_m: float = 0.0,
        max_force_n:    float = 20.0,
    ) -> GripperResult:
        """
        Close the gripper toward a target finger separation.

        Closing against an object (force control) naturally stops when the
        contact force reaches max_force_n — this prevents crushing objects.

        Implementation guide:
            1. Clamp target_width_m to [self.min_width, self.max_width].
            2. Convert target_width_m to individual joint target positions.
               For a parallel-jaw gripper each finger moves half the total width.
            3. Drive joints with POSITION_CONTROL and the specified force limit:
                   p.setJointMotorControl2(
                       self._robot_body_id, joint_idx,
                       p.POSITION_CONTROL,
                       targetPosition=target_joint_pos,
                       force=max_force_n,
                       physicsClientId=self._client,
                   )
            4. Step until convergence. Check contact force each step.
            5. Measure actual width and contact force via get_state().
            6. Return GripperResult with the actual measured values.

        Args:
            target_width_m : Desired finger separation in metres.
                             0.0 means fully closed; use max_width to open.
            max_force_n    : Maximum joint force in Newtons. Prevents crushing.

        Returns:
            GripperResult describing the outcome.
        """
        ...

    @abstractmethod
    def get_state(self) -> GripperState:
        """
        Read and return the current physical state of the gripper.

        Implementation guide:
            1. For each finger joint, read joint state:
                   pos, vel, reaction, applied = p.getJointState(
                       self._robot_body_id, joint_idx, physicsClientId=self._client
                   )
            2. Compute total finger width from joint positions.
            3. Read contact points for the finger links:
                   contacts = p.getContactPoints(
                       bodyA=self._robot_body_id,
                       physicsClientId=self._client,
                   )
               Sum the normal forces of contacts involving finger links.
            4. Determine is_open: finger_width > (self.max_width * 0.9).
            5. Return GripperState(...).

        Returns:
            GripperState with current width, contact force, and open/closed status.
        """
        ...

    # ── Optional hooks (may override) ─────────────────────────────────────────

    def _resolve_finger_joints(self) -> list[int]:
        """
        Discover and return the joint indices of the gripper finger joints.

        Default implementation returns an empty list. Subclasses MUST override
        this and call it during __init__ to populate self._finger_joint_indices.

        Example for Franka Hand (two finger joints named "panda_finger_joint1/2"):
            def _resolve_finger_joints(self) -> list[int]:
                import pybullet as p
                indices = []
                for i in range(p.getNumJoints(self._robot_body_id,
                                              physicsClientId=self._client)):
                    info = p.getJointInfo(self._robot_body_id, i,
                                         physicsClientId=self._client)
                    joint_name = info[1].decode("utf-8")
                    if "finger" in joint_name.lower():
                        indices.append(i)
                return indices

        Returns:
            List of integer joint indices for all finger joints.
        """
        return []

    def is_object_grasped(self, force_threshold_n: float = 0.5) -> bool:
        """
        Convenience wrapper: True if the gripper is holding an object.

        Args:
            force_threshold_n : Minimum contact force (N) to consider a grasp valid.

        Returns:
            True if contact force exceeds threshold and gripper is not fully open.
        """
        state = self.get_state()
        return (not state.is_open) and (state.contact_force_n >= force_threshold_n)

    # ── Shared helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _elapsed(start: float) -> float:
        """Return milliseconds elapsed since `start` (from time.perf_counter())."""
        return (time.perf_counter() - start) * 1000

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"model='{self.model_name}', "
                f"robot_body_id={self._robot_body_id})")
