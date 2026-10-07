"""
simulation_backend/robots/gripper/franka_hand.py
-------------------------------------------------
Franka Hand parallel-jaw gripper controller for the Franka Panda arm.

Franka URDF joint structure (from franka_panda/panda.urdf):
    Index 0–6  : panda_joint1 – panda_joint7   (revolute, arm DOF)
    Index 7    : panda_joint8                   (fixed)
    Index 8    : panda_hand_joint               (fixed)
    Index 9    : panda_finger_joint1            (prismatic, left finger)
    Index 10   : panda_finger_joint2            (prismatic, right finger)
    Index 11   : panda_grasptarget_hand         (fixed)

    Link 9  = panda_leftfinger
    Link 10 = panda_rightfinger

Physical limits per finger (from URDF):
    position : 0.0 m (closed) → 0.04 m (open)
    force    : 20 N max
    velocity : 0.2 m/s max

Total gap = left_pos + right_pos (both fingers move symmetrically).
Full open = 0.04 + 0.04 = 0.08 m = 8 cm.
Full close = 0.00 + 0.00 = 0.00 m.

How open() works:
    Drives both finger joints to 0.04 m using POSITION_CONTROL.
    Steps the simulation for up to _open_steps iterations or until
    both joints are within _pos_tol of 0.04.

How close() works:
    Drives both finger joints toward target_width_m / 2 each.
    Uses max_force_n as the joint force limit — this is how PyBullet
    simulates compliance. When the fingers meet an object the contact
    reaction force resists further motion naturally.
    Steps the simulation until convergence or max steps reached.
    Returns the actual measured width and contact force.

How get_state() works:
    Reads joint positions via p.getJointState() → computes total width.
    Reads contact points via p.getContactPoints() for both finger links
    and sums all normal forces to get contact_force_n.

Public interface:
    open()                          → GripperResult
    close(target_width_m, force_n)  → GripperResult
    get_state()                     → GripperState
    is_object_grasped()             → bool  (inherited from GripperBase)

Usage (called from FrankaPanda):
    gripper = FrankaHand(physics_client=client, robot_body_id=body_id)
    gripper.open()
    # ... arm descends to object ...
    result = gripper.close(target_width_m=0.0, max_force_n=20.0)
    state  = gripper.get_state()
    if state.grasp_successful:
        # object is in hand
"""

import logging
import time

import pybullet as p

from simulation_backend.robots.gripper.gripper_base import (
    GripperBase,
    GripperResult,
    GripperState,
)

logger = logging.getLogger(__name__)


class FrankaHand(GripperBase):
    """
    Franka Hand parallel-jaw gripper controller.

    Both fingers are prismatic joints on the same URDF body as the arm.
    Each finger travels independently from 0.0 m (closed) to 0.04 m (open).
    Total grip width = sum of both finger positions.

    Simulation stepping:
        All open/close commands step the PyBullet simulation internally
        until the joints converge. The number of steps is bounded by
        _open_steps and _close_steps to prevent infinite loops.
        Each step advances the simulation by one physics timestep
        (default 1/240 s in PyBullet).
    """

    model_name = "franka_hand"

    # ── Physical constants from panda.urdf ────────────────────────────────────
    # Joint indices (0-based, same body as the arm)
    _FINGER_JOINT_1 = 9   # panda_finger_joint1 — left finger
    _FINGER_JOINT_2 = 10  # panda_finger_joint2 — right finger

    # Link indices for contact point queries
    _FINGER_LINK_1  = 9   # panda_leftfinger
    _FINGER_LINK_2  = 10  # panda_rightfinger

    # Per-finger position limits (metres)
    _FINGER_OPEN    = 0.04   # 4 cm per finger = 8 cm total gap
    _FINGER_CLOSED  = 0.00   # fully closed

    # Inherited class attributes (override GripperBase defaults)
    max_width = 0.08   # 8 cm total (left + right)
    min_width = 0.00

    def __init__(
        self,
        physics_client: int,
        robot_body_id:  int,
        config:         dict = None,
    ) -> None:
        """
        Args:
            physics_client : PyBullet client ID from p.connect().
            robot_body_id  : Body ID of the loaded panda.urdf.
            config         : Optional config dict. Recognised keys:
                               open_force_n      (default 20.0)
                               close_force_n     (default 20.0)
                               open_steps        (default 120)
                               close_steps       (default 200)
                               position_tol      (default 0.001)
        """
        super().__init__(physics_client, robot_body_id, config)

        self._open_force    = float(self._cfg.get("open_force_n",   20.0))
        self._close_force   = float(self._cfg.get("close_force_n",  20.0))
        self._open_steps    = int(self._cfg.get("open_steps",       120))
        self._close_steps   = int(self._cfg.get("close_steps",      200))
        self._pos_tol       = float(self._cfg.get("position_tol",   0.001))

        # Populate the inherited _finger_joint_indices list
        self._finger_joint_indices = [self._FINGER_JOINT_1, self._FINGER_JOINT_2]

        logger.info(
            f"[franka_hand] Initialised — "
            f"joints=[{self._FINGER_JOINT_1}, {self._FINGER_JOINT_2}]  "
            f"open_force={self._open_force}N  close_force={self._close_force}N"
        )

    # ── open ──────────────────────────────────────────────────────────────────

    def open(self) -> GripperResult:
        """
        Fully open the gripper to 0.08 m (4 cm per finger).

        Drives both prismatic joints to their upper limit (0.04 m)
        using POSITION_CONTROL, then steps the simulation until both
        joints reach within _pos_tol of the target.

        Returns:
            GripperResult with final measured width and latency.
        """
        start = time.perf_counter()

        self._set_finger_positions(
            target_pos=self._FINGER_OPEN,
            force=self._open_force,
        )

        self._step_until_converged(
            target_pos=self._FINGER_OPEN,
            max_steps=self._open_steps,
        )

        state = self.get_state()
        lat   = self._elapsed(start)

        logger.debug(
            f"[franka_hand] open() done — "
            f"width={state.finger_width_m:.4f}m  {lat:.0f}ms"
        )

        return GripperResult(
            success    = True,
            action     = "open",
            width_m    = state.finger_width_m,
            force_n    = state.contact_force_n,
            latency_ms = lat,
            message    = f"Gripper opened to {state.finger_width_m:.4f}m",
        )

    # ── close ─────────────────────────────────────────────────────────────────

    def close(
        self,
        target_width_m: float = 0.0,
        max_force_n:    float = 20.0,
    ) -> GripperResult:
        """
        Close the gripper toward a target total finger separation.

        Drives each finger joint to (target_width_m / 2) with max_force_n
        as the force limit. When fingers contact an object the physics
        engine naturally resists further motion — the joint force acts as
        a compliance limit preventing object crushing.

        Args:
            target_width_m : Desired total gap in metres. Default 0.0 (fully closed).
                             Clamped to [0.0, 0.08].
            max_force_n    : Maximum force per finger joint. Default 20.0 N.

        Returns:
            GripperResult with actual measured width, contact force, and latency.
        """
        start = time.perf_counter()

        # Clamp to physical limits
        target_total = max(self.min_width, min(self.max_width, target_width_m))
        # Each finger moves to half the total gap
        target_per_finger = target_total / 2.0

        self._set_finger_positions(
            target_pos=target_per_finger,
            force=max_force_n,
        )

        self._step_until_converged(
            target_pos=target_per_finger,
            max_steps=self._close_steps,
        )

        state = self.get_state()
        lat   = self._elapsed(start)

        logger.debug(
            f"[franka_hand] close() done — "
            f"width={state.finger_width_m:.4f}m  "
            f"force={state.contact_force_n:.3f}N  {lat:.0f}ms"
        )

        return GripperResult(
            success    = True,
            action     = "close",
            width_m    = state.finger_width_m,
            force_n    = state.contact_force_n,
            latency_ms = lat,
            message    = (
                f"Gripper closed to {state.finger_width_m:.4f}m "
                f"(contact: {state.contact_force_n:.3f}N)"
            ),
        )

    # ── get_state ─────────────────────────────────────────────────────────────

    def get_state(self) -> GripperState:
        """
        Read the current physical state of the Franka Hand.

        Reads joint positions for both finger joints and sums them to
        get total finger width. Queries all contact points on both finger
        links and sums normal forces to get total contact force.

        Returns:
            GripperState with is_open, finger_width_m, and contact_force_n.
        """
        # ── Finger positions → total width ────────────────────────────────
        pos1 = p.getJointState(
            self._robot_body_id,
            self._FINGER_JOINT_1,
            physicsClientId=self._client,
        )[0]
        pos2 = p.getJointState(
            self._robot_body_id,
            self._FINGER_JOINT_2,
            physicsClientId=self._client,
        )[0]
        total_width = pos1 + pos2

        # ── Contact force from both finger links ──────────────────────────
        contact_force = 0.0
        for link_idx in (self._FINGER_LINK_1, self._FINGER_LINK_2):
            contacts = p.getContactPoints(
                bodyA=self._robot_body_id,
                linkIndexA=link_idx,
                physicsClientId=self._client,
            )
            if contacts:
                for contact in contacts:
                    # contact[9] = normal force magnitude
                    contact_force += contact[9]

        # ── is_open: width within 10% of max ──────────────────────────────
        is_open = total_width >= (self.max_width * 0.9)

        return GripperState(
            is_open          = is_open,
            finger_width_m   = round(total_width, 5),
            contact_force_n  = round(contact_force, 4),
            max_width_m      = self.max_width,
            min_width_m      = self.min_width,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _set_finger_positions(self, target_pos: float, force: float) -> None:
        """
        Send a POSITION_CONTROL command to both finger joints simultaneously.

        Args:
            target_pos : Target position per finger in metres.
                         0.0 = closed, 0.04 = fully open.
            force      : Maximum joint force in Newtons.
        """
        for joint_idx in self._finger_joint_indices:
            p.setJointMotorControl2(
                bodyIndex     = self._robot_body_id,
                jointIndex    = joint_idx,
                controlMode   = p.POSITION_CONTROL,
                targetPosition= target_pos,
                force         = force,
                physicsClientId = self._client,
            )

    def _step_until_converged(self, target_pos: float, max_steps: int) -> None:
        """
        Step the simulation until both finger joints reach target_pos
        (within _pos_tol) or max_steps is exhausted.

        Each call to p.stepSimulation() advances the physics by one
        timestep (1/240 s by default). This runs synchronously — the
        caller blocks until convergence or timeout.

        Args:
            target_pos : Per-finger target position in metres.
            max_steps  : Maximum number of simulation steps before giving up.
        """
        for _ in range(max_steps):
            p.stepSimulation(physicsClientId=self._client)

            pos1 = p.getJointState(
                self._robot_body_id,
                self._FINGER_JOINT_1,
                physicsClientId=self._client,
            )[0]
            pos2 = p.getJointState(
                self._robot_body_id,
                self._FINGER_JOINT_2,
                physicsClientId=self._client,
            )[0]

            if (abs(pos1 - target_pos) < self._pos_tol and
                    abs(pos2 - target_pos) < self._pos_tol):
                return

        logger.debug(
            f"[franka_hand] _step_until_converged: "
            f"max_steps={max_steps} reached without full convergence."
        )

    def _resolve_finger_joints(self) -> list[int]:
        """
        Return the two finger joint indices for the Franka Hand.

        Overrides GripperBase._resolve_finger_joints().
        The indices are hard-coded from the URDF because the Franka URDF
        structure is fixed and well-documented. If the URDF ever changes,
        this can be switched to dynamic discovery via p.getJointInfo().
        """
        return [self._FINGER_JOINT_1, self._FINGER_JOINT_2]
