"""
workspace.py
------------
Builds the physical workspace in PyBullet — table surface, walls, and floor.

Reads workspace dimensions and appearance from scene_config.yaml.
Called once at simulation startup by simulation.py before any objects
or robots are loaded.

All dimensions are in metres. The workspace origin (0, 0, 0) is the
centre of the table surface.

Usage:
    from simulation_backend.simulation_environment.workspace import Workspace

    workspace = Workspace(physics_client, workspace_config)
    workspace.build()
"""

import logging
import pybullet as p
import pybullet_data

logger = logging.getLogger(__name__)


class Workspace:
    """
    Constructs the physical simulation environment.

    Builds in this order:
        1. Floor plane (if enabled)
        2. Table surface
        3. Perimeter walls (if enabled)

    All bodies are static (useFixedBase=True, mass=0) since the
    workspace itself never moves.
    """

    def __init__(self, physics_client: int, config: dict):
        """
        Args:
            physics_client: PyBullet client ID from p.connect()
            config:         The 'workspace' section of scene_config.yaml
        """
        self._client = physics_client
        self._cfg    = config
        self._body_ids: list[int] = []

    def build(self) -> None:
        """
        Build the full workspace — floor, table, walls.
        Logs each component as it's created.
        """
        self._load_floor()
        self._load_table()
        if self._cfg.get("walls", {}).get("enabled", True):
            self._load_walls()
        logger.info("Workspace built successfully.")

    # ── Floor ──────────────────────────────────────────────────────────────────

    def _load_floor(self) -> None:
        """Load the ground plane beneath the table."""
        floor_cfg = self._cfg.get("floor", {})
        if not floor_cfg.get("enabled", True):
            return

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self._client,
        )

        floor_id = p.loadURDF(
            "plane.urdf",
            basePosition=[0, 0, -self._cfg.get("height_m", 0.75)],
            useFixedBase=True,
            physicsClientId=self._client,
        )

        color = floor_cfg.get("color", [0.5, 0.5, 0.5, 1.0])
        p.changeVisualShape(
            floor_id,
            linkIndex=-1,
            rgbaColor=color,
            physicsClientId=self._client,
        )

        self._body_ids.append(floor_id)
        logger.debug(f"Floor loaded — body_id={floor_id}")

    # ── Table ──────────────────────────────────────────────────────────────────

    def _load_table(self) -> None:
        """Load the table surface from URDF or build from primitives."""
        urdf_path = self._cfg.get("urdf")

        if urdf_path:
            self._load_table_from_urdf(urdf_path)
        else:
            self._load_table_from_primitive()

    def _load_table_from_urdf(self, urdf_path: str) -> None:
        """Load table from a URDF file."""
        position = self._cfg.get("position", [0.0, 0.0, 0.0])

        try:
            table_id = p.loadURDF(
                urdf_path,
                basePosition=position,
                useFixedBase=True,
                physicsClientId=self._client,
            )
            color = self._cfg.get("color", [0.76, 0.60, 0.42, 1.0])
            p.changeVisualShape(
                table_id,
                linkIndex=-1,
                rgbaColor=color,
                physicsClientId=self._client,
            )
            self._body_ids.append(table_id)
            logger.debug(f"Table loaded from URDF — body_id={table_id}")

        except Exception as e:
            logger.warning(f"Could not load table URDF ({e}), falling back to primitive.")
            self._load_table_from_primitive()

    def _load_table_from_primitive(self) -> None:
        """
        Build table surface as a PyBullet box primitive.
        Used as fallback when table.urdf is not found.
        """
        width     = self._cfg.get("width_m", 2.0)
        depth     = self._cfg.get("depth_m", 1.5)
        thickness = self._cfg.get("thickness_m", 0.05)
        position  = self._cfg.get("position", [0.0, 0.0, 0.0])
        color     = self._cfg.get("color", [0.76, 0.60, 0.42, 1.0])

        half_extents = [width / 2, depth / 2, thickness / 2]

        col_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self._client,
        )
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self._client,
        )

        # Position the table so its top surface is at z=0
        table_position = [
            position[0],
            position[1],
            position[2] - thickness / 2,
        ]

        table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=table_position,
            physicsClientId=self._client,
        )

        self._body_ids.append(table_id)
        logger.debug(f"Table built from primitive — body_id={table_id}")

    # ── Walls ──────────────────────────────────────────────────────────────────

    def _load_walls(self) -> None:
        """
        Build four perimeter walls around the workspace.

        Walls are thin static boxes placed at the four edges of the
        table surface to prevent objects sliding off.
        """
        walls_cfg = self._cfg.get("walls", {})
        width     = self._cfg.get("width_m", 2.0)
        depth     = self._cfg.get("depth_m", 1.5)
        thickness = walls_cfg.get("thickness_m", 0.02)
        height    = walls_cfg.get("height_m", 0.15)
        color     = walls_cfg.get("color", [0.85, 0.85, 0.85, 0.6])

        half_h = height / 2
        half_t = thickness / 2

        # Define four walls: (half_extents, position)
        walls = [
            # Front wall  (+Y edge)
            ([width / 2, half_t, half_h], [0.0,  depth / 2, half_h]),
            # Back wall   (-Y edge)
            ([width / 2, half_t, half_h], [0.0, -depth / 2, half_h]),
            # Left wall   (-X edge)
            ([half_t, depth / 2, half_h], [-width / 2, 0.0, half_h]),
            # Right wall  (+X edge)
            ([half_t, depth / 2, half_h], [ width / 2, 0.0, half_h]),
        ]

        for half_extents, position in walls:
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self._client,
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=color,
                physicsClientId=self._client,
            )
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=position,
                physicsClientId=self._client,
            )
            self._body_ids.append(wall_id)

        logger.debug(f"Built 4 perimeter walls.")

    # ── Utilities ──────────────────────────────────────────────────────────────

    def get_body_ids(self) -> list[int]:
        """Return PyBullet body IDs of all workspace components."""
        return self._body_ids.copy()

    def reset(self) -> None:
        """
        Workspace components are static — nothing to reset.
        Kept for interface consistency with other simulation components.
        """
        pass
