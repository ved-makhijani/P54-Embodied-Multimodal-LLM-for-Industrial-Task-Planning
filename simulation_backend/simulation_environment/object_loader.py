"""
object_loader.py
----------------
Loads all simulation objects from scene_config.yaml into PyBullet and
registers them in the ObjectRegistry.

Responsibilities:
    - Parse the objects list from scene_config.yaml
    - Call pybullet.loadURDF() for each object
    - Apply RGBA colour via changeVisualShape()
    - Register each loaded body in the ObjectRegistry

This file is called once at simulation startup by simulation.py.
Adding a new object to the scene requires only a new entry in
scene_config.yaml — no code changes here.

Usage:
    from simulation_backend.simulation_environment.object_loader import ObjectLoader

    loader   = ObjectLoader(physics_client, config, registry)
    body_ids = loader.load_all()
"""

import logging
import pybullet as p
from pathlib import Path
from typing import Optional

from .object_registry import ObjectRegistry, ObjectEntry

logger = logging.getLogger(__name__)


class ObjectLoader:
    """
    Loads and registers all scene objects from scene_config.yaml.

    Each object entry in config must have:
        id, label, urdf, color, position, orientation,
        mass_kg, graspable, attributes

    The URDF defines the physical shape and collision geometry.
    Color is applied post-load via changeVisualShape() so all blocks
    can share a single block.urdf with different colours.
    """

    def __init__(
        self,
        physics_client: int,
        objects_config: list[dict],
        registry:       ObjectRegistry,
    ):
        """
        Args:
            physics_client: PyBullet physics client ID from p.connect()
            objects_config: The 'objects' list parsed from scene_config.yaml
            registry:       ObjectRegistry instance to populate
        """
        self._client  = physics_client
        self._config  = objects_config
        self._registry = registry

    def load_all(self) -> list[int]:
        """
        Load all objects from config into PyBullet and register them.

        Returns:
            List of PyBullet body IDs in the order they were loaded.
        """
        body_ids = []

        for obj_cfg in self._config:
            body_id = self._load_one(obj_cfg)
            if body_id is not None:
                body_ids.append(body_id)

        logger.info(f"Loaded {len(body_ids)} objects into simulation.")
        return body_ids

    def _load_one(self, cfg: dict) -> Optional[int]:
        """
        Load a single object from its config entry.

        Args:
            cfg: Single object dict from scene_config.yaml objects list

        Returns:
            PyBullet body ID, or None if loading failed.
        """
        label       = cfg["label"]
        urdf_path   = cfg["urdf"]
        color       = cfg["color"]
        position    = cfg["position"]
        orientation = cfg.get("orientation", [0.0, 0.0, 0.0, 1.0])
        mass_kg     = cfg.get("mass_kg", 0.1)
        graspable   = cfg.get("graspable", True)
        attributes  = cfg.get("attributes", {})

        # Validate URDF exists
        if not Path(urdf_path).exists():
            logger.error(f"URDF not found for '{label}': {urdf_path}")
            return None

        try:
            # Load the URDF into PyBullet
            # useFixedBase=True for static objects (mass=0), False for dynamic
            use_fixed_base = (mass_kg == 0.0)

            body_id = p.loadURDF(
                urdf_path,
                basePosition=position,
                baseOrientation=orientation,
                useFixedBase=use_fixed_base,
                physicsClientId=self._client,
            )

            # Apply colour — all blocks share block.urdf but get different colours
            # linkIndex=-1 means the base link (the object body itself)
            p.changeVisualShape(
                body_id,
                linkIndex=-1,
                rgbaColor=color,
                physicsClientId=self._client,
            )

            # Register in the object registry
            self._registry.register(
                body_id=body_id,
                label=label,
                color=color,
                position=tuple(position),
                graspable=graspable,
                mass_kg=mass_kg,
                attributes=attributes,
            )

            logger.debug(
                f"Loaded '{label}' — body_id={body_id} "
                f"pos={position} fixed={use_fixed_base}"
            )
            return body_id

        except Exception as e:
            logger.error(f"Failed to load '{label}' from {urdf_path}: {e}")
            return None

    def reset_positions(self) -> None:
        """
        Reset all loaded objects to their original positions from config.
        Called on simulation reset between pipeline runs.
        """
        for cfg in self._config:
            label    = cfg["label"]
            position = cfg["position"]
            orientation = cfg.get("orientation", [0.0, 0.0, 0.0, 1.0])

            entry = self._registry.get_by_label(label)
            if entry is None:
                continue

            p.resetBasePositionAndOrientation(
                entry.body_id,
                position,
                orientation,
                physicsClientId=self._client,
            )
            self._registry.update_position(entry.body_id, tuple(position))
            logger.debug(f"Reset '{label}' to {position}")
