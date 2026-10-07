"""
simulation_environment
----------------------
Sub-package responsible for building and managing the physical
simulation environment in PyBullet.

Modules:
    workspace       : Table surface, floor, and perimeter walls
    object_loader   : Loads object URDFs from scene_config.yaml
    object_registry : Maps PyBullet body IDs to labels and attributes
    scene_builder   : Assembles the rich scene dict from detections
"""
