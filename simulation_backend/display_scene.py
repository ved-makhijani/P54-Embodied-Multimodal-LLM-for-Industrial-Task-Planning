"""
display_scene.py
----------------
Standalone visual debugging tool for the simulation environment.

Launches the PyBullet GUI window and an OpenCV camera feed window
showing the current simulation scene state. Nothing from the main
pipeline is involved — this tool is purely for visual inspection.

ONLY runnable as a module from the project root:
    python -m simulation_backend.display_scene

Do NOT import or call this from simulation.py or any other pipeline file.

What is shown:
    PyBullet GUI window:
        - Full 3D physics simulation with all scene objects
        - Floating object labels above each object (configurable)
        - Robot state debug text overlay (configurable)

    OpenCV camera feed window:
        - Live camera frame from the Camera class (camera.py)
        - Detection bounding boxes overlaid (if VISION_DETECTOR is set)
        - Segmentation mask overlay (if SEGMENTATION_MASK=true)

    Terminal output:
        - Current scene dict printed at configured rate
        - Robot state (position, held object) printed at configured rate

Press Q in the camera window or Ctrl+C in terminal to exit.

Config:
    All display settings are in scene_config.yaml under the 'display' key.
    Camera settings are under the 'camera' key — owned entirely by Camera.
    Set VISION_DETECTOR in .env to activate a primary detector.
    Set SEGMENTATION_MASK=true in .env to enable the seg mask overlay.
"""

import os
import time
import logging
import signal

import pybullet as p
import cv2
import yaml

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config path ────────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scene_config.yaml"
)

# ── Globals ────────────────────────────────────────────────────────────────────
_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received.")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)


# ── Config loader ──────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── PyBullet setup ─────────────────────────────────────────────────────────────

def _connect_gui() -> int:
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setRealTimeSimulation(1, physicsClientId=client)
    logger.info("PyBullet GUI connected.")
    return client


# ── Object label overlays ──────────────────────────────────────────────────────

def _add_object_labels(
    client:    int,
    registry,
    label_cfg: dict,
) -> list[int]:
    """
    Add floating debug text labels above each object in the PyBullet GUI.

    Returns list of debug item IDs so they can be removed on cleanup.
    Controlled by display.object_labels.enabled in scene_config.yaml.
    """
    if not label_cfg.get("enabled", True):
        logger.info("Object labels disabled in config.")
        return []

    color    = label_cfg.get("color",    [1.0, 1.0, 1.0])
    size     = label_cfg.get("size",     1.2)
    offset_z = label_cfg.get("offset_z", 0.15)

    debug_ids = []
    for entry in registry.all_entries():
        x, y, z = entry.position
        debug_id = p.addUserDebugText(
            text=entry.label,
            textPosition=[x, y, z + offset_z],
            textColorRGB=color,
            textSize=size,
            physicsClientId=client,
        )
        debug_ids.append(debug_id)

    logger.info(f"Added {len(debug_ids)} object labels to PyBullet GUI.")
    return debug_ids


def _remove_object_labels(client: int, debug_ids: list[int]) -> None:
    """Remove all object label debug items from the PyBullet GUI."""
    for debug_id in debug_ids:
        p.removeUserDebugItem(debug_id, physicsClientId=client)


# ── Robot state overlay ────────────────────────────────────────────────────────

def _update_robot_state_overlay(
    client:      int,
    overlay_id:  int | None,
    state_text:  str,
    overlay_cfg: dict,
) -> int | None:
    """
    Update the robot state debug text in the PyBullet GUI.

    Removes the previous text item and creates a new one each frame so
    the position and content stay current. Returns the new item ID.
    """
    if not overlay_cfg.get("enabled", True):
        return overlay_id

    if overlay_id is not None:
        p.removeUserDebugItem(overlay_id, physicsClientId=client)

    new_id = p.addUserDebugText(
        text=state_text,
        textPosition=overlay_cfg.get("position", [0.0, 0.0, 1.5]),
        textColorRGB=overlay_cfg.get("color",    [0.0, 1.0, 0.5]),
        textSize=overlay_cfg.get("size",          1.0),
        physicsClientId=client,
    )
    return new_id


# ── Terminal output ────────────────────────────────────────────────────────────

def _print_scene(scene: dict) -> None:
    """Print current scene dict to terminal."""
    print("\n─── Scene State ───────────────────────────────────────────")
    for obj in scene.get("detected_objects", []):
        coords = obj["position"]["coordinates_3d"]
        print(
            f"  {obj['label']:<20} "
            f"x={coords['x']:>6.3f}  y={coords['y']:>6.3f}  z={coords['z']:>6.3f}  "
            f"source={obj['source']}"
        )
    print(f"  Timestamp: {scene.get('scene_timestamp', 'unknown')}")
    print("───────────────────────────────────────────────────────────")


# ── Main display loop ──────────────────────────────────────────────────────────

def run() -> None:
    global _running

    cfg         = _load_config()
    cam_cfg     = cfg.get("camera", {})
    display_cfg = cfg.get("display", {})
    ws_cfg      = cfg.get("workspace", {})
    obj_cfg     = cfg.get("objects", [])

    label_cfg   = display_cfg.get("object_labels",       {})
    overlay_cfg = display_cfg.get("robot_state_overlay",  {})
    cv_cfg      = display_cfg.get("opencv_window",        {})
    term_cfg    = display_cfg.get("terminal",             {})

    # ── Connect to PyBullet GUI ────────────────────────────────────────────────
    client = _connect_gui()

    # ── Build workspace and load objects ───────────────────────────────────────
    from simulation_backend.simulation_environment.workspace       import Workspace
    from simulation_backend.simulation_environment.object_loader   import ObjectLoader
    from simulation_backend.simulation_environment.object_registry import ObjectRegistry

    registry  = ObjectRegistry()
    workspace = Workspace(client, ws_cfg)
    workspace.build()

    loader = ObjectLoader(client, obj_cfg, registry)
    loader.load_all()

    # ── Add object labels to PyBullet GUI ─────────────────────────────────────
    label_ids = _add_object_labels(client, registry, label_cfg)

    # ── Scene builder + ground truth ──────────────────────────────────────────
    from simulation_backend.simulation_environment.scene_builder import SceneBuilder
    from simulation_backend.vision.ground_truth                  import GroundTruth

    ground_truth = GroundTruth(client, registry)
    builder      = SceneBuilder(registry)

    # ── Camera (replaces the old _capture_frame helper) ───────────────────────
    from simulation_backend.vision.camera import Camera

    camera = Camera(physics_client=client, config=cam_cfg)
    logger.info(f"Camera ready: {camera}")

    # ── Primary detector (optional — set VISION_DETECTOR in .env) ─────────────
    from simulation_backend.vision.detection_base import get_detector

    detector = get_detector(registry=registry, config=cfg)
    if detector:
        logger.info(f"Primary detector active: {detector}")
    else:
        logger.info("No primary detector set (VISION_DETECTOR not set).")

    # ── Segmentation setup (optional — set SEGMENTATION_MASK=true in .env) ────
    seg_enabled = os.getenv("SEGMENTATION_MASK", "false").lower() == "true"
    if seg_enabled:
        logger.info("Segmentation mask overlay enabled.")

    # ── OpenCV window setup ───────────────────────────────────────────────────
    cv_enabled  = cv_cfg.get("enabled",        True)
    window_name = cv_cfg.get("window_name",    "Simulation Camera Feed")
    cv_rate     = cv_cfg.get("update_rate_hz", 30)

    if cv_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ── Timing ────────────────────────────────────────────────────────────────
    term_rate        = term_cfg.get("update_rate_hz", 2)
    last_cv_time     = 0.0
    last_term_time   = 0.0
    robot_overlay_id = None

    logger.info("Display scene running. Press Q in camera window or Ctrl+C to exit.")
    print("\n" + "=" * 60)
    print("  simulation_backend — Display Scene")
    print(f"  Objects loaded:  {len(registry)}")
    print(f"  Camera:          {camera.width}×{camera.height}  fov={cam_cfg.get('fov', 60)}°")
    print(f"  Detector:        {detector.name if detector else 'none'}")
    print(f"  Segmentation:    {seg_enabled}")
    print("  Press Q in the camera window or Ctrl+C to exit.")
    print("=" * 60 + "\n")

    while _running:
        now = time.time()
        p.stepSimulation(physicsClientId=client)

        # ── OpenCV update ──────────────────────────────────────────────────────
        if cv_enabled and (now - last_cv_time) >= 1.0 / cv_rate:

            # Camera.capture() returns a CameraFrame with .bgr, .depth, .seg
            cam_frame = camera.capture()
            display   = cam_frame.bgr.copy()

            # Segmentation mask overlay — colour each object by its registry colour
            if seg_enabled and cv_cfg.get("show_segmentation", True):
                body_id_to_color = {
                    entry.body_id: (
                        int(entry.color[0] * 255),
                        int(entry.color[1] * 255),
                        int(entry.color[2] * 255),
                    )
                    for entry in registry.all_entries()
                }
                seg_vis = cam_frame.seg_colourmap(body_id_to_color)
                display = cv2.addWeighted(display, 0.5, seg_vis, 0.5, 0)

            # Primary detector — draw bounding boxes if active
            if detector and cv_cfg.get("show_detections", True):
                try:
                    detections = detector.detect(cam_frame)
                    display    = detector.draw_detections(display, detections)
                except Exception as exc:
                    logger.warning(f"Detector error (skipping frame): {exc}")

            cv2.imshow(window_name, display)
            last_cv_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Q pressed — exiting.")
                _running = False
                break

        # ── Robot state overlay update ─────────────────────────────────────────
        if overlay_cfg.get("enabled", True):
            state_text = "Robot: idle | Held: None | Pos: (0.0, 0.0, 0.0)"
            robot_overlay_id = _update_robot_state_overlay(
                client, robot_overlay_id, state_text, overlay_cfg
            )

        # ── Terminal update ────────────────────────────────────────────────────
        if (term_cfg.get("enabled", True)
                and (now - last_term_time) >= 1.0 / term_rate):

            gt_detections = ground_truth.get_all()
            scene = builder.build(
                detector_results=[],
                segmentation_results=None,
                ground_truth_results=gt_detections,
            )

            if term_cfg.get("print_scene", True):
                _print_scene(scene)

            last_term_time = now

        time.sleep(1.0 / 240.0)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    logger.info("Cleaning up...")
    if cv_enabled:
        cv2.destroyAllWindows()
    if label_ids:
        _remove_object_labels(client, label_ids)
    p.disconnect(physicsClientId=client)
    logger.info("Display scene exited cleanly.")


if __name__ == "__main__":
    run()
