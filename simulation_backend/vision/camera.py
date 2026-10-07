"""
camera.py
---------
Eye-to-hand camera for the simulation backend.

Wraps PyBullet's getCameraImage() into a proper class that owns its
configuration, computes view/projection matrices once on init (recomputed
if config changes), and exposes typed capture results.

Returned frames are always BGR (OpenCV convention) so they can be passed
directly to cv2.imshow() or any DetectorBase subclass without conversion.

Usage:
    from simulation_backend.vision.camera import Camera, CameraFrame

    cam   = Camera(physics_client=client, config=cfg["camera"])
    frame = cam.capture()

    cv2.imshow("feed", frame.bgr)         # live display
    detections = detector.detect(frame)   # pass to any detector

Config keys (all from scene_config.yaml → camera:):
    mode          : "eye_to_hand" (eye_in_hand reserved for future)
    position      : [x, y, z]  camera mount in world coords (metres)
    target        : [x, y, z]  point the camera looks at
    up_vector     : [x, y, z]  which world axis is "up" in the image
    fov           : degrees     vertical field of view
    aspect_ratio  : float       width / height  (e.g. 1.333 for 640×480)
    near_val      : float       near clipping plane (metres)
    far_val       : float       far clipping plane (metres)
    width         : int         capture width in pixels
    height        : int         capture height in pixels

Coordinate conventions:
    - World space  : metres, right-hand, Z-up
    - Image space  : pixels, origin top-left, x right, y down
    - Depth buffer : raw PyBullet [0, 1] — linearised to metres in CameraFrame
    - BGR image    : uint8 numpy array, shape (height, width, 3)
    - Seg mask     : int32 numpy array, shape (height, width)
                     each pixel value is the PyBullet body_id of the object
                     visible at that pixel, or -1 for background
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pybullet as p

logger = logging.getLogger(__name__)


# ── Frame dataclass ────────────────────────────────────────────────────────────

@dataclass
class CameraFrame:
    """
    One captured frame from the simulation camera.

    All three arrays share the same (height, width) spatial extent and
    are produced by a single p.getCameraImage() call, so they are
    perfectly aligned — no reprojection needed.

    Fields:
        bgr      : uint8  (H, W, 3)  OpenCV-convention colour image
        depth    : float32 (H, W)    linearised depth in metres (0 = camera, far_val = max)
        seg      : int32  (H, W)     PyBullet body_id per pixel; -1 = background
        width    : capture width in pixels (convenience copy of bgr.shape[1])
        height   : capture height in pixels (convenience copy of bgr.shape[0])

    Usage examples:
        frame.bgr                           # pass to cv2.imshow
        frame.depth[y, x]                  # metric depth at pixel (x, y)
        mask = (frame.seg == body_id)       # isolate one object
        frame.depth_colourmap()            # visualise depth as colour image
    """
    bgr:    np.ndarray          # (H, W, 3) uint8
    depth:  np.ndarray          # (H, W)    float32  metres
    seg:    np.ndarray          # (H, W)    int32    body_id
    width:  int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self) -> None:
        self.height, self.width = self.bgr.shape[:2]

    def depth_colourmap(
        self,
        near: float = 0.0,
        far:  float = 5.0,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Convert the metric depth map to a false-colour uint8 BGR image.

        Useful for visualising depth alongside the colour feed.

        Args:
            near     : depth value mapped to 0 (cool colour in JET)
            far      : depth value mapped to 255 (warm colour in JET)
            colormap : any cv2.COLORMAP_* constant

        Returns:
            (H, W, 3) uint8 BGR false-colour image
        """
        clipped  = np.clip(self.depth, near, far)
        normalised = ((clipped - near) / max(far - near, 1e-6) * 255).astype(np.uint8)
        return cv2.applyColorMap(normalised, colormap)

    def seg_colourmap(self, body_id_to_color: dict[int, tuple]) -> np.ndarray:
        """
        Render the segmentation mask as a colour image.

        Args:
            body_id_to_color : {body_id: (r, g, b)}  each component 0-255

        Returns:
            (H, W, 3) uint8 BGR colour-coded segmentation image
        """
        canvas = np.zeros((*self.seg.shape, 3), dtype=np.uint8)
        for body_id, (r, g, b) in body_id_to_color.items():
            mask = (self.seg == body_id)
            canvas[mask] = [b, g, r]   # OpenCV BGR
        return canvas


# ── Camera class ───────────────────────────────────────────────────────────────

class Camera:
    """
    Eye-to-hand simulation camera backed by PyBullet's renderer.

    Reads all configuration from the 'camera' section of scene_config.yaml.
    View and projection matrices are built once in __init__ and cached;
    call reconfigure() if you move the camera at runtime.

    Thread safety: not thread-safe — call capture() from a single thread.

    Typical lifecycle:
        cam   = Camera(client, cfg["camera"])
        frame = cam.capture()          # → CameraFrame

    Intrinsic matrix (for computer-vision use):
        K = cam.intrinsics()           # 3×3 numpy float64
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, physics_client: int, config: dict) -> None:
        """
        Args:
            physics_client : PyBullet client ID from p.connect()
            config         : The 'camera' section of scene_config.yaml
        """
        self._client = physics_client
        self._cfg    = config

        self._width:  int   = int(config.get("width",  640))
        self._height: int   = int(config.get("height", 480))
        self._near:   float = float(config.get("near_val", 0.1))
        self._far:    float = float(config.get("far_val",  10.0))
        self._fov:    float = float(config.get("fov", 60))
        self._aspect: float = float(config.get("aspect_ratio", self._width / self._height))

        self._position:  list[float] = config.get("position",  [1.5, 0.0, 1.8])
        self._target:    list[float] = config.get("target",    [0.0, 0.0, 0.0])
        self._up_vector: list[float] = config.get("up_vector", [0.0, 0.0, 1.0])

        self._view_matrix: tuple = ()
        self._proj_matrix: tuple = ()
        self._build_matrices()

        logger.info(
            f"Camera initialised — {self._width}×{self._height}  "
            f"fov={self._fov}°  pos={self._position}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def capture(self) -> CameraFrame:
        """
        Render one frame from the camera's viewpoint and return a CameraFrame.

        Calls p.getCameraImage() which steps the PyBullet renderer for this
        view only — it does NOT advance the physics simulation.

        Returns:
            CameraFrame  with .bgr, .depth (metres), .seg (body_id mask)
        """
        _, _, rgba_flat, depth_flat, seg_flat = p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )

        # ── Colour ────────────────────────────────────────────────────────────
        rgba = np.reshape(rgba_flat, (self._height, self._width, 4)).astype(np.uint8)
        bgr  = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)

        # ── Depth — linearise from NDC to metres ──────────────────────────────
        # PyBullet returns values in [0, 1] (NDC depth buffer).
        # The standard OpenGL linearisation formula recovers metric depth.
        ndc_depth = np.reshape(depth_flat, (self._height, self._width)).astype(np.float32)
        depth_m   = self._linearise_depth(ndc_depth)

        # ── Segmentation ──────────────────────────────────────────────────────
        seg = np.reshape(seg_flat, (self._height, self._width)).astype(np.int32)

        return CameraFrame(bgr=bgr, depth=depth_m, seg=seg)

    def reconfigure(self, config: dict) -> None:
        """
        Update camera parameters at runtime (e.g. moving to eye-in-hand).

        Rebuilds the internal view and projection matrices.

        Args:
            config : New camera config dict (same format as __init__)
        """
        self._cfg        = config
        self._width      = int(config.get("width",  self._width))
        self._height     = int(config.get("height", self._height))
        self._near       = float(config.get("near_val", self._near))
        self._far        = float(config.get("far_val",  self._far))
        self._fov        = float(config.get("fov", self._fov))
        self._aspect     = float(config.get("aspect_ratio", self._width / self._height))
        self._position   = config.get("position",  self._position)
        self._target     = config.get("target",    self._target)
        self._up_vector  = config.get("up_vector", self._up_vector)
        self._build_matrices()
        logger.info(f"Camera reconfigured — pos={self._position}")

    def intrinsics(self) -> np.ndarray:
        """
        Return the 3×3 camera intrinsic matrix K in pixel units.

        Computed analytically from fov, aspect, and resolution — consistent
        with the projection matrix used internally.

            K = [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]

        where:
            fy = (height / 2) / tan(fov_rad / 2)
            fx = fy   (square pixels assumed)
            cx = width  / 2
            cy = height / 2

        Returns:
            (3, 3) float64 numpy array
        """
        fov_rad = np.deg2rad(self._fov)
        fy = (self._height / 2.0) / np.tan(fov_rad / 2.0)
        fx = fy                             # square pixels
        cx = self._width  / 2.0
        cy = self._height / 2.0
        return np.array([
            [fx,   0.0, cx],
            [0.0,  fy,  cy],
            [0.0,  0.0,  1.0],
        ], dtype=np.float64)

    def pixel_to_world(
        self,
        u:     int,
        v:     int,
        depth: Optional[float] = None,
        frame: Optional[CameraFrame] = None,
    ) -> tuple[float, float, float]:
        """
        Back-project a pixel (u, v) to a 3-D world point.

        Provide either a scalar `depth` value in metres or a full
        `frame` (the depth is read from frame.depth[v, u]).

        Args:
            u      : pixel column  (x right)
            v      : pixel row     (y down)
            depth  : metric depth in metres at this pixel
            frame  : CameraFrame to read depth from (alternative to depth)

        Returns:
            (x, y, z) world coordinates in metres

        Raises:
            ValueError : if neither depth nor frame is provided
        """
        if depth is None:
            if frame is None:
                raise ValueError("Provide either 'depth' or 'frame'.")
            depth = float(frame.depth[v, u])

        K     = self.intrinsics()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # Camera-space ray
        xc = (u - cx) * depth / fx
        yc = (v - cy) * depth / fy
        zc = depth

        # Camera→world via inverse view matrix
        # p.computeViewMatrix returns a column-major 4×4 flat tuple
        view = np.array(self._view_matrix, dtype=np.float64).reshape(4, 4).T
        view_inv = np.linalg.inv(view)

        cam_point  = np.array([xc, yc, zc, 1.0])
        world_point = view_inv @ cam_point
        return (
            round(float(world_point[0]), 4),
            round(float(world_point[1]), 4),
            round(float(world_point[2]), 4),
        )

    # ── Properties (read-only) ─────────────────────────────────────────────────

    @property
    def width(self)  -> int:   return self._width

    @property
    def height(self) -> int:   return self._height

    @property
    def near(self)   -> float: return self._near

    @property
    def far(self)    -> float: return self._far

    @property
    def position(self) -> list[float]: return list(self._position)

    @property
    def target(self)   -> list[float]: return list(self._target)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_matrices(self) -> None:
        """Build and cache the PyBullet view and projection matrices."""
        self._view_matrix = p.computeViewMatrix(
            cameraEyePosition=self._position,
            cameraTargetPosition=self._target,
            cameraUpVector=self._up_vector,
            physicsClientId=self._client,
        )
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=self._fov,
            aspect=self._aspect,
            nearVal=self._near,
            farVal=self._far,
            physicsClientId=self._client,
        )
        logger.debug("Camera matrices rebuilt.")

    def _linearise_depth(self, ndc: np.ndarray) -> np.ndarray:
        """
        Convert PyBullet's normalised depth buffer values [0, 1] to
        metric depth in metres using the standard OpenGL formula.

            z_eye = (2 * far * near) / (far + near - ndc * (far - near))

        Args:
            ndc : (H, W) float32 array of raw depth buffer values

        Returns:
            (H, W) float32 array of metric depth values in metres
        """
        near, far = self._near, self._far
        return (2.0 * far * near) / (far + near - ndc * (far - near))

    def __repr__(self) -> str:
        return (
            f"Camera(pos={self._position}, target={self._target}, "
            f"{self._width}×{self._height}, fov={self._fov}°, "
            f"near={self._near}m, far={self._far}m)"
        )
