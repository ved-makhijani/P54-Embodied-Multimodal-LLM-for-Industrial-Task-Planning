"""
camera_interface.py
-------------------
Unified camera interface.
CAMERA_BACKEND=realsense | webcam | picamera
"""
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CameraFrame:
    def __init__(self, rgb: np.ndarray, depth: np.ndarray = None,
                 intrinsics: dict = None):
        self.rgb        = rgb
        self.depth      = depth
        self.intrinsics = intrinsics

    @property
    def has_depth(self) -> bool:
        return self.depth is not None


def get_camera():
    backend = os.getenv("CAMERA_BACKEND", "webcam").lower()
    if backend == "realsense":
        from camera_backend.realsense_camera import RealSenseCamera
        return RealSenseCamera()
    elif backend == "picamera":
        from camera_backend.picamera_camera import PiCamera
        return PiCamera()
    else:
        from camera_backend.webcam_camera import WebcamCamera
        return WebcamCamera()