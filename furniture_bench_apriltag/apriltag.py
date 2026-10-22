"""dt_apriltags wrapper.

Vendored from furniture-bench@a87960a2:furniture_bench/perception/apriltag.py.
Dropped the pyrealsense2 fast-path (we feed numpy RGB straight from the ZED).
"""

import numpy as np
import cv2
from dt_apriltags import Detector


class AprilTag:
    def __init__(self, tag_size, family="tag36h11", nthreads=4,
                 quad_decimate=1.0):
        self.at_detector = Detector(
            families=family,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            debug=0,
        )
        self.tag_size = tag_size

    def detect(self, frame, intr_param):
        """Detect AprilTag.

        Args:
            frame: HxWx3 uint8 RGB or HxW uint8 grayscale.
            intr_param: [fx, fy, cx, cy] camera intrinsics.
        Returns:
            List of dt_apriltags detections with hamming < 2.
        """
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        detections = self.at_detector.detect(
            frame, True, intr_param, self.tag_size)
        return [d for d in detections if d.hamming < 2]

    def detect_id(self, frame, intr_param):
        return {d.tag_id: d for d in self.detect(frame, intr_param)}
