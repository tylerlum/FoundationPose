"""Minimal vendored slice of FurnitureBench's AprilTag tracking stack.

Vendored from https://github.com/clvrai/furniture-bench (commit a87960a2),
trimmed to only what's needed to recover a 6-DoF pose for a FurnitureBench
part (anchor + auxiliary tags glued to its faces) from a single RGB frame.
Original code is MIT-licensed; see THIRD_PARTY_LICENSES.

Heavy deps removed: pyrealsense2, torch, numba, gym, polymetis, furniture_bench
runtime config. The vendored API is purely numpy + cv2 + dt_apriltags.
"""

from .apriltag import AprilTag  # noqa: F401
from .pose import comp_avg_pose, get_mat  # noqa: F401
from . import config  # noqa: F401
from .parts.square_table_leg import SquareTableLeg  # noqa: F401
