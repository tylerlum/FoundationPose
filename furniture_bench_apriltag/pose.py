"""Minimal pose helpers used by Part / Leg.

Vendored from furniture-bench@a87960a2:furniture_bench/utils/pose.py — kept
only get_mat / rot_mat (used by the part-frame configs) and comp_avg_pose
(used by the tracker to fuse multi-tag pose estimates).
"""

from typing import List, Union

import numpy as np
import numpy.typing as npt

from . import transform as T
from .average_quaternions import average_quaternions


def rot_mat(angles, hom: bool = False):
    """(x, y, z) Euler angles (radians) -> 3x3 or 4x4 rotation matrix."""
    x, y, z = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    if hom:
        M = np.zeros((4, 4), dtype=np.float32)
        M[:3, :3] = R
        M[3, 3] = 1.0
        return M
    return R


def get_mat(pos: List[float],
            angles: Union[List[float], npt.NDArray[np.float32]]):
    """4x4 homogeneous from (x,y,z) position + (x,y,z) Euler angles (or
    3x3 / 4x4 matrix)."""
    transform = np.zeros((4, 4), dtype=np.float32)
    if not isinstance(angles, np.ndarray) or not len(angles.shape) == 2:
        transform[:3, :3] = np.eye(3) if not np.any(angles) else rot_mat(angles)
    else:
        if len(angles[0, :]) == 4:
            transform[:4, :4] = angles
        else:
            transform[:3, :3] = angles
    transform[3, 3] = 1.0
    transform[:3, 3] = pos
    return transform


def comp_avg_pose(poses):
    """Average a list of 4x4 poses. Rotations are averaged via Markley's
    eigenvector method; positions are arithmetic-mean. Returns 4x4 float64
    (the rotation block is float32 from quat2mat; promoted on assignment)."""
    quats = []
    positions = []
    for pose in poses:
        if pose is None:
            continue
        quats.append(T.convert_quat(T.mat2quat(pose[:3, :3]), "wxyz"))
        positions.append(pose[:3, 3])
    quats = np.stack(quats, axis=0)
    positions = np.stack(positions, axis=0)
    avg_quat = average_quaternions(quats).astype(np.float32)
    avg_rot = T.quat2mat(T.convert_quat(avg_quat, "xyzw"))
    avg_pos = positions.mean(axis=0)
    return T.to_homogeneous(avg_pos, avg_rot)
