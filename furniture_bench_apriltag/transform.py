"""Minimal pose-math helpers.

Vendored from furniture-bench@a87960a2:furniture_bench/utils/transform.py.
Kept: to_homogeneous, convert_quat, mat2quat, quat2mat. Dropped numba JIT
(we don't need the speedup for ~1 call per frame on a 3x3 matrix).
"""

import math

import numpy as np

EPS = np.finfo(np.float32).eps * 4.0


def to_homogeneous(pos, rot):
    """Stack position (3,) and rotation (3,3) into a 4x4 homogeneous matrix."""
    if isinstance(pos, list):
        pos = np.array(pos)
    transform = np.zeros((4, 4))
    if pos.ndim == 2:
        transform[:3, 3:] = pos
    else:
        assert pos.ndim == 1
        transform[:3, 3] = pos
    transform[:3, :3] = rot
    transform[3, 3] = 1
    return transform


def convert_quat(q, to="xyzw"):
    """Reorder a quaternion between xyzw and wxyz conventions."""
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise ValueError("convert_quat: choose `to` from {'xyzw', 'wxyz'}")


def mat2quat(rmat):
    """3x3 rotation matrix -> (x, y, z, w) quaternion."""
    M = np.asarray(rmat).astype(np.float32)[:3, :3]
    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def quat2mat(quaternion):
    """(x, y, z, w) quaternion -> 3x3 rotation matrix."""
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]
    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3).astype(np.float32)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ],
        dtype=np.float32,
    )
