"""Quaternion averaging (Markley 2007 / NASA tech report).

Verbatim port from furniture-bench@a87960a2:furniture_bench/utils/
averageQuaternions.py, which itself ports
https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py.

Q is an N x 4 array of quaternions in (w, x, y, z) order. Returns the
average as a (4,) array, also (w, x, y, z). Signs may be flipped (q and -q
describe the same orientation).
"""

import numpy as np
import numpy.matlib as npm


def average_quaternions(Q):
    M = Q.shape[0]
    A = npm.zeros(shape=(4, 4))
    for i in range(M):
        q = Q[i, :]
        A = np.outer(q, q) + A
    A = (1.0 / M) * A
    eigen_values, eigen_vectors = np.linalg.eig(A)
    eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]
    return np.real(eigen_vectors[:, 0].A1)
