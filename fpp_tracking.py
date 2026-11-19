"""FoundationPose++-style aids for track_one() on a held tool (18 Nov 2026), used by live_tracking_robust_2_objects.py.

Idea credit: FoundationPose++ (github.com/teal024/FoundationPose-plus-plus): FoundationPose's tracking only refines from
the previous pose, so give it a real tracker. That repo has no licence, so this is an independent implementation.

Why here: on the sweep task the brush's track left the brush when the hand covered it (pose slid onto the fingers and
down to the table, 18 Nov take 164617) and, in the low ZED2 view, onto the orange KUKA link behind the lifted brush
(take 173827), and never came back. SAM2 (already run every frame by --sam2_tool) keeps the brush's 2D mask through
those frames, so before each track_one():

  anchor_xy      move FoundationPose's starting pose sideways so its centre projects onto the centre of SAM2's mask
                 (depth kept), as FoundationPose++ does with its 2D tracker's box centre.
  masked_depth   blank the depth outside SAM2's mask (dilated), so the refiner can only fit the tool's own pixels and
                 cannot slide onto the fingers, the arm or the table. (Not in FoundationPose++.)
  PoseFilter     constant-velocity prediction of the starting pose from the last track_one() results (translation and
                 rotation vector), so fast motion starts closer; the published pose stays the raw track_one() result.

All three need a usable SAM2 mask (enough pixels, not exploded); otherwise the frame runs as plain FoundationPose.
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def mask_centre(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def anchor_xy(pose_last, K, uv, max_shift_px=None):
    """pose_last: FoundationPose's est.pose_last (torch, (4,4) or (1,4,4); translation = mesh centre in the camera).
    Moves its x, y so the centre projects to uv = (u, v); z unchanged. Returns (new pose_last, shift in px) or
    (pose_last, shift) unchanged when the shift exceeds max_shift_px."""
    P = pose_last.clone()
    t = P.reshape(-1, 4, 4)[0, :3, 3]
    z = float(t[2])
    if not np.isfinite(z) or z <= 0.01:
        return pose_last, float('nan')
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u0, v0 = float(t[0]) / z * fx + cx, float(t[1]) / z * fy + cy
    shift = float(np.hypot(uv[0] - u0, uv[1] - v0))
    if max_shift_px is not None and shift > max_shift_px:
        return pose_last, shift
    t[0] = (uv[0] - cx) * z / fx
    t[1] = (uv[1] - cy) * z / fy
    return P, shift


def masked_depth(depth, mask, dilate_px=8):
    """Copy of depth with everything outside the (dilated) mask set to 0 (FoundationPose treats 0 as no depth)."""
    m = mask.astype(np.uint8)
    if dilate_px > 0:
        m = cv2.dilate(m, np.ones((2 * dilate_px + 1,) * 2, np.uint8))
    d = depth.copy()
    d[m == 0] = 0
    return d


class PoseFilter:
    """Constant-velocity predictor on (t, rotation vector) with a per-step blend; predict() gives the next starting pose.

    update(T) takes a track_one() result (4x4, any frame consistent between calls). Velocities are low-passed
    (alpha) and clipped (max_v m/step, max_w rad/step) so one bad frame cannot fling the prediction."""

    def __init__(self, alpha=0.5, max_v=0.05, max_w=np.radians(20)):
        self.alpha, self.max_v, self.max_w = alpha, max_v, max_w
        self.reset()

    def reset(self):
        self.T, self.v, self.w = None, np.zeros(3), np.zeros(3)

    def update(self, T):
        T = np.asarray(T, np.float64)
        if not np.isfinite(T).all():
            return
        if self.T is not None:
            dv = T[:3, 3] - self.T[:3, 3]
            dw = R.from_matrix(T[:3, :3] @ self.T[:3, :3].T).as_rotvec()
            self.v = self.alpha * dv + (1 - self.alpha) * self.v
            self.w = self.alpha * dw + (1 - self.alpha) * self.w
            nv, nw = np.linalg.norm(self.v), np.linalg.norm(self.w)
            if nv > self.max_v:
                self.v *= self.max_v / nv
            if nw > self.max_w:
                self.w *= self.max_w / nw
        self.T = T.copy()

    def predict(self):
        if self.T is None:
            return None
        P = np.eye(4)
        P[:3, :3] = R.from_rotvec(self.w).as_matrix() @ self.T[:3, :3]
        P[:3, 3] = self.T[:3, 3] + self.v
        return P
