"""Checks and corrections for FoundationPose on a held tool (29 Oct 2026), used by
live_tracking_robust_2_objects.py (a copy of live_tracking_new_gui_2_objects.py; the original is unchanged).

Why: on the robot the screwdriver's track dies in the hand (29 Oct, 4 of 4 lifts): the fingers hide most of the
driver, often leaving only its flat end in view, FoundationPose drifts off, never recovers, and the policy lets go.
Offline on the 03:22 clip, SAM2 (Tyler's real-time fork, tiny, ~18 ms / frame at 960 x 540) kept the driver's mask
through the grasp, the end-on hold, the fall and the table; the IoU of its mask with FoundationPose's silhouette was
0.82 while tracked on the table, ~0 once lost, and 0.0-0.5 end-on in the hand while FoundationPose was roughly right
(a small disk 10-20 px off) -- so "lost" needs BOTH a low IoU and the two centres far apart, for several frames.

  SamTool     SAM2 camera predictor on object 1 (prompt: the registration mask's box + its innermost point).
  ToolCheck   FoundationPose's silhouette (mesh projected at its pose) vs SAM2's mask: iou, centre distance (px),
              badness 0..1, and lost after lost_frames bad frames in a row.
  pose_last_from_pose   FoundationPose's internal state for a pose in the mesh frame (to restart track_one from it).

SAM2 code and weights: /home/tylerlum/.deps_sam2rt_20261029 (a copy of ~/github_repos/segment-anything-2-real-time
from bohg-ws-19 without its compiled _C; hydra-core in pylibs/), so the drl venv itself is unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import cv2

SAM2_ROOT = Path('/home/tylerlum/.deps_sam2rt_20261029')
SAM2_CKPT = SAM2_ROOT / 'segment-anything-2-real-time/checkpoints/sam2_hiera_tiny.pt'
SAM2_CFG = 'sam2_hiera_t.yaml'


def mask_prompt(mask):
    """The box (x0, y0, x1, y1) of a mask and its innermost pixel (x, y)."""
    m = np.asarray(mask).astype(np.uint8)
    ys, xs = np.nonzero(m)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    py, px = np.unravel_index(int(dist.argmax()), dist.shape)
    return box, np.array([[px, py]], np.float32)


class SamTool:
    """SAM2's camera predictor for one object. reset(rgb, mask) starts it on a frame; track(rgb) -> bool mask."""

    def __init__(self, ckpt=SAM2_CKPT, cfg=SAM2_CFG, device='cuda'):
        for p in (SAM2_ROOT / 'pylibs', SAM2_ROOT / 'segment-anything-2-real-time'):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        import torch
        from sam2.build_sam import build_sam2_camera_predictor
        self.torch = torch
        # FoundationPose's stack makes CUDA the default device; SAM2's frame preparation mixes torch.from_numpy (CPU)
        # with torch.tensor (default device). Run SAM2 with a CPU default, as it runs standalone.
        with torch.device('cpu'):
            self.pred = build_sam2_camera_predictor(cfg, str(ckpt), device=device)
        self.pred.fill_hole_area = 0            # hole filling needs the compiled _C, which is not copied
        self.started = False
        self.last = None

    def _ctx(self):
        t = self.torch
        return t.inference_mode(), t.autocast('cuda', dtype=t.bfloat16), t.device('cpu')

    def reset(self, rgb, mask):
        box, pt = mask_prompt(mask)
        a, b, c = self._ctx()
        with a, b, c:
            if self.started:
                self.pred.reset_state()
            self.pred.load_first_frame(rgb)
            self.pred.add_new_prompt(frame_idx=0, obj_id=1, bbox=box, points=pt,
                                     labels=np.array([1], np.int32))
        self.started = True
        self.last = np.asarray(mask).astype(bool)
        return box, pt

    def track(self, rgb):
        a, b, c = self._ctx()
        with a, b, c:
            _, masks = self.pred.track(rgb)
        self.last = (masks[0, 0] > 0).cpu().numpy()
        return self.last


class ToolCheck:
    """FoundationPose's silhouette vs a 2D mask. A frame is bad when the IoU is below lost_iou AND the centres are
    more than lost_px apart; lost after lost_frames bad frames in a row. badness: 0 when they agree (iou >= 2 x
    lost_iou or centres within lost_px), 1 when clearly apart (iou 0 and centres >= 2 x lost_px)."""

    def __init__(self, mesh, K, W, H, lost_iou=0.1, lost_px=30.0, lost_frames=4, min_mask_px=60):
        self.v = np.asarray(mesh.vertices, np.float64)
        self.f = np.asarray(mesh.faces, np.int64)
        self.K, self.W, self.H = np.asarray(K, np.float64), int(W), int(H)
        self.lost_iou, self.lost_px, self.lost_frames, self.min_mask_px = lost_iou, lost_px, lost_frames, min_mask_px
        self.n_bad = 0

    def silhouette(self, T):
        m = np.zeros((self.H, self.W), np.uint8)
        if T is None or not np.isfinite(T).all():
            return m.astype(bool)
        p = self.v @ T[:3, :3].T + T[:3, 3]
        if (p[:, 2] <= 0.01).any():
            return m.astype(bool)
        uv = (p[:, :2] / p[:, 2:3]) * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]]
        tri = np.round(uv[self.f]).astype(np.int32)
        lo, hi = tri.min(axis=1), tri.max(axis=1)
        keep = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < self.W) & (lo[:, 1] < self.H)
        for t in tri[keep]:
            cv2.fillConvexPoly(m, t, 1)
        return m.astype(bool)

    def check(self, mask, T):
        sil = self.silhouette(T)
        out = dict(iou=float('nan'), dpx=float('nan'), sam_px=int(mask.sum()) if mask is not None else 0,
                   fp_px=int(sil.sum()), badness=0.0, bad=False, lost=False, usable=False)
        if mask is None or out['sam_px'] < self.min_mask_px:
            self.n_bad = 0                       # no usable 2D mask (hidden or SAM2 lost it): no verdict
            return out
        out['usable'] = True
        inter, union = int((mask & sil).sum()), int((mask | sil).sum())
        out['iou'] = inter / union if union else 0.0
        ys, xs = np.nonzero(mask)
        c_m = np.array([xs.mean(), ys.mean()])
        if out['fp_px'] > 0:
            ys2, xs2 = np.nonzero(sil)
            out['dpx'] = float(np.linalg.norm(c_m - [xs2.mean(), ys2.mean()]))
        else:
            out['dpx'] = float('inf')            # FoundationPose's pose is off screen or behind the camera
        a = np.clip(1.0 - out['iou'] / (2 * self.lost_iou), 0.0, 1.0)
        b = np.clip(out['dpx'] / self.lost_px - 1.0, 0.0, 1.0)
        out['badness'] = float(a * b)
        out['bad'] = out['iou'] < self.lost_iou and out['dpx'] > self.lost_px
        self.n_bad = self.n_bad + 1 if out['bad'] else 0
        out['lost'] = self.n_bad >= self.lost_frames
        return out


def pose_last_from_pose(pose, model_center):
    """FoundationPose keeps pose_last for the CENTRED mesh: pose (mesh frame) = pose_last @ T(-model_center)."""
    T = np.asarray(pose, np.float64).copy()
    T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.asarray(model_center, np.float64).reshape(3)
    return T


# ---- end-face tag (29 Oct) -------------------------------------------------------------------------------------------
def _rot_deg(Ra, Rb):
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


class EndFaceTag:
    """A small ArUco tag stuck on the tool's flat end (the driver: the handle end at x = end_x, facing -x), its TOP
    edge toward the tool's +y (the red fin). detect(gray, depth, pose_fp) -> the tool pose from the tag, or None.

    Tag frame (OpenCV ArUco): x right, y toward the top edge, z out of the face. In the tool frame: x_tag = +z,
    y_tag = +y, z_tag = -x, centre at (end_x, 0, 0); roll_deg turns the tag about the shaft (the sticker's error,
    found by calibrate()). The face's range comes from the depth image (a 15 mm tag's own range is poor); of the two
    planar solutions (IPPE) the one nearer FoundationPose's pose is kept (or the one facing the camera)."""

    def __init__(self, K, tag_id=1, dict_name='DICT_4X4_50', size_m=0.015, end_x=-0.05, roll_deg=0.0,
                 roi_px=0, min_side_px=12.0):
        self.K = np.asarray(K, np.float64)
        self.tag_id, self.size, self.end_x, self.roi_px, self.min_side_px = tag_id, size_m, end_x, roi_px, min_side_px
        ar = cv2.aruco
        params = ar.DetectorParameters()
        params.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX
        self.det = ar.ArucoDetector(ar.getPredefinedDictionary(getattr(ar, dict_name)), params)
        h = size_m / 2
        self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
        self.set_roll(roll_deg)
        self.cal = []

    def set_roll(self, roll_deg):
        self.roll_deg = float(roll_deg)
        c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        R = np.zeros((3, 3))
        R[:, 0], R[:, 1], R[:, 2] = [0, 0, 1], [0, 1, 0], [-1, 0, 0]      # x_tag, y_tag, z_tag in the tool frame
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.T_D_tag = np.eye(4)
        self.T_D_tag[:3, :3] = R @ Rz
        self.T_D_tag[:3, 3] = [self.end_x, 0.0, 0.0]
        self.T_tag_D = np.linalg.inv(self.T_D_tag)

    def _roi(self, gray, pose_fp):
        H, W = gray.shape[:2]
        if not self.roi_px or pose_fp is None or not np.isfinite(pose_fp).all():
            return gray, 0, 0
        p = pose_fp[:3, :3] @ [self.end_x, 0, 0] + pose_fp[:3, 3]
        if p[2] <= 0.05:
            return gray, 0, 0
        u, v = self.K[0, 0] * p[0] / p[2] + self.K[0, 2], self.K[1, 1] * p[1] / p[2] + self.K[1, 2]
        x0, y0 = int(max(0, u - self.roi_px)), int(max(0, v - self.roi_px))
        x1, y1 = int(min(W, u + self.roi_px)), int(min(H, v + self.roi_px))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return gray, 0, 0
        return gray[y0:y1, x0:x1], x0, y0

    def detect(self, gray, depth=None, pose_fp=None):
        img, x0, y0 = self._roi(gray, pose_fp)
        corners, ids, _ = self.det.detectMarkers(img)
        if ids is None or self.tag_id not in ids.ravel():
            return None
        c = corners[list(ids.ravel()).index(self.tag_id)].reshape(4, 2).astype(np.float64) + [x0, y0]
        side = float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))
        if side < self.min_side_px:
            return None
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(self.obj, c, self.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not n:
            return None
        ctr = c.mean(axis=0)
        z_depth = None
        if depth is not None:
            m = np.zeros(depth.shape[:2], np.uint8)
            cv2.fillConvexPoly(m, np.round(c).astype(np.int32), 1)
            d = depth[m.astype(bool)]
            d = d[(d > 0.1) & (d < 3.0)]
            if len(d) >= 5:
                z_depth = float(np.median(d))
        cands = []
        for rv, tv, e in zip(rvecs, tvecs, np.ravel(errs)):
            T = np.eye(4)
            T[:3, :3] = cv2.Rodrigues(rv)[0]
            T[:3, 3] = tv.ravel()
            if z_depth is not None:          # the face's range from depth, along the ray through the tag centre
                ray = np.linalg.inv(self.K) @ [ctr[0], ctr[1], 1.0]
                T[:3, 3] = ray * (z_depth / ray[2])
            cands.append((T @ self.T_tag_D, float(e), T))
        if pose_fp is not None and np.isfinite(pose_fp).all():
            cands.sort(key=lambda x: _rot_deg(x[0][:3, :3], pose_fp[:3, :3]))
        else:                                # the solution whose face normal points at the camera more
            cands.sort(key=lambda x: x[2][:3, 2] @ (x[2][:3, 3] / np.linalg.norm(x[2][:3, 3])))
        T_C_D, err, T_C_tag = cands[0]
        return dict(T=T_C_D, side_px=side, reproj_px=err, depth=z_depth is not None, n_solutions=len(cands))

    def compare(self, T_tag, pose_fp):
        """Distance of the end-face centres (m) and the rotation angle (deg) between the tag's and FoundationPose's
        tool poses."""
        e = np.array([self.end_x, 0.0, 0.0])
        dpos = float(np.linalg.norm((T_tag[:3, :3] @ e + T_tag[:3, 3]) - (pose_fp[:3, :3] @ e + pose_fp[:3, 3])))
        return dpos, _rot_deg(T_tag[:3, :3], pose_fp[:3, :3])

    def calibrate(self, T_tag, pose_fp, n=15, max_spread_deg=10.0):
        """Learn the sticker's roll about the shaft from frames where FoundationPose is trusted. Returns the corrected
        roll for set_roll() (deg) once n samples agree within max_spread_deg, else None."""
        R_rel = pose_fp[:3, :3].T @ T_tag[:3, :3]          # the tag-derived tool pose in FoundationPose's tool frame
        ang = float(np.degrees(np.arctan2(R_rel[2, 1], R_rel[1, 1])))  # its turn about the tool's x (the shaft)
        self.cal.append(ang)
        if len(self.cal) < n:
            return None
        a = np.radians(self.cal[-n:])
        mean = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))
        spread = float(np.degrees(np.sqrt(-2 * np.log(max(np.hypot(np.sin(a).mean(), np.cos(a).mean()), 1e-9)))))
        if spread > max_spread_deg:
            return None
        self.cal = []
        return self.roll_deg - mean          # the tag-derived pose is turned by -(roll error) about the shaft

    def fuse(self, T_tag, pose_fp):
        """FoundationPose's shaft direction + the tag's roll about it + the tag's end-face position (a 15 mm tag's
        own tilt is ambiguous: its two planar solutions differ by up to ~120 deg, but both give the roll about the
        true shaft within ~2 deg). Returns (T, roll_diff_deg, dpos_m) with roll_diff = the tag's roll - FP's."""
        e = np.array([self.end_x, 0.0, 0.0])
        x = pose_fp[:3, 0] / np.linalg.norm(pose_fp[:3, 0])
        y = T_tag[:3, 1] - (T_tag[:3, 1] @ x) * x
        if np.linalg.norm(y) < 0.3:                   # the tag's y nearly along FP's shaft: no roll information
            return None, float('nan'), float('nan')
        y /= np.linalg.norm(y)
        R = np.stack([x, y, np.cross(x, y)], axis=1)
        p_end = T_tag[:3, :3] @ e + T_tag[:3, 3]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p_end - R @ e
        y_fp = pose_fp[:3, 1]
        roll_diff = float(np.degrees(np.arctan2(np.cross(y_fp, y) @ x, y_fp @ y)))
        dpos = float(np.linalg.norm(p_end - (pose_fp[:3, :3] @ e + pose_fp[:3, 3])))
        return T, roll_diff, dpos
