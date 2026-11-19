"""Per-frame validity of a tracked payload (the sweep cube) from the frame itself.

FoundationPose's track_one() always returns a pose, including after the object is hidden or lost (a hidden cube's
tracker follows the arm). PayloadCheck decides, per captured frame, whether the frame actually supports the pose:

  colour   pixels in the payload's HSV range near the pose's silhouette   >= min_pixels
  iou      silhouette (mesh rendered at the pose) vs those colour pixels   >= min_silhouette_iou
  depth    observed depth pixels inside the (eroded) silhouette            >= min_depth_pixels
  surface  fraction of those within max_surface_residual_m of the render   >= min_surface_fraction
  jump     distance from the last VALID position                           <= max_speed_m_s * dt + jump_slack_m

Defaults are the sweep deployment's own payload contract (policy_futures_dev
robot_deploy/local/tasks/sweep_payload_29mm.json, "cube"), so a frame this marks valid is one that deployment's
RGB-D payload node would also accept. HSV is OpenCV's (H 0-179) on the RGB image.
"""
import numpy as np
import cv2
import torch

from Utils import nvdiffrast_render

SWEEP_CUBE_29MM = dict(hue_min=4, hue_max=38, saturation_min=130, value_min=45, min_pixels=60,
                       min_depth_pixels=30, max_surface_residual_m=0.005, min_surface_fraction=0.65,
                       min_silhouette_iou=0.45, max_speed_m_s=2.0, jump_slack_m=0.02)


class PayloadCheck:
    def __init__(self, mesh_tensors, K, W, H, glctx, hue_min, hue_max, saturation_min, value_min, min_pixels,
                 min_depth_pixels, max_surface_residual_m, min_surface_fraction, min_silhouette_iou,
                 max_speed_m_s, jump_slack_m, roi_pad_px=12, erode_px=2):
        self.mesh_tensors, self.K, self.W, self.H, self.glctx = mesh_tensors, np.asarray(K, np.float64), W, H, glctx
        self.hsv_lo = np.array([hue_min, saturation_min, value_min], np.uint8)
        self.hsv_hi = np.array([hue_max, 255, 255], np.uint8)
        self.min_pixels, self.min_depth_pixels = min_pixels, min_depth_pixels
        self.max_residual, self.min_surface = max_surface_residual_m, min_surface_fraction
        self.min_iou, self.max_speed, self.jump_slack = min_silhouette_iou, max_speed_m_s, jump_slack_m
        self.pad, self.erode_px = roi_pad_px, erode_px
        self.config = dict(hue_min=hue_min, hue_max=hue_max, saturation_min=saturation_min, value_min=value_min,
                           min_pixels=min_pixels, min_depth_pixels=min_depth_pixels,
                           max_surface_residual_m=max_surface_residual_m, min_surface_fraction=min_surface_fraction,
                           min_silhouette_iou=min_silhouette_iou, max_speed_m_s=max_speed_m_s,
                           jump_slack_m=jump_slack_m, roi_pad_px=roi_pad_px, erode_px=erode_px)
        self.reset()

    def reset(self):
        """Forget the last valid position (call after a (re-)registration)."""
        self.last_pos, self.last_t = None, None

    def render_depth(self, pose_cam):
        ob = torch.as_tensor(np.asarray(pose_cam, np.float32)[None], device='cuda')
        _, depth, _ = nvdiffrast_render(K=self.K, H=self.H, W=self.W, ob_in_cams=ob, glctx=self.glctx,
                                        mesh_tensors=self.mesh_tensors)
        return depth[0].data.cpu().numpy()

    def check(self, rgb, depth, pose_cam, t_s, use_depth=True):
        """Verdict for one frame. rgb uint8 HxWx3 (RGB), depth metres (None / use_depth=False skips the depth
        terms), pose_cam 4x4 in the camera frame, t_s the frame's capture time in seconds."""
        out = dict(valid=False, reason='', color_px=0, iou=float('nan'), depth_px=0,
                   surface_frac=float('nan'), jump_m=float('nan'))
        pose_cam = np.asarray(pose_cam, np.float64)
        if pose_cam.shape != (4, 4) or not np.isfinite(pose_cam).all() or pose_cam[2, 3] <= 0:
            out['reason'] = 'no_pose'
            return out
        rendered = self.render_depth(pose_cam)
        sil = rendered > 0
        vs, us = np.nonzero(sil)
        if len(vs) == 0:
            out['reason'] = 'off_image'
            return out
        v0, v1 = max(vs.min() - self.pad, 0), min(vs.max() + self.pad + 1, self.H)
        u0, u1 = max(us.min() - self.pad, 0), min(us.max() + self.pad + 1, self.W)
        sil_roi = sil[v0:v1, u0:u1]
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb[v0:v1, u0:u1]), cv2.COLOR_RGB2HSV)
        color = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi) > 0
        out['color_px'] = int(color.sum())
        union = int((sil_roi | color).sum())
        out['iou'] = float((sil_roi & color).sum() / union) if union else 0.0
        if use_depth and depth is not None:
            core = sil_roi
            if self.erode_px > 0:
                k = np.ones((2 * self.erode_px + 1,) * 2, np.uint8)
                eroded = cv2.erode(sil_roi.astype(np.uint8), k) > 0
                core = eroded if eroded.sum() >= self.min_depth_pixels else sil_roi
            d_obs = np.asarray(depth[v0:v1, u0:u1], np.float64)
            have = core & np.isfinite(d_obs) & (d_obs > 0.05)
            out['depth_px'] = int(have.sum())
            if out['depth_px']:
                resid = np.abs(d_obs[have] - rendered[v0:v1, u0:u1][have])
                out['surface_frac'] = float((resid <= self.max_residual).mean())
        pos = pose_cam[:3, 3]
        if self.last_pos is not None:
            out['jump_m'] = float(np.linalg.norm(pos - self.last_pos))
        if out['color_px'] < self.min_pixels:
            out['reason'] = 'color_px'
        elif out['iou'] < self.min_iou:
            out['reason'] = 'iou'
        elif use_depth and depth is not None and out['depth_px'] < self.min_depth_pixels:
            out['reason'] = 'depth_px'
        elif use_depth and depth is not None and out['surface_frac'] < self.min_surface:
            out['reason'] = 'surface'
        elif (self.last_pos is not None and t_s is not None and self.last_t is not None
              and out['jump_m'] > self.max_speed * max(t_s - self.last_t, 0.0) + self.jump_slack):
            out['reason'] = 'jump'
        else:
            out['valid'] = True
            self.last_pos, self.last_t = pos.copy(), t_s
        return out
