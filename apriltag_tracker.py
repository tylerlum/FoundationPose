"""AprilTag-based pose tracker for one FurnitureBench part.

Per-frame pipeline:
  1. dt_apriltags.Detector.detect(rgb_grayscale) -> list of tag detections
  2. For each detected tag in `part.tag_ids`:
        T_cam_object = T_cam_tag @ inv(rel_pose_from_center[tag_id])
  3. Average across tags via Markley's quaternion-eigenvector method
     (`furniture_bench_apriltag.comp_avg_pose`).

Returns the 4x4 camera-frame pose of the part, plus the number of tags
used for the estimate (0 -> no detection that frame).
"""

import numpy as np

from furniture_bench_apriltag import AprilTag, SquareTableLeg, comp_avg_pose
from furniture_bench_apriltag import config as fba_config


# Map config["class"] -> Part subclass. Add new entries here when
# vendoring new FB parts.
_PART_CLASSES = {
    "SquareTableLeg": SquareTableLeg,
}


def build_part(part_name):
    """Look up a part name in the vendored FB config and instantiate its
    Part subclass. Raises ValueError if the name is unknown."""
    if part_name not in fba_config.PARTS:
        known = sorted(fba_config.PARTS)
        raise ValueError(
            f"Unknown FB part {part_name!r}. Available: {known}. "
            f"Add new parts in furniture_bench_apriltag/config.py.")
    part_cfg = fba_config.PARTS[part_name]
    cls_name = part_cfg["class"]
    if cls_name not in _PART_CLASSES:
        raise ValueError(
            f"Part {part_name!r} declares class={cls_name!r} but no such "
            f"Part subclass is registered in apriltag_tracker._PART_CLASSES.")
    return _PART_CLASSES[cls_name](part_cfg)


class AprilTagTracker:
    """Wraps a vendored FB Part + dt_apriltags Detector for one object."""

    def __init__(self, part_name, K, family="tag36h11", nthreads=4,
                 quad_decimate=1.0, tag_size_override=None):
        self.part = build_part(part_name)
        tag_size = (float(tag_size_override) if tag_size_override is not None
                    else self.part.tag_size)
        self.detector = AprilTag(
            tag_size, family=family, nthreads=nthreads,
            quad_decimate=quad_decimate)
        K = np.asarray(K, dtype=np.float64)
        self.intr_param = [float(K[0, 0]), float(K[1, 1]),
                           float(K[0, 2]), float(K[1, 2])]
        # Cache inverses so we don't recompute them every frame.
        self._inv_rel = {
            tid: np.linalg.inv(self.part.rel_pose_from_center[tid])
            for tid in self.part.tag_ids
        }

    def track(self, rgb):
        """Run one detection on an HxWx3 uint8 RGB frame.

        Returns:
            (pose_cam_4x4 | None, n_tags_used)
            pose_cam is the 4x4 transform of the part's center frame in
            camera coordinates, or None if no part tag was detected.
        """
        detections = self.detector.detect_id(rgb, self.intr_param)
        poses = []
        for tid in self.part.tag_ids:
            det = detections.get(tid)
            if det is None:
                continue
            T_cam_tag = np.eye(4, dtype=np.float64)
            T_cam_tag[:3, :3] = np.asarray(det.pose_R, dtype=np.float64)
            T_cam_tag[:3, 3] = np.asarray(det.pose_t, dtype=np.float64).reshape(3)
            poses.append(T_cam_tag @ self._inv_rel[tid])
        if not poses:
            return None, 0
        if len(poses) == 1:
            return poses[0], 1
        return comp_avg_pose(poses), len(poses)
