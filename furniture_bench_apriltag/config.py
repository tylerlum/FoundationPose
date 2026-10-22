"""Vendored FurnitureBench config for AprilTag tracking.

Vendored from furniture-bench@a87960a2:furniture_bench/config.py, trimmed
to just what AprilTag tracking needs: base-tag layout and the specific
furniture parts we support. Add new parts here by copying their
{name, ids, half_width / tag_offset / ...} entry from FB's config.py and
ensuring the matching Part subclass is registered in PART_CLASSES.

Units: meters.
"""

import numpy as np

# Base tags on the FB tabletop. Each tag sits at a fixed offset from the
# table-center coordinate frame (rel_pose_from_coordinate). For now we
# don't auto-calibrate the camera against these (live_tracking_apriltag.py
# uses T_RC_DEFAULT) — kept here for the future `--calibrate_via_base_tags`
# extension.
BASE_TAGS = [0, 1, 2, 3]
BASE_TAG_SIZE = 0.048  # meters
BASE_REL_POSE_FROM_COORDINATE = {
    0: np.array([[1, 0, 0, -0.03],
                 [0, 1, 0, -0.03],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1.0]], dtype=np.float32),
    1: np.array([[1, 0, 0, 0.03],
                 [0, 1, 0, -0.03],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1.0]], dtype=np.float32),
    2: np.array([[1, 0, 0, -0.03],
                 [0, 1, 0, 0.03],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1.0]], dtype=np.float32),
    3: np.array([[1, 0, 0, 0.03],
                 [0, 1, 0, 0.03],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1.0]], dtype=np.float32),
}

# All square_table furniture parts share the same physical AprilTag size.
SQUARE_TABLE_TAG_SIZE = 0.0195  # meters

# Per-part configs. Mirrors FB's config['furniture'][<furniture>][<part>]
# dicts, with only the fields the AprilTag tracker actually consumes
# (tag_ids, half_width / tag_offset, the matching Part subclass key).
PARTS = {
    "square_table_leg2": {
        "name": "square_table_leg2",
        "ids": [12, 13, 14, 15],   # FB config.py:265
        "tag_size": SQUARE_TABLE_TAG_SIZE,
        "half_width": 0.015,        # FB config.py:268
        # `tag_offset` for square_table_leg is hardcoded to 0.015 m in
        # FB's SquareTableLeg subclass (see square_table_leg.py:7). The
        # vendored SquareTableLeg sets the same value.
        "class": "SquareTableLeg",
    },
}
