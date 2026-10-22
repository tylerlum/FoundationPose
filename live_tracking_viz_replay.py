"""Standalone viser replay for a saved live-tracking run.

Usage:
    python third_party/FoundationPose/live_tracking_viz_replay.py <run_dir>

`<run_dir>` is a directory produced by `live_tracking.py --visualize`
(it must contain `viz_data.npz`). The script unprojects the saved
strided RGB / depth into a per-frame colored point cloud, loads the
mesh referenced at capture time, and launches a viser server with a
frame slider, the tracked-pose axes, and the mesh overlay.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir', type=str,
                   help='Run directory containing viz_data.npz')
    p.add_argument('--point_size', type=float, default=0.003)
    p.add_argument('--port', type=int, default=None,
                   help='viser port (default: pick a free one).')
    p.add_argument('--no_extrinsics', action='store_true',
                   help='Do not orient the viser world-up axis using the '
                        'default camera-to-robot transform (T_RC_DEFAULT).')
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    npz_path = run_dir / 'viz_data.npz'
    if not npz_path.is_file():
        sys.exit(f'no viz_data.npz at {npz_path}')

    data = np.load(npz_path, allow_pickle=False)
    rgb_s = data['rgb']         # (N, Hs, Ws, 3) uint8
    depth_s = data['depth']     # (N, Hs, Ws) float32
    poses_arr = data['poses']   # (N, 4, 4) float64 with NaN for missing
    K = data['K']
    stride = int(data['stride'])
    mesh_path = str(data['mesh_path']) or None

    pose_log = [None if np.isnan(T).any() else T for T in poses_arr]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # Import from live_tracking_replay (numpy/viser/trimesh only) rather than
    # live_tracking_utils, which would pull in the FoundationPose stack
    # (nvdiffrast, estimater, warp) just to view a saved run.
    from live_tracking_replay import (unproject_strided, launch_viser_replay,
                                      T_RC_DEFAULT)

    print(f'[replay] {rgb_s.shape[0]} frames, stride={stride}, '
          f'mesh={mesh_path}')
    pc_log = [unproject_strided(rgb_s[i], depth_s[i], K, stride=stride)
              for i in range(rgb_s.shape[0])]
    if mesh_path and not Path(mesh_path).is_file():
        print(f'[replay] mesh path {mesh_path} not found — skipping mesh overlay')
        mesh_path = None
    launch_viser_replay(pc_log, pose_log, K,
                        mesh_path=mesh_path,
                        point_size=args.point_size,
                        port=args.port,
                        T_RC=None if args.no_extrinsics else T_RC_DEFAULT)


if __name__ == '__main__':
    main()
