"""Replay-only helpers for live tracking.

Split out of `live_tracking_utils.py` so that the standalone replay script
(`live_tracking_viz_replay.py`) doesn't transitively import the FoundationPose
stack (nvdiffrast, the `estimater` module, warp, etc.). Those modules cost
several seconds and a GPU context to load and aren't needed for replay.

Everything here is pure-Python / numpy / trimesh / viser / socket.
"""

import os
import time
from pathlib import Path

import numpy as np


# Default camera->robot transform, used to map a camera-frame object pose into
# the robot frame: robot_pose = T_RC_DEFAULT @ cam_pose. Since 13 Oct 2026 it is
# read from the station repo (github.com/tylerlum/kuka_sharpa_station), which
# keeps every calibration, dated, and the one in use as calibrations/current.txt:
# a camera move is a commit there, not an edit here. KUKA_SHARPA_STATION points
# at its clone; by default the clone sits next to this repo.
STATION_DIR = Path(os.environ.get(
    'KUKA_SHARPA_STATION', Path(__file__).resolve().parents[3] / 'kuka_sharpa_station'))
T_RC_FILE = STATION_DIR / 'calibrations' / 'current.txt'

# Used only when T_RC_FILE is missing: the 13 Oct 2026 viser calibration
# (the station repo's calibrations/2026-10-13_1918_viser.txt).
T_RC_BUILTIN = np.array([
    [-0.40178874053474328, -0.11760770567462986,  0.90814879592798803, -0.35799999999999998],
    [-0.91441628899472904,  0.10467816465046974, -0.39100553994351189, -0.57300000000000006],
    [-0.04907808472844947, -0.98752767526351326, -0.14960090971658008,  0.72049999999999992],
    [ 0.00000000000000000,  0.00000000000000000,  0.00000000000000000,  1.00000000000000000],
])


def _load_T_RC():
    if T_RC_FILE.is_file():
        return np.loadtxt(T_RC_FILE).reshape(4, 4)
    print(f'[live_tracking_replay] WARNING: no {T_RC_FILE}; using the built-in T_RC '
          '(13 Oct 2026). Clone github.com/tylerlum/kuka_sharpa_station next to this '
          'repo, or set KUKA_SHARPA_STATION.', flush=True)
    return T_RC_BUILTIN.copy()


T_RC_DEFAULT = _load_T_RC()

# The calibration from /juno/u/kedia/FoundationPose/live_tracking_with_ros_reset.py,
# in use until the camera moved on 13 Oct 2026.
T_RC_BEFORE_20261013 = np.array([
    [ 0.95627894385143042, -0.17782179758182126,  0.23218524899563026, -0.49520504226664308],
    [-0.28556601086548311, -0.39643057160653450,  0.87252212312014887, -1.44857156913606078],
    [-0.06310812138518884, -0.90067874972183481, -0.42987806970668574,  1.03018932829980048],
    [ 0.0,                  0.0,                  0.0,                  1.0                ],
])


def unproject_strided(rgb_s, depth_s, K, stride, max_depth_m=3.0):
    """Unproject already-stride-sampled arrays. `rgb_s` and `depth_s` are
    sub-sampled from full resolution at `stride` (e.g. `rgb[::stride, ::stride]`);
    `K` is the FULL-resolution intrinsics matrix. Pixels with depth above
    `max_depth_m` are dropped (Fast-FS occasionally hallucinates depths in
    the 10-20 m range from low-texture regions, and they wreck viser's
    auto-camera). Returns (points (N,3) float32, colors (N,3) float32 in [0,1])."""
    H_s, W_s = depth_s.shape
    vs_s, us_s = np.mgrid[0:H_s, 0:W_s]
    valid = (depth_s > 0) & np.isfinite(depth_s) & (depth_s <= max_depth_m)
    vs_s = vs_s[valid]
    us_s = us_s[valid]
    ds = depth_s[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us = us_s * stride
    vs = vs_s * stride
    xs = (us - cx) * ds / fx
    ys = (vs - cy) * ds / fy
    points = np.stack([xs, ys, ds], axis=-1).astype(np.float32)
    colors = rgb_s[vs_s, us_s].astype(np.float32) / 255.0
    return points, colors


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def first_free_port(start, tries=50):
    """The first port from `start` up that nothing is listening on, so a script's viser URL stays the same from
    one run to the next (each script has its own `start`). A port left in TIME_WAIT by the last run counts as
    free. Falls back to any free port."""
    import socket
    for port in range(int(start), int(start) + tries):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    return _free_port()


def launch_viser_replay(pc_log, pose_log, K, mesh_path=None,
                        point_size=0.01, port=None, T_RC=None):
    """Open a viser server showing the recorded run. Blocks until interrupt.

    pc_log: list of (points (N,3) float32, colors (N,3) float32 in [0,1])
    pose_log: list of (4,4) float64 or None per frame
    K: (3,3) intrinsics (unused here but kept for downstream consumers)
    T_RC: optional (4,4) camera-to-robot transform. When set, the viser
        world-up axis is aligned with robot +Z (so gravity points down in the
        viewer) and the initial client camera is placed at the ZED origin
        looking down +Z (the ZED's own POV).
    """
    import viser
    import trimesh
    if not pc_log:
        print('[viser] no point clouds logged — skipping replay.')
        return
    if port is None:
        port = _free_port()
    server = viser.ViserServer(port=port)
    print(f'[viser] replay at http://localhost:{port}  (Ctrl+C to stop)')

    n_frames = len(pc_log)

    # Scene is in camera (ZED) coords: +Z forward, +X right, +Y down. By
    # default viser uses +Z up, which leaves the replay tilted. If a
    # camera-to-robot transform is provided, align viser's world up with
    # robot +Z so gravity points down in the viewer.
    if T_RC is not None:
        T_RC = np.asarray(T_RC, dtype=np.float64).reshape(4, 4)
        R_RC = T_RC[:3, :3]
        # Robot +Z expressed in camera coords = column 2 of R_CR = R_RC^T,
        # i.e. row 2 of R_RC.
        up_cam = R_RC[2, :]
        server.scene.set_up_direction(tuple(float(x) for x in up_cam))

        # Look-at target: prefer the mean tracked-object position; fall back
        # to the point-cloud centroid; final fallback ~1 m down +Z.
        valid_xyz = [p[:3, 3] for p in pose_log if p is not None]
        if valid_xyz:
            look_at_target = np.mean(np.stack(valid_xyz), axis=0)
        else:
            pts0 = pc_log[0][0]
            look_at_target = (pts0.mean(axis=0) if len(pts0)
                              else np.array([0.0, 0.0, 1.0]))

        @server.on_client_connect
        def _set_initial_view(client):
            # Start at the ZED's POV (origin in camera coords).
            client.camera.position = (0.0, 0.0, 0.0)
            client.camera.up_direction = tuple(float(x) for x in up_cam)
            client.camera.look_at = tuple(float(x) for x in look_at_target)

    points0, colors0 = pc_log[0]
    pc_handle = server.scene.add_point_cloud(
        '/scene/points', points=points0, colors=colors0,
        point_size=point_size,
    )

    # Object pose axes — defer wxyz computation to update fn.
    pose_handle = server.scene.add_frame(
        '/object', show_axes=True, axes_length=0.1, axes_radius=0.003,
    )

    # Optional mesh at the tracked pose.
    mesh_handle = None
    if mesh_path is not None:
        try:
            mesh = trimesh.load(mesh_path, process=False)
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                mesh_handle = server.scene.add_mesh_simple(
                    '/object/mesh',
                    vertices=np.asarray(mesh.vertices, dtype=np.float32),
                    faces=np.asarray(mesh.faces, dtype=np.uint32),
                    color=(180, 180, 220), opacity=0.6,
                )
        except Exception as e:
            print(f'[viser] mesh load failed ({e}); continuing without it.')

    # Pose trajectory polyline (positions only).
    positions = np.array([p[:3, 3] for p in pose_log if p is not None],
                         dtype=np.float32)
    if len(positions) >= 2:
        try:
            server.scene.add_spline_catmull_rom(
                '/trajectory', positions=positions, color=(255, 80, 80),
                line_width=2.0,
            )
        except Exception:
            pass

    def _apply_pose(i):
        T = pose_log[i]
        if T is None:
            pose_handle.visible = False
            if mesh_handle is not None:
                mesh_handle.visible = False
            return
        pose_handle.visible = True
        pose_handle.position = tuple(float(x) for x in T[:3, 3])
        quat = trimesh.transformations.quaternion_from_matrix(T)
        pose_handle.wxyz = (float(quat[0]), float(quat[1]),
                            float(quat[2]), float(quat[3]))
        if mesh_handle is not None:
            mesh_handle.visible = True
            mesh_handle.position = pose_handle.position
            mesh_handle.wxyz = pose_handle.wxyz

    slider = server.gui.add_slider(
        'frame', min=0, max=n_frames - 1, step=1, initial_value=0)

    @slider.on_update
    def _on_frame(_):
        i = int(slider.value)
        pts, cols = pc_log[i]
        pc_handle.points = pts
        pc_handle.colors = cols
        _apply_pose(i)

    _apply_pose(0)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print('[viser] shutting down')
