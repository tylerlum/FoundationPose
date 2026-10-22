"""Standalone viser replay for a saved TWO-object live-tracking take.

Usage:
    python third_party/FoundationPose/live_tracking_viz_replay_2_objects.py <run_dir>

`<run_dir>` is a take written by live_tracking_new_gui_2_objects.py with
--viz_data (it must contain viz_data.npz). Like live_tracking_viz_replay.py,
but it shows both objects: each one's axes, its mesh at the tracked pose (in
the mesh's own colour when the mesh has vertex colours), and its trajectory.

Imports only numpy / trimesh / viser (via live_tracking_replay), not the
FoundationPose stack.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Fallback mesh colours when a mesh has none, and the trajectory colours.
_OBJ_COLORS = ((200, 70, 70), (70, 170, 200))


def _mesh_color(mesh, k):
    vc = getattr(getattr(mesh, 'visual', None), 'vertex_colors', None)
    if vc is not None and len(vc) and getattr(mesh.visual, 'kind', None) == 'vertex':
        return tuple(int(c) for c in np.median(np.asarray(vc)[:, :3], axis=0))
    return _OBJ_COLORS[k]


def launch_viser_replay_2(pc_log, pose_logs, K, mesh_paths=(None, None),
                          point_size=0.003, port=None, T_RC=None):
    """Open a viser server showing a recorded two-object run. Blocks until
    interrupted.

    pc_log: list of (points (N,3) float32, colors (N,3) float32 in [0,1])
    pose_logs: two lists of (4,4) float64 or None per frame (camera frame)
    T_RC: optional camera-to-robot transform; when set, the viewer's up axis
        is the robot's +Z and the camera starts at the ZED's position.
    """
    import trimesh
    import viser
    from live_tracking_replay import first_free_port

    if not pc_log:
        print('[viser] no point clouds logged; skipping replay.')
        return
    port = first_free_port(port or 8120)       # the same URL on every rerun
    server = viser.ViserServer(port=port)
    print(f'[viser] two-object replay at http://localhost:{port}  (Ctrl+C to stop)')
    n_frames = len(pc_log)

    if T_RC is not None:
        R_RC = np.asarray(T_RC, dtype=np.float64).reshape(4, 4)[:3, :3]
        up_cam = R_RC[2, :]            # robot +Z in camera coordinates
        server.scene.set_up_direction(tuple(float(x) for x in up_cam))
        valid = [p[:3, 3] for log in pose_logs for p in log if p is not None]
        target = (np.mean(np.stack(valid), axis=0) if valid
                  else np.array([0.0, 0.0, 1.0]))

        @server.on_client_connect
        def _set_initial_view(client):
            client.camera.position = (0.0, 0.0, 0.0)
            client.camera.up_direction = tuple(float(x) for x in up_cam)
            client.camera.look_at = tuple(float(x) for x in target)

    points0, colors0 = pc_log[0]
    pc_handle = server.scene.add_point_cloud(
        '/scene/points', points=points0, colors=colors0, point_size=point_size)

    frames, meshes = [], []
    for k in range(2):
        frames.append(server.scene.add_frame(
            f'/object_{k + 1}', show_axes=True, axes_length=0.08,
            axes_radius=0.003))
        handle = None
        path = mesh_paths[k] if k < len(mesh_paths) else None
        if path:
            try:
                mesh = trimesh.load(path, process=False)
                handle = server.scene.add_mesh_simple(
                    f'/object_{k + 1}/mesh',
                    vertices=np.asarray(mesh.vertices, dtype=np.float32),
                    faces=np.asarray(mesh.faces, dtype=np.uint32),
                    color=_mesh_color(mesh, k), opacity=0.7)
            except Exception as e:
                print(f'[viser] object {k + 1} mesh load failed ({e}); '
                      'continuing without it.')
        meshes.append(handle)
        positions = np.array([p[:3, 3] for p in pose_logs[k] if p is not None],
                             dtype=np.float32)
        if len(positions) >= 2:
            try:
                server.scene.add_spline_catmull_rom(
                    f'/trajectory_{k + 1}', positions=positions,
                    color=_OBJ_COLORS[k], line_width=2.0)
            except Exception:
                pass

    def _apply(i):
        pts, cols = pc_log[i]
        pc_handle.points = pts
        pc_handle.colors = cols
        for k in range(2):
            T = pose_logs[k][i] if i < len(pose_logs[k]) else None
            frames[k].visible = T is not None
            if T is None:
                continue
            frames[k].position = tuple(float(x) for x in T[:3, 3])
            q = trimesh.transformations.quaternion_from_matrix(T)
            frames[k].wxyz = tuple(float(x) for x in q)

    slider = server.gui.add_slider('frame', min=0, max=n_frames - 1, step=1,
                                   initial_value=0)
    play = server.gui.add_checkbox('play', initial_value=False)

    @slider.on_update
    def _on_frame(_):
        _apply(int(slider.value))

    _apply(0)
    try:
        while True:
            if play.value:
                slider.value = (int(slider.value) + 1) % n_frames
            time.sleep(1.0 / 15)
    except KeyboardInterrupt:
        print('[viser] replay stopped.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir', type=str, help='Take directory with viz_data.npz')
    p.add_argument('--point_size', type=float, default=0.003)
    p.add_argument('--port', type=int, default=8120,
                   help='viser port: the first free one from this up.')
    p.add_argument('--no_extrinsics', action='store_true',
                   help='Do not orient the viewer with T_RC_DEFAULT.')
    args = p.parse_args()

    npz_path = Path(args.run_dir) / 'viz_data.npz'
    if not npz_path.is_file():
        sys.exit(f'no viz_data.npz at {npz_path}')
    data = np.load(npz_path, allow_pickle=False)
    rgb_s, depth_s, K, stride = data['rgb'], data['depth'], data['K'], int(data['stride'])
    poses_1 = data['poses_1'] if 'poses_1' in data else data['poses']
    poses_2 = data['poses_2'] if 'poses_2' in data else np.full_like(poses_1, np.nan)
    pose_logs = [[None if np.isnan(T).any() else T for T in arr] for arr in (poses_1, poses_2)]
    mesh_paths = []
    for key in ('mesh_path_1', 'mesh_path_2'):
        m = str(data[key]) if key in data else ''
        if m and not Path(m).is_file():
            print(f'[replay] mesh {m} not found; skipping its overlay')
            m = ''
        mesh_paths.append(m or None)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from live_tracking_replay import T_RC_DEFAULT, unproject_strided

    print(f'[replay] {rgb_s.shape[0]} frames, stride={stride}, meshes={mesh_paths}')
    pc_log = [unproject_strided(rgb_s[i], depth_s[i], K, stride=stride)
              for i in range(rgb_s.shape[0])]
    launch_viser_replay_2(pc_log, pose_logs, K, mesh_paths=mesh_paths,
                          point_size=args.point_size, port=args.port,
                          T_RC=None if args.no_extrinsics else T_RC_DEFAULT)


if __name__ == '__main__':
    main()
