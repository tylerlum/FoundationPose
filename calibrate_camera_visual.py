"""Live calibration convergence viewer; the original calibrate_camera.py is unchanged.

record: passive joint-state subscriber, continuous RGB/depth preview, capture at stops.
replay: explicitly supplied saved poses, offline dry-run unless --execute is supplied.
offline: reconstruct convergence from samples.npz without ROS, ZED, or publishers.

See calibrate_camera_visual.md for commands, uncertainty semantics and limitations.
"""
from __future__ import annotations

import argparse
from collections import deque
import faulthandler
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import select
import signal
import sys
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import calibrate_camera as cal


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_camera_intrinsics(directory, K, width, height, serial_number, camera_resolution,
                           camera_upsidedown=False, provenance=None):
    """Keep resolution-specific intrinsics beside the extrinsics for this run.

    The SDK rectified-left K is already used by calibration. This exports it;
    it does not recalibrate the lens or replace it with another camera's K.
    """
    directory = Path(directory)
    K = np.asarray(K, dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or width <= 0 or height <= 0):
        raise ValueError('Invalid intrinsic matrix or image size')
    sizes = [(int(width), int(height)), (160, 90)]
    if camera_resolution == 'HD1080':
        sizes.append((1920, 1080))
    by_resolution = {}
    for w, h in dict.fromkeys(sizes):
        matrix = np.diag([w/width, h/height, 1.]) @ K
        name = f'{w}x{h}'
        np.savetxt(directory/f'K_{name}.txt', matrix, fmt='%.12f')
        np.save(directory/f'K_{name}.npy', matrix)
        by_resolution[name] = dict(width=w, height=h, K=matrix,
            fx=matrix[0, 0], fy=matrix[1, 1], cx=matrix[0, 2], cy=matrix[1, 2],
            source='recorded calibration K' if (w, h) == (width, height) else
                   'derived by full-frame pure scaling from recorded calibration K; not a separate lens calibration')
    metadata = dict(schema_version=1, serial_number=str(serial_number),
        capture_resolution=camera_resolution, calibration_image_size=[int(width), int(height)],
        calibration_K=K, stream='ZED rectified left', intrinsic_units='pixels',
        image_rotation_degrees=180 if camera_upsidedown else 0,
        distortion=dict(model='rectified pinhole', coefficients=[0., 0., 0., 0., 0.],
                        note='Zero distortion applies to rectified images; raw-lens coefficients were not logged.'),
        camera_frame='optical: x right, y down, z forward',
        paired_extrinsics_file='T_RC.txt',
        extrinsics_convention='p_robot_base = T_RC @ p_camera_optical; translation in metres',
        intrinsics_by_resolution=by_resolution,
        resizing_note='K at other sizes uses the capture pipeline scaling convention. Full frame, no crop.',
        provenance=provenance or dict(K_source='SDK rectified-left intrinsics used by this recording'))
    (directory/'camera_intrinsics.json').write_text(json.dumps(metadata, default=json_value, indent=2)+'\n')
    return metadata


def pose_delta(a, b):
    d = cal.inv(a) @ b
    return (float(1000 * np.linalg.norm(d[:3, 3])),
            float(np.degrees(Rotation.from_matrix(d[:3, :3]).magnitude())))


def previous_fit(results, new_sample_n):
    candidates = [s for s in results if 'T_RC' in s and s['n'] < new_sample_n]
    return max(candidates, key=lambda s: s['n']) if candidates else None


def score_unseen(fit, A, B, sample_n):
    """Freeze BOTH fitted transforms and score before submitting the new sample.

    No re-estimation of the tag mounting is allowed here. This is a prospective
    prediction diagnostic, not ground truth or an independent accuracy certificate.
    """
    row = dict(sample_n=sample_n, time=time.time(), fit_n=None)
    if fit is None:
        return row
    if fit['n'] >= sample_n:
        raise ValueError('Prediction must use a fit that has not seen this sample')
    error = cal.inv(np.asarray(fit['T_RC']) @ B) @ (A @ np.asarray(fit['T_ee_tag']))
    row.update(fit_n=fit['n'], fit_time=fit.get('time'),
               mm=float(1000*np.linalg.norm(error[:3, 3])),
               deg=float(np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude())),
               fit_well_constrained=fit.get('uncertainty', {}).get('valid', False))
    return row


def rolling_prediction_rms(rows, window=10):
    valid = [r for r in rows if r['fit_n'] is not None][-window:]
    if not valid:
        return None
    return dict(count=len(valid), **{k: float(np.sqrt(np.mean([r[k]**2 for r in valid]))) for k in ('mm', 'deg')})


def error_plot(xs, series, title, unit, xlabel, revision, prior_counts=None):
    """Readable linear Plotly chart; preserve the operator's zoom on updates."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for name, values, color in series:
        hover = f'Sample %{{x}}<br>{name}: %{{y:.3f}} {unit}'
        if prior_counts is not None:
            hover += '<br>Earlier fit used %{customdata} samples'
        fig.add_trace(go.Scatter(x=list(xs), y=list(values), name=name, mode='lines+markers',
            line=dict(color=color, width=2), marker=dict(size=5),
            customdata=prior_counts, hovertemplate=hover+'<extra></extra>'))
    # The heading lives outside Plotly, so it cannot collide with a wrapping
    # legend in Viser's narrow control panel.
    fig.update_layout(template='plotly_white', meta=dict(title=title),
        font=dict(family='Arial, sans-serif', size=12, color='#243244'),
        margin=dict(l=60, r=14, t=62, b=58), autosize=True, showlegend=True,
        legend=dict(orientation='h', x=0, y=1.03, xanchor='left', yanchor='bottom', font=dict(size=11)),
        hovermode='x unified', dragmode='zoom', uirevision=revision,
        paper_bgcolor='white', plot_bgcolor='white')
    axes = dict(type='linear', showgrid=True, gridcolor='#dce3eb', gridwidth=1,
                showline=True, linecolor='#718096', ticks='outside', tickcolor='#718096',
                automargin=True, zeroline=True, zerolinecolor='#a0aec0', tickfont=dict(size=11))
    span = max(xs)-min(xs) if len(xs) > 1 else 1
    fig.update_xaxes(**axes, title_text=xlabel, title_standoff=10,
                     dtick=max(1, math.ceil(span/6)), tick0=0)
    fig.update_yaxes(**axes, title_text=f'Error ({unit})', title_standoff=10, rangemode='tozero')
    return fig


def fixed_validation_split(n, test_count=10, first_only=False):
    """One held-out pose per equal time-order bin; selection ignores fit errors."""
    if first_only:
        test_count = 1
    if test_count < 1 or n-test_count < 5:
        raise ValueError('Need at least one test pose and five training poses')
    test = np.array([0]) if first_only else np.floor((np.arange(test_count)+.5)*n/test_count).astype(int)
    train = np.setdiff1d(np.arange(n), test)
    return train, test


def fixed_test_errors(fit, A, B):
    """Evaluate frozen camera AND mounting transforms on the unchanged test set."""
    errors = np.array([pose_delta(fit['T_RC'] @ b, a @ fit['T_ee_tag']) for a, b in zip(A, B)])
    return dict(test_rms_mm=float(np.sqrt(np.mean(errors[:, 0]**2))),
                test_rms_deg=float(np.sqrt(np.mean(errors[:, 1]**2))),
                test_mm=errors[:, 0], test_deg=errors[:, 1])


def run_fixed_validation(args, data, view, directory, first_only=False):
    train, test = fixed_validation_split(len(data['A']), args.fixed_test_poses, first_only=first_only)
    counts = sorted({n for n in [5, 8, 10, 15, 20, 25, 30, 35, 40, 45, len(train)] if n <= len(train)})
    report = dict(samples=str(args.samples), first_pose_only=first_only,
                  split='first recorded pose only' if first_only else 'one test pose at the center of each equal recording-order bin',
                  train_sample_numbers=(train+1).tolist(), test_sample_numbers=(test+1).tolist(),
                  fit_counts=counts, results=[])
    panel = view.add_validation_panel(report)
    filename = 'first_pose_learning_curve.json' if first_only else 'fixed_test_learning_curve.json'
    worker = SolverWorker()
    try:
        for n in counts:
            if view.stop_requested.is_set():
                break
            indices = train[:n]
            assert not np.intersect1d(indices, test).size
            worker.submit(data['A'][indices], data['B'][indices])
            while not view.stop_requested.is_set():
                results = worker.poll()
                result = next((r for r in results if r['n'] == n and 'restarting' not in r.get('error', '')), None)
                if result is None:
                    time.sleep(.03)
                    continue
                result['train_sample_numbers'] = (indices+1).tolist()
                if 'error' not in result:
                    result.update(fixed_test_errors(result, data['A'][test], data['B'][test]))
                    print(f'[{"first pose" if first_only else "fixed test"}] {n} training poses: '
                          f'{result["test_rms_mm"]:.2f} mm / {result["test_rms_deg"]:.2f} deg '
                          f'RMS on the same {len(test)} test poses', flush=True)
                else:
                    print(f'[fixed test] {n} training poses: {result["error"]}', flush=True)
                report['results'].append(result)
                (directory/filename).write_text(json.dumps(report, default=json_value, indent=2))
                view.update_validation_panel(report, panel)
                break
    finally:
        worker.close()
    return report


def fit_uncertainty(A, B, sol):
    """Local, model-based covariance, marginalizing the unknown EE/tag transform.

    Linearize the same scaled residual as the original solver. Estimate its noise
    variance with 6*N-12 degrees of freedom. This is a local least-squares
    approximation at the robust optimum, NOT absolute calibration accuracy or a
    robust confidence interval. Tag size, FK bias and correlated errors are absent.
    Refuse a finite ellipsoid for a rank-deficient system rather than using pinv.
    """
    X, Y = sol['T_RC'], sol['T_ee_tag']
    x = np.r_[Rotation.from_matrix(X[:3, :3]).as_rotvec(), X[:3, 3],
              Rotation.from_matrix(Y[:3, :3]).as_rotvec(), Y[:3, 3]]
    eps = 1e-6
    J = np.column_stack([(cal.residuals(x + np.eye(12)[i]*eps, A, B)
                          - cal.residuals(x - np.eye(12)[i]*eps, A, B))/(2*eps)
                         for i in range(12)])
    _, s, vt = np.linalg.svd(J, full_matrices=False)
    rank = int(np.count_nonzero(s > s[0]*1e-6))
    out = dict(rank=rank, condition=float(s[0]/max(s[-1], 1e-30)), valid=False)
    if rank != 12 or len(A)*6 <= 12:
        out['reason'] = 'Underconstrained: rotate the wrist about more independent axes.'
        return out
    r = cal.residuals(x, A, B)
    cov = ((vt.T / (s*s)) @ vt) * float(r @ r)/(len(r)-12)
    # Convert rotation-vector parameter covariance into a base-frame small-angle
    # covariance; include the full 12-variable inverse before selecting camera DOFs.
    G = np.zeros((6, 12))
    G[:3, 3:6] = np.eye(3)
    for i in range(3):
        dr = np.eye(3)[i]*eps
        Rp = Rotation.from_rotvec(x[:3]+dr).as_matrix()
        Rm = Rotation.from_rotvec(x[:3]-dr).as_matrix()
        G[3:, i] = (Rotation.from_matrix(Rp @ X[:3, :3].T).as_rotvec()
                    - Rotation.from_matrix(Rm @ X[:3, :3].T).as_rotvec())/(2*eps)
    camera_cov = G @ cov @ G.T
    out.update(valid=True, covariance_base=camera_cov,
               position_radii95_mm=1000*np.sqrt(7.814728*np.maximum(
                   0, np.linalg.eigvalsh(camera_cov[:3, :3]))),
               rotation_radii95_deg=np.degrees(np.sqrt(7.814728*np.maximum(
                   0, np.linalg.eigvalsh(camera_cov[3:, 3:])))))
    if max(out['rotation_radii95_deg']) > 30 or max(out['position_radii95_mm']) > 500:
        out.update(valid=False, reason='Very weak constraints: uncertainty exceeds the local approximation. '
                   'Collect rotations about more independent axes; finite shapes hidden.')
    return out


def solve_snapshot(A, B):
    start = time.perf_counter()
    result = cal.solve(A, B)
    result['uncertainty'] = fit_uncertainty(A, B, result)
    result['solve_seconds'] = time.perf_counter() - start
    result['n'] = len(A)
    result['time'] = time.time()
    return result


def _solver_main(requests, results):
    faulthandler.enable(all_threads=True)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # These are tiny 12-variable fits. Avoid native thread pools competing with
    # camera capture and ROS, and keep numerical work isolated from the GUI.
    cv2.setNumThreads(1)
    from threadpoolctl import threadpool_limits
    thread_limit = threadpool_limits(limits=1)  # Keep the controller alive for this process.
    while True:
        job = requests.get()
        if job is None:
            return
        A, B = job
        try:
            results.put(solve_snapshot(A, B))
        except Exception as exc:
            results.put(dict(n=len(A), error=f'{type(exc).__name__}: {exc}', time=time.time()))


class SolverWorker:
    """Bounded pending work, isolated native solver, bounded crash recovery."""

    def __init__(self):
        self.latest = self.pending = None
        self.restarts = 0
        self.notices = []
        self._start()

    def _start(self):
        ctx = mp.get_context('spawn')
        self.requests, self.results = ctx.Queue(maxsize=1), ctx.Queue()
        self.process = ctx.Process(target=_solver_main, args=(self.requests, self.results), daemon=True)
        self.process.start()

    def submit(self, A, B):
        if len(A) >= 3:
            self.latest = self.pending = (np.array(A, copy=True), np.array(B, copy=True))
            self.flush()

    def flush(self):
        if not self.process.is_alive():
            code = self.process.exitcode
            if self.restarts >= 3:
                raise RuntimeError(f'Calibration worker repeatedly failed (exit code {code}); samples retained')
            self.restarts += 1
            n = len(self.latest[0]) if self.latest is not None else 0
            error = f'Solver worker exited with code {code}; restarting ({self.restarts}/3) and retrying latest data'
            print(f'[solve] {error}', flush=True)
            self.notices.append(dict(n=n, error=error, time=time.time()))
            self.process.join(timeout=1)
            self.requests.cancel_join_thread()
            self.requests.close()
            self.results.close()
            self._start()
            self.pending = self.latest
        if self.pending is not None:
            try:
                self.requests.put_nowait(self.pending)
                self.pending = None
            except queue.Full:
                pass

    def poll(self):
        self.flush()
        out, self.notices = self.notices, []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        # Callers drain the final requested solve before closing. Bound shutdown.
        try:
            self.requests.put(None, timeout=1)
        except queue.Full:
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.requests.cancel_join_thread()
        self.requests.close()
        self.results.close()


class PoseSampler:
    """Accept only contiguous visible, fresh, stationary frames, with pose diversity."""

    def __init__(self, settle_s, hold_s, tolerance_deg, min_move_deg):
        self.settle_s, self.hold_s = settle_s, hold_s
        self.tolerance = math.radians(tolerance_deg)
        self.min_move = math.radians(min_move_deg)
        self.history = deque()
        self.window = []
        self.last_q = None

    def reset(self):
        self.history.clear()
        self.window.clear()

    def update(self, now, q, T, fresh, enabled=True):
        if not fresh or q is None or not enabled:
            self.reset()
            return None, 'Joint states stale' if not fresh else 'Moving / capture paused'
        self.history.append((now, q.copy()))
        while len(self.history) > 1 and self.history[1][0] < now-self.settle_s:
            self.history.popleft()
        still = (now-self.history[0][0] >= self.settle_s and
                 max(np.max(np.abs(qq-q)) for _, qq in self.history) <= self.tolerance)
        moved = self.last_q is None or np.max(np.abs(q-self.last_q)) > self.min_move
        if not still or not moved or T is None:
            self.window.clear()
            return None, ('Moving' if not still else 'Pose already captured' if not moved else 'Waiting for tag')
        self.window.append((now, q.copy(), T.copy()))
        if now-self.window[0][0] < self.hold_s or len(self.window) < 3:
            return None, 'Averaging visible tag'
        # Reject movement across the entire averaging window too.
        qs = np.stack([v[1] for v in self.window])
        if np.max(np.ptp(qs, axis=0)) > self.tolerance:
            self.window.clear()
            return None, 'Moved during averaging'
        Ts = [v[2] for v in self.window]
        mean_T = cal.mean_pose(Ts)
        result = dict(q=qs.mean(axis=0), T_cam_tag=mean_T, frames=len(Ts),
                      spread_mm=float(1000*max(np.linalg.norm(t[:3, 3]-mean_T[:3, 3]) for t in Ts)))
        self.last_q = q.copy()
        self.window.clear()
        return result, 'Sample captured'


def canonical_arm(msg):
    names = [n.replace('iiwa14_', 'iiwa_') for n in msg.name]
    wanted = [f'iiwa_joint_{i}' for i in range(1, 8)]
    if len(names) != 7 or set(names) != set(wanted) or len(msg.position) != 7:
        raise ValueError(f'Unexpected arm joint names: {msg.name}')
    q = np.array([msg.position[names.index(n)] for n in wanted], dtype=float)
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite arm joint positions')
    return q


class RobotFeed:
    def __init__(self, urdf):
        import rospy
        from sensor_msgs.msg import JointState
        self.urdf, self.lock = urdf, threading.Lock()
        self.arm = self.hand = None
        self.arm_time = self.hand_time = 0.
        self.error = ''
        rospy.Subscriber('/iiwa/joint_states', JointState, self._arm, queue_size=1)
        rospy.Subscriber('/sharpa/joint_states', JointState, self._hand, queue_size=1)

    def _arm(self, msg):
        with self.lock:
            try:
                canonical_arm(msg)
            except ValueError as exc:
                self.error = str(exc)
                return
            self.arm, self.arm_time, self.error = msg, time.monotonic(), ''

    def _hand(self, msg):
        with self.lock:
            self.hand, self.hand_time = msg, time.monotonic()

    def snapshot(self, require_hand=False):
        with self.lock:
            arm, hand = self.arm, self.hand
            fresh = arm is not None and time.monotonic()-self.arm_time < .25 and not self.error
            hand_fresh = hand is not None and time.monotonic()-self.hand_time < .25
            if require_hand:
                fresh = fresh and hand_fresh and len(hand.position) == 22 and len(set(hand.name)) == 22
                fresh = fresh and np.isfinite(hand.position).all()
            q = canonical_arm(arm) if arm is not None else None
            joints = dict(zip(cal.ARM_JOINTS, q)) if q is not None else {}
            if hand_fresh:
                joints.update(cal.joint_map(self.urdf, None, hand))
            return q, joints, bool(fresh), hand_fresh

    def current(self):
        q, _, fresh, _ = self.snapshot(require_hand=True)
        if not fresh:
            raise RuntimeError('Fresh named arm + 22 hand joint states required; publication stopped')
        return q


def cloud_points(rgb, depth, K, stride=4, max_depth=2.5):
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > .1) & (z < max_depth)
    xyz = np.stack(((u[valid]-K[0, 2])*z[valid]/K[0, 0],
                    (v[valid]-K[1, 2])*z[valid]/K[1, 1], z[valid]), axis=1)
    return xyz.astype(np.float32), rgb[::stride, ::stride][valid]


def optical_frustum(K, width, height, length=.20):
    pixels = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    corners = np.c_[(pixels[:, 0]-K[0, 2])*length/K[0, 0],
                    (pixels[:, 1]-K[1, 2])*length/K[1, 1], np.full(4, length)]
    return np.array([[np.zeros(3), p] for p in corners] +
                    [[corners[i], corners[(i+1)%4]] for i in range(4)])


class CalibrationView:
    def __init__(self, args, urdf, K, offline=False):
        import trimesh
        import viser
        self.args, self.urdf, self.K = args, urdf, K
        self.server = viser.ViserServer(host='0.0.0.0', port=args.view_port,
                                        label='Calibration convergence' + (' | OFFLINE' if offline else ' | LIVE'))
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_frame('/robot_base', axes_length=.2, axes_radius=.004, origin_radius=.006)
        self.server.scene.add_grid('/ground', width=2., height=2., position=(0., -.6, 0.))
        self.robot = {}
        for link, visuals in urdf.visuals.items():
            for i, (path, T, scale) in enumerate(visuals):
                try:
                    mesh = trimesh.load(str(path), force='mesh', process=False, skip_materials=True)
                    vertices = np.asarray(mesh.vertices)*scale
                    vertices = vertices @ T[:3, :3].T + T[:3, 3]
                    handle = self.server.scene.add_mesh_simple(f'/robot/{link}_{i}',
                        vertices=vertices.astype(np.float32), faces=np.asarray(mesh.faces),
                        color=(190, 190, 200), opacity=.65)
                    self.robot.setdefault(link, []).append(handle)
                except Exception as exc:
                    print(f'[view] mesh skipped {path.name}: {exc}', flush=True)
        self.camera = self.server.scene.add_frame('/camera', axes_length=.1, axes_radius=.002, origin_radius=.003)
        self.frustum_lines = optical_frustum(K, args.width, args.height)
        self.frustum = self.server.scene.add_line_segments('/camera/frustum',
            points=self.frustum_lines, colors=(160, 160, 160), thickness=2., thickness_units='screen')
        self.cloud = self.server.scene.add_point_cloud('/camera/cloud',
            points=np.zeros((1, 3), np.float32), colors=np.zeros((1, 3), np.uint8), point_size=.003)
        self.tag_observed = self.server.scene.add_frame('/observed_tag', axes_length=.06, axes_radius=.002, visible=False)
        self.tag_predicted = self.server.scene.add_frame('/predicted_tag', axes_length=.09, axes_radius=.002, visible=False)
        sphere = trimesh.creation.icosphere(subdivisions=2)
        self.sphere_vertices, self.sphere_faces = sphere.vertices, sphere.faces
        self.ellipsoid = self.server.scene.add_mesh_simple('/uncertainty/position', vertices=sphere.vertices,
            faces=sphere.faces, color=(245, 160, 40), opacity=.25, visible=False)
        self.cone = self.server.scene.add_line_segments('/uncertainty/direction',
            points=np.zeros((1, 2, 3)), colors=(245, 160, 40), thickness=1., thickness_units='screen', visible=False)
        gui = self.server.gui
        self.status = gui.add_markdown('**Waiting for images and joint states**', order=0)
        self.metrics = gui.add_markdown('**UNCALIBRATED GUESS**. Need diverse poses before estimating T_RC.', order=2)
        self.stop_requested = threading.Event()
        gui.add_button('Finish recording and solve' if not offline else 'Close offline viewer', order=3).on_click(
            lambda _: self.stop_requested.set())
        with gui.add_folder('Display controls', order=4, expand_by_default=False):
            self.show_rgb = gui.add_checkbox('Update camera image', initial_value=True)
            self.show_cloud = gui.add_checkbox('Update point cloud', initial_value=not offline)
            self.show_robot = gui.add_checkbox('Update robot', initial_value=True)
            self.show_uncertainty = gui.add_checkbox('Show approximate fit uncertainty', initial_value=True)
            self.show_history = gui.add_checkbox('Show previous camera estimates', initial_value=True)
            self.uncertainty_scale = gui.add_slider('Uncertainty display magnification', min=1., max=20.,
                                                   step=1., initial_value=1.)
            gui.add_markdown('Orange: approximate 95% local fit uncertainty. Scale 1 = actual size. '
                            'Excludes tag-size error, FK bias and depth error. The cone shows viewing-direction '
                            'uncertainty; roll uncertainty is included in the numbers below.')
        self.image = gui.add_image(np.zeros((args.height, args.width, 3), np.uint8), label='Live RGB + AprilTag',
                                   format='jpeg', jpeg_quality=80, order=1)
        gui.add_markdown('Green outline: detected tag. Orange outline: tag predicted from robot FK and '
                        'the displayed calibration. Large 3D tag axes: predicted; small axes: observed.', order=1.1)
        if offline:
            gui.add_markdown('**Offline:** saved joint poses and sample images. No depth was saved in this '
                             'recording, so no recorded point cloud is available. No ROS or camera connection.')
        with gui.add_folder('Solve history'):
            self.follow = gui.add_checkbox('Follow latest estimate', initial_value=True)
            self.history_idx = gui.add_slider('Samples used by estimate', min=0, max=1, step=1, initial_value=0)
            plot_config = dict(displaylogo=False, responsive=True, displayModeBar='hover',
                               modeBarButtonsToRemove=['select2d', 'lasso2d'])
            self.fit_plots = {}
            for unit, title in [('mm', 'Fit residual: position'), ('deg', 'Fit residual: orientation')]:
                gui.add_markdown(f'**{title}**')
                self.fit_plots[unit] = gui.add_plotly(error_plot([], [], title, unit,
                    'Samples used in fit', f'fit_{unit}'), aspect=1.35, config=plot_config)
            self.details = gui.add_markdown('Waiting for first solve.')
        with gui.add_folder('Predict new poses before fitting them', order=3.5):
            self.prediction_status = gui.add_markdown('Waiting for a fitted estimate and a new sample.')
            self.prediction_plots = {}
            for unit, title in [('mm', 'Unseen-pose prediction: position'), ('deg', 'Unseen-pose prediction: orientation')]:
                gui.add_markdown(f'**{title}**')
                self.prediction_plots[unit] = gui.add_plotly(error_plot([], [], title, unit,
                    'New measurement number', f'prediction_{unit}'), aspect=1.35, config=plot_config)
            gui.add_markdown('Both camera pose and '
                            'tag mounting are frozen before this sample enters the solver. '
                            'Look for low errors on diverse new poses, alongside a stable T_RC. '
                            'All axes are linear. Drag to zoom; double-click to restore the full range. '
                            'Hover for values; click legend entries to hide or show a line.')
        self.history, self.ghosts = [], []
        self.predictions = []
        self.on_select = None
        self.last_draw = self.last_cloud = 0.
        self.selected = None
        self.last_q = self.last_tag = None
        self.offline = offline
        T = cal.load_T(args.calibration) if args.calibration else cal.look_at(args.cam_xyz, args.look_at)
        self.T = T
        self.apply_pose(self.camera, T)
        self.trail_positions = [T[:3, 3].copy()]
        self.trail = self.server.scene.add_line_segments('/history/path',
            points=np.array([[T[:3, 3], T[:3, 3]]]), colors=(60, 140, 210), thickness=2., thickness_units='screen')
        initial = self.server.scene.add_line_segments('/history/initial_guess',
            points=self.frustum_lines, colors=(130, 130, 130), thickness=1., thickness_units='screen')
        self.apply_pose(initial, T)
        self.ghosts.append(initial)
        self.history_idx.on_update(lambda _: self.select_history())
        self.follow.on_update(lambda _: self.select_history())
        self.show_uncertainty.on_update(lambda _: self.select_history())
        self.uncertainty_scale.on_update(lambda _: self.select_history())
        self.show_history.on_update(lambda _: self.update_ghosts())

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (.9, -1.7, 1.15)
            client.camera.look_at = (0., -.6, .55)
            client.camera.up_direction = (0., 0., 1.)
            client.camera.fov = .9

        print(f'[view] http://localhost:{self.server.get_port()}', flush=True)

    @staticmethod
    def apply_pose(handle, T):
        handle.position = tuple(T[:3, 3])
        handle.wxyz = tuple(Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]])

    def update_ghosts(self):
        for ghost in self.ghosts:
            ghost.visible = self.show_history.value
        self.trail.visible = self.show_history.value

    def add_result(self, result):
        if 'error' in result:
            self.metrics.content = f'**Solve at {result["n"]} samples failed:** {result["error"]}. Collect more diverse poses.'
            return
        for key in ('T_RC', 'T_ee_tag'):
            result[key] = np.asarray(result[key], float)
        previous = self.history[-1] if self.history else None
        result['delta_mm'], result['delta_deg'] = pose_delta(previous['T_RC'], result['T_RC']) if previous else (None, None)
        self.history.append(result)
        T = result['T_RC']
        ghost = self.server.scene.add_line_segments(f'/history/estimate_{len(self.history)}',
            points=self.frustum_lines*.6, colors=(155, 165, 180), thickness=1., thickness_units='screen',
            visible=self.show_history.value)
        self.apply_pose(ghost, T)
        self.ghosts.append(ghost)
        self.trail_positions.append(T[:3, 3].copy())
        self.trail.points = np.array(list(zip(self.trail_positions[:-1], self.trail_positions[1:])))
        self.history_idx.max = max(1, result['n'])
        if self.follow.value:
            self.history_idx.value = result['n']
        self.select_history()
        self.draw_chart()

    def select_history(self):
        if not self.history:
            return
        index = (len(self.history)-1 if self.follow.value else
                 max((i for i, s in enumerate(self.history) if s['n'] <= self.history_idx.value), default=0))
        s = self.history[index]
        self.selected, self.T = s, s['T_RC']
        self.apply_pose(self.camera, self.T)
        u = s['uncertainty']
        color = (50, 190, 100) if u['valid'] else (245, 160, 40)
        self.frustum.colors = np.tile(np.array(color, np.uint8), (len(self.frustum_lines), 2, 1))
        spread = np.round(s['wrist_spread_deg'], 1).tolist()
        coverage = 'Limited rotation coverage' if min(spread) < 5 else 'Rotation coverage heuristic met'
        delta = 'first solve' if s['delta_mm'] is None else f'{s["delta_mm"]:.2f} mm / {s["delta_deg"]:.2f} deg'
        self.metrics.content = (f'**Estimate from {s["n"]} samples** | solve {s["solve_seconds"]:.2f} s\n\n'
            f'RMS: **{s["rms_mm"]:.2f} mm / {s["rms_deg"]:.2f} deg**  \n'
            f'Max: **{max(s["mm"]):.2f} mm / {max(s["deg"]):.2f} deg**  \n'
            f'Change since previous solve: {delta}  \n'
            f'Rotation spread: {spread} deg. **{coverage}**.  \n'
            f'Jacobian rank: {u["rank"]}/12. '
            + ('Approximate fit uncertainty shown in orange.' if u['valid'] else u['reason']))
        uncertainty_text = (f'Position ellipsoid radii (mm): {np.round(u["position_radii95_mm"], 2).tolist()}  \n'
                            f'Rotation radii (deg): {np.round(u["rotation_radii95_deg"], 2).tolist()}'
                            if u['valid'] else 'No finite uncertainty region displayed.')
        self.details.content = (uncertainty_text + '\n\nCamera optical -> robot base, metres:\n```\n'
                                + np.array2string(self.T, precision=6) + '\n```\n'
                                'Residuals and convergence do not establish absolute accuracy.')
        self.draw_uncertainty(s)
        if self.on_select is not None:
            self.on_select(s)
        self.update_tags()

    def draw_uncertainty(self, sol):
        u = sol['uncertainty']
        visible = u['valid'] and self.show_uncertainty.value
        self.ellipsoid.visible = self.cone.visible = visible
        if not visible:
            return
        covariance = np.asarray(u['covariance_base'])
        scale = self.uncertainty_scale.value
        values, vectors = np.linalg.eigh(covariance[:3, :3])
        radii = np.sqrt(7.814728*np.maximum(values, 0))*scale
        self.ellipsoid.vertices = ((self.sphere_vertices*radii) @ vectors.T).astype(np.float32)
        self.ellipsoid.position = tuple(self.T[:3, 3])
        z = self.T[:3, 2]
        cross = np.array([[0., -z[2], z[1]], [z[2], 0., -z[0]], [-z[1], z[0], 0.]])
        values, vectors = np.linalg.eigh(cross @ covariance[3:, 3:] @ cross.T)
        theta = np.linspace(0, 2*np.pi, 41)
        radii = np.sqrt(5.991465*np.maximum(values[-2:], 0))*scale
        offsets = (np.c_[np.cos(theta), np.sin(theta)]*radii) @ vectors[:, -2:].T
        origin = self.T[:3, 3]
        rim = origin + .25*(z+offsets)
        self.cone.points = np.array([[rim[i], rim[i+1]] for i in range(len(rim)-1)] +
                                    [[origin, rim[i]] for i in range(0, len(rim)-1, 5)])

    def draw_chart(self):
        ns = [s['n'] for s in self.history]
        for unit, rms_key, title in [('mm', 'rms_mm', 'position'), ('deg', 'rms_deg', 'orientation')]:
            self.fit_plots[unit].figure = error_plot(ns, [
                ('Fit RMS', [s[rms_key] for s in self.history], '#2563eb'),
                ('Worst pose', [max(s[unit]) for s in self.history], '#dc2626')],
                f'Fit residual: {title}', unit, 'Samples used in fit', f'fit_{unit}')

    def add_validation_panel(self, report):
        gui = self.server.gui
        first_only = report.get('first_pose_only', False)
        panel = dict(report=report, plots={})
        with gui.add_folder('First pose held out: learning curve' if first_only else 'Fixed test set: learning curve',
                            order=.4 if first_only else .5):
            gui.add_markdown(
                f'**Same {len(report["test_sample_numbers"])} test poses at every point.** '
                'These poses never enter any fit in this analysis. The remaining poses enter in recording order. '
                'Both fitted transforms are frozen for testing. The 3D scene continues to show the original full-data fit.')
            if first_only:
                gui.add_markdown('**Only the first recorded pose is tested here.** This measures agreement at that '
                                 'one wrist pose; it can look good even when the camera pose is poorly constrained.')
            gui.add_markdown('Reserved sample numbers: '+', '.join(map(str, report['test_sample_numbers'])))
            panel['range'] = gui.add_dropdown('Plot range',
                options=['All fits', '8+ training poses (zoom)'],
                initial_value='8+ training poses (zoom)' if not first_only and len(report['train_sample_numbers']) >= 8 else 'All fits')
            panel['summary'] = gui.add_markdown('The full results table below includes every fit.')
            panel['status'] = gui.add_markdown('Computing the independent fits...', order=99)
            for unit, title in [('mm', 'Fixed-test RMS: position'), ('deg', 'Fixed-test RMS: orientation')]:
                gui.add_markdown(f'**{title}**')
                panel['plots'][unit] = gui.add_plotly(error_plot([], [], title, unit,
                    'Number of training poses', f'fixed_test_{unit}'), aspect=1.35,
                    config=dict(displaylogo=False, responsive=True, displayModeBar='hover',
                                modeBarButtonsToRemove=['select2d', 'lasso2d']))
            gui.add_markdown('Linear axes; raw RMS, no smoothing or enforced decrease. Drag to zoom, '
                             'double-click to reset. This tests held-out poses from the same recording; '
                             'it does not measure absolute accuracy or independence from shared measurement biases.')
        panel['range'].on_update(lambda _: self.update_validation_panel(panel['report'], panel))
        return panel

    def update_validation_panel(self, report, panel):
        panel['report'] = report
        rows = report['results']
        shown = [r for r in rows if panel['range'].value == 'All fits' or r['n'] >= 8]
        if rows and 'test_rms_mm' in rows[0]:
            first = rows[0]
            panel['summary'].content = (
                f'First fit ({first["n"]} training poses): **{first["test_rms_mm"]:.2f} mm / '
                f'{first["test_rms_deg"]:.2f} deg** test RMS. '
                + ('**Plot zoom: 8+ training poses.** Select All fits to include the earliest fit. '
                   if panel['range'].value != 'All fits' else '**Showing all fits.** ')
                + 'The complete results table is below the plots.')
        for unit, title in [('mm', 'position'), ('deg', 'orientation')]:
            panel['plots'][unit].figure = error_plot([r['n'] for r in shown], [
                (f'Test RMS ({len(report["test_sample_numbers"])} fixed poses)',
                 [r.get(f'test_rms_{unit}') for r in shown], '#7c3aed')],
                f'Fixed-test RMS: {title}', unit, 'Number of training poses',
                f'fixed_test_{unit}_{panel["range"].value}')
        lines = ['| Training poses | Test RMS (mm) | Test RMS (deg) |', '| --- | --- | --- |']
        for r in rows:
            lines.append(f'| {r["n"]} | {r["test_rms_mm"]:.2f} | {r["test_rms_deg"]:.2f} |'
                         if 'test_rms_mm' in r else f'| {r["n"]} | Fit failed | - |')
        panel['status'].content = '\n'.join(lines)

    def add_prediction(self, row):
        self.predictions.append(row)
        rolling = rolling_prediction_rms(self.predictions)
        if row['fit_n'] is None:
            self.prediction_status.content = f'Sample {row["sample_n"]}: no previous fit available yet.'
            return
        caution = '' if row['fit_well_constrained'] else ' **Previous fit was weakly constrained.**'
        self.prediction_status.content = (
            f'**New sample {row["sample_n"]}, predicted by fit from {row["fit_n"]} samples**  \n'
            f'Error before refitting: **{row["mm"]:.2f} mm / {row["deg"]:.2f} deg**  \n'
            f'Last {rolling["count"]} predictions RMS: **{rolling["mm"]:.2f} mm / {rolling["deg"]:.2f} deg**.' + caution)
        valid = [r for r in self.predictions if r['fit_n'] is not None]
        ns = [r['sample_n'] for r in valid]
        for unit, title in [('mm', 'position'), ('deg', 'orientation')]:
            self.prediction_plots[unit].figure = error_plot(ns, [
                ('New pose error', [r[unit] for r in valid], '#d97706'),
                ('Rolling RMS (10)', [rolling_prediction_rms(valid[:i+1])[unit]
                                      for i in range(len(valid))], '#059669')],
                f'Unseen-pose prediction: {title}', unit, 'New measurement number',
                f'prediction_{unit}', prior_counts=[r['fit_n'] for r in valid])

    def update_tags(self):
        valid = self.selected is not None and self.last_tag is not None and self.last_q is not None
        self.tag_observed.visible = self.tag_predicted.visible = valid
        if valid:
            self.apply_pose(self.tag_observed, self.T @ self.last_tag)
            ee = self.urdf.fk(self.last_q)[self.args.ee_link]
            self.apply_pose(self.tag_predicted, ee @ self.selected['T_ee_tag'])

    def update(self, rgb, depth, corners, T_tag, joints, fresh, hand_fresh, state, count):
        now = time.monotonic()
        if now-self.last_draw < 1/self.args.view_hz:
            return
        self.last_draw = now
        self.status.content = (f'**{"OFFLINE" if self.offline else "LIVE"} | Samples {count} | '
            f'{"TAG SEEN" if T_tag is not None else "TAG NOT SEEN"}**  \n{state}  \n'
            f'Arm: {"fresh" if fresh else "STALE"}. Hand: {"fresh" if hand_fresh else "unavailable; displayed at zero"}.')
        self.last_q, self.last_tag = joints if fresh else None, T_tag
        if self.show_robot.value and joints:
            poses = self.urdf.fk(joints)
            for link, handles in self.robot.items():
                for handle in handles:
                    self.apply_pose(handle, poses[link])
        self.update_tags()
        if self.show_rgb.value:
            frame = rgb.copy()
            if corners is not None:
                cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 2)
            if fresh and self.selected is not None:
                ee = self.urdf.fk(joints)[self.args.ee_link]
                tag = cal.inv(self.T) @ ee @ self.selected['T_ee_tag']
                if tag[2, 3] > .05:
                    half = self.args.tag_size/2
                    pts = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
                    uv, _ = cv2.projectPoints(pts, Rotation.from_matrix(tag[:3, :3]).as_rotvec(),
                                              tag[:3, 3], self.K, None)
                    if np.isfinite(uv).all() and np.max(np.abs(uv)) < 1e5:
                        cv2.polylines(frame, [uv.reshape(-1, 2).astype(np.int32)], True, (255, 160, 0), 2)
            self.image.image = frame
        self.cloud.visible = self.show_cloud.value and depth is not None
        if self.cloud.visible and now-self.last_cloud >= 1/self.args.cloud_hz:
            self.last_cloud = now
            self.cloud.points, self.cloud.colors = cloud_points(rgb, depth, self.K, self.args.stride, self.args.max_depth)

    def close(self):
        self.server.stop()


class LiveSession:
    """Camera/detection/preview thread. Motion never waits for a solve or GUI update."""

    def __init__(self, args, urdf, feed, cap, directory):
        self.args, self.urdf, self.feed, self.cap, self.directory = args, urdf, feed, cap, directory
        self.view = CalibrationView(args, urdf, cap.K)
        try:
            self.detector = cal.make_detector()
            self.worker = SolverWorker()
        except Exception:
            self.view.close()
            raise
        self.sampler = PoseSampler(args.settle_s, args.hold_s, args.still_tol_deg, args.min_move_deg)
        self.stop_event = threading.Event()
        self.error = None
        self.lock = threading.Lock()
        self.enabled = args.mode == 'record'
        self.source_index = -1
        self.target = None
        self.samples, self.stream, self.results, self.skipped = [], [], [], []
        self.video = None
        self.last_video = 0.
        self.thread = threading.Thread(target=self._loop, name='calibration-camera-preview', daemon=True)
        self.thread.start()

    def capture(self, enabled, source_index=-1, target=None):
        with self.lock:
            self.enabled, self.source_index, self.target = enabled, source_index, target
            self.sampler.reset()

    def check(self):
        if self.error:
            raise RuntimeError('Camera/recording worker failed') from self.error
        if self.view.stop_requested.is_set():
            raise KeyboardInterrupt

    def save_samples(self):
        tmp = self.directory/'samples.tmp.npz'
        q = np.array([s['q'] for s in self.samples]).reshape(-1, 7)
        np.savez(tmp, q=q, T_cam_tag=np.array([s['T_cam_tag'] for s in self.samples]).reshape(-1, 4, 4),
                 T_base_ee=np.array([s['T_base_ee'] for s in self.samples]).reshape(-1, 4, 4),
                 K=self.cap.K, tag_size=self.args.tag_size, tag_id=self.args.tag_id,
                 ee_link=self.args.ee_link, urdf=str(self.args.urdf), joint_names=np.array(cal.ARM_JOINTS),
                 depth_refine=False, schema_version=2, serial_number=str(self.args.serial_number),
                 stream=np.array(self.stream).reshape(-1, 24),
                 source_indices=np.array([s['source_index'] for s in self.samples]),
                 skipped_source_indices=np.array(self.skipped),
                 transform_convention='T_parent_child maps child coordinates into parent coordinates')
        tmp.replace(self.directory/'samples.npz')

    def poll_results(self):
        for result in self.worker.poll():
            self.results.append(result)
            self.view.add_result(result)
            with (self.directory/'estimate_history.jsonl').open('a') as f:
                f.write(json.dumps(result, default=json_value)+'\n')
            if 'error' not in result:
                print(f'[solve] n={result["n"]} RMS={result["rms_mm"]:.2f} mm / '
                      f'{result["rms_deg"]:.2f} deg; {result["solve_seconds"]:.2f}s', flush=True)

    def _loop(self):
        seq, last_camera = 0, time.monotonic()
        try:
            while not self.stop_event.is_set():
                self.poll_results()
                frame = self.cap.latest(since_seq=seq, timeout=.1)
                if frame is None:
                    if time.monotonic()-last_camera > 3:
                        raise RuntimeError('No camera frames for 3 seconds')
                    continue
                seq, _, rgb, depth = frame
                now = last_camera = time.monotonic()
                q, joints, fresh, hand_fresh = self.feed.snapshot(require_hand=self.args.mode == 'replay')
                T, corners = cal.detect_tag(rgb, depth, self.cap.K, self.args.tag_size,
                                            self.args.tag_id, self.detector, depth_refine=False)
                self.stream.append(np.r_[time.time(), q if q is not None else np.full(7, np.nan),
                                         (T if T is not None else np.full((4, 4), np.nan)).ravel()])
                with self.lock:
                    enabled = self.enabled
                    if self.target is not None and q is not None:
                        enabled = enabled and np.max(np.abs(q-self.target)) <= math.radians(self.args.capture_tolerance_deg)
                    sample, state = self.sampler.update(now, q, T, fresh, enabled)
                    if sample is not None:
                        sample['source_index'] = self.source_index
                        if self.args.mode == 'replay':
                            self.enabled = False  # At most one sample at each commanded stop.
                if sample is not None:
                    sample['T_base_ee'] = self.urdf.fk(dict(zip(cal.ARM_JOINTS, sample['q'])))[self.args.ee_link]
                    n = len(self.samples)+1
                    prediction = score_unseen(previous_fit(self.results, n), sample['T_base_ee'], sample['T_cam_tag'], n)
                    self.samples.append(sample)
                    if not cv2.imwrite(str(self.directory/f'sample_{n:02d}_raw.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError('Could not save calibration sample image')
                    self.save_samples()
                    with (self.directory/'prediction_history.jsonl').open('a') as f:
                        f.write(json.dumps(prediction)+'\n')
                    self.view.add_prediction(prediction)
                    self.worker.submit([s['T_base_ee'] for s in self.samples], [s['T_cam_tag'] for s in self.samples])
                    print(f'[record] sample {n}: {sample["frames"]} frames, spread {sample["spread_mm"]:.2f} mm', flush=True)
                self.view.update(rgb, depth, corners, T, joints, fresh, hand_fresh, state, len(self.samples))
                if self.args.video and now-self.last_video >= .1:
                    self.last_video = now
                    bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (rgb.shape[1]//2, rgb.shape[0]//2))
                    if self.video is None:
                        self.video = cv2.VideoWriter(str(self.directory/'record.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                                    10., (bgr.shape[1], bgr.shape[0]))
                        if not self.video.isOpened():
                            raise RuntimeError('Could not open calibration video output')
                    self.video.write(bgr)
        except Exception as exc:
            self.error = exc

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            # Do not race a network write by saving simultaneously from another thread.
            raise RuntimeError('Recording thread still finishing I/O; samples already checkpointed on each stop')
        try:
            if self.video is not None:
                self.video.release()
            self.save_samples()
            if len(self.samples) >= 3:
                self.worker.submit([s['T_base_ee'] for s in self.samples], [s['T_cam_tag'] for s in self.samples])
                deadline = time.monotonic()+30
                while time.monotonic() < deadline:
                    self.poll_results()
                    if (self.results and self.results[-1]['n'] == len(self.samples)
                            and 'restarting' not in self.results[-1].get('error', '')):
                        break
                    time.sleep(.05)
                final = next((s for s in reversed(self.results) if s['n'] == len(self.samples) and 'error' not in s), None)
                if final is not None:
                    cal.save_T(final['T_RC'], str(self.directory/'T_RC'))
                    np.savetxt(self.directory/'T_ee_tag.txt', final['T_ee_tag'])
                    cal.report(final)
                else:
                    print('[solve] No valid final fit; raw samples saved for offline solve.', flush=True)
            print(f'[record] saved {len(self.samples)} samples -> {self.directory}', flush=True)
        finally:
            self.worker.close()
            self.view.close()


def load_samples(path, urdf):
    with np.load(path, allow_pickle=False) as data:
        q = np.asarray(data['q'], float)
        if 'joint_names' in data:
            names = list(data['joint_names'].astype(str))
            q = q[:, [names.index(n) for n in cal.ARM_JOINTS]]
        ee = str(data['ee_link'])
        A = (np.asarray(data['T_base_ee'], float) if 'T_base_ee' in data else
             np.array([urdf.fk(dict(zip(cal.ARM_JOINTS, row)))[ee] for row in q]))
        return dict(q=q, A=A, B=np.asarray(data['T_cam_tag'], float), K=np.asarray(data['K'], float),
                    tag_size=float(data['tag_size']), tag_id=int(data['tag_id']), ee_link=ee)


def run_offline(args):
    urdf = cal.Urdf(args.urdf)
    data = load_samples(args.samples, urdf)
    args.tag_size, args.ee_link = data['tag_size'], data['ee_link']
    view = CalibrationView(args, urdf, data['K'], offline=True)
    worker = None if args.use_saved_fits else SolverWorker()
    saved = {}
    saved_predictions = {}
    if args.use_saved_fits:
        for line in (args.samples.parent/'estimate_history.jsonl').read_text().splitlines():
            fit = json.loads(line)
            if 'T_RC' in fit:
                n = int(fit['n'])
                if not 3 <= n <= len(data['A']):
                    raise ValueError('Saved fit has an incompatible sample count')
                X, Y = np.array(fit['T_RC']), np.array(fit['T_ee_tag'])
                x = np.r_[Rotation.from_matrix(X[:3, :3]).as_rotvec(), X[:3, 3],
                          Rotation.from_matrix(Y[:3, :3]).as_rotvec(), Y[:3, 3]]
                residual = cal.residuals(x, data['A'][:n], data['B'][:n]).reshape(-1, 6)
                rms = 1000*np.sqrt(np.mean(np.sum(residual[:, 3:]**2, axis=1)))
                if not np.isclose(rms, fit['rms_mm'], rtol=1e-4, atol=1e-5):
                    raise ValueError('Saved fit history does not match these samples')
            if 'T_RC' in fit or fit['n'] not in saved:
                saved[fit['n']] = fit
        prediction_file = args.samples.parent/'prediction_history.jsonl'
        if prediction_file.is_file():
            for line in prediction_file.read_text().splitlines():
                row = json.loads(line)
                n, fit_n = row['sample_n'], row['fit_n']
                if not 1 <= n <= len(data['A']) or (fit_n is not None and not 3 <= fit_n < n):
                    raise ValueError('Saved prediction has an incompatible sample count')
                if fit_n is not None:
                    expected = score_unseen(saved[fit_n], data['A'][n-1], data['B'][n-1], n)
                    if not all(np.isclose(row[k], expected[k], rtol=1e-4, atol=1e-5) for k in ('mm', 'deg')):
                        raise ValueError('Saved prediction history does not match these samples')
                saved_predictions[n] = row
    directory = make_directory(args, 'offline')
    completed = []
    image_cache = {}

    def show_sample(result):
        n = result['n']
        if n not in image_cache:
            path = args.samples.parent/f'sample_{n:02d}_raw.png'
            bgr = cv2.imread(str(path)) if path.is_file() else None
            image_cache[n] = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None
                              else np.zeros((args.height, args.width, 3), np.uint8))
        view.last_draw = 0.
        view.update(image_cache[n], None, None, data['B'][n-1],
                    dict(zip(cal.ARM_JOINTS, data['q'][n-1])), True, False,
                    f'Saved sample {n}; green detection unavailable offline, orange = prediction', n)

    view.on_select = show_sample
    try:
        for n in range(3, len(data['q'])+1):
            if view.stop_requested.is_set():
                break
            prediction = saved_predictions.get(n)
            if prediction is None:
                prediction = score_unseen(previous_fit(completed, n), data['A'][n-1], data['B'][n-1], n)
            view.add_prediction(prediction)
            with (directory/'prediction_history.jsonl').open('a') as f:
                f.write(json.dumps(prediction)+'\n')
            if worker is not None:
                worker.submit(data['A'][:n], data['B'][:n])
            show_sample(dict(n=n))
            while not view.stop_requested.is_set():
                results = worker.poll() if worker is not None else [saved.get(n, dict(n=n, error='No saved fit for this sample count'))]
                for result in results:
                    completed.append(result)
                    view.add_result(result)
                    with (directory/'estimate_history.jsonl').open('a') as f:
                        f.write(json.dumps(result, default=json_value)+'\n')
                if any(r['n'] == n and 'restarting' not in r.get('error', '') for r in results):
                    break
                time.sleep(.03)
        print(f'[offline] convergence history -> {directory}', flush=True)
        if args.fixed_test_poses and not view.stop_requested.is_set():
            run_fixed_validation(args, data, view, directory)
        if args.first_pose_test and not view.stop_requested.is_set():
            run_fixed_validation(args, data, view, directory, first_only=True)
        if args.exit_when_done:
            return
        print('[offline] Finished. Use Follow latest / Samples used by estimate to inspect convergence. Ctrl-C to close.', flush=True)
        while not view.stop_requested.wait(.1):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        if worker is not None:
            worker.close()
        view.close()


def make_directory(args, prefix):
    directory = Path(args.out_dir)/f'{prefix}_visual_{time.strftime("%Y%m%d_%H%M%S")}'
    directory.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (directory/'config.json').write_text(json.dumps(config, indent=2))
    return directory


def run_live(args):
    urdf = cal.Urdf(args.urdf)
    if args.mode == 'replay':
        # Reuse the original file validation, timing and sampled table checks.
        import replay_calibration as replay
        poses, args.tag_size, args.tag_id, args.ee_link = replay.load_poses(args.poses, urdf)
        if not args.execute:
            validate_start(urdf, args.start_q)
            checker = cal.TableCheck(urdf, args.table_z, args.table_box)
            clearance = replay.check_paths(poses, checker, args.min_clear, args.start_q)
            durations = [replay.duration(a, b, args.move_s, args.max_joint_speed_deg) for a, b in zip(poses[:-1], poses[1:])]
            first = args.first_move_s if args.start_q is None else replay.duration(np.array(args.start_q), poses[0], args.first_move_s, args.max_joint_speed_deg)
            print(f'DRY RUN: {len(poses)} poses, estimated minimum {first+sum(durations)+len(poses)*(args.settle_s+args.hold_s):.1f}s; '
                  f'sampled table clearance {clearance*100:.2f}cm. No ROS, camera, or publishers.\n'
                  + ('Current-to-first path not checked.' if args.start_q is None else 'Supplied start path checked.'))
            return
    import rospy
    rospy.init_node('camera_calibration_visual', anonymous=True, disable_signals=True)
    feed = RobotFeed(urdf)
    deadline = time.monotonic()+10
    while not feed.snapshot(require_hand=args.mode == 'replay')[2]:
        if time.monotonic() > deadline:
            raise RuntimeError('Fresh joint states not available')
        time.sleep(.05)
    start = feed.current() if args.mode == 'replay' else None
    if args.mode == 'replay':
        validate_start(urdf, start)
        checker = cal.TableCheck(urdf, args.table_z, args.table_box, cal.joint_map(urdf, None, feed.hand))
        replay.check_paths(poses, checker, args.min_clear, start)
    directory = make_directory(args, args.mode)
    cap = cal.open_camera(args)
    session = None
    try:
        save_camera_intrinsics(directory, cap.K, args.width, args.height, args.serial_number,
                               args.camera_resolution, args.camera_upsidedown)
        session = LiveSession(args, urdf, feed, cap, directory)
        if args.mode == 'replay':
            execute_replay(args, replay, poses, start, feed, session, rospy)
        else:
            print('[record] Passive recording. No command publishers. Hold each pose for about 2 seconds.', flush=True)
            while not rospy.is_shutdown():
                session.check()
                time.sleep(.05)
    except KeyboardInterrupt:
        print('[record] Finishing and saving.', flush=True)
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            cap.stop_capture()


def validate_start(urdf, start):
    if start is None:
        return
    start = np.asarray(start, float)
    lower, upper = urdf.arm_limits()
    if start.shape != (7,) or not np.isfinite(start).all() or np.any(start < lower) or np.any(start > upper):
        raise ValueError('Starting arm pose is invalid or outside URDF joint limits')


def execute_replay(args, replay, poses, start, feed, session, rospy):
    print('POSITION CONTROL required. Checked table paths only; cameras, cables, bowl and self-collision '
          'are not modeled. Preview is live now. No commands sent yet.', flush=True)
    print(f'Type REPLAY to move through {len(poses)} poses: ', end='', flush=True)
    while not select.select([sys.stdin], [], [], .1)[0]:
        session.check()  # The GUI Finish button remains usable while waiting.
    if sys.stdin.readline().strip() != 'REPLAY':
        return
    session.check()
    if np.max(np.abs(feed.current()-start)) > math.radians(.5):
        raise RuntimeError('Robot moved since preflight; restart to check the new path')
    from sensor_msgs.msg import JointState
    arm_pub = rospy.Publisher('/iiwa/joint_cmd', JointState, queue_size=1)
    hand_pub = rospy.Publisher('/sharpa/joint_cmd', JointState, queue_size=1)
    hand_names, hand_q = list(feed.hand.name), list(feed.hand.position)

    def send(q, guard=True):
        actual = feed.current()
        if rospy.is_shutdown():
            raise RuntimeError('ROS stopped')
        if guard and np.max(np.abs(q-actual)) > math.radians(args.max_tracking_error_deg):
            raise RuntimeError('Arm tracking error exceeded guard; stopping publication')
        arm, hand = JointState(), JointState()
        arm.header.stamp = hand.header.stamp = rospy.Time.now()
        arm.name, arm.position = [f'iiwa_joint_{i}' for i in range(1, 8)], list(q)
        hand.name, hand.position = hand_names, hand_q
        arm_pub.publish(arm)
        hand_pub.publish(hand)

    try:
        deadline = time.monotonic()+5
        while not (arm_pub.get_num_connections() and hand_pub.get_num_connections()):
            session.check()
            feed.current()
            if time.monotonic() > deadline:
                raise RuntimeError('No arm/hand command subscribers')
            time.sleep(.05)
        current = start.copy()
        np.save(session.directory/'requested_q.npy', poses)
        for i, target in enumerate(poses):
            session.capture(False)
            seconds = replay.duration(current, target, args.first_move_s if i == 0 else args.move_s, args.max_joint_speed_deg)
            print(f'[replay] {i+1}/{len(poses)}: move {seconds:.1f}s', flush=True)
            t0 = time.monotonic()
            while True:
                session.check()
                phase = min((time.monotonic()-t0)/seconds, 1.)
                send(current+.5*(1-math.cos(math.pi*phase))*(target-current))
                if phase >= 1:
                    break
                time.sleep(1/60)
            current = target.copy()
            count = len(session.samples)
            session.capture(True, i, target.copy())
            deadline = time.monotonic()+args.settle_s+args.hold_s+args.capture_extra_s
            while len(session.samples) == count and time.monotonic() < deadline:
                session.check()
                send(target)
                time.sleep(1/60)
            session.capture(False)
            if np.max(np.abs(feed.current()-target)) > math.radians(args.capture_tolerance_deg):
                raise RuntimeError('Commanded pose not reached within capture tolerance')
            if len(session.samples) == count:
                session.skipped.append(i)
                print(f'[replay] SKIP {i+1}: no stable visible tag; continuing checked path', flush=True)
    finally:
        # Preserve original replay behavior: measured-position hold, no return move.
        try:
            if arm_pub.get_num_connections():
                send(feed.current(), guard=False)
        except Exception as exc:
            print(f'[replay] Final measured-position hold unavailable: {exc}', flush=True)
        arm_pub.unregister()
        hand_pub.unregister()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    for mode in ('record', 'replay', 'offline'):
        s = sub.add_parser(mode)
        s.add_argument('--urdf', default=str(cal.URDF))
        s.add_argument('--ee-link', '--ee_link', default='iiwa14_link_7')
        s.add_argument('--view-port', '--view_port', type=int, default=8111)
        s.add_argument('--view-hz', type=float, default=10.)
        s.add_argument('--cloud-hz', type=float, default=5.)
        s.add_argument('--stride', type=int, default=4)
        s.add_argument('--max-depth', type=float, default=2.5)
        s.add_argument('--width', type=int, default=960)
        s.add_argument('--height', type=int, default=540)
        s.add_argument('--calibration', help='Initial display guess only; does not seed or constrain the solve')
        s.add_argument('--cam-xyz', type=float, nargs=3, default=cal.CAM_GUESS_XYZ)
        s.add_argument('--look-at', type=float, nargs=3, default=cal.CAM_GUESS_LOOK_AT)
        s.add_argument('--out-dir', '--out_dir', default='outputs/camera_calibration_visual')
        if mode == 'offline':
            s.add_argument('--samples', required=True, type=Path)
            s.add_argument('--exit-when-done', action='store_true')
            s.add_argument('--use-saved-fits', action='store_true', help='Inspect adjacent estimate_history.jsonl without rerunning solves')
            s.add_argument('--fixed-test-poses', type=int, default=0,
                           help='Reserve this many fixed test poses and refit training prefixes for a learning curve (e.g. 10)')
            s.add_argument('--first-pose-test', action='store_true',
                           help='Also show a learning curve with only the first recorded pose held out')
            continue
        s.add_argument('--serial-number', '--serial_number', required=True)
        s.add_argument('--camera-resolution', '--camera_resolution', default='HD1080', choices=['HD1080', 'HD720', 'VGA'])
        s.add_argument('--camera-upsidedown', '--camera_upsidedown', action='store_true')
        s.add_argument('--exposure', type=int, default=25)
        s.add_argument('--gain', type=int, default=40)
        s.add_argument('--no-video', dest='video', action='store_false')
        s.add_argument('--settle-s', type=float, default=.7)
        s.add_argument('--hold-s', type=float, default=1.)
        s.add_argument('--still-tol-deg', type=float, default=.05)
        s.add_argument('--min-move-deg', type=float, default=3.)
        if mode == 'record':
            s.add_argument('--tag-size', '--tag_size', required=True, type=float)
            s.add_argument('--tag-id', '--tag_id', type=int, default=0)
        else:
            s.add_argument('--poses', required=True, type=Path)
            modes = s.add_mutually_exclusive_group()
            modes.add_argument('--execute', action='store_true')
            modes.add_argument('--dry-run', action='store_true')
            s.add_argument('--start-q', nargs=7, type=float)
            s.add_argument('--first-move-s', type=float, default=10.)
            s.add_argument('--move-s', type=float, default=4.)
            s.add_argument('--max-joint-speed-deg', type=float, default=10.)
            s.add_argument('--max-tracking-error-deg', type=float, default=10.)
            s.add_argument('--capture-tolerance-deg', type=float, default=1.)
            s.add_argument('--capture-extra-s', type=float, default=1.)
            s.add_argument('--table-z', type=float, default=.53)
            s.add_argument('--table-box', nargs=4, type=float, default=[-.6, .6, -1.4, -.55])
            s.add_argument('--min-clear', type=float, default=0.)
    return p


def main():
    p = parser()
    args = p.parse_args()
    for name in ('view_hz', 'cloud_hz', 'stride', 'width', 'height', 'max_depth', 'tag_size', 'settle_s',
                 'hold_s', 'still_tol_deg', 'min_move_deg', 'first_move_s', 'move_s', 'max_joint_speed_deg',
                 'max_tracking_error_deg', 'capture_tolerance_deg', 'capture_extra_s'):
        if hasattr(args, name) and (not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            p.error(f'{name} must be positive and finite')
    if args.mode == 'replay':
        if args.first_move_s < 10 or args.move_s < 4:
            p.error('First move must be >= 10s and subsequent moves >= 4s')
        if not np.isfinite(args.min_clear) or args.min_clear < 0:
            p.error('min-clear must be finite and nonnegative')
    if args.mode == 'offline':
        if args.fixed_test_poses < 0:
            p.error('fixed-test-poses must be nonnegative')
        run_offline(args)
    else:
        run_live(args)


if __name__ == '__main__':
    main()
