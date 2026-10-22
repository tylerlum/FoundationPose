"""Offline regression checks. No ROS, camera, network, or motion publishers."""
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import io

import numpy as np
from scipy.spatial.transform import Rotation

import calibrate_camera_visual as vis


def synthetic(n=16):
    rng = np.random.default_rng(10)
    X = vis.cal.from_xyz_rpy([-.1, -.8, .9], [110, 10, 175])
    Y = vis.cal.from_xyz_rpy([.01, 0, .24], [10, 20, 30])
    A = [vis.cal.make(rng.normal(0, .35, 3), rng.normal(0, .12, 3)) for _ in range(n)]
    B = [vis.cal.inv(X) @ a @ Y for a in A]
    return np.array(A), np.array(B), X, Y


class CalibrationVisualTests(unittest.TestCase):
    def test_intrinsics_export_preserves_recorded_K_and_labels_scaled_sizes(self):
        import json
        K = np.array([[528.8661499, 0., 486.25753784], [0., 528.8661499, 276.28417969], [0., 0., 1.]])
        with TemporaryDirectory() as tmp:
            vis.save_camera_intrinsics(tmp, K, 960, 540, '27432424', 'HD1080')
            np.testing.assert_array_equal(np.load(Path(tmp)/'K_960x540.npy'), K)
            np.testing.assert_allclose(np.loadtxt(Path(tmp)/'K_1920x1080.txt'), np.diag([2, 2, 1]) @ K)
            np.testing.assert_allclose(np.loadtxt(Path(tmp)/'K_160x90.txt'), np.diag([1/6, 1/6, 1]) @ K)
            metadata = json.loads((Path(tmp)/'camera_intrinsics.json').read_text())
            self.assertEqual(metadata['serial_number'], '27432424')
            self.assertEqual(metadata['calibration_image_size'], [960, 540])
            self.assertEqual(metadata['paired_extrinsics_file'], 'T_RC.txt')

    def test_fixed_test_split_and_frozen_transform_scoring(self):
        train, test = vis.fixed_validation_split(48, 10)
        self.assertEqual(len(train), 38)
        self.assertEqual(len(test), 10)
        self.assertEqual(len(np.intersect1d(train, test)), 0)
        np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(48))
        # Each test point lies in a different recording-order bin.
        np.testing.assert_array_equal(np.floor(test*10/48), np.arange(10))
        first_train, first_test = vis.fixed_validation_split(48, first_only=True)
        np.testing.assert_array_equal(first_test, [0])
        np.testing.assert_array_equal(first_train, np.arange(1, 48))
        A, B, X, Y = synthetic(48)
        fit = dict(T_RC=X, T_ee_tag=Y)
        perfect = vis.fixed_test_errors(fit, A[test], B[test])
        self.assertLess(perfect['test_rms_mm'], 1e-9)
        self.assertLess(perfect['test_rms_deg'], 1e-9)
        # Move all held-out observations, without letting their common offset
        # be absorbed by re-estimating the mounting or camera transform.
        disturbed = B[test].copy()
        disturbed[:, 0, 3] += .02
        scored = vis.fixed_test_errors(fit, A[test], disturbed)
        self.assertAlmostEqual(scored['test_rms_mm'], 20., places=8)
        self.assertLess(scored['test_rms_deg'], 1e-9)

    def test_live_preview_continues_without_tag_and_saves_empty_run(self):
        class View:
            def __init__(self, *args):
                self.stop_requested = threading.Event()
                self.frames = 0
                self.closed = False
            def update(self, *args):
                self.frames += 1
            def add_result(self, result):
                pass
            def add_prediction(self, row):
                pass
            def close(self):
                self.closed = True
        class Camera:
            K = np.eye(3)
            def latest(self, since_seq=0, timeout=.1):
                time.sleep(.01)
                return since_seq+1, time.time_ns(), np.zeros((10, 10, 3), np.uint8), np.ones((10, 10), np.float32)
        class Feed:
            def snapshot(self, require_hand=False):
                return np.zeros(7), dict(zip(vis.cal.ARM_JOINTS, np.zeros(7))), True, False
        args = SimpleNamespace(mode='record', settle_s=.1, hold_s=.1, still_tol_deg=.05,
                               min_move_deg=3., tag_size=.055, tag_id=0, video=False,
                               ee_link='iiwa14_link_7', urdf=str(vis.cal.URDF), serial_number='fake')
        with TemporaryDirectory() as tmp, patch.object(vis, 'CalibrationView', View), \
                patch.object(vis.cal, 'detect_tag', return_value=(None, None)) as detect:
            session = vis.LiveSession(args, vis.cal.Urdf(), Feed(), Camera(), Path(tmp))
            time.sleep(.25)
            session.check()
            session.close()
            self.assertGreater(session.view.frames, 5)
            self.assertTrue(session.view.closed)
            self.assertFalse(session.worker.process.is_alive())
            self.assertTrue(all(call.kwargs['depth_refine'] is False for call in detect.call_args_list))
            with np.load(Path(tmp)/'samples.npz') as data:
                self.assertEqual(data['q'].shape, (0, 7))
                self.assertGreater(len(data['stream']), 5)

    def test_solver_preserves_original_math_and_recovers_known_pose(self):
        A, B, X, _ = synthetic()
        # Add small observation noise so uncertainty is nonzero.
        rng = np.random.default_rng(20)
        for b in B:
            b[:3, 3] += rng.normal(0, .0003, 3)
        expected = vis.cal.solve(A, B)
        actual = vis.solve_snapshot(A, B)
        np.testing.assert_allclose(actual['T_RC'], expected['T_RC'], atol=1e-10)
        self.assertLess(vis.pose_delta(X, actual['T_RC'])[0], 3.)
        self.assertTrue(actual['uncertainty']['valid'])
        cov = actual['uncertainty']['covariance_base']
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-12)

    def test_unseen_prediction_freezes_both_transforms_and_rejects_leakage(self):
        A, B, X, Y = synthetic()
        fit = dict(T_RC=X, T_ee_tag=Y, n=10)
        good = vis.score_unseen(fit, A[10], B[10], 11)
        self.assertLess(good['mm'], 1e-8)
        self.assertLess(good['deg'], 1e-8)
        shifted = B[10].copy(); shifted[0, 3] += .02
        bad = vis.score_unseen(fit, A[10], shifted, 11)
        self.assertAlmostEqual(bad['mm'], 20., places=8)
        np.testing.assert_array_equal(fit['T_ee_tag'], Y)
        with self.assertRaises(ValueError):
            vis.score_unseen(fit, A[9], B[9], 10)
        self.assertIsNone(vis.score_unseen(None, A[0], B[0], 1)['fit_n'])
        self.assertIs(vis.previous_fit([dict(error='failed', n=11), fit, dict(T_RC=X, n=12)], 11), fit)

    def test_prediction_rolling_window_uses_only_recent_valid_predictions(self):
        rows = [dict(fit_n=None, sample_n=1)] + [dict(fit_n=i, mm=float(i), deg=2.*i) for i in range(1, 13)]
        summary = vis.rolling_prediction_rms(rows)
        self.assertEqual(summary['count'], 10)
        self.assertAlmostEqual(summary['mm'], np.sqrt(np.mean(np.arange(3, 13)**2)))

    def test_identical_poses_have_no_finite_uncertainty_region(self):
        A, B, X, Y = synthetic(1)
        u = vis.fit_uncertainty(np.repeat(A, 8, axis=0), np.repeat(B, 8, axis=0),
                                dict(T_RC=X, T_ee_tag=Y))
        self.assertFalse(u['valid'])
        self.assertLess(u['rank'], 12)

    def test_single_axis_poses_are_underconstrained(self):
        _, _, X, Y = synthetic(1)
        A = [vis.cal.make([0, 0, t], [.1, .2, .3]) for t in np.linspace(-.5, .5, 15)]
        B = [vis.cal.inv(X) @ a @ Y for a in A]
        u = vis.fit_uncertainty(A, B, dict(T_RC=X, T_ee_tag=Y))
        self.assertFalse(u['valid'])

    def test_sampler_requires_fresh_still_contiguous_tag_and_new_pose(self):
        sampler = vis.PoseSampler(.2, .3, .05, 3)
        q, T = np.zeros(7), np.eye(4)
        for t in np.arange(0, .5, .05):
            sample, _ = sampler.update(float(t), q, T, True)
            self.assertIsNone(sample)
        sampler.update(.5, q, None, True)  # A lost tag resets averaging.
        sample = None
        for t in np.arange(.55, 1.0, .05):
            candidate, _ = sampler.update(float(t), q, T, True)
            sample = candidate if candidate is not None else sample
        self.assertIsNotNone(sample)
        for t in np.arange(1., 2., .05):
            self.assertIsNone(sampler.update(float(t), q, T, True)[0])
        q2 = q.copy(); q2[0] = .1
        for t in np.arange(2., 3., .05):
            self.assertIsNone(sampler.update(float(t), q2, T, False)[0])
        for t in np.arange(3., 4., .05):
            self.assertIsNone(sampler.update(float(t), q2, T, True, enabled=False)[0])
        samples = [sampler.update(float(t), q2, T, True)[0] for t in np.arange(4., 5., .05)]
        self.assertEqual(sum(s is not None for s in samples), 1)

    def test_cloud_and_frustum_use_actual_principal_point(self):
        K = np.array([[10, 0, 1.5], [0, 20, .8], [0, 0, 1.]])
        depth = np.ones((3, 4), np.float32)
        depth[0, 0] = np.nan
        rgb = np.zeros((3, 4, 3), np.uint8)
        xyz, _ = vis.cloud_points(rgb, depth, K, 1)
        self.assertEqual(len(xyz), 11)
        uv = xyz[:, :2]/xyz[:, 2:]*np.array([10, 20])+np.array([1.5, .8])
        np.testing.assert_allclose(uv, np.round(uv), atol=1e-6)
        frustum = vis.optical_frustum(K, 4, 3)
        corner = frustum[0, 1]
        np.testing.assert_allclose(K @ corner / corner[2], [0, 0, 1.], atol=1e-12)

    def test_joint_order_is_by_name_not_array_position(self):
        msg = SimpleNamespace(name=[f'iiwa_joint_{i}' for i in range(7, 0, -1)], position=list(range(7, 0, -1)))
        np.testing.assert_equal(vis.canonical_arm(msg), np.arange(1, 8))
        msg.name[0] = 'bad'
        with self.assertRaises(ValueError):
            vis.canonical_arm(msg)

    def test_replay_cancellation_creates_no_publishers(self):
        session = SimpleNamespace(check=lambda: None)
        feed = SimpleNamespace(current=lambda: np.zeros(7))
        def forbidden(*args, **kwargs):
            self.fail('Publisher created before replay confirmation')
        ros = SimpleNamespace(Publisher=forbidden)
        with patch.object(vis.select, 'select', return_value=([True], [], [])), \
                patch.object(vis.sys, 'stdin', io.StringIO('CANCEL\n')):
            vis.execute_replay(SimpleNamespace(), None, np.zeros((3, 7)), np.zeros(7), feed, session, ros)

    def test_replay_start_validation_rejects_bad_current_pose(self):
        robot = vis.cal.Urdf()
        for q in (np.full(7, np.nan), np.full(7, 100.), np.zeros(6)):
            with self.assertRaises(ValueError):
                vis.validate_start(robot, q)

    def test_worker_finishes_latest_snapshot_without_blocking_submit(self):
        A, B, _, _ = synthetic()
        worker = vis.SolverWorker()
        try:
            start = time.monotonic()
            for n in range(3, 17):
                worker.submit(A[:n], B[:n])
            self.assertLess(time.monotonic()-start, .5)
            results = []
            deadline = time.monotonic()+15
            while time.monotonic() < deadline and not any(s['n'] == 16 for s in results):
                results.extend(worker.poll())
                time.sleep(.02)
            self.assertTrue(any(s['n'] == 16 and 'error' not in s for s in results))
        finally:
            worker.close()
        self.assertFalse(worker.process.is_alive())

    def test_worker_recovers_from_native_process_exit(self):
        A, B, _, _ = synthetic()
        worker = vis.SolverWorker()
        try:
            worker.submit(A, B)
            worker.process.terminate()  # Simulate a native crash, not a Python exception.
            worker.process.join(timeout=2)
            results = []
            deadline = time.monotonic()+15
            while time.monotonic() < deadline and not any('T_RC' in s for s in results):
                results.extend(worker.poll())
                time.sleep(.02)
            self.assertEqual(worker.restarts, 1)
            self.assertTrue(any('error' in s for s in results))
            self.assertTrue(any(s['n'] == len(A) and 'T_RC' in s for s in results))
        finally:
            worker.close()


if __name__ == '__main__':
    unittest.main()
