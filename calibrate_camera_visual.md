# Visual AprilTag camera calibration

`calibrate_camera_visual.py` is a separate entry point. The original
`calibrate_camera.py` and `replay_calibration.py` are unchanged. It imports their
solver, FK, camera capture, replay pose validation, trajectory timing, and sampled
table-clearance checks. No environment changes are needed in the existing `drl` venv.

## Viewer

Default URL: `http://localhost:8111`. For a remote browser:

```bash
ssh -L 8111:localhost:8111 tylerlum@bohg-ws-16
```

The robot base is the fixed world frame. `T_RC` maps camera optical coordinates
(x right, y down, z forward) into the robot base, in metres. This is not the
simulator environment frame. The default camera pose is explicitly an uncalibrated
guess; `--calibration PATH` can supply a better display guess, but does not constrain
or initialize the numerical solve.

- Robot joints and camera images update at up to 10 Hz, including while moving or
  when the tag is absent. Missing hand states are labeled and displayed at zero.
- A subsampled live depth cloud updates at 5 Hz. The cloud and calibrated frustum
  use the SDK's rectified-left intrinsics, including the actual principal point.
- Green tag outline is observed. Orange is predicted using FK and the fitted
  camera/tag mounting transforms. The 3D frames show the same two estimates.
- Each accepted pose triggers a background-process solve. At most one solve is
  running with bounded pending work. The camera and replay motion do not wait for it.
  Native numerical libraries use one thread in that worker. An unexpected worker
  exit is logged, keeps the previous valid display, and retries the latest snapshot
  in a fresh process. Recovery is limited to three restarts per run; repeated failures
  stop the session with checkpointed data intact.
- The panel shows RMS/max translation and orientation residuals, solve duration,
  rotation spread, camera estimate change, and a residual chart.
- Previous camera frustums and a connected blue trail show convergence from the
  gray initial display guess. Disable **Follow latest estimate** and use
  **Samples used by estimate** to inspect history. Offline inspection also changes
  the recorded robot pose and image to the corresponding sample.
- **Predict new poses before fitting them** scores each newly accepted measurement
  with the latest completed fit that has not seen it. Both the camera transform and
  the end-effector/tag transform are frozen. The panel identifies the sample count
  used by that fit, shows the new position/orientation error, and plots individual
  errors plus a rolling RMS of the last 10 predictions. Position and orientation
  have separate interactive Plotly charts with linear axes, labeled ticks and
  units, gridlines, and legends. Hover for values, drag to zoom into later samples,
  and double-click to restore the full range. Zoom survives incoming updates.
  The fitting-residual plots use the same presentation. Plots update on new
  samples/results, not on every camera frame.
  Different arriving poses have different difficulty, so this is a prospective
  diagnostic, not a fixed test set or a promise of monotonic improvement. A slow
  solver may use a fit older than N-1; the actual count is shown and saved.
- Display checkboxes skip image, robot, or cloud update work. The uncertainty
  magnification defaults to 1 (actual scale), and is explicitly adjustable.

## Uncertainty interpretation

The translucent orange ellipsoid is approximate 95% position fit uncertainty.
The orange cone shows approximate 95% viewing-direction fit uncertainty. Full
rotation uncertainty, including roll, is also printed numerically. These are
local linearized least-squares estimates at the robust solver's optimum, using
the original residual scaling (0.05 m/rad) and residual variance. The full
12-parameter covariance is inverted before extracting the camera marginal, so
the unknown end-effector/tag mounting transform is included as a nuisance variable.

This is a diagnostic approximation, not measured absolute accuracy or a robust
confidence interval. It excludes tag-size error, robot-model bias, synchronization
bias, correlated samples, and depth error. Rank-deficient fits have no finite
ellipsoid. Very broad estimates (over 30 degrees or 0.5 m radius) are explicitly
marked outside the useful local approximation and the shapes are hidden.

Low residuals alone do not establish readiness. Inspect diversity, convergence,
and new independent validation poses. The three rotation-spread values summarize
principal directions of collected wrist rotations, not necessarily robot XYZ.
Depth is used only for the displayed cloud. AprilTag depth refinement remains OFF.

## Passive collection

The robot must publish fresh `/iiwa/joint_states`; `/sharpa/joint_states` is optional
for visualizing the hand. This mode creates no command publishers.

```bash
source /home/tylerlum/.drl_env.sh
drl
python third_party/FoundationPose/calibrate_camera_visual.py record \
  --serial-number 27432424 --camera-resolution HD1080 \
  --tag-size 0.055 --tag-id 0 --view-port 8111 \
  --out-dir /juno/u/tylerlum/camera_2_calibration_27432424
```

Hold each new pose for about 2 seconds. Samples require fresh correctly named
joint states, a stationary arm, continuous tag visibility during averaging, and
at least 3 degrees of joint change from the last accepted pose. Tag loss resets
the averaging window. The GUI Finish button or one Ctrl-C saves and finishes.

## Automatic replay, without changing the original replay script

First inspect the exact pose file without connecting to ROS or the camera:

```bash
python third_party/FoundationPose/calibrate_camera_visual.py replay \
  --poses /juno/u/tylerlum/camera_2_calibration_27432424/record_20261017_172107/samples.npz \
  --serial-number 27432424 --dry-run
```

The saved 51-pose sequence passes the existing sampled table check at a minimum
of 1.28 cm clearance with zero hand joints in the offline check. Nominal minimum
duration is 302.4 seconds; timeouts for missed tags can extend it. An offline check
without `--start-q` cannot check the current-to-first path.

For execution, arm position control and the Sharpa state/command nodes must be
running. Passive FRI recording mode cannot execute joint targets. The script checks
fresh arm/hand states and the current-to-first path before asking for `REPLAY`.
Neither commands nor publishers are created before that confirmation.

```bash
python third_party/FoundationPose/calibrate_camera_visual.py replay \
  --poses /juno/u/tylerlum/camera_2_calibration_27432424/record_20261017_172107/samples.npz \
  --serial-number 27432424 --camera-resolution HD1080 \
  --view-port 8111 --hold-s 0.5 --execute \
  --out-dir /juno/u/tylerlum/camera_2_calibration_27432424
```

This starts the display from the generic guess, without loading an earlier
calibration. The solver always starts from the samples' closed-form estimate,
not the display guess. The 0.5-second averaging window normally captures 16 camera
frames after 0.7 seconds of settling. With these holds, the nominal minimum replay
duration is about 277 seconds; missed tags and scheduling add time.

First move is at least 10 seconds; subsequent moves at least 4 seconds. The existing
cosine interpolation, speed limit, tracking-error checks, and final measured-position
hold are retained. A missing tag skips that sample after a bounded hold and continues
the checked trajectory. Stale joints, camera/recording errors, or tracking failure stop
the replay. There is no automatic return move. Model/table checks do not model the
camera, bowl, people, cables, or self-collision. Inspect the physical paths, including
the start transition. A path jogged joint by joint is not the same as straight joint
interpolation between its recorded stops.

## Offline convergence inspection

No ROS, camera connection, or motion. Existing recordings need no recollection:

```bash
python third_party/FoundationPose/calibrate_camera_visual.py offline \
  --samples /juno/u/tylerlum/camera_2_calibration_27432424/record_20261017_172107/samples.npz \
  --view-port 8113 --out-dir outputs/calibration_convergence
```

It solves chronological prefixes of 3 through N samples, shows each estimate, then
leaves the viewer open. `--exit-when-done` closes it automatically for verification.
It uses the exact saved `T_base_ee` when available. Saved raw images are loaded by
their exact filename, never by a recursive directory scan. No cloud is fabricated:
the earlier recorder did not save depth, so this offline mode has no point cloud.
For repeated analysis of network-resident data, stage the known files locally first.
`--use-saved-fits` instead reads the adjacent `estimate_history.jsonl`, verifies its
fits against the samples, and displays the completed fit history without solving
again. When the live `prediction_history.jsonl` is present, it validates and shows
those exact pre-update errors and the fit counts that were available then. Otherwise
offline prediction errors are reconstructed from chronological prefix fits.

### Fixed-test-set learning curve

Add `--fixed-test-poses 10` in offline mode to show a separate learning curve near
the top of the viewer. It reserves one sample at the center of each of 10 equal
recording-order bins, without looking at fit errors. All reserved samples are
excluded from every fit in this analysis. The other samples remain chronological.
It fits 5, 8, 10, 15, 20, 25, 30, 35, and all available training samples (omitting
counts beyond the dataset), then evaluates each frozen camera/mounting transform
pair on exactly the same reserved poses. Both plotted axes are linear and the RMS
points are unsmoothed. Failed fits are reported and leave gaps. The 3D scene still
shows the original full-data fit, not these validation-only subset fits.
The initial plot view is explicitly labeled **8+ training poses (zoom)** so an
unstable 5-pose fit does not flatten all later errors on a linear axis. Its error
is printed above the plot; select **All fits** to include it. Every fit remains
in the results table and saved JSON.

For the 48-sample run above, a locally staged copy can be analyzed with:

```bash
python third_party/FoundationPose/calibrate_camera_visual.py offline \
  --samples /tmp/camera2_guess_run_preview/samples.npz --use-saved-fits \
  --fixed-test-poses 10 --view-port 8111 --out-dir /tmp/camera2_fixed_test
```

The output `fixed_test_learning_curve.json` saves the exact split, transforms,
per-test-pose errors, RMS values, and fit failures. This is a held-out pose
consistency check within one recording. Nearby poses and common measurement/model
biases remain correlated; the curve is not an absolute-accuracy measurement.

On `replay_visual_20261017_175834`, the fixed test sample numbers are
3, 8, 13, 17, 22, 27, 32, 37, 41, and 46. The 5-pose fit is unstable (1406 mm /
62.40 deg test RMS). With 10, 15, 20, and 38 training poses, test RMS is
14.98/1.41, 4.75/0.92, 3.61/0.88, and 2.98/0.91 mm/deg, respectively.
The 8-pose fit has lower position error than the 10-pose fit (8.96 mm / 1.65 deg),
so the raw curve is not monotonic. All 14 regression tests pass,
including split exclusion and frozen-transform scoring. The saved report also
verifies every fit excludes all 10 reserved poses.

`--first-pose-test` adds a separate curve that holds out only recorded sample 1
and trains on chronological prefixes of samples 2 through N. It can be used alone
or together with `--fixed-test-poses 10`. Results are saved to
`first_pose_learning_curve.json`. Its default plot range includes every fit. One
pose tests a much narrower part of the workspace and can have low error even when
the camera and mounting transforms remain poorly constrained elsewhere.
For this run it gives 1.46 mm / 0.39 deg with 5 training poses, 0.60 mm / 0.37 deg
with 8, and 1.62 mm / 0.78 deg with all other 47 poses. This single-pose curve does
not steadily decrease. Both learning-curve reports are also stored beside this
run's `samples.npz` on Juno. All 15 regression tests pass, and the first-pose report
was checked to exclude sample 1 from every one of its 11 fits.

## Saved output and tests

Each live run creates a new timestamped directory, never overwriting its input:

- `samples.npz`: accepted q, `T_cam_tag`, explicit FK `T_base_ee`, K, tag size/id,
  camera serial, joint names, source pose indices, skipped indices, and the per-frame
  timestamp/joint/tag stream. Checkpointed atomically after each sample and on exit.
- `sample_NN_raw.png`, optional `record.mp4` (nominal 10 fps, half resolution).
- `estimate_history.jsonl`: transforms, per-pose errors, fit metrics, solve timing,
  covariance and observability diagnostics for each completed solve, including failures.
- `prediction_history.jsonl`: new sample number, earlier fit sample count and time,
  position/orientation error before refitting, and whether that fit was constrained.
- `config.json`; replay also writes `requested_q.npy`.
- `camera_intrinsics.json` and `K_WIDTHxHEIGHT.txt` / `.npy`: the exact rectified-left
  K used by this recording, with camera serial, pixel dimensions, rotation,
  provenance, and the paired `T_RC.txt` convention. The visual recorder now writes
  these automatically at startup. Scaled 160x90 matrices and, for HD1080 recordings,
  1920x1080 matrices are explicitly labeled as derived full-frame scalings. Raw-lens
  distortion coefficients were not logged; rectified images use zero distortion.
- `T_RC.txt`, `T_RC.npy`, `T_ee_tag.txt`: final fit, if enough samples yield a valid solve.

Depth is not saved in these bundles. Displaying a live cloud does not enable depth
refinement or persistent depth logging. Original source sample files are read-only.

Intrinsics were also exported beside the existing dated calibrations. Camera-specific
indexes on Juno list exact matched intrinsics/extrinsics paths:

- Camera 15107: `/juno/u/tylerlum/camera_calibration/camera_15107_calibrations.json`,
  including `replay_20261016_161152/camera_intrinsics.json`.
- Camera 27432424: `/juno/u/tylerlum/camera_2_calibration_27432424/camera_27432424_calibrations.json`,
  including `record_20261017_172107`, `replay_visual_20261017_174836`, and
  `replay_visual_20261017_175834`, each with its own `camera_intrinsics.json`.

Use the K from the same recording as T_RC. For example, camera 2's earlier manual
recording has fx=529.416 at 960x540, while both subsequent replays have fx=528.866.
The old NPZ format did not store serial or capture mode; backfilled JSONs explicitly
identify those fields as recording-session provenance rather than NPZ metadata.

Run offline regression checks from the repository root:

```bash
python third_party/FoundationPose/test_calibrate_camera_visual.py -v
```

The checks cover solver parity, known-pose recovery, covariance degeneracy,
missing-tag/stale-joint sampling behavior, continuous preview without a tag,
intrinsics/frame geometry, name-based joint ordering, and asynchronous final-fit delivery.
The tests also force a worker exit to verify crash reporting and recovery, and
check unseen-pose scoring with both transforms frozen and no use of future samples.

Validation on 2026-10-17: 11 offline tests passed, the 51-pose dry-run passed,
and the full saved dataset reproduced 2.10373 mm / 1.207 deg RMS. Browser rendering,
point-cloud updates with synthetic depth, and display-disable controls were checked.
Two native solver-worker SIGSEGVs occurred during development; their library-level
cause was not established. Worker thread limits, diagnostic traces, and bounded
restart/retry handling were added. A subsequent stress test completed 98 successive
prefix fits, and the forced-exit recovery test passed.

Hardware validation: `replay_visual_20261017_174836` under
`/juno/u/tylerlum/camera_2_calibration_27432424` completed the 51-pose sequence with
0.5-second holds, accepted 48 poses, and skipped requested poses 7, 14, and 29.
Final fitting RMS was 1.81552 mm / 0.725718 deg. The result differed from the prior
manual fit by 1.71 mm / 0.11 deg. The solve at 5 samples failed from insufficient
rotation diversity; later fits recovered. There were no worker crashes in this run.
On a common unseen set (accepted poses 21 through 48), fits using the first 10 and
20 poses predicted RMS errors of 13.68 mm / 2.46 deg and 4.77 mm / 0.93 deg,
respectively, with both fitted transforms frozen. These are consistency checks,
not absolute calibration accuracy. The prediction/history update passes 13 offline
tests. The original calibration and replay scripts are unchanged.

The next run, `replay_visual_20261017_175834` in the same camera-2 directory,
loaded no initial calibration file. It accepted 48 poses, skipped requested poses
7, 14, and 41, and finished at 1.97042 mm / 0.685844 deg fitting RMS. Its final
10 pre-update predictions had RMS 1.64836 mm / 0.497649 deg. The final camera pose
was within 0.366 mm / 0.044 deg of the preceding replay's fit. Both command-topic
publisher lists were empty after completion. The new Plotly viewer was checked
in a browser on this run's saved data: all four charts rendered with linear axes,
grids, units, and legends, including at the narrow default panel width. It also
validated the saved live prediction errors against the saved prefix fits.
