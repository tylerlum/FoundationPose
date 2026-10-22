# The robot's trackers (bohg-ws-16)

This branch adds the FoundationPose trackers and camera tools that run on the perception PC of the Kuka iiwa 14 +
Sharpa robot (bohg-ws-16, RTX 4090), as of 17 Oct 2026. They come from depthbasedRL branch
`2026_10_17_HardwareChangesPose` (commit 1c51281), where they live in `third_party/FoundationPose/`. That clone has
no git remote, so this branch is the shared copy.

In depthbasedRL the scripts run from the depthbasedRL root (`python third_party/FoundationPose/<script>.py ...`).
Here the same files are at the repo root: read `third_party/FoundationPose/<script>.py` in the docs as `<script>.py`.

## What runs on the robot

```
 ZED (opened directly, pyzed) --> live_tracking_new_gui.py (FoundationPose; viser panel) --> /robot_frame/current_object_pose
                                       | auto-record while rl_policy_node.py publishes /iiwa/joint_cmd
                                       v
                                  <save_dir>/<take>/ (videos, poses, K.txt; viz_data.npz with --viz_data)
                                       v
                                  live_tracking_viz_replay*.py <take>   (replay in the browser)
```

| Script | What it does | Viser port |
|---|---|---|
| `live_tracking_new_gui.py` | The tracker in use (one object). ZED in, pose out on ROS (`--ros`), text-prompted masks (SAM3, `--text_prompt`) or a box / points, browser panel (re-register, save, swap object mid-run), auto-record | 8090 |
| `live_tracking_new_gui_2_objects.py` | The same for two objects (a tool and a workpiece), plus the live 3D / 2D view | 8100 |
| `live_tracking_new_autorecord.py` | Policy-gated recording, no GUI | |
| `live_tracking_new.py` | The original tracker | |
| `live_tracking_live_view.py` | The panel's live views: the point cloud and the meshes at their tracked poses (3D), the image with the meshes drawn on (2D) | |
| `live_tracking_viz_replay.py`, `live_tracking_viz_replay_2_objects.py` | Replay a saved take (`viz_data.npz`) in the browser: point cloud, tracked mesh and axes, frame slider | 2 objects: 8120 |
| `live_pointcloud_viser.py` | The live ZED point cloud, with the same capture and depth backend as the trackers | |
| `calibrate_camera.py` | T_RC (camera -> robot base): by eye with sliders over the live cloud, or with an AprilTag on the hand (collect + solve) | 8110 |
| `calibrate_camera_visual.py`, `replay_calibration.py` | Calibration convergence viewer; replay saved calibration poses (dry run by default) | |
| `apriltag_tracker.py`, `live_tracking_apriltag.py`, `furniture_bench_apriltag/` | AprilTag tracking instead of FoundationPose | |
| `text_mask.py`, `point_prompt_mask.py` | Masks from a text prompt (SAM3) or from a box / points | |
| `fp_trt_runner.py`, `scripts/export_refine_onnx.py`, `scripts/build_trt_engine.py` | Optional TensorRT refiner (`--fp_refiner_trt <engine>`) | |
| `make_colored_screwdriver.py` | Print parts and the matching tracking mesh for the four-colour fin screwdriver | |
| `*_TYLER_hacks.py` | Experiments; not used on the robot | |

Commands, flags and measured numbers: [docs/bohg-ws-16/RUN.md](docs/bohg-ws-16/RUN.md) (start here),
[LIVE_TRACKING_RUNBOOK.md](docs/bohg-ws-16/LIVE_TRACKING_RUNBOOK.md), [LOCAL_SETUP.md](docs/bohg-ws-16/LOCAL_SETUP.md)
(how the bohg-ws-16 environment was built), [live_tracking_troubleshooting.md](docs/bohg-ws-16/live_tracking_troubleshooting.md).

## Running these outside depthbasedRL

- **Meshes:** the docs' `--mesh_path assets/urdf/...` are depthbasedRL paths. Pass your object's mesh (in metres, in
  the policy's object frame).
- **Camera calibration (T_RC):** `--calibration <T_RC.txt>`. Without it the replay helpers read
  `kuka_sharpa_station/calibrations/current.txt` from a clone next to this repo (or next to depthbasedRL), or from
  `$KUKA_SHARPA_STATION`; else they warn and use a built-in 13 Oct 2026 calibration.
- **calibrate_camera.py** needs the robot URDF: `--urdf`, or `ROBOT_URDF=<SimToolReal repo>/assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf`.
- **Fast-FoundationStereo depth** (`--depth_backend fast_fs`) expects `Fast-FoundationStereo/` next to this folder
  (its weights and TensorRT engines inside). `--depth_backend zed_neural` needs only the ZED SDK.
- **Environment:** bohg-ws-16 runs these in a Python 3.11 uv venv with torch 2.7 + CUDA 12.6 and rospy from
  rospypi (docs/bohg-ws-16/LOCAL_SETUP.md); this repo's `requirements.txt` pins the standalone install. SAM3 (`facebook/sam3`) is a gated Hugging Face model: request
  access, or use the box / point masks.
- **ZED:** the trackers open the camera directly (`pyzed`), so no other process can hold it. On the robot, run them
  on the perception PC only.

## How the core files were merged (21 Oct 2026)

The bohg-ws-16 copy and this fork had both changed some FoundationPose files. Both sets of changes are kept:

| File | Kept from this fork | Added from bohg-ws-16 |
|---|---|---|
| `estimater.py` | `max_initial_candidates`, rotation-grid arguments (`extract_poses.py`) | each instance writes its mesh to its own temp folder (shared `/tmp` clash) |
| `learning/training/predict_pose_refine.py` | `max_pose_batch` chunking (`extract_poses.py`) | `trt_engine_path` (TensorRT refiner); with `max_pose_batch=None` the loop is as on bohg-ws-16 (no extra `empty_cache`) |
| `learning/training/predict_score.py` | `max_pose_batch` | (no change) |
| `learning/datasets/h5_dataset.py` | | crop -> full image -> crop xyz in chunks, so register fits in memory at 1920x1080 |
| `generate_mask.py` | | SAM weights cached per process (no reload on every re-register) |
| `live_tracking_utils.py` | | faster saving (uncompressed npz option, x264 preset / CRF, optional overlay video) |
| `live_tracking_replay.py` | | T_RC from the station repo, `first_free_port` (stable viser ports); the station path now works in both layouts |
| `live_tracking_2_objects.py`, `live_tracking_utils_2_objects.py` | paths relative to this folder (no `$DEPTHBASEDRL_ROOT`) | (otherwise the same) |
| `calibrate_camera.py` | | new file; `ROBOT_URDF` override added here |
| `requirements.txt`, `.gitignore` | the standalone pins | a note on the bohg-ws-16 environment; its ignore lines |

Before this merge, `live_tracking.py` and `live_tracking_2_objects.py` on branch
`2026-07-30_TwoObjectTracking_FrameBufferCap` passed `trt_engine_path=` to a refiner that did not take it; they now
run.

## Tested (bohg-ws-19, RTX 4090, 21 Oct 2026)

- Every `.py` file compiles (the FoundationPose conda env, Python 3.9).
- The SimToolReal deploy kit's FoundationPose replay test (`scripts/kit/run_fp_replay_test.sh`: claw hammer, rendered
  ZED frames, 20 s) with this branch as `FOUNDATIONPOSE_ROOT`: 1.8 mm / 1.3 deg median error, 0 % lost, 21.7 Hz;
  the same as the kit's reference run on the previous branch (1.8 mm / 1.3 deg, 0 %, 19.5 Hz). This runs
  `estimater.py`, `predict_pose_refine.py`, `predict_score.py` and `h5_dataset.py`.
- `live_tracking_viz_replay.py` on a synthetic `viz_data.npz`: loads it and serves the viser page.
- `test_calibrate_camera_visual.py`: 15 / 15 pass (with `ROBOT_URDF` set).

Not tested here: anything that opens the ZED (no camera on this machine), SAM3, the TensorRT engines and the
Fast-FoundationStereo backend. On bohg-ws-16 the original files in depthbasedRL are unchanged.
