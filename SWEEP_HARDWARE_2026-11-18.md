# Sweep task hardware testing, 17-18 Nov 2026 (bohg-ws-16 perception side)

What was run, the exact cameras / calibrations / meshes / commands, what broke, and what to try next.
Written for Kuan (and his agent) to reproduce the perception side and iterate in sim.

Task: the robot (iiwa14 + Sharpa hand) picks up the red brush and sweeps the 29 mm gold cube into the red dustpan.
Perception runs on **bohg-ws-16** (RTX 4090). The policy and planner run on **bohg-ws-19**
(`/home/tylerlum/github_repos/policy_futures_dev`, branch `2026-11-17_HardwareTesting`). ROS master: `http://bohg-ws-2.stanford.edu:11311`.

```
 ZED camera ──USB──> bohg-ws-16: live_tracking_new_gui_2_objects.py (FoundationPose, Fast-FS depth, viser :8100)
                          │  publishes (ROS, master bohg-ws-2)
                          ├── /robot_frame/current_object_pose      PoseStamped  brush, frame robot_frame
                          ├── /robot_frame/current_object_pose_2    PoseStamped  cube
                          ├── /perception/payload/pose_robot        PoseStamped  cube (same msg as _2)
                          ├── /perception/payload/status            String JSON  {valid, stamp, reason, ...}
                          └── /camera_frame/current_object_pose[_2] PoseStamped  camera frame
                          ▼
 bohg-ws-19: policy_futures_dev robot_deploy (sweep policy + planner) ──> /iiwa/joint_cmd ──> robot
 tracker watches /iiwa/joint_cmd to auto-record one "take" per policy rollout
```

## 1. Code, branches, environments

| What | Where | Branch / env |
|---|---|---|
| Live tracker, calibration, tools (this doc) | bohg-ws-16 `/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose` (local-only repo) | `2026-11-18_SweepHardwareTesting`; env: `source /home/tylerlum/.drl_env.sh && drl` |
| Same files, on GitHub | `github.com/tylerlum/FoundationPose` | `2026-11-18_SweepHardwareTesting_bohg-ws-16` (snapshot of the folder above) |
| Payload topics + capture stamps change | commit "Two-object tracker: ZED capture stamps on every pose, and the sweep payload topics" | in both branches |
| Policy / planner / site config | bohg-ws-19 `/home/tylerlum/github_repos/policy_futures_dev` | `2026-11-17_HardwareTesting`, `.venv_release` |
| SAM2 (masks) | bohg-ws-16 `/home/tylerlum/github_repos/segment-anything-2-real-time` | `SimToolReal`, `.venv` (uv, py3.10, torch cu126) |
| SAM3D (metric mesh from one RGB-D frame) | bohg-ws-16 `/home/tylerlum/github_repos/sam-3d-objects` | `SimToolReal`, `.venv311` (uv, py3.11, torch 2.5.1 cu121) |
| Full FoundationStereo (offline depth, not live) | bohg-ws-16 `/home/tylerlum/github_repos/FoundationStereo` | `.venv` (torch 2.4.1 cu121), weights `pretrained_models/pretrained_models/23-51-11` |

Always pin the tracker to the good cores: `taskset -c 0-9,12-31` (faulty-core note in `.drl_env.sh`).

## 2. Cameras: serials, image sizes, intrinsics, extrinsics

Conventions (all files below):
- Camera frame = ZED **rectified left**, OpenCV optical frame: x right, y down, z forward. Zero distortion (rectified).
- `T_RC.txt`: 4x4, `p_robot = T_RC @ p_camera`, metres. Robot frame = iiwa base (`robot_frame`).
- Capture is HD1080 (1920x1080) at 30 FPS; FoundationPose and depth run at **960x540**. Use `K_960x540` with the tracker.
- Copies of every file are in `calibration_snapshots_20261118/` next to this doc.

| Camera | Serial | K at 960x540 (fx=fy, cx, cy) | Current T_RC | Camera position in robot frame | Looks |
|---|---|---|---|---|---|
| **ZED2 (current)** | 27432424 | 528.866, 486.258, 276.284 | `/juno/u/tylerlum/camera_2_calibration_27432424/record_visual_20261118_172707/T_RC.txt` | (-0.090, -1.304, 0.902) | back along the table toward the robot, 25 deg down |
| ZED2 (superseded 16:11, camera since moved) | 27432424 | same | `.../record_visual_20261118_161129/T_RC.txt` | (0.351, -1.232, 0.918) | from front-right toward the robot, 20 deg down |
| ZED1 | 15107 | 692.022, 480.106, 282.767 | `/juno/u/tylerlum/camera_calibration/replay_20261016_161152/T_RC.txt` (16 Oct; verify before trusting) | (-0.379, -0.769, 0.912) | from the robot's left side, 39 deg down |

Current ZED2 `T_RC` (record_visual_20261118_172707):
```
 0.996055  0.011647  0.087968 -0.090268
-0.074377 -0.431090  0.899238 -1.304057
 0.048396 -0.902234 -0.428523  0.901944
 0         0         0         1
```
ZED1 `T_RC` (replay_20261016_161152):
```
-0.364505 -0.570603  0.735900 -0.378759
-0.930532  0.253161 -0.264613 -0.769090
-0.035312 -0.781231 -0.623242  0.912204
 0         0         0         1
```

Calibration quality (passive manual mode, AprilTag 36h11 id 0, 55 mm, on the hand):

| Calibration | Poses | Fit RMS (worst) | Hand rotation spread (deg) | Table check from depth |
|---|---|---|---|---|
| ZED2 172707 (current) | 29 | 1.37 mm / 0.48 deg (2.4 mm / 1.0 deg) | 19.5, 10.1, 6.4 | z 0.5405 m, tilt 0.78 deg |
| ZED2 161129 (superseded) | 22 | 2.36 mm / 0.69 deg (4.4 mm / 1.3 deg) | 20.2, 6.3, **2.2** (weak axis) | z 0.542 m, tilt 2.0 deg |

The measured table top is **z = 0.540 m** in robot frame; the ws-19 site file uses 0.53 (sim value). The cube centre reads z 0.552-0.557.

For sim: use `T_RC` as the camera extrinsics (robot base -> camera), K_960x540 at 960x540. If the sim camera uses an OpenGL/USD convention (x right, y up, z backward), right-multiply by `diag(1,-1,-1,1)`.

## 3. Meshes and object poses

| Object | Mesh | Notes |
|---|---|---|
| Red brush (tool, object 1) | `/juno/u/tylerlum/tracking_meshes_20261026/red_brush/red_brush.obj` | same mesh as `simtoolreal_deploy_kit/.../brush/red_brush/red_brush.obj` |
| Gold cube (payload, object 2) | `/juno/u/tylerlum/tracking_meshes_20261026/cube_29mm_gold.obj` | 29 mm, vertex colour #E4BD68 (228,189,104), `--upright_meshes cube_29mm_gold` |
| Red dustpan (fixture) | `/juno/u/tylerlum/kuan_handoff_20261118/red_dustpan_mesh_20261118.zip` (obj+mtl+png+glb) | SAM3D, metric, 0.220 x 0.378 x 0.056 m; frame = SAM3D object frame |

Dustpan pose (FoundationPose on one frame, SAM2 mask, IoU 0.90), after the final adjustment, 18 Nov ~16:20:
- camera (ZED2 at its 16:11 position): xyz (-0.2161, 0.1818, 0.4457), quat xyzw (-0.2745, 0.7411, -0.5833, -0.1876)
- **robot frame** (with T_RC 161129): xyz **(-0.0404, -1.0958, 0.5913)**, quat xyzw **(-0.0191, 0.0481, 0.9986, -0.0086)**
- files: `/juno/u/tylerlum/kuan_handoff_20261118/dustpan/fp_v5_adjusted/fp/pose_robot_T_RC_20261118_161129.txt`
- The robot-frame pose stays valid only if the dustpan has not moved since. Also copied to ws-19:
  `policy_futures_dev/robot_deploy/local/dustpan_20261118/` (with `HANDOFF.md`).

## 4. How to run

### 4a. Live two-object tracker (brush + cube), ZED2

```bash
source /home/tylerlum/.drl_env.sh && drl          # cd's into depthbasedRL, sets ROS_MASTER_URI=bohg-ws-2
M=/juno/u/tylerlum/tracking_meshes_20261026
taskset -c 0-9,12-31 python third_party/FoundationPose/live_tracking_new_gui_2_objects.py \
  --serial_number 27432424 \
  --calibration /juno/u/tylerlum/camera_2_calibration_27432424/record_visual_20261118_172707/T_RC.txt \
  --depth_backend fast_fs --fast_fs_trt_dir third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4 \
  --camera_resolution HD1080 --camera_fps 30 --fps 40 \
  --mesh_path $M/red_brush/red_brush.obj \
  --mesh_path_2 $M/cube_29mm_gold.obj --object_name_2 cube_29mm \
  --mask_mode points --upright_meshes cube_29mm_gold \
  --save_dir /home/tylerlum/kuan_takes --ros --payload_object 2 \
  --payload_config /home/tylerlum/kuan_takes/sweep_payload_fp_10mm.json
```
- ZED1: `--serial_number 15107 --calibration /juno/u/tylerlum/camera_calibration/replay_20261016_161152/T_RC.txt`.
- Never run two trackers with `--ros` at once (same topic names).
- Panel: `http://bohg-ws-16.stanford.edu:8100`. Turning off the 2D/3D views there saves a few FPS.
- `--ros` also arms auto-record: one take per policy rollout (start after 2 msgs on `/iiwa/joint_cmd`, stop 0.5 s after the last).
  Rolling buffer 60 s (2400 frames at the 40 FPS cap). Stop with one Ctrl-C (saves the last take).
- Each take dir: `raw.mp4`, `overlay.mp4`, `poses_object_{1,2}.npy` (4x4, camera frame), `frame_log.jsonl`
  (capture stamp + payload verdict per frame), `metadata.json`, `K.txt`. No depth is saved.
- Normal speed: 28-30 FPS (track ~10-15 ms per object). Lower if other GPU viewers are open.
- `sweep_payload_fp_10mm.json` = `{"cube": {"max_surface_residual_m": 0.010}}` (Fast-FS depth is too noisy for 5 mm).

Payload validity (`payload_check.py`, cube only) per frame: colour pixels in the HSV range near the rendered silhouette, silhouette IoU,
depth pixels, surface residual, jump vs the last valid pose. Status JSON stamp == pose stamp (ZED capture time, host clock).

### 4b. Camera calibration (passive: you move the arm by hand)

```bash
source /home/tylerlum/.drl_env.sh && drl
taskset -c 0-9,12-31 python third_party/FoundationPose/calibrate_camera_visual.py record \
  --serial-number 27432424 --camera-resolution HD1080 --tag-size 0.055 --tag-id 0 --view-port 8111 \
  --out-dir /juno/u/tylerlum/camera_2_calibration_27432424
```
Needs fresh `/iiwa/joint_states` (else "Fresh joint states not available"). Panel on :8111. Hold each pose ~2 s, 20+ poses,
rotate the hand > 10 deg about every axis, press Finish. Output `.../record_visual_<timestamp>/T_RC.txt`.
Check: `tools_sweep_20261118/capture_rgbd_zed.py` + a table plane fit should give z ~0.540 and tilt < 1 deg.

### 4c. One-frame tools (`tools_sweep_20261118/`, run in the drl env unless noted)

| Script | Does |
|---|---|
| `capture_rgbd_zed.py SERIAL OUT N` | N frames: `rgb/*.png`, Fast-FS depth `depth/*.npy` (m) + `depth_png/*.png` (mm), `cam_K.txt` (960x540) |
| `capture_stereo_zed.py SERIAL OUT` | rectified left/right at 1920x1080 and 960x540, ZED NEURAL depth, `K_full.txt`, `cam_K.txt`, `baseline_m.txt` |
| `fp_register_one.py RGB DEPTH MASK K MESH OUT` | FoundationPose register on one frame -> `pose_cam.txt` + overlays (mesh blend, 3D box) |
| `depth_overlay.py DEPTH K MESH POSE OUT.png` | mesh at the pose over the depth image + residual map |
| `viz_rgbd_mesh.py --video_dir D --mesh NAME=MESH@POSE ...` | viser: RGB-D points, SAM mask, meshes (SAM2 venv) |
| `viz_compare.py --root D --variants ...` | viser: compare depth sources (point clouds + meshes) |
| `payload_take_stats.py TAKE...` / `payload_sheet.py` | cube validity summary / contact sheet of a take |
| `mesh_outline_sheet.py TAKE i0 i1 step OUT x0 y0 x1 y1` | brush/cube mesh outlines at the logged poses over raw.mp4 (best way to see tracking failures) |
| `run_sam3d.sh IN OUT LOG` | SAM3D on `IN/{rgb,depth(mm png),masks,cam_K.txt}` (first frame) |
| `make_depth_variants.py`, `run_fs.py` | depth-source comparison on one stereo pair (Fast-FS / ZED NEURAL / full FoundationStereo) |

SAM2 mask without clicks: `video_sam2.py --input_dir D --output_dir M --use_second_prompt --prompt_x X --prompt_y Y --second_prompt_x X2 --second_prompt_y Y2`.

### 4d. Mesh from one RGB-D frame (SAM2 + SAM3D)

1. Capture (`capture_rgbd_zed.py`), keep the object fully in frame.
2. SAM2 mask (above). 3. `run_sam3d.sh` -> `OUT/mesh/mesh.obj` (metric: SAM3D uses the depth and cam_K).
   `OUT/posed_mesh` is in the PyTorch3D camera frame: flip x and y (`diag(-1,-1,1,1)`) to get OpenCV.
   FoundationPose on the mesh gives a much better pose than SAM3D's own (2.7 mm vs 16 mm median point-to-mesh).
4. SAM3D env gotchas on this box: drop the NGC index (DNS fails) and `nvidia-pyindex`; pin `setuptools<80` (`pkg_resources`)
   and `numpy<2` (kaolin); build `nvdiffrast` from source with `--no-cache` (a cached wheel was built against another torch).
   Fits in 24 GB (docs say 32 GB). ~2-3 min per run. HF checkpoints are gated (`facebook/sam-3d-objects`).

## 5. What happened / findings

Takes are in `/home/tylerlum/kuan_takes/` (ws-16 local) and copied to `/juno/u/tylerlum/kuan_handoff_20261118/takes/`.

1. **Brush lost under hand occlusion (take 20261118_164617, ZED2 at 16:11 position).** At 2.65 s the brush is in the hand
   (handle sticking up) but its pose swings down to the table next to the cube (z 0.60 -> 0.54 in 0.1 s) and stays along the
   fingers/forearm until ~3.5 s. FoundationPose latches onto the white fingers when only a few brush pixels are visible.
   The brush has no validity check, so the policy got that pose.
2. **Brush latches onto the orange KUKA link and never recovers (take 20261118_173827, current ZED2 view).** The new view looks
   along the table at the robot; when the brush is lifted (z ~0.67 m, near the top of the image) the orange link is right
   behind it (same red-orange as the handle). At 4.4 s the pose stays on the link; the next take (173850) starts already lost
   (brush z 0.31 to -0.02, y up to +0.45 for 10 s). The tracker never re-registers by itself.
3. **Cube check false alarms near red objects.** HSV hue 4-38 also matches the dustpan (hue 2-8, 80% inside the range) and the
   brush handle; the cube is hue 19-23. Near them the IoU drops below 0.45 with a correct pose. Proposed: hue 14-32
   (keeps 100% of cube pixels, 0% of dustpan). Same range is in ws-19 `robot_deploy/local/tasks/sweep_payload_29mm.json`.
   Elsewhere the check was right (cube moved by hand and lost; robot hand covering it). In the current ZED2 view: 99-100% valid.
4. **Tracker stall to ~0.1 FPS (twice).** Both times right after a short-lived rospy subscriber (killed by `timeout`) left a
   CLOSE-WAIT socket on the tracker's publisher; one thread at 100% CPU, GPU idle. Restart fixes it. Do not attach throwaway
   ROS readers to the live tracker; read takes / the panel instead. Root cause not proven.
5. **FPS** drops to ~21-24 when other GPU viewers (viser 3D pages) are open; ~28-30 otherwise.
6. **Depth.** Live depth is Fast-FoundationStereo TensorRT at 384x224, 4 iters: thin parts (dustpan rim, brush) are blurred
   (mesh-vs-depth residual median ~10 mm, p90 ~45 mm). Full FoundationStereo is installed for offline comparison (not done).
7. **ZED1 vs ZED2 calibration** disagree by ~8 mm in cube height (ZED1's is a month old). Recalibrate ZED1 before multi-camera use.

## 6. Next steps (ordered)

1. Re-register before every rollout (panel button, brush flat on the table) so one lost track does not ruin later takes.
2. **Mask the robot out of FoundationPose's input**: render the arm+hand from `/iiwa/joint_states` + `/sharpa/joint_states` and the
   URDF into the camera, zero those depth pixels / exclude from the mask. Targets findings 1 and 2 directly.
3. **Kinematic in-hand prior for the brush**: after the grasp, `T_ee_brush` is ~constant; predict the brush from FK, use
   FoundationPose for small corrections, reject jumps (> ~1 m/s relative to the hand, pose under the table).
4. Brush validity + auto re-register (below table / outside workspace / jump -> invalid -> re-register from SAM2).
5. Cube hue 14-32 (config only).
6. Camera placement: a higher view looking down on the table keeps the lifted brush in front of table/floor, not the robot.
7. Two cameras (ZED1 + ZED2, ~90-150 deg apart, one lower): per-object pick/average by quality score, re-seed a lost tracker from the
   other camera. `live_tracking_robust_2_objects.py` already has SAM2 lost-detection/re-registration and a two-camera hand-off
   selector (`--reseed_on_message`) to build on.
8. FoundationPose++ ideas (github.com/teal024/FoundationPose-plus-plus): a 2D tracker per frame re-anchors the translation before
   `track_one`, plus a Kalman filter on the 6D pose. See branch `2026-11-18_FoundationPosePlusPlus` (if present) and §7.

## 7. Sim experiments worth running

- Camera placement sweep for the sweep task: render the brush in hand from candidate camera poses (the three in §2 plus higher /
  lower variants) and measure visible brush pixels during lift and sweep, and how much of the robot is directly behind the brush.
- Multi-camera: same with ZED1 + ZED2 poses together; score = max visibility over cameras per frame.
- Perception noise in the policy: inject the observed failure modes (brush pose snaps to the table near the cube for ~1 s;
  brush pose frozen on the robot link; cube invalid for 10-50 frames) and check the policy's response.

## 8. FoundationPose++-style tracking (branch `2026-11-18_FoundationPosePlusPlus`, 18 Nov evening)

Tracker: `live_tracking_robust_2_objects.py` (now also has the payload topics + capture stamps, so it can drive ws-19).
Recommended flags on top of the usual ZED2 command (§4a), all object 1 = brush:
```
--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --save_depth
--payload_object 2 --payload_config /home/tylerlum/kuan_takes/sweep_payload_fp_10mm.json --ros
```
What each does: SAM2 (tiny, real-time fork) tracks the brush's 2D mask at 640 px, GPU frame prep, on its own thread
and CUDA stream, every 2nd frame. The lost check compares FoundationPose's silhouette with that mask by coverage
(fraction of the mask inside the silhouette; IoU false-alarms under heavy hand occlusion). When lost for 4 frames:
first re-anchor the last pose onto the mask and refine (keeps orientation), else re-register from the mask.
`--save_depth` writes `<take>/replay/` so a real rollout can be re-run offline with `--replay_dir`.

Evidence so far (SYNTHETIC sequences built from one real ZED1 frame; ground truth = image shift; labelled as such):

| Sequence | Plain FoundationPose | Recommended flags | Original FoundationPose++ (Cutie, +KF) |
|---|---|---|---|
| brush teleports 150 px | lost for good: 228 mm median, 80/149 frames > 3 cm | recovers in 3 frames: 12 mm median, rot max 12.7 deg | translation 12-14 mm, rotation 65-74 deg median (flips to 180) |
| 86% held occlusion + orange link | 4.6 mm median, 7.6 max | same (no regression) | not run |
| pan up to 75 px/frame | 23 mm max | same | not run |

Replay FPS on the 4090 with the GPU free (no camera/depth): plain 38, SAM2 check sync 22.5, async 28.3,
async + every 2 = 32.5. Live adds ~7 ms/frame (capture + Fast-FS).

Dropped (hurt on synthetic): blanking depth outside the mask (rotation error x2), constant-velocity start pose
(100 deg flip), FoundationPose++'s per-frame centre anchor (mask centre != object centre under occlusion).
Opt-in still available: `--fpp_anchor_xy` (silhouette delta gated on coverage), `--fpp_mask_depth_px`, `--fpp_filter`.

Speed fixes: SAM2 prepare_data on the GPU (35 -> 21 ms), 640 px (-> 13 ms), nvdiffrast silhouette for the check
(11.4 -> ~1-3 ms), check / anchor maths in a bounding box (12.5 -> 1.9 ms, 2.9 -> 0.9 ms).
FoundationPose++ itself runs in a clone of the drl venv: `/home/tylerlum/github_repos/FoundationPose-plus-plus/.venv_fpp`
(+ hydra-core, easydict, Cutie `--no-deps`; set PYTHONPATH to our FoundationPose so its compiled code is used).
