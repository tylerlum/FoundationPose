# Live tracking - how to run it

Quick reference for the FoundationPose live tracking setup on bohg-ws-16.
Deeper detail in docs/LIVE_TRACKING_RUNBOOK.md (all options, measured
numbers, and the post-save stall post-mortem) and docs/LOCAL_SETUP.md
(how this environment was built and why).

## The command

```bash
drl
python third_party/FoundationPose/live_tracking_new_gui.py \
  --text_prompt "black spoon" \
  --depth_backend zed_neural --camera_resolution HD1080 --camera_fps 30 --fps 40 \
  --mesh_path assets/urdf/dextoolbench/spatula/spoon_spatula/spoon_spatula.obj \
  --save_dir /juno/u/tylerlum/2026-09-19_videos/Sept_19 --ros
```

`drl` is a shell function (from `~/.bashrc`) that activates the venv, points
every cache at local disk, sets the CUDA/gcc toolchain and ROS vars, and cds
to the repo.

On startup it prints the viser panel URL, e.g.
`http://bohg-ws-16.stanford.edu:<port>` — open that for re-register, save,
discard, auto-record toggle, live FPS, and mid-run object/prompt swapping.
Each script has its own port, so the link stays the same across reruns: GUI 8090, two-object GUI 8100,
calibrate_camera.py viser 8110, two-object replay 8120. If it is taken, the next free one (+1).

## What it does by default

- **Text masking** — `--text_prompt` names the object, no clicking, ever.
  SAM3 (`facebook/sam3`), ~0.22 s per call once warm. Runs at register only,
  not per frame.
- **Auto-record** — buffers only while `rl_policy_node.py` publishes joint
  targets on `/iiwa/joint_cmd`, and writes the take when it stops. Tracking
  never pauses, so a new rollout costs no re-register.
- **Saves block the loop** (deliberately, ~1–2 s) and print how long they
  took. Videos are `veryfast`/`crf 20`; `viz_data.npz` is off.

## The three scripts

| script | use |
|---|---|
| `live_tracking_new.py` | the original, untouched |
| `live_tracking_new_autorecord.py` | policy-gated recording, no GUI |
| `live_tracking_new_gui.py` | **this one** — adds the browser panel + mid-run swaps |

## Flags worth knowing

| flag | why |
|---|---|
| `--no_auto_record` | buffer continuously; save by hand |
| `--mask_mode center_box` | skip SAM3 entirely, fixed centre box |
| `--text_fallback_click` | fall back to the click picker if a prompt misses |
| `--no_overlay_video` | halves encode time, halves file size |
| `--encode_crf 26` | smaller/faster videos |
| `--viz_data` | write the point-cloud npz (needed for `--visualize`) |
| `--no_gui` | no panel; falls back to the OpenCV keyboard window |
| `--mesh_choices a.obj b.obj` | extra meshes in the swap dropdown |

Prompt wording matters more than you'd expect: on the lab scene
`"black spoon"` scored 0.97, `"spoon"` 0.95, and `"ladle"` matched **nothing**
— same object.

## If something goes wrong

**`CAMERA NOT DETECTED`** — a stale capture child is holding `/dev/video0`.

```bash
fuser -v /dev/video0
bash /home/tylerlum/cleanup_tracking.sh
```

**Ctrl-C once, then wait.** Mashing it produces bogus `UnboundLocalError`s
from inside Python's regex compiler — see `docs/live_tracking_troubleshooting.md`.

**A loop that collapses to ~1.6 Hz** was a rospy bug (a subscriber whose
publisher exits spins at 100% CPU holding the GIL). Fixed 2026-09-19 in
commit `8933d88`; the runbook has the full story.

## Two objects (13 Oct)

`live_tracking_new_gui_2_objects.py` is the same GUI script for two objects: one panel,
both objects registered on one frame and tracked every loop. Object 1 publishes on
`current_object_pose`, object 2 on `current_object_pose_2` (`--pose_topic_2 workpiece_pose`
for the tool program node). Two track_one calls per frame: expect about half the loop rate.

```bash
drl
M=/juno/u/tylerlum/kit_tracking_meshes_20261013
python third_party/FoundationPose/live_tracking_new_gui_2_objects.py \
  --depth_backend zed_neural --camera_resolution HD1080 --camera_fps 30 --fps 30 \
  --mesh_path $M/open_wrench_18.obj --text_prompt "gray wrench" \
  --mesh_path_2 $M/bolt_thick_head.obj --text_prompt_2 "small gray cube" \
  --save_dir /juno/u/tylerlum/2026-10-13_kit_tracking_2obj --ros
```

- Screw head: `--mesh_path_2 $M/screw_p1_head.obj --text_prompt_2 "gray block" --text_join_px_2 60`
  (SAM3 sees its two halves as two instances; this joins them). Driver: `"black screwdriver"`.
- The panel re-registers either object or both, and swaps either object's prompt or mesh.
- The wrench and the heads register upright (`--upright_meshes`, `--no_upright`): FoundationPose
  can return them turned over.
- A take adds `poses_object_1.npy`, `poses_object_2.npy` and `K.txt`. With `--viz_data`,
  `python third_party/FoundationPose/live_tracking_viz_replay_2_objects.py <take>` replays both.
- `--replay_dir <dir with rgb/ depth/ cam_K.txt>` runs it on recorded frames, without the camera.

## If the joint states stop (FRI drops, 13 Oct)

The robot app closes FRI at the first quality dip, and the teach pad only says "FRI connection closed". The FRI pane
on bohg-ws-2 now says so itself (FRI LOST / back / quality / timing), and a boot service keeps the CPU settings that
cut the drops. Everything about the station -- the robot PC setup, FRI notes, the watchdog, the camera calibrations
(calibrations/current.txt, which the trackers here read) -- is in github.com/tylerlum/kuka_sharpa_station (private),
cloned next to this repo (/home/tylerlum/github_repos/kuka_sharpa_station): kuka/README.md, calibrations/README.md.

## Two objects, the command that works (15 Oct)

Wrench + bolt head, Fast-FS depth, the 14 Oct calibration, the head kept upright. Click each object in the popups.

```bash
drl
M=/juno/u/tylerlum/kit_tracking_meshes_20261013
python third_party/FoundationPose/live_tracking_new_gui_2_objects.py \
  --depth_backend fast_fs \
  --fast_fs_trt_dir third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4 \
  --camera_resolution HD1080 --camera_fps 30 --fps 40 \
  --mesh_path $M/open_wrench_18_red.obj --mask_mode points \
  --mesh_path_2 $M/bolt_thick_head_bluegrey.obj \
  --upright_meshes bolt_thick_head_bluegrey --force_upright \
  --calibration /juno/u/tylerlum/camera_calibration/replay_20261016_161152/T_RC.txt \
  --save_dir /juno/u/tylerlum/2026-10-15_videos/Dummy2 --ros
```

Screwdriver + screw head (15 Oct: the screw head tracked well; the screwdriver did not, see below):

```bash
drl
M=/juno/u/tylerlum/kit_tracking_meshes_20261013
python third_party/FoundationPose/live_tracking_new_gui_2_objects.py \
  --depth_backend fast_fs \
  --fast_fs_trt_dir third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4 \
  --camera_resolution HD1080 --camera_fps 30 --fps 40 \
  --mesh_path $M/round_screwdriver_p1_red.obj --mask_mode points \
  --mesh_path_2 $M/screw_p1_head_bluegrey.obj \
  --upright_meshes screw_p1_head_bluegrey --force_upright \
  --calibration /juno/u/tylerlum/camera_calibration/replay_20261016_161152/T_RC.txt \
  --save_dir /juno/u/tylerlum/2026-10-15_videos/Dummy2 --ros
```

- `--lock_position` (with `--upright_meshes`): after register, the head's centre stays where register put it and only
  its heading about +z is tracked (implies `--force_upright`; re-registering moves the lock). For a screw or bolt that
  sits in a fixture and only turns.
- `round_screwdriver_p1_red` is nearly symmetric about its shaft. The only feature that fixes its roll is the flag
  along the shaft, and when it is perpendicular to the image (edge-on to the camera) it is hard to see, so the
  pose flips between roll angles. The fin version (`round_screwdriver_p1_fin`, a triangle 90 deg from the flag) still
  flipped: both are thin plates, so face-on they look the same from either side. Fixed on 17 Oct by colour (below).
- Panel: http://bohg-ws-16.stanford.edu:8100 (re-register, save, and the live 3D / 2D view; both views on by default).
- Drop `--force_upright` if a head may really be tilted or lying down: it forces +z up regardless.
- Once `kuka_sharpa_station/calibrations/current.txt` is the 14 Oct calibration, `--calibration` can go.

## Live view, Fast-FS depth, the point-cloud viewer (15 Oct)

**Live view in the two-object panel.** The panel (port 8100) has a "Live view (off = no cost)" folder: a **3D view**
(the ZED point cloud and both meshes at their tracked poses, in the robot frame) and a **2D view** (the image with
both meshes and their axes drawn on, red where there is no depth). Each switches on and off mid-run; both off costs
the loop one bool check per frame. When on, a background thread draws at the panel's rate (default 5 Hz). Both start
on (`--live_view 3d` / `2d` for one, `--live_view off` for neither); on 17 Oct, with a browser on this machine, the
tracker held 30 FPS with both on, the same as off. The 3D view shows the camera image in the camera's frustum, with
the 2D view's pose annotations ('frustum: annotated' off for the plain image). The 3D view starts at x = y = 0, the camera's height,
looking where the camera looks. Code: `third_party/FoundationPose/live_tracking_live_view.py`.

**Pass the calibration.** Without `--calibration` the trackers use the station's `calibrations/current.txt`, which is
the 13 Oct one and stale. The camera moved again before 16 Oct (12.8 cm, 13° from the 14 Oct calibration); the 16 Oct
calibration (36 poses, fit RMS 1.69 mm / 0.21°) matches it:
`--calibration /juno/u/tylerlum/camera_calibration/replay_20261016_161152/T_RC.txt`. It sets the robot-frame poses,
the 3D view and "up" for `--upright_meshes`.

**Fast-FoundationStereo depth.** ZED neural loses everything closer than ~0.4 m (the bottom third of the image here)
and small grey parts. Fast-FS (TensorRT) fills both:

```bash
--depth_backend fast_fs --fast_fs_trt_dir third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4
```

| on 15 Oct, same frame | ZED neural | Fast-FS TRT, 4 iters |
|---|---|---|
| pixels with depth, whole image | 62% | 83% |
| near (bottom third) | 12% | 77% |
| the grey piece on the fixture | 39% | 100% |
| time per frame, RTX 4090 | (in the capture process) | 7.7 ms (8 iters: 10.3 ms) |

Where both have depth they agree to a median 1.2 mm. Fast-FS runs in the tracking process, so the loop gets ~8 ms
slower. The engines were built on an RTX 6000 Ada; TensorRT warns but they run. Figures:
`/juno/u/tylerlum/2026-10-15_depth_comparison/`.

**Point-cloud viewer.** `python third_party/FoundationPose/live_pointcloud_viser.py --calibration <T_RC.txt>` (port
8125): the live ZED cloud in the robot frame, the image with no-depth pixels in red, and a checkbox for FoundationPose's
own depth filter. It needs the camera, so not while a tracker runs.

**Mesh frames.** Poses are the OBJ's own frame. `round_screwdriver_p1_red`: x along the length (20 cm), origin 5 cm
from one end. `screw_p1_head_bluegrey`: z along the screw axis, origin 2.4 cm below the head; the slot runs along x.

**Heads registering sideways: `--force_upright`.** The bolt and screw heads are near-cubes, so a sideways pose fits
the image almost as well as the upright one, and FoundationPose often picks it (67-70 deg off vertical on the 15 Oct
fixture frame). With `--force_upright`, a mesh in `--upright_meshes` registers from upright guesses only (+z up,
24 headings) and keeps +z exactly up every frame (turned about the mesh centre), so only its heading and position
are tracked. Needs the right `--calibration`: "up" comes from it.

## Second camera (ZED 2, S/N 27432424) and the taped screwdriver (17 Oct)

The fin screwdriver with its flag wrapped in blue tape and its fin in black tracks its roll well. The mesh
`$M/round_screwdriver_p1_fin_taped.obj` is `round_screwdriver_p1_fin_red.obj` with the flag's faces (22, 80, 155) and
the fin's (18, 17, 22), colours sampled from a photo, vertices unmerged so the colours stop at the plate edges
(preview: `$M/round_screwdriver_p1_fin_taped_preview.png`). The camera's calibrations are in
`/juno/u/tylerlum/camera_2_calibration_27432424/` (three runs, within 1.5 mm / 0.07 deg of each other).

```bash
drl
M=/juno/u/tylerlum/kit_tracking_meshes_20261013
python third_party/FoundationPose/live_tracking_new_gui_2_objects.py \
  --serial_number 27432424 \
  --depth_backend fast_fs \
  --fast_fs_trt_dir third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4 \
  --camera_resolution HD1080 --camera_fps 30 --fps 40 \
  --mesh_path $M/round_screwdriver_p1_fin_taped.obj --mask_mode points \
  --mesh_path_2 $M/screw_p1_head_bluegrey.obj \
  --upright_meshes screw_p1_head_bluegrey --force_upright --lock_position \
  --calibration /juno/u/tylerlum/camera_2_calibration_27432424/replay_visual_20261017_175834/T_RC.txt \
  --save_dir /juno/u/tylerlum/2026-10-17_videos/Dummy2_cam2_taped --ros
```

- `$M/round_screwdriver_p1_fin_taped_yellow.obj` adds the yellow stripe taped along the shaft on the bare side
  (centred opposite the gap between the plates, about 190 deg of the shaft, from the handle to x = 125 mm, colour
  (223, 184, 6)); use it with the yellow-taped screwdriver. Preview: `..._taped_yellow_preview.png`, which matches
  the photos from both sides (so the print is not mirrored relative to the mesh).
- Half resolution (the default 960x540) is the better trade: on this camera full resolution (`--width 1920 --height
  1080`) ran at 22 FPS (track 14 + 14 ms) against 30 FPS (9 + 11 ms), and register at full resolution needed the
  chunked xyz warp in `learning/datasets/h5_dataset.py` to fit in GPU memory.
- On this camera the 3D view's start position (x = y = 0 at the camera's height) looks at the arm, not the table.

## Four-colour screwdriver print (17 Oct)

`third_party/FoundationPose/make_colored_screwdriver.py` (run with the simtoolreal_ros env's python, which has
manifold3d) splits `round_screwdriver_p1_fin_red.obj` into four single-colour parts for the X1C's AMS, with the same
shape: handle and shaft quadrants red / gray / gold / black (Bambu PLA Basic; boundaries 45 deg either side of the flag),
flag black, fin gold. Outputs in `$M/screwdriver_fin_quad_print/`: `print_{red,gray,gold,black}.stl` (+ a zip of the four) (mm, already in print
orientation: lying down, the gap between the plates facing up), the tracking mesh `round_screwdriver_p1_fin_quad.obj`
(outer surface only, same area as the source) and a preview from four sides.
