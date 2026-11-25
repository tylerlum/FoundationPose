# Up to three objects: live_tracking_new_gui_n_objects.py (26 Oct 2026)

**bohg-ws-16 has a faulty CPU core (logical CPUs 10-11; found 26 Oct): start the trackers with
`taskset -c 0-9,12-31 python ...`, or disable those CPUs first** (docs/ROBOT_WEEK_2026-10-26.md section 0).

`live_tracking_new_gui_n_objects.py` is `live_tracking_new_gui_2_objects.py` for 2 or 3 objects. It is a new file;
the two-object script and its panel are unchanged. Its panel is `live_tracking_gui_panel_n_objects.py` (port 8105, so
the two-object GUI on 8100 can run beside it). Backup copies: `/juno/u/tylerlum/tracking_meshes_20261026/tracker/`.

New flags:
- `--mesh_path_3`, `--text_prompt_3`, `--text_join_px_3`, `--object_name_3`, `--pose_topic_3` (default
  `current_object_pose_3`) add the third object. Without `--mesh_path_3` it tracks two objects.
- Upright rules PER OBJECT, by mesh file stem:
  - `--upright_meshes STEMS`: register upright only (keep the copy with +z up), then track freely.
  - `--force_upright_meshes STEMS`: also keep +z exactly up every frame.
  - `--lock_position_meshes STEMS`: also lock the position where registration put it.
  - The global `--force_upright` / `--lock_position` keep their two-object meaning (every upright mesh).
- `--mask_mode replay` reads `masks_<k>/<frame>.png` for k = 1..N.

Topics: `camera_frame/<topic>` and `robot_frame/<topic>` for each object (stamped at publish time, not at image time).

## Commands for the robot (meshes in the policies' frames: /juno/u/tylerlum/tracking_meshes_20261026)

Stir teacher, the spoon + the bowl (two objects; the two-object script works the same way with these flags):

    drl
    M=/juno/u/tylerlum/tracking_meshes_20261026
    taskset -c 0-9,12-31 python third_party/FoundationPose/live_tracking_new_gui_n_objects.py \
      --serial_number 27432424 --calibration <that camera's T_RC.txt> \
      --depth_backend fast_fs --fast_fs_trt_dir third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4 \
      --camera_resolution HD1080 --camera_fps 30 --fps 40 \
      --mesh_path $M/wooden_spoon_gold.obj --mesh_path_2 $M/bowl_grey.obj --pose_topic_2 bowl_pose \
      --mask_mode points --upright_meshes wooden_spoon_gold bowl_grey \
      --save_dir /juno/u/tylerlum/2026-10-26_videos/stir_teacher --ros

Sweep teacher, the red brush + two gold 2 cm cubes:

    taskset -c 0-9,12-31 python third_party/FoundationPose/live_tracking_new_gui_n_objects.py \
      <the same camera / depth flags> \
      --mesh_path $M/red_brush/red_brush.obj \
      --mesh_path_2 $M/cube_20mm_gold.obj --mesh_path_3 $M/cube_20mm_gold.obj \
      --object_name_2 cube_1 --object_name_3 cube_2 \
      --mask_mode points --upright_meshes cube_20mm_gold \
      --save_dir /juno/u/tylerlum/2026-10-26_videos/sweep_teacher --ros

Leave the brush out of `--upright_meshes`: its +z is the bristle normal, which points down when it lies on its bristles.

## Replay test (MuJoCo renders, the proposed az330 camera, 600 frames at 30 Hz; scripts/perception/stir_replay_dir_3obj.py)

- Three objects tracked at 25 FPS (track_one p50 ~10 ms per object; the replay's own frame reads from /juno add
  ~5-27 ms). With a camera expect ~25 FPS with Fast-FS depth.
- Spoon (`--upright_meshes`): registered with +z up; it then rotates freely in the hand. The paddle point is within
  3.6 mm (p90 5.7 mm) while lifted. But FoundationPose switches to the spoon's turned copy (180 deg about its long
  axis) at the first lifted frame and keeps it (100 % of lifted frames, 0 % on the table). The paddle point is
  unaffected; the policy runtime's flip guard (simtoolreal_dev deployment/tool_use/flip_guard.py) undoes the turn.
- Bowl: `--upright_meshes` 0.6 mm at registration, 0.3 mm median. Forced upright registration (`--force_upright_meshes`
  or `--lock_position_meshes`) put it 107 mm off on the first frame; tracking recovers (0.4 mm) unless the position is
  locked (then it stays 117 mm off). So: the bowl with `--upright_meshes`, not locked.
- Cubes (2 cm, 141-255 mask px from this camera): 1.6-2.7 mm while visible. Once the arm hides a cube, its tracker
  follows the arm and never comes back (65-250 mm). Re-register it from the panel ('Re-register object k') when its
  overlay leaves the cube, and place the camera so the hand rarely covers the cubes.
- Over ssh without an X display, `--no_gui` needs `--headless` too (as the two-object script).
