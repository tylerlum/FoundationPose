# MuJoCo test bench for the held-tool tracker (19 Nov 2026)

Synthetic ZED2 takes of the sweep cell (iiwa14 + Sharpa hand holding the red brush, 29 mm gold cube, table at
0.540 m), rendered from the 172707 ZED2 T_RC and K (960x540), with exact ground truth. Depth comes from the live
tracker's own Fast-FS TRT engine run on a rendered stereo pair (12 cm baseline), so it blurs like the real one
(median error vs true depth 8-14 mm). The tracker runs unchanged in replay mode; `--replay_realtime_scale` makes
it skip frames like the live camera.

Envs: generation in the RoboStack env (`/home/tylerlum/.micromamba/envs/simtoolreal_ros`, has mujoco), everything
else in drl (`depthbasedRL/.venv`). Set `FPSIM_ROOT` to a scratch dir for takes/ and runs/ (default: this folder).

```bash
# 1. takes (scenarios: pick_lift, occlusion_tour, fast_wave, heldout_<seed>); speeds capped to real p95 (0.9 m/s)
MUJOCO_GL=egl $RE/python gen_take.py --scenario heldout_5 --out $FPSIM_ROOT/takes/heldout_5
python stereo_depth.py $FPSIM_ROOT/takes/heldout_5          # Fast-FS depth (+ replay_gtdepth/ with exact depth)
# 2. tracker configs (plain v4 v6 v7 v8); RT=0.5 = frame skipping like live, when the GPU is shared
RT=0.5 bash run_configs.sh heldout_5 replay plain v8
python score_sim.py $FPSIM_ROOT/takes/heldout_5 $FPSIM_ROOT/runs/heldout_5_rt/*/2026*/
# 3. full ROS path on a PRIVATE master (port 11411), recorded and scored in the robot frame
bash run_ros_test.sh fast_wave <tracker flags>; python score_ros.py takes/fast_wave runs/ros/fast_wave/topics.jsonl
```

`pose_search.json`: 269 reachable held-brush arm poses with the brush's visible fraction (used by the scenarios).
`color_signal.py`: shows the rendered-colour error separates right from wrong poses (right ~0.07, >5 cm off 0.23-0.41).

Results, 8 takes, realtime replay (frames within 2 cm / frames > 5 cm off / recovered by the end):
plain 25% / 74% / 1 of 8; v4 30% / 67% / 2 of 8; v7-v8 (colour arbitration) 49-71% / 22-47% / 6-7 of 8.
Identical configs vary a lot run to run (async SAM2 + timing): compare over several takes and repeats.
Limits: kinematic grasp (no finger contact), MuJoCo shading, white hand, plain background; replay at 0.5x scale.

## 19 Nov update
- SAM prompt: box + 1 point made SAM2 AND SAM3 segment arm + hand + brush when the brush is held (sim mask IoU
  0.25-0.38). `robust_tool.SamTool` now prompts box + 4 positive + 6 negative points (IoU 0.82-0.91, same 13 ms);
  `--sam2_simple_prompt` restores the old one. SAM3 tracker with a mask prompt: IoU 0.88-0.95 but 80 ms/frame.
  `sam_bench.py`, `sam_video.py` (side-by-side videos, sim + real).
- WARNING, colour check (`--arbitrate color`): thresholds are from sim. On REAL takes (`real_color.py`) a correctly
  tracked brush on the table scores 0.28-0.31 (sim 0.07) because the mesh texture is dark and its handle is red where
  the real one is black, and the brush stuck on the orange KUKA link scores the same. Do not use the absolute
  thresholds live until recalibrated / replaced (see the 19 Nov message).
- `FPSIM_CAM` = zed2_172707 (current) | zed2_161129 (previous diagonal) | zed1 selects the camera; `batch_cam.sh`.
- `timing.py`: register 0.57 s (iter 2) / 0.98 s (iter 5), track_one ~20-40 ms, colour check 2 ms (shared GPU).
