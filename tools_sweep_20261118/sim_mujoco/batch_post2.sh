#!/bin/bash
cd "$(dirname "$0")"
PY=/home/tylerlum/github_repos/depthbasedRL/.venv/bin/python
OPT=/home/tylerlum/github_repos/FoundationPose_speedopt_20261119
# 1. restore the realtime baseline runs that were overwritten
for t in heldout_4 occlusion_tour pick_lift; do RT=0.5 bash run_configs.sh $t replay plain; done
echo RESTORE_DONE
# 2. speed edits: same poses? (plain, sequential) then timing, alternating
mkdir -p runs_speed
for t in heldout_4 occlusion_tour; do
  for v in base opt; do
    if [ $v = opt ]; then FPD=$OPT; else FPD=/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose; fi
    FP_DIR=$FPD bash run_configs.sh $t replay plain; rm -rf runs_speed/${v}_$t; mv runs/$t/plain runs_speed/${v}_$t
  done
done
for k in 1 2 3; do
  for v in base opt; do
    if [ $v = opt ]; then FPD=$OPT; else FPD=/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose; fi
    FP_DIR=$FPD bash run_configs.sh pick_lift replay plain; rm -rf runs_speed/time_${v}_$k; mv runs/pick_lift/plain runs_speed/time_${v}_$k
  done
done
echo SPEED_DONE
# 3. ROS end to end with the recommended live config + logging + save_depth, then the real-take comparer on its output
bash run_ros_test.sh fast_wave --sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --log_checks --save_depth > post_ros.log 2>&1
TK=$(ls -d runs/ros/fast_wave/2026*/ | head -1)
echo "ros take: $TK"; ls $TK; ls $TK/replay | head
CONFIGS="plain recommended v8" bash /home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose/tools_sweep_20261118/compare_replay_configs.sh ${TK%/} /juno/u/tylerlum/camera_2_calibration_27432424/record_visual_20261118_172707/T_RC.txt > post_compare.log 2>&1
echo BATCH_DONE
