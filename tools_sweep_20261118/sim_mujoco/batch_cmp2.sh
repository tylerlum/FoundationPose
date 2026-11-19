#!/bin/bash
cd "$(dirname "$0")"
until grep -q BATCH_DONE batch_ros2.log 2>/dev/null; do sleep 30; done
TK=$(ls -d runs/ros/fast_wave/2026*/ | head -1)
echo "take $TK"; ls $TK/replay
CONFIGS="plain recommended v4hc" bash /home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose/tools_sweep_20261118/compare_replay_configs.sh ${TK%/} /juno/u/tylerlum/camera_2_calibration_27432424/record_visual_20261118_172707/T_RC.txt > post_compare2.log 2>&1
echo BATCH_DONE
