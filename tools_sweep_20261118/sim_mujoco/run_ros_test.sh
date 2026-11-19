#!/bin/bash
# bash run_ros_test.sh TAKE CONFIG_FLAGS...   Full ROS path on a PRIVATE master (port 11411): tracker --ros (replay of
# the MuJoCo take) -> topics -> recorder. Never touches the robot's master.
S=$(cd "$(dirname "$0")" && pwd); R=${FPSIM_ROOT:-$S}; TAKE=$1; shift  # R holds takes/ and runs/
export ROS_MASTER_URI=http://localhost:11411 ROS_HOSTNAME=localhost
RE=/home/tylerlum/.micromamba/envs/simtoolreal_ros/bin
PY=/home/tylerlum/github_repos/depthbasedRL/.venv/bin/python
FP=/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose
M=/juno/u/tylerlum/tracking_meshes_20261026
out=$R/runs/ros/$TAKE; rm -rf $out; mkdir -p $out
$RE/python $RE/rosmaster --core -p 11411 > $out/master.log 2>&1 & RM=$!
sleep 3
$RE/python $S/ros_record.py $out/topics.jsonl > $out/recorder.log 2>&1 & RR=$!
sleep 2
cd $FP && timeout 900 $PY live_tracking_robust_2_objects.py --replay_dir $R/takes/$TAKE/replay --mask_mode replay \
  --headless --no_auto_record --mesh_path $M/red_brush/red_brush.obj --mesh_path_2 $M/cube_29mm_gold.obj \
  --object_name_2 cube_29mm --upright_meshes cube_29mm_gold --save_dir $out --ros \
  --calibration /juno/u/tylerlum/camera_2_calibration_27432424/record_visual_20261118_172707/T_RC.txt \
  --payload_object 2 --payload_config /home/tylerlum/kuan_takes/sweep_payload_fp_10mm.json "$@" > $out/log.txt 2>&1
echo "tracker exit $?"
sleep 2; kill $RR; sleep 1; kill $RM; wait $RR $RM 2>/dev/null
echo "recorded $(wc -l < $out/topics.jsonl) messages"; cat $out/recorder.log
