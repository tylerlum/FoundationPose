#!/bin/bash
# Screw + wrench takes for the general tracker, current ZED2 and ZED1 (same motion), + hand poses + stereo depth.
cd "$(dirname "$0")"
PYM=/home/tylerlum/.micromamba/envs/simtoolreal_ros/bin/python
PY=/home/tylerlum/github_repos/depthbasedRL/.venv/bin/python
gen() { task=$1; sc=$2; cam=$3; name=$4
  FPSIM_TASK=$task FPSIM_CAM=$cam MUJOCO_GL=egl timeout 1800 $PYM gen_take.py --scenario $sc --out takes/$name 2>&1 | grep -E "wrote|Error|angle" > takes_$name.log
  FPSIM_TASK=$task FPSIM_CAM=$cam MUJOCO_GL=egl timeout 900 $PYM gen_take.py --scenario $sc --out takes/$name --hand_only 2>&1 | grep -E "hand poses|Error" >> takes_$name.log; }
LIST="screw:screw_task screw:tour_1 screw:tour_2 wrench:wrench_task wrench:tour_1 wrench:tour_2"
for cam in zed2_172707 zed1; do
  n=0
  for it in $LIST; do task=${it%%:*}; sc=${it##*:}
    nm=${task}_${sc#${task}_}; [ $cam = zed1 ] && nm=zed1__$nm
    rm -rf takes/$nm; gen $task $sc $cam $nm & n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait
  done; wait
done
ALL=$(for p in "" zed1__; do for t in screw_task screw_tour_1 screw_tour_2 wrench_task wrench_tour_1 wrench_tour_2; do echo takes/$p$t; done; done)
$PY stereo_depth.py $ALL 2>&1 | grep -E "frames|Error"
echo BATCH_DONE
