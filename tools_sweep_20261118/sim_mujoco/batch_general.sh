#!/bin/bash
# General tracker experiments: same flags for sweep / screw / wrench; single camera (ZED2 front, ZED1) and two cameras.
cd "$(dirname "$0")"
until grep -q BATCH_DONE batch_gen_general.log 2>/dev/null; do sleep 30; done
PY=/home/tylerlum/github_repos/depthbasedRL/.venv/bin/python
export RT=0.5
GEN="screw_task screw_tour_1 screw_tour_2 wrench_task wrench_tour_1 wrench_tour_2"
SW="pick_lift occlusion_tour fast_wave heldout_1 heldout_2 heldout_4 heldout_5 heldout_6"
# 1. single camera, same flags everywhere
for t in $GEN; do for p in "" zed1__; do bash run_configs.sh $p$t replay plain v4np v4hc; done; done
for t in $SW; do bash run_configs.sh zed1__$t replay v4hc; done
echo PHASE1_DONE
# 2. two cameras: each camera restarts from the other camera's v4hc pose when lost
mkdir -p rsf
for t in $GEN $SW; do
  a=$(ls -d runs/${t}_rt/v4hc/2026*/ 2>/dev/null | head -1); b=$(ls -d runs/zed1__${t}_rt/v4hc/2026*/ 2>/dev/null | head -1)
  [ -n "$b" ] && $PY export_robot_poses.py $b takes/zed1__$t/gt/T_RC.txt rsf/$t.npz
  [ -n "$a" ] && $PY export_robot_poses.py $a takes/$t/gt/T_RC.txt rsf/zed1__$t.npz
done
for t in $GEN $SW; do bash run_configs.sh $t replay v4hcx; bash run_configs.sh zed1__$t replay v4hcx; done
echo BATCH_DONE
