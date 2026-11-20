#!/bin/bash
# After batch_general: regenerate screw/wrench takes (workpiece column, correct heights), rerun general configs.
cd "$(dirname "$0")"
until grep -q BATCH_DONE batch_general.log 2>/dev/null; do sleep 30; done
PY=/home/tylerlum/github_repos/depthbasedRL/.venv/bin/python
mkdir -p runs_old_general; for d in runs/*screw_* runs/*wrench_*; do mv $d runs_old_general/ 2>/dev/null; done
rm -f batch_gen_general.log; SKIP_WAIT=1 bash batch_gen_general.sh > batch_gen_general.log 2>&1
echo REGEN_DONE
export RT=0.5
GEN="screw_task screw_tour_1 screw_tour_2 wrench_task wrench_tour_1 wrench_tour_2"
for t in $GEN; do for p in "" zed1__; do bash run_configs.sh $p$t replay plaincal v4npcal gen; done; done
for t in $GEN; do
  a=$(ls -d runs/${t}_rt/gen/2026*/ | head -1); b=$(ls -d runs/zed1__${t}_rt/gen/2026*/ | head -1)
  $PY export_robot_poses.py $b takes/zed1__$t/gt/T_RC.txt rsf/$t.npz; $PY export_robot_poses.py $a takes/$t/gt/T_RC.txt rsf/zed1__$t.npz
done
for t in $GEN; do bash run_configs.sh $t replay genx; bash run_configs.sh zed1__$t replay genx; done
echo PHASE2_DONE
# sequential (every frame) gen on the takes Point2Pose will run on, for a like-for-like comparison
unset RT
for t in fast_wave heldout_4 occlusion_tour screw_task screw_tour_1 wrench_tour_1; do bash run_configs.sh $t replay gen; done
echo BATCH_DONE
