#!/bin/bash
# Redo after the AFS token expiry: regenerate screw/wrench takes, general runs, Point2Pose, gated two-camera test.
cd "$(dirname "$0")"
rm -f batch_gen_general.log batch_general2.log batch_p2p.log batch_v4hcxv.log
ALL=$(for p in "" zed1__; do for t in screw_task screw_tour_1 screw_tour_2 wrench_task wrench_tour_1 wrench_tour_2; do echo takes/$p$t; done; done)
/home/tylerlum/github_repos/depthbasedRL/.venv/bin/python stereo_depth.py $ALL 2>&1 | grep -E "frames|Error" > batch_gen_general.log
echo BATCH_DONE >> batch_gen_general.log
sed -i 's|^until grep -q BATCH_DONE batch_general.log 2>/dev/null; do sleep 30; done$||; s|^rm -f batch_gen_general.log; SKIP_WAIT=1 bash batch_gen_general.sh > batch_gen_general.log 2>&1$||; s|^mkdir -p runs_old_general; for d in runs/\*screw_\* runs/\*wrench_\*; do mv $d runs_old_general/ 2>/dev/null; done$||' batch_general2.sh
for d in runs/*screw_* runs/*wrench_*; do rm -rf $d; done
bash batch_general2.sh > batch_general2.log 2>&1
bash batch_p2p.sh > batch_p2p.log 2>&1
bash batch_v4hcxv.sh > batch_v4hcxv.log 2>&1
echo REDO_DONE
