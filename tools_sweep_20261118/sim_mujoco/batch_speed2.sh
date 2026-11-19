#!/bin/bash
# Round-2 speed edits: identical poses? then timing (alternating), sequential plain replays.
cd "$(dirname "$0")"
until grep -q BATCH_DONE batch_v4hct.log 2>/dev/null; do sleep 30; done
BASE=/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose
OPT=/home/tylerlum/github_repos/FoundationPose_speedopt2_20261119
mkdir -p runs_speed2
for t in heldout_4 occlusion_tour; do
  for v in base opt; do
    if [ $v = opt ]; then FPD=$OPT; else FPD=$BASE; fi
    FP_DIR=$FPD bash run_configs.sh $t replay plain; rm -rf runs_speed2/${v}_$t; mv runs/$t/plain runs_speed2/${v}_$t
  done
done
for k in 1 2 3; do
  for v in base opt; do
    if [ $v = opt ]; then FPD=$OPT; else FPD=$BASE; fi
    FP_DIR=$FPD bash run_configs.sh pick_lift replay plain; rm -rf runs_speed2/time_${v}_$k; mv runs/pick_lift/plain runs_speed2/time_${v}_$k
  done
done
echo BATCH_DONE
