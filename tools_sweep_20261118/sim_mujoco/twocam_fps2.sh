#!/bin/bash
# plain config (lighter: no SAM2) so two trackers fit beside the live tracker
cd "$(dirname "$0")"
FP=/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose; PY=/home/tylerlum/github_repos/depthbasedRL/.venv/bin/python
M=/juno/u/tylerlum/tracking_meshes_20261026; S=$PWD
run() { out=$S/runs_twocam/$2; rm -rf $out; mkdir -p $out; (cd $FP && $PY live_tracking_robust_2_objects.py --replay_dir $1 --mask_mode replay --headless --no_auto_record --mesh_path $M/red_brush/red_brush.obj --mesh_path_2 $M/cube_29mm_gold.obj --object_name_2 cube_29mm --upright_meshes cube_29mm_gold --save_dir $out > $out/log.txt 2>&1); }
run $S/takes/heldout_4/replay plain_single
run $S/takes/heldout_4/replay plain_pair_a & run $S/takes/zed1__heldout_4/replay plain_pair_b & wait
for d in plain_single plain_pair_a plain_pair_b; do echo "$d: $(grep -E '^\s+(track_1|track_2|total)' runs_twocam/$d/log.txt | tr -s ' ' | tr '\n' ' ') $(grep -c OutOfMemory runs_twocam/$d/log.txt) OOM"; done
