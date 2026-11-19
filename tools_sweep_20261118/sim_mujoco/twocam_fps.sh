#!/bin/bash
# Two trackers at once (= two cameras on one GPU) vs one, sequential replays (every frame), v4 + new prompt.
cd "$(dirname "$0")"
FP=/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose; PY=/home/tylerlum/github_repos/depthbasedRL/.venv/bin/python
M=/juno/u/tylerlum/tracking_meshes_20261026
CFG="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor"
run() { out=runs_twocam/$2; rm -rf $out; mkdir -p $out; 
  (cd $FP && $PY live_tracking_robust_2_objects.py --replay_dir $1 --mask_mode replay --headless --no_auto_record --mesh_path $M/red_brush/red_brush.obj --mesh_path_2 $M/cube_29mm_gold.obj --object_name_2 cube_29mm --upright_meshes cube_29mm_gold --save_dir $OLDPWD/$out $CFG > $OLDPWD/$out/log.txt 2>&1); }
S=$PWD
run $S/takes/heldout_4/replay single_zed2
run $S/takes/heldout_4/replay pair_zed2 & run $S/takes/zed1__heldout_4/replay pair_zed1 & wait
for d in single_zed2 pair_zed2 pair_zed1; do echo "$d: $(grep -E '^\s+(track_1|track_2|total)' runs_twocam/$d/log.txt | tr -s ' ' | tr '\n' ' ')"; done
