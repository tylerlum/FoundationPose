#!/bin/bash
# bash run_configs.sh TAKE_NAME [replay|replay_gtdepth] [config ...]   -> runs/<take>[_gt]/<config>/<take dir>
S=$(cd "$(dirname "$0")" && pwd); R=${FPSIM_ROOT:-$S}
FP=${FP_DIR:-/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose}
PY=/home/tylerlum/github_repos/depthbasedRL/.venv/bin/python
M=/juno/u/tylerlum/tracking_meshes_20261026
TAKE=$1; RP=${2:-replay}; shift; shift
CFGS=${@:-plain v4}
declare -A CFG=(
  [plain]=""
  [v4]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor"
  [v5]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --arbitrate"
  [v6]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --arbitrate --flip_check_every 3"
  [v7]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --arbitrate color --flip_check_every 3"
  [v8]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --arbitrate color --flip_check_every 3 --lost_max_cooldown_s 2"
  [v8b]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --arbitrate color --flip_check_every 3 --lost_max_cooldown_s 2"
  [v8kf]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --arbitrate color --flip_check_every 3 --lost_max_cooldown_s 2 --fpp_filter --fpp_filter_alpha 0.8"
  [v8kfb]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --arbitrate color --flip_check_every 3 --lost_max_cooldown_s 2 --fpp_filter --fpp_filter_alpha 0.8"
  [v4np]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor"
  [v4npb]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor"
  [rec]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --log_checks"
  [v4_sync]="--sam2_tool --sam2_image_size 640 --sam2_every 2 --lost_metric cover --lost_try_anchor"
)
suffix=""; [ "$RP" = replay_gtdepth ] && suffix=_gt; [ -n "$RT" ] && [ "$RT" != "0" ] && suffix="${suffix}_rt"
for name in $CFGS; do
  out=$R/runs/$TAKE$suffix/$name; rm -rf "$out"; mkdir -p "$out"
  cd $FP && $PY live_tracking_robust_2_objects.py --replay_dir "$R/takes/$TAKE/$RP" --replay_realtime_scale ${RT:-0} --mask_mode replay \
    --headless --no_auto_record --mesh_path $M/red_brush/red_brush.obj --mesh_path_2 $M/cube_29mm_gold.obj \
    --object_name_2 cube_29mm --upright_meshes cube_29mm_gold --save_dir "$out" ${CFG[$name]} $EXTRA > "$out/log.txt" 2>&1
  echo "$TAKE$suffix $name exit $?"
done
