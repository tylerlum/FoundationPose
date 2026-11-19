#!/bin/bash
# Re-run one real take (recorded with --save_depth) under several tracker configs and score each.
#   bash compare_replay_configs.sh TAKE_DIR [T_RC.txt]
# Writes TAKE_DIR/compare/<config>/<take>/ and prints score_real_replay.py's table. Needs the drl env; no ROS, no camera.
# Do not run while a live tracker is recording on the same GPU (it will slow both down).
set -u
PY=${PY:-$(command -v python || echo /home/tylerlum/github_repos/depthbasedRL/.venv/bin/python)}   # works without activating drl
TAKE=$1
TRC=${2:-/juno/u/tylerlum/camera_2_calibration_27432424/record_visual_20261118_172707/T_RC.txt}
HERE=$(cd "$(dirname "$0")" && pwd)
FP=$(cd "$HERE/.." && pwd)
M=${MESH_DIR:-/juno/u/tylerlum/tracking_meshes_20261026}
declare -A CFG=(
  [plain]=""
  [sam2_iou_reregister]="--sam2_tool"
  [recommended]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor"
  [recommended_anchorxy]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --fpp_anchor_xy"
  [v4_oldprompt]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --sam2_simple_prompt"
  [v8]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --arbitrate color --flip_check_every 3 --lost_max_cooldown_s 2"
  [v4hc]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2"
  [v8_filter]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --arbitrate color --flip_check_every 3 --lost_max_cooldown_s 2 --fpp_filter --fpp_filter_alpha 0.8"
)
# v4hc needs <take>/replay/hand_center.npy (written by --save_depth when the take ran with --hand_check ros).
# RT=1 (or 0.5 when the GPU is shared): replay skips frames like the live camera ("pretend live"); RT unset/0 =
# every frame in order (sequential, simplest). CONFIGS="plain v8" picks a subset. --log_checks logs colour/appearance.
for name in ${CONFIGS:-plain recommended v4hc v8}; do
  out=$TAKE/compare/$name; rm -rf "$out"; mkdir -p "$out"
  taskset -c 0-9,12-31 $PY "$FP/live_tracking_robust_2_objects.py" --replay_dir "$TAKE/replay" --calibration "$TRC" --replay_realtime_scale ${RT:-0} --log_checks --mask_mode replay \
    --headless --no_auto_record --mesh_path $M/red_brush/red_brush.obj --mesh_path_2 $M/cube_29mm_gold.obj \
    --object_name_2 cube_29mm --upright_meshes cube_29mm_gold --save_dir "$out" ${CFG[$name]} > "$out/log.txt" 2>&1
  echo "$name exit $?"
done
$PY "$HERE/score_real_replay.py" "$TRC" "$TAKE" $TAKE/compare/*/2*/
