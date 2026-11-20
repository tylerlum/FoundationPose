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
  [v9]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --arbitrate appearance --flip_check_every 3 --flip_judge appearance --lost_max_cooldown_s 2"
  [v9h]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --arbitrate appearance --flip_check_every 3 --flip_judge appearance --lost_max_cooldown_s 2 --hand_check replay @CAL@"
  [v9hct]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --arbitrate appearance --flip_check_every 3 --flip_judge appearance --lost_max_cooldown_s 2 --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_text_redetect --redetect_prompt red_brush"
  [v4h]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@"
  [v4hc]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2"
  [v9b]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --arbitrate appearance --flip_check_every 3 --flip_judge appearance --lost_max_cooldown_s 2"
  [v9hb]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --arbitrate appearance --flip_check_every 3 --flip_judge appearance --lost_max_cooldown_s 2 --hand_check replay @CAL@"
  [v9hctb]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --arbitrate appearance --flip_check_every 3 --flip_judge appearance --lost_max_cooldown_s 2 --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_text_redetect --redetect_prompt red_brush"
  [v4hb]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@"
  [v4hcb]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2"
  [v9nf]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --arbitrate appearance --lost_max_cooldown_s 2"
  [v9hcnf]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 --arbitrate appearance"
  [v4hcc]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2"
  [v8c]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --arbitrate color --flip_check_every 3 --lost_max_cooldown_s 2"
  [v4npc]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor"
  [plainb]=""
  [v4hct]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 --lost_text_redetect --redetect_prompt red_brush"
  [v4hctb]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 --lost_text_redetect --redetect_prompt red_brush"
  [v4hcx]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 @RSF@"
  [v4hcxb]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 @RSF@"
  [plaincal]="@CAL@"
  [v4npcal]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor @CAL@"
  [gen]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 --sym_axes auto"
  [genb]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 --sym_axes auto"
  [genx]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 --sym_axes auto @RSF@"
  [v4hcxv]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 @RSF@ --reseed_min_cover 0.5"
  [genm]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 --sym_axes auto --mask_obj1_for_obj2"
  [genmb]="--sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check replay @CAL@ --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 --sym_axes auto --mask_obj1_for_obj2"
  [v4_sync]="--sam2_tool --sam2_image_size 640 --sam2_every 2 --lost_metric cover --lost_try_anchor"
)
suffix=""; [ "$RP" = replay_gtdepth ] && suffix=_gt; [ "$RP" = replay_fsit8 ] && suffix=_it8; [ -n "$RT" ] && [ "$RT" != "0" ] && suffix="${suffix}_rt"
for name in $CFGS; do
  out=$S/runs/$TAKE$suffix/$name; rm -rf "$out"; mkdir -p "$out"
  # meshes per task (the tracker flags are the same for every task)
  TASKN=$(cat $R/takes/$TAKE/gt/task.txt 2>/dev/null || echo sweep)
  KM=/juno/u/tylerlum/kit_tracking_meshes_20261013
  case $TASKN in
    screw)  M1=$KM/screwdriver_fin_quad_print/round_screwdriver_p1_fin_quad.obj; M2=$KM/screw_p1_head_bluegrey.obj; N2=screw_head; U2=screw_p1_head_bluegrey ;;
    wrench) M1=$KM/open_wrench_18_red.obj; M2=$KM/bolt_thick_head_bluegrey.obj; N2=bolt_head; U2=bolt_thick_head_bluegrey ;;
    *)      M1=$M/red_brush/red_brush.obj; M2=$M/cube_29mm_gold.obj; N2=cube_29mm; U2=cube_29mm_gold ;;
  esac
  cd $FP && $PY live_tracking_robust_2_objects.py --replay_dir "$S/takes/$TAKE/$RP" --replay_realtime_scale ${RT:-0} --mask_mode replay \
    --headless --no_auto_record --mesh_path $M1 --mesh_path_2 $M2 \
    --object_name_2 $N2 --upright_meshes $U2 --save_dir "$out" $(echo "${CFG[$name]}" | sed "s|@CAL@|--calibration $R/takes/$TAKE/gt/T_RC.txt|; s|@RSF@|--reseed_file $R/rsf/$TAKE.npz|") $EXTRA > "$out/log.txt" 2>&1
  echo "$TAKE$suffix $name exit $?"
done
