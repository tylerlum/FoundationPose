#!/bin/bash
cd "$(dirname "$0")"
until grep -q BATCH_DONE batch_speed2.log 2>/dev/null; do sleep 30; done
FAKE_JOINTS=1 bash run_ros_test.sh fast_wave --sam2_tool --sam2_image_size 640 --sam2_async --sam2_every 2 --lost_metric cover --lost_try_anchor --hand_check ros --catchup_iter 4 --rereg_rot_subsample 3 --lost_max_cooldown_s 2 --log_checks --save_depth > post_ros_hand.log 2>&1
echo BATCH_DONE
