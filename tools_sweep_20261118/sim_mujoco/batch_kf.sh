#!/bin/bash
cd "$(dirname "$0")"
until grep -q BATCH_DONE batch_cam.log; do sleep 30; done
export RT=0.5
for t in pick_lift occlusion_tour fast_wave heldout_1 heldout_2 heldout_4 heldout_5 heldout_6; do bash run_configs.sh $t replay v8kf v8kfb; done
echo BATCH_DONE
