#!/bin/bash
cd "$(dirname "$0")"
bash twocam_fps2.sh > twocam_fps2.log 2>&1
export RT=0.5
for t in pick_lift occlusion_tour fast_wave heldout_1 heldout_2 heldout_4 heldout_5 heldout_6; do bash run_configs.sh $t replay v9nf v9hcnf v4hcc v8c v4npc plainb; done
echo BATCH_DONE
