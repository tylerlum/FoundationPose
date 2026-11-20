#!/bin/bash
cd "$(dirname "$0")"
until grep -q BATCH_DONE batch_p2p.log 2>/dev/null; do sleep 30; done
export RT=0.5
for t in pick_lift occlusion_tour fast_wave heldout_1 heldout_2 heldout_4 heldout_5 heldout_6; do bash run_configs.sh $t replay v4hcxv; bash run_configs.sh zed1__$t replay v4hcxv; done
echo BATCH_DONE
