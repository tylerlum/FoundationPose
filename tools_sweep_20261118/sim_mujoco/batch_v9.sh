#!/bin/bash
cd "$(dirname "$0")"; export RT=0.5
for t in pick_lift occlusion_tour fast_wave heldout_1 heldout_2 heldout_4 heldout_5 heldout_6; do bash run_configs.sh $t replay v4h v4hc v9 v9h v9hct v4hb v4hcb v9b v9hb v9hctb; done
echo BATCH_DONE
