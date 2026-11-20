#!/bin/bash
cd "$(dirname "$0")"; export RT=0.5
for t in screw_task screw_tour_1 screw_tour_2 wrench_task wrench_tour_1 wrench_tour_2 pick_lift occlusion_tour fast_wave heldout_1 heldout_2 heldout_4 heldout_5 heldout_6; do bash run_configs.sh $t replay genfu; done
echo BATCH_DONE
