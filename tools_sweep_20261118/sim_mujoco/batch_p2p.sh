#!/bin/bash
# Point2Pose (model-based) on 6 takes, sequential, after the general batch (GPU memory).
cd "$(dirname "$0")"
until grep -q BATCH_DONE batch_general2.log 2>/dev/null; do sleep 30; done
P2P=/home/tylerlum/github_repos/ext_trackers/point-to-pose/.venv/bin/python
M=/juno/u/tylerlum/tracking_meshes_20261026; KM=/juno/u/tylerlum/kit_tracking_meshes_20261013
for t in fast_wave heldout_4 occlusion_tour screw_task screw_tour_1 wrench_tour_1; do
  case $t in screw*) mesh=$KM/screwdriver_fin_quad_print/round_screwdriver_p1_fin_quad.obj;; wrench*) mesh=$KM/open_wrench_18_red.obj;; *) mesh=$M/red_brush/red_brush.obj;; esac
  rm -rf runs_p2p/$t; mkdir -p runs_p2p/$t
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 1800 $P2P p2p_replay.py takes/$t $mesh runs_p2p/$t > runs_p2p/$t/log.txt 2>&1
  echo "$t exit $? $(tail -1 runs_p2p/$t/log.txt)"
done
echo BATCH_DONE
