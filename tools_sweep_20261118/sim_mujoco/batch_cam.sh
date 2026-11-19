#!/bin/bash
cd "$(dirname "$0")"
PYM=/home/tylerlum/.micromamba/envs/simtoolreal_ros/bin/python
PY=/home/tylerlum/github_repos/depthbasedRL/.venv/bin/python
SC="pick_lift occlusion_tour fast_wave heldout_1 heldout_2 heldout_4 heldout_5 heldout_6"
for cam in zed2_161129 zed1; do
  n=0
  for sc in $SC; do
    t=${cam}__$sc; rm -rf takes/$t
    FPSIM_CAM=$cam MUJOCO_GL=egl timeout 1500 $PYM gen_take.py --scenario $sc --out takes/$t 2>&1 | grep -E "wrote|Error" > takes_$t.log &
    n=$((n+1)); [ $((n % 4)) -eq 0 ] && wait
  done; wait
done
$PY stereo_depth.py $(for cam in zed2_161129 zed1; do for sc in $SC; do echo takes/${cam}__$sc; done; done) 2>&1 | grep frames
export RT=0.5
for t in $SC; do bash run_configs.sh $t replay v8 v8b; done            # current camera, new SAM prompt
for cam in zed2_161129 zed1; do for sc in $SC; do bash run_configs.sh ${cam}__$sc replay plain v8 v8b; done; done
echo BATCH_DONE
