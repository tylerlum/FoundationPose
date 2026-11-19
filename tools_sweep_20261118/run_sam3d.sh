#!/bin/bash
# run_sam3d.sh INPUT_DIR OUTPUT_DIR LOG : run SAM3D run_inference.py, stop its blocking viser once the mesh is saved.
IN=$1; OUT=$2; LOG=$3
SAM3D_REPO=${SAM3D_REPO:-/home/tylerlum/github_repos/sam-3d-objects}
cd "$SAM3D_REPO" && source .venv311/bin/activate
export PYTHONPATH=$SAM3D_REPO LIDRA_SKIP_INIT=1 WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp}
python -u run_inference.py --input_dir "$IN" --output_dir "$OUT" > "$LOG" 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  if grep -aq "http://localhost" "$LOG"; then sleep 2; kill -INT $PID; sleep 3; kill -KILL $PID 2>/dev/null; break; fi
  sleep 2
done
wait $PID 2>/dev/null
grep -a "^size" "$LOG" && ls "$OUT/mesh/mesh.obj" && echo "SAM3D_OK $OUT" || { echo "SAM3D_FAIL $OUT"; tr '\r' '\n' < "$LOG" | tail -5; }
