#!/usr/bin/env bash
# Install the depthbasedRL FoundationPose live-tracking environment onto
# bohg-ws-16's LOCAL NVMe, for tylerlum.
#
#   bash install_drl.sh            # run every stage in order
#   bash install_drl.sh 30 40      # run only stages 30 and 40
#
# Versions below are pinned to exactly what is installed in kedia's working
# /juno/u/kedia/depthbasedRL/.venv_isaacsim, so this reproduces that env
# minus Isaac Sim (which live_tracking_new.py does not import).
set -uo pipefail

# Root dir on the machine's LOCAL NVMe. /home/tylerlum once the one-time
# `sudo bash drl_bootstrap.sh` has run, else /tmp/tylerlum (same disk, but
# cleared on reboot). Override with DRL_LOCAL_OVERRIDE=...
if [ -n "${DRL_LOCAL_OVERRIDE:-}" ]; then
  LOCAL=$DRL_LOCAL_OVERRIDE
elif [ -d /home/tylerlum ] && [ -w /home/tylerlum ]; then
  LOCAL=/home/tylerlum
else
  LOCAL=/tmp/tylerlum
fi
echo "### DRL_LOCAL = $LOCAL"
ROOT=$LOCAL/github_repos/depthbasedRL
VENV=$ROOT/.venv
PY=$VENV/bin/python
UV=$LOCAL/.local/bin/uv
LOG=$LOCAL/install_logs
mkdir -p "$LOG"

TORCH_IDX=https://download.pytorch.org/whl/cu126
ROSPYPI_IDX=https://rospypi.github.io/simple/

# ---- shared env (mirrors ~/.drl_env.sh) ------------------------------------
export XDG_CACHE_HOME=$LOCAL/.cache
export UV_CACHE_DIR=$LOCAL/.cache/uv
export UV_PYTHON_INSTALL_DIR=$LOCAL/.local/share/uv/python
export PIP_CACHE_DIR=$LOCAL/.cache/pip
export TORCH_HOME=$LOCAL/.cache/torch
export TORCH_EXTENSIONS_DIR=$LOCAL/.cache/torch_extensions
export TRITON_CACHE_DIR=$LOCAL/.cache/triton
export NUMBA_CACHE_DIR=$LOCAL/.cache/numba
export HF_HOME=$LOCAL/.cache/huggingface
export CUDA_CACHE_PATH=$LOCAL/.cache/nv
export WARP_CACHE_PATH=$LOCAL/.cache/warp
export MPLCONFIGDIR=$LOCAL/.cache/matplotlib
export CUDA_HOME=/usr/local/cuda-12.6
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="8.9"
export PATH=$CUDA_HOME/bin:$LOCAL/.local/bin:$PATH

say() { echo; echo "############ $* ############"; }
run() { echo "+ $*"; "$@"; local rc=$?; [ $rc -ne 0 ] && echo "!! FAILED (rc=$rc): $*"; return $rc; }

stage_00_dirs() {
  say "00  local dirs + uv on local disk"
  mkdir -p "$LOCAL"/.cache/{uv,pip,torch,torch_extensions,torchinductor,triton,numba,huggingface,nv,matplotlib,ov}
  mkdir -p "$LOCAL/.local/bin" "$LOCAL/.local/share/uv/python" "$LOCAL/github_repos"
  # uv currently lives in AFS; keep a local copy so it is not read over AFS.
  if [ ! -x "$UV" ]; then
    cp "$(command -v uv)" "$UV" && chmod +x "$UV"
  fi
  run "$UV" --version
}

stage_10_venv() {
  say "10  local CPython 3.11 + venv + torch 2.7.0 cu126"
  # CRITICAL: uv will happily reuse the python3.11 already on PATH, which on
  # this box is /afs/.../.local/bin/python3.11 -> a symlink into /juno NFS.
  # A venv built on it reads the interpreter and the whole stdlib over the
  # network on every import, and drags AFS/NFS -L paths into every native
  # link step. Force a managed interpreter into $UV_PYTHON_INSTALL_DIR
  # (local NVMe) and point the venv at it by absolute path.
  run "$UV" python install 3.11.13 --reinstall || return 1
  LOCAL_PY=$(ls -d "$UV_PYTHON_INSTALL_DIR"/cpython-3.11.13-*/bin/python3.11 2>/dev/null | head -1)
  if [ -z "$LOCAL_PY" ]; then
    echo "!! no managed python under $UV_PYTHON_INSTALL_DIR"; return 1
  fi
  case "$(readlink -f "$LOCAL_PY")" in
    /afs/*|/juno/*) echo "!! managed python still resolves off local disk: $(readlink -f "$LOCAL_PY")"; return 1 ;;
  esac
  echo "    local interpreter: $LOCAL_PY"

  rm -rf "$VENV"
  run "$UV" venv "$VENV" --python "$LOCAL_PY" --relocatable || return 1

  # Prove the venv is entirely on local disk before installing 5 GB into it.
  run "$PY" -c "
import sys, sysconfig
base = sys.base_prefix
print('base_prefix:', base)
print('stdlib     :', sysconfig.get_paths()['stdlib'])
assert not base.startswith(('/afs', '/juno')), 'venv base interpreter is NOT on local disk'
print('OK: interpreter is on local disk')
" || return 1

  run "$UV" pip install --python "$PY" \
      "torch==2.7.0" "torchvision==0.22.0" --index-url "$TORCH_IDX" || return 1
  run "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda)"
}

stage_20_deps() {
  say "20  FoundationPose + Fast-FoundationStereo + misc deps"
  # numpy is pinned INSIDE the resolve (not re-pinned afterwards like the
  # original doc does) so the whole env is internally consistent. kedia's env
  # has numpy 1.26.4 next to an opencv that declares numpy>=2 -- that only
  # "works" because the downgrade happened in a later, separate invocation.
  #
  # numba is deliberately absent: it is NOT imported anywhere in the
  # FoundationPose code we run (the only mentions are comments saying its JIT
  # was removed). In kedia's env it arrived as an Isaac Sim transitive dep.
  # Keeping it would force numpy<1.27 AND block every modern opencv.
  #
  # Hard pins are only the ones that actually matter:
  #   tensorrt-cu12  -- must match the prebuilt .engine files
  #   cuda-python    -- fp_trt_runner.py does `from cuda import cudart`;
  #                     live_tracking_new.py loads the bundled TRT refiner
  #                     (refine_net_fp16.plan) BY DEFAULT, so this is required,
  #                     not optional
  #   transformers   -- SAM mask generation
  #   numpy          -- matches kedia's working env
  run "$UV" pip install --python "$PY" \
    "numpy==1.26.4" \
    "tensorrt-cu12==10.16.1.11" \
    "cuda-python==12.9.4" \
    "transformers==5.8.1" \
    "fvcore==0.1.5.post20221221" \
    "pyrender==0.1.45" \
    kornia open3d opencv-python opencv-contrib-python \
    PyOpenGL PyOpenGL_accelerate \
    scikit-learn scikit-image ninja pybind11 \
    ruamel.yaml colorama joblib roma transformations trimesh \
    imageio imageio-ffmpeg h5py psutil pyglet Rtree timm \
    warp-lang matplotlib scipy pillow viser omegaconf einops \
    onnxruntime-gpu gdown pyyaml setuptools wheel || return 1

  say "20b ROS 1 Noetic python client (rospypi wheels)"
  run "$UV" pip install --python "$PY" --extra-index-url "$ROSPYPI_IDX" \
    rospy rosgraph rosmaster \
    std_msgs sensor_msgs geometry_msgs nav_msgs \
    tf2_ros tf2_msgs cv_bridge || return 1

  say "20c verify numpy did not get pulled forward by the ROS wheels"
  run "$PY" -c "
import numpy, torch, cv2, warp
print('numpy      ', numpy.__version__)
print('torch      ', torch.__version__, torch.cuda.is_available())
print('cv2        ', cv2.__version__)
print('warp       ', warp.__version__)
assert numpy.__version__.startswith('1.26'), 'numpy drifted off 1.26.x'
"
}

stage_30_builds() {
  say "30  source builds: pytorch3d + nvdiffrast (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
  echo "    CUDA_HOME=$CUDA_HOME  CXX=$CXX"
  run "$UV" pip install --python "$PY" --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git" || return 1
  run "$UV" pip install --python "$PY" --no-build-isolation \
      "git+https://github.com/NVlabs/nvdiffrast" || return 1
  run "$PY" -c "import pytorch3d,nvdiffrast;print('pytorch3d',pytorch3d.__version__)"
}

stage_40_mycpp() {
  say "40  mycpp pybind11 extension"
  ( cd "$ROOT" && PYTHON="$PY" bash third_party/FoundationPose/build_all.sh ) || return 1
  ls -l "$ROOT"/third_party/FoundationPose/mycpp/build/mycpp*.so
}

stage_50_repo() {
  say "50  editable install of repo-local packages"
  ( cd "$ROOT" && run "$UV" pip install --python "$PY" -e . --no-deps )
}

stage_60_pyzed() {
  say "60  pyzed from the system ZED SDK"
  if [ ! -r /usr/local/zed/get_python_api.py ]; then
    echo "!! /usr/local/zed is not readable by $(whoami)."
    echo "   Needs: sudo usermod -aG zed tylerlum   (then log out and back in)"
    return 1
  fi
  ( cd "$LOCAL" && run "$PY" /usr/local/zed/get_python_api.py )
  run "$PY" -c "import pyzed.sl as sl;print('ZED SDK',sl.Camera.get_sdk_version())"
}

stage_70_weights() {
  say "70  FoundationPose pretrained weights (~248 MB, public NVlabs Drive folder)"
  W=$ROOT/third_party/FoundationPose/weights
  mkdir -p "$W"
  if [ -r "$W/2023-10-28-18-33-37/model_best.pth" ] && [ -r "$W/2024-01-11-20-02-45/model_best.pth" ]; then
    echo "    already present, skipping"
  else
    run "$PY" -m gdown --folder \
      'https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i' \
      -O "$W/" || return 1
    if [ -d "$W/no_diffusion" ]; then
      mv "$W/no_diffusion/2023-10-28-18-33-37" "$W/" 2>/dev/null
      mv "$W/no_diffusion/2024-01-11-20-02-45" "$W/" 2>/dev/null
      rmdir "$W/no_diffusion" 2>/dev/null
    fi
  fi
  ls -l "$W"/*/ 2>&1 | head -20
}

stage_80_smoke() {
  say "80  smoke tests"
  ( cd "$ROOT" && "$PY" - <<'PY'
import sys, os
sys.path.insert(0, 'third_party/FoundationPose')
import torch
print('torch      :', torch.__version__, 'cuda avail:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
import numpy; print('numpy      :', numpy.__version__)
import pytorch3d; print('pytorch3d  :', pytorch3d.__version__)
import nvdiffrast.torch as dr; print('nvdiffrast : ok')
from Utils import mycpp
print('mycpp      :', 'ok' if mycpp is not None else 'MISSING (rerun stage 40)')
from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
print('estimater  : imports ok')
try:
    import pyzed.sl as sl; print('pyzed      :', sl.Camera.get_sdk_version())
except Exception as e:
    print('pyzed      : FAIL', e)
import rospy, geometry_msgs.msg; print('rospy      : ok')
import tensorrt; print('tensorrt   :', tensorrt.__version__)
import transformers; print('transformers:', transformers.__version__)
import viser; print('viser      : ok')
PY
  )
  say "80b FoundationPose weight load"
  ( cd "$ROOT" && "$PY" -c "
import sys; sys.path.insert(0,'third_party/FoundationPose')
from estimater import ScorePredictor, PoseRefinePredictor
ScorePredictor(); PoseRefinePredictor(); print('FP weights load OK')
" )
}

STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
  STAGES=(00 10 20 30 40 50 60 70 80)
fi
for s in "${STAGES[@]}"; do
  fn=$(declare -F | awk '{print $3}' | grep "^stage_${s}_" | head -1)
  if [ -z "$fn" ]; then echo "no such stage: $s"; continue; fi
  "$fn" 2>&1 | tee "$LOG/stage_${s}.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then echo "### stage $s FAILED (rc=$rc); see $LOG/stage_${s}.log"; fi
done
echo; echo "install_drl.sh done. Logs in $LOG"
