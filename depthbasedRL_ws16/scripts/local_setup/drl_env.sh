# ~/.drl_env.sh  --  depthbasedRL / FoundationPose live-tracking environment
# for tylerlum on bohg-ws-16.
#
# Everything here deliberately points at the machine's LOCAL NVMe rather than
# /juno (NFS) or the AFS home: compiles and cache-heavy imports are orders of
# magnitude slower on those.
#
#   drl            # activate env + cd to the repo
#   drl --no-cd    # activate env, stay put
#
# DRL_LOCAL resolves to /home/tylerlum once that exists (needs the one-time
# `sudo bash drl_bootstrap.sh`), otherwise /tmp/tylerlum. Both are the same
# physical NVMe; /tmp is just cleared on reboot.

_drl_pick_local() {
    if [ -n "${DRL_LOCAL_OVERRIDE:-}" ]; then echo "$DRL_LOCAL_OVERRIDE"; return; fi
    if [ -d /home/tylerlum ] && [ -w /home/tylerlum ]; then echo /home/tylerlum; return; fi
    echo /tmp/tylerlum
}

drl() {
    DRL_LOCAL=$(_drl_pick_local); export DRL_LOCAL
    export DRL_ROOT=$DRL_LOCAL/github_repos/depthbasedRL
    export DRL_VENV=$DRL_ROOT/.venv

    # ---- caches: local NVMe, never NFS/AFS -------------------------------
    export XDG_CACHE_HOME=$DRL_LOCAL/.cache
    export UV_CACHE_DIR=$DRL_LOCAL/.cache/uv
    export UV_PYTHON_INSTALL_DIR=$DRL_LOCAL/.local/share/uv/python
    export PIP_CACHE_DIR=$DRL_LOCAL/.cache/pip
    export TORCH_HOME=$DRL_LOCAL/.cache/torch
    export TORCH_EXTENSIONS_DIR=$DRL_LOCAL/.cache/torch_extensions
    export TORCHINDUCTOR_CACHE_DIR=$DRL_LOCAL/.cache/torchinductor
    export TRITON_CACHE_DIR=$DRL_LOCAL/.cache/triton
    export NUMBA_CACHE_DIR=$DRL_LOCAL/.cache/numba
    export HF_HOME=$DRL_LOCAL/.cache/huggingface
    export CUDA_CACHE_PATH=$DRL_LOCAL/.cache/nv
    # warp ignores XDG_CACHE_HOME and falls back to $HOME/.cache/warp,
    # which is AFS on this box. It JIT-compiles FoundationPose's
    # erode_depth / bilateral_filter_depth kernels, so this matters.
    export WARP_CACHE_PATH=$DRL_LOCAL/.cache/warp
    # Private TMPDIR. trimesh unpacks OBJ textures into the system temp
    # dir with fixed names (material_0.png, material.mtl); /tmp already
    # holds kedia-owned copies from his runs, so the shared /tmp gives
    # PermissionError on mesh load.
    export TMPDIR=$DRL_LOCAL/.cache/tmp
    mkdir -p "$TMPDIR"
    export MPLCONFIGDIR=$DRL_LOCAL/.cache/matplotlib

    # ---- toolchain for the three source builds ---------------------------
    # CUDA 12.6, not the 12.8 that /usr/local/cuda points at, so nvcc matches
    # the cu126 torch wheel exactly.
    export CUDA_HOME=/usr/local/cuda-12.6
    # gcc-11: g++-12 is installed here but its cc1plus is missing.
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11
    export CUDAHOSTCXX=/usr/bin/g++-11
    # One GPU on this box (RTX 4090 = sm_89). Building that single arch
    # instead of "8.0;8.6;8.9;9.0" cuts the pytorch3d build ~4x.
    export TORCH_CUDA_ARCH_LIST="8.9"

    export LD_LIBRARY_PATH=/usr/local/zed/lib:${LD_LIBRARY_PATH:-}
    export PATH=$CUDA_HOME/bin:$DRL_LOCAL/.local/bin:$PATH

    # ---- ROS 1 Noetic (rospypi wheels inside the venv) -------------------
    # rospy defaults its log dir to $HOME/.ros, i.e. AFS -- it warns and
    # then loses logs. Keep ROS state on local disk too.
    export ROS_HOME=$DRL_LOCAL/.ros
    export ROS_LOG_DIR=$DRL_LOCAL/.ros/log
    mkdir -p "$ROS_LOG_DIR"
    export ROS_MASTER_URI=${ROS_MASTER_URI:-http://bohg-ws-2.stanford.edu:11311}
    ROS_IP=$(hostname -I 2>/dev/null | awk '{print $1}'); export ROS_IP

    if [ -f "$DRL_VENV/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$DRL_VENV/bin/activate"
    else
        echo "drl: warning: no venv at $DRL_VENV (not installed yet)" >&2
    fi
    [ "${1:-}" != "--no-cd" ] && cd "$DRL_ROOT"
    return 0
}
