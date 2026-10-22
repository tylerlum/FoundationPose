#!/usr/bin/env bash
# Build the FoundationPose mycpp pybind11 extension into mycpp/build/.
#
# Pass PYTHON=<absolute-path-to-venv-python> on the command line; the script
# resolves the path before cd-ing into the build directory. See
# docs/venv_isaacsim_install.md §4b for the unified-env install procedure.
#
# Requirements:
#   - System CUDA toolkit on PATH (CUDA_HOME=/usr/local/cuda is what the doc
#     uses; CUDA 12.x is fine — upstream FoundationPose's CUDA 11.8 pin only
#     applies to its torch 2.0 stack, not the mycpp pybind11 extension).
#   - A C++ compiler new enough for C++14 (system gcc-13 works).
#   - Boost (system + program_options), Eigen3, libomp, pybind11 — Boost,
#     Eigen3, libomp from apt; pybind11 from the venv.
set -euo pipefail

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Resolve PYTHON to an absolute path *before* we cd into the build dir below.
PYTHON_INPUT="${PYTHON:-python}"
PYTHON="$(command -v "$PYTHON_INPUT" || { [ -x "$PYTHON_INPUT" ] && readlink -f "$PYTHON_INPUT"; })"
if [ -z "$PYTHON" ]; then
    echo "build_all.sh: cannot resolve PYTHON='$PYTHON_INPUT' to an absolute path" >&2
    exit 1
fi

PYBIND11_DIR=$("$PYTHON" -c "import pybind11; print(pybind11.get_cmake_dir())")

cd "${PROJ_ROOT}/mycpp"
rm -rf build
mkdir -p build
cd build
PYTHON_BIN="$("$PYTHON" -c 'import sys; print(sys.executable)')"

# If anaconda's base env is autosourced from ~/.bashrc, its Boost / libstdc++
# at $CONDA_PREFIX/lib will outrank the system libs at link time and get
# baked in as the .so's RUNPATH. The resulting mycpp.so then fails at import
# with `GLIBCXX_3.4.32 not found` because anaconda's libstdc++ is older than
# gcc-13's. Tell CMake to ignore the anaconda prefix entirely.
IGNORE_PATHS=""
if [ -n "${CONDA_PREFIX:-}" ]; then
    IGNORE_PATHS="-DCMAKE_IGNORE_PATH=${CONDA_PREFIX};${CONDA_PREFIX}/lib;${CONDA_PREFIX}/include"
    IGNORE_PATHS="$IGNORE_PATHS -DCMAKE_IGNORE_PREFIX_PATH=${CONDA_PREFIX}"
fi

cmake .. \
    -Dpybind11_DIR="${PYBIND11_DIR}" \
    -DPYBIND11_FINDPYTHON=ON \
    -DPython_EXECUTABLE="${PYTHON_BIN}" \
    -DPython3_EXECUTABLE="${PYTHON_BIN}" \
    $IGNORE_PATHS
make -j"$(nproc)"

cd "${PROJ_ROOT}"
