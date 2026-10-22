# Local-SSD setup on bohg-ws-16 (tylerlum)

Self-contained FoundationPose live-tracking env on bohg-ws-16's local NVMe,
independent of kedia's account. Built 2026-09-19.

```
/home/tylerlum/
├── github_repos/depthbasedRL/     # this repo (528 MB, code + meshes + engines)
│   └── .venv/                     # python 3.11 venv (~21 GB with torch+CUDA)
├── .local/share/uv/python/        # the CPython itself -- NOT on AFS/NFS
├── .cache/{uv,torch,torch_extensions,warp,huggingface,triton,numba,tmp,...}
├── .drl_env.sh                    # the `drl` shell function
└── install_logs/                  # per-stage install logs
```

## Daily use

`drl` is sourced from `~/.bashrc` (guarded, so it is a no-op elsewhere):

```bash
drl                 # activate venv, set caches/toolchain/ROS, cd to the repo
drl --no-cd         # same but stay in the current directory
```

Then the command this whole setup exists for:

```bash
python third_party/FoundationPose/live_tracking_new.py \
  --depth_backend zed_neural --camera_resolution HD1080 --camera_fps 30 --fps 40 \
  --mesh_path assets/urdf/pose_tracker_eval/fabrica_beam/fabrica_beam.obj \
  --save_dir videos/my_run/ --ros
```

Verify at any time: `python /home/tylerlum/verify_env.py` (from the repo root).

## What this env is

A **uv venv on Python 3.11**, not conda. ROS 1 Noetic comes from **rospypi pip
wheels** (`rospy`, `rosmaster`, `sensor_msgs`, ...), not RoboStack and not apt
-- that is what `docs/venv_isaacsim_install.md` specifies and what kedia's
working env actually uses.

**Isaac Sim is deliberately not installed.** `live_tracking_new.py` imports
none of it, and it is ~15 GB. To add it later:
`uv pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com`

Verified versions (match kedia's `.venv_isaacsim`):

| package | version | |
|---|---|---|
| torch | 2.7.0+cu126 | CUDA available, RTX 4090 sm_89 |
| numpy | 1.26.4 | |
| pytorch3d | 0.7.9 | source build |
| nvdiffrast | 0.4.0 | source build |
| tensorrt-cu12 | 10.16.1.11 | **must** match the prebuilt `.engine` files |
| pyzed | 4.2 | ZED SDK 4.2.5 |
| transformers | 5.8.1 | SAM mask generation |
| warp-lang | 1.17.0 | FP depth kernels |

Measured end-to-end on the real camera (HD720, `--track_refine_iter 2`):
capture ready 3.5 s, register 1.8 s, `track_one` **15.5 ms mean -> 64.5 Hz**.

## Why it is fast now (and was not before)

`docs/live_tracking_troubleshooting.md` §4 identified the cause: iowait on an
idle box, because `$HOME` is AFS and the repo is on `/juno` NFS. Four concrete
fixes:

1. **The Python interpreter itself was on NFS.** uv reuses whatever
   `python3.11` is on `$PATH`, which here is
   `/afs/.../.local/bin/python3.11` -> a symlink into `/juno`. The whole
   stdlib was read over the network on every import. This also caused a hard
   build failure: nvdiffrast died with `collect2: fatal error: ld terminated
   with signal 11` because the link line carried `-L/afs/.../python/lib`.
   Forcing a managed CPython onto local disk fixed the segfault outright.
   `install_drl.sh` stage 10 now asserts the interpreter is not under
   `/afs` or `/juno` before installing anything into the venv.
2. **All caches pinned to local NVMe** -- `TORCH_EXTENSIONS_DIR` (nvdiffrast
   and FP's `mycuda` JIT-compile CUDA there), `WARP_CACHE_PATH`, `HF_HOME`,
   `TRITON_CACHE_DIR`, `UV_CACHE_DIR`, `NUMBA_CACHE_DIR`, `MPLCONFIGDIR`.
   Warp needs its own variable: it ignores `XDG_CACHE_HOME` and falls back to
   `$HOME/.cache/warp`, i.e. AFS.
3. **`TORCH_CUDA_ARCH_LIST="8.9"`** instead of `"8.0;8.6;8.9;9.0"`. This box
   has exactly one GPU. pytorch3d builds in **1m12s**.
4. **`CUDA_HOME=/usr/local/cuda-12.6`**, not the `/usr/local/cuda` symlink
   (12.8), so nvcc matches the cu126 torch wheel. And `gcc-11`/`g++-11`,
   because `g++-12` on this box is missing its `cc1plus`. No conda anywhere
   on `PATH`.

## Deviations from kedia's env (deliberate)

- **`numba` dropped.** The original recipe pins `numba==0.59.1`, which is
  *unsatisfiable* with any modern opencv (`numba` wants `numpy<1.27`,
  `opencv-python>=4.12` wants `numpy>=2`). It is imported nowhere in the
  FoundationPose code we run -- the only mentions are comments saying its JIT
  was removed. It reached kedia's env as an Isaac Sim transitive dep. Dropping
  it lets numpy 1.26.4 be pinned *inside* the resolve, so the env is
  internally consistent rather than relying on a post-hoc downgrade.
- **`estimater.py` patched** (one hunk, `reset_object`). Upstream writes the
  mesh to `/tmp/<uuid>.obj`; the `.obj` name is unique but trimesh writes its
  sidecar materials next to it under *fixed* names (`material_0.png`,
  `material.mtl`), which collide with kedia's copies already in the shared
  `/tmp` and raise `PermissionError` on every mesh load. Now each instance
  gets its own `tempfile.mkdtemp()`. `TMPDIR` is also set per-user in `drl`.
- **`.git` not copied.** The 215 MB history is kedia's; `live_tracking_new.py`
  is untracked there anyway. Re-clone from
  `https://github.com/kushal2000/depthbasedRL.git` if you want it.
- **Result dirs not copied** (`videos/`, `train_dir/`, `May21_tests*/`, ...),
  nor `isaacgym/`, `pretrained_policy/`, `rl_games/`. 90 GB -> 528 MB.

## Gotchas that bite

- **Import order: `rospy` BEFORE the FoundationPose stack.** Reversing it
  segfaults reliably -- `live_tracking_utils.py` documents this, and it is
  real in this env (it ate my first smoke test).
- **Group membership.** `/dev/video0` is `root:video` and `/usr/local/zed` is
  group `zed`. Both required; granted via `usermod -aG video,zed tylerlum`.
  Group changes only apply to **new** login sessions.
- **`/tmp` is the same physical disk as `/home`** on this box
  (`/dev/nvme0n1p2`). It is only cleared on reboot -- it was never slower.
- **Camera contention.** A stale capture child holding `/dev/video0` segfaults
  the next run (troubleshooting doc §3). Check with `fuser -v /dev/video0`.

## Rebuilding from scratch

```bash
bash /home/tylerlum/install_drl.sh           # all stages
bash /home/tylerlum/install_drl.sh 30 40     # just the source builds
```
Stages: 00 dirs/uv · 10 python+venv+torch · 20 deps+ROS · 30 pytorch3d+nvdiffrast
· 40 mycpp · 50 editable repo · 60 pyzed · 70 FP weights · 80 smoke tests.

Note stage 60 needs `pip`, which uv venvs lack; the wheel it downloads is
installed directly instead:
`uv pip install /home/tylerlum/pyzed-4.2-cp311-cp311-linux_x86_64.whl`
