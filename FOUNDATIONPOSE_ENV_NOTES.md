# FoundationPose Env Notes

This repo was installed and validated in the Conda env `foundationpose` under:

`/home/tylerlum/miniforge3/envs/foundationpose`

## Conda Path

The originally suggested Conda path did not exist:

`/home/tylerlum/miniconda3/bin/conda`

The working Conda executable on this machine is:

`/home/tylerlum/miniforge3/bin/conda`

Use:

```bash
eval "$(/home/tylerlum/miniforge3/bin/conda shell.bash hook)"
conda activate foundationpose
```

## Non-README Fixes Needed

These were needed in addition to the upstream README.

### `mycpp` build dependencies

The C++ extension build needed Conda-side packages that were not called out in the README:

```bash
conda install -n foundationpose boost boost-cpp eigen pybind11
```

### `transformers` compatibility

The unpinned latest `transformers` version was incompatible with the repo's `torch==2.0.0` setup used here. The working fix was:

```bash
pip install transformers==4.39.3 tokenizers==0.15.2
```

### `numpy` ABI compatibility

At one point Conda upgraded `numpy` to `2.0.2`, which broke SciPy compatibility in this env. The working version here is:

```bash
pip install --force-reinstall --no-deps numpy==1.26.4
```

After RoboStack install, `numpy` remained at `1.26.4`, which is fine.

## RoboStack / ROS Notes

Installed packages:

```bash
conda install ros-noetic-desktop ros-noetic-geometry-msgs ros-noetic-std-msgs -y
```

### What broke during install

The RoboStack install did not work cleanly on the first try. Conda link scripts incorrectly looked for files under the base Miniforge prefix:

- `/home/tylerlum/miniforge3/bin/x86_64-conda-linux-gnu-g++`
- `/home/tylerlum/miniforge3/bin/x86_64-conda-linux-gnu-cpp`
- `/home/tylerlum/miniforge3/setup.sh`
- `/home/tylerlum/miniforge3/etc/conda/activate.d/ros_setup_util.py`

Those files should effectively have been resolved relative to the target env, but the install scripts were not doing that correctly during transaction execution.

### Temporary install-only workarounds

To get the RoboStack transaction through, temporary symlinks were created under the base Miniforge prefix:

1. Symlinks for `x86_64-conda-linux-gnu-*` compiler wrappers from:
   `/home/tylerlum/miniforge3/envs/foundationpose/bin/`
   into:
   `/home/tylerlum/miniforge3/bin/`
2. A temporary `setup.sh` symlink in:
   `/home/tylerlum/miniforge3/setup.sh`
3. A temporary `ros_setup_util.py` symlink in:
   `/home/tylerlum/miniforge3/etc/conda/activate.d/ros_setup_util.py`

These were only used to complete package installation.

### Cleanup status

Those temporary base-prefix symlinks were removed after the ROS install succeeded.

Current status:

- `rostopic` works from the Conda env
- `roscore` works from the Conda env
- there are no remaining temporary base-prefix compiler/setup symlinks from the install workaround

## ROS Runtime Usage

`rostopic list` worked reliably when the master was explicitly bound to localhost:

```bash
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=127.0.0.1
```

Then in one shell:

```bash
eval "$(/home/tylerlum/miniforge3/bin/conda shell.bash hook)"
conda activate foundationpose
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=127.0.0.1
roscore
```

And in another shell:

```bash
eval "$(/home/tylerlum/miniforge3/bin/conda shell.bash hook)"
conda activate foundationpose
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=127.0.0.1
rostopic list
```

Expected output with only the core master running:

```text
/rosout
/rosout_agg
```

## Demo Validation

This checkout does not contain `demo.py` or `run_demo.py`.

The validated demo path in this repo is:

```bash
python extract_poses.py \
  --video_dir demo_data/kinect_driller_seq \
  --mesh_path demo_data/kinect_driller_seq/mesh/textured_mesh.obj \
  --mask_path demo_data/kinect_driller_seq/masks/0000001.png \
  --output_path demo_data/kinect_driller_seq/poses_pred_recheck.json \
  --debug 0
```

This completed successfully and wrote:

`demo_data/kinect_driller_seq/poses_pred_recheck.json`

with:

- `744` poses

## Version State After ROS Install

Important env versions after the successful RoboStack install:

- `python 3.9.18`
- `numpy 1.26.4`
- `boost 1.78.0`
- `boost-cpp 1.78.0`

These differ from the earlier pre-ROS state because RoboStack pulled older compatible packages.
