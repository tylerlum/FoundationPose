# FoundationPose (SimToolReal Fork)

Fork of [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) adapted for the [SimToolReal](https://simtoolreal.github.io/) perception pipeline. This repo provides four user-facing scripts for recording RGB-D videos, visualizing RGB-D/mesh/pose data, extracting 6D object poses, and running real-time pose tracking with ROS.

## Installation

### 1. Create Conda Environment

```bash
conda create -n foundationpose python=3.9 -y
conda activate foundationpose
```

### 2. Install Compiler Toolchain

CUDA 11.8 requires GCC <= 11. Install both the CUDA toolkit and a compatible GCC via conda:

```bash
# CUDA 11.8 toolkit (must match PyTorch CUDA version)
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y

# GCC 11 (required by CUDA 11.8 nvcc)
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch3D (from source)

PyTorch3D must be compiled from source. Set `CUDA_HOME`, `CC`, and `CXX` to use the conda-installed toolchain:

```bash
export CUDA_HOME=$CONDA_PREFIX
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
```

This takes several minutes to compile CUDA kernels.

### 5. Install nvdiffrast (from source)

```bash
export CUDA_HOME=$CONDA_PREFIX
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
```

### 6. Build C++ Extensions

```bash
bash build_all_conda.sh
```

This builds `mycpp` (pose clustering used by FoundationPose).

### 7. Download Model Weights

Download the FoundationPose pretrained weights from the [original repo](https://github.com/NVlabs/FoundationPose#model-weights) and place them in `weights/`:

```
weights/
  ├── 2023-10-28-18-33-37/   # Scorer model
  └── 2024-01-11-20-02-45/   # Refiner model
```

### 8. Optional: ZED SDK

Required for `record_video.py` and `live_tracking_with_ros.py`:
- Install from [stereolabs.com](https://www.stereolabs.com/developers/release)
- Then: `pip install pyzed`

### 9. Optional: ROS Noetic (via RoboStack)

Required for `live_tracking_with_ros.py`. Install via [RoboStack](https://robostack.github.io/) into the same conda environment:

```bash
conda config --env --add channels robostack-staging
conda config --env --add channels conda-forge
conda config --env --set channel_priority flexible

conda install ros-noetic-desktop
conda install ros-noetic-geometry-msgs ros-noetic-std-msgs
```

After installation, ROS is automatically sourced when you activate the conda environment.

### Verify Installation

```bash
cd /path/to/FoundationPose
python -c "from Utils import *; from estimater import *; from generate_mask import generate_binary_mask_box; import mycpp; print('All imports OK')"
```

## Scripts

### Script 1: Record Video

Record an RGB-D video from a ZED stereo camera.

```bash
python record_video.py \
    --save_dir recordings/ \
    --serial_number 15107 \
    --fps 30
```

Press `Ctrl+C` to stop recording. Output directory structure:
```
recordings/<timestamp>/
  ├── rgb/          # RGB frames as PNGs
  ├── depth/        # Depth frames as 16-bit PNGs (mm)
  ├── cam_K.txt     # 3x3 camera intrinsics
  └── rgb.mp4       # RGB video
```

**Options:** `--width` (960), `--height` (540), `--exposure` (25), `--gain` (40), `--camera_upsidedown`

### Script 2: Visualize RGB-D, Meshes, and Poses

Use `visualize.py` before running FoundationPose to sanity-check the RGB-D data, camera intrinsics, mesh scale, and coordinate conventions. The viewer runs a local Viser web server and shows the RGB-D point cloud, camera frame, optional object mesh, and optional object poses.

First check the RGB-D frames and `cam_K.txt` alone:

```bash
python visualize.py \
    --video_dir recordings/<timestamp>/
```

Then add a mesh without poses to check scale and rough placement. Without a pose JSON, the mesh is placed at the image-center ray and median first-frame depth; this is only for scale/orientation sanity checks, not a predicted pose.

```bash
python visualize.py \
    --video_dir recordings/<timestamp>/ \
    --mesh_path /path/to/object.obj
```

If the mesh appears too small or too large, use a temporary scale factor:

```bash
python visualize.py \
    --video_dir recordings/<timestamp>/ \
    --mesh_path /path/to/object.obj \
    --mesh_scale 20
```

After pose extraction, inspect poses without a mesh to verify trajectory and coordinate frames:

```bash
python visualize.py \
    --video_dir recordings/<timestamp>/ \
    --pose_json_path outputs/<run>/poses.json
```

Finally, visualize RGB-D, mesh, and poses together:

```bash
python visualize.py \
    --video_dir recordings/<timestamp>/ \
    --mesh_path /path/to/object.obj \
    --pose_json_path outputs/<run>/poses.json
```

Open the printed Viser URL in a browser. Use the frame controls to play, pause, step through frames, and toggle the object or point cloud.

The pose JSON can be the output from `extract_poses.py` (`poses_cam`/`poses_robot`) or a raw list of poses. By default, `--pose_key auto` selects `poses_cam`, then `poses_world`, then `poses_robot` when present.

If RGB and depth image sizes differ, `visualize.py` prints a warning and resizes RGB to the depth image size before constructing the point cloud. Depth and `cam_K.txt` are treated as the authoritative geometry.

**Options:** `--pose_key auto|poses_cam|poses_robot|poses_world`, `--port` (8081), `--dt` (0.0333), `--start_idx` (0), `--point_cloud_stride` (1), `--point_cloud_subsample` (1), `--max_depth_m` (5.0), `--mesh_opacity <float>`, `--mesh_scale` (1.0), `--show_point_cloud/--no-show_point_cloud`

For camera-frame data, `visualize.py` uses the identity transform by default so FoundationPose `poses_cam` and RGB-D points appear in the camera frame. To visualize in another world frame, edit `T_W_C` near the top of `visualize.py`.

### Script 3: Extract Poses

Extract 6D object poses from a recorded RGB-D video and an object mesh.

```bash
python extract_poses.py \
    --video_dir recordings/<timestamp>/ \
    --mesh_path /path/to/object.obj \
    --calibration calibration/T_RC_example.txt \
    --output_path poses.json
```

On the first frame, an interactive window opens where you click 4 corners of a bounding box around the object for SAM-based segmentation. FoundationPose then tracks the object through all frames. To skip interactive selection and use a pre-existing mask, pass `--mask_path <mask.png>`.

If RGB, depth, and mask image sizes differ, `extract_poses.py` prints warnings, resizes RGB to the depth image size, and resizes masks to the depth image size with nearest-neighbor interpolation. As in the visualizer, depth and `cam_K.txt` are treated as the authoritative geometry.

**Output format** (`poses.json`):
```json
{
  "poses_cam": [
    [x, y, z, qx, qy, qz, qw],
    ...
  ],
  "poses_robot": [
    [x, y, z, qx, qy, qz, qw],
    ...
  ]
}
```

**Options:** `--mask_path <mask.png>` (skip interactive SAM), `--est_refine_iter` (5), `--track_refine_iter` (2), `--debug` (0/1/2), `--show_window`, `--max_initial_candidates <int>`, `--max_pose_batch <int>`

### Script 4: Live Tracking with ROS

Run real-time 6D pose tracking from a ZED camera and publish poses to ROS topics.

```bash
python live_tracking_with_ros.py \
    --mesh_path /path/to/object.obj \
    --calibration calibration/T_RC_example.txt
```

**Published ROS topics:**
- `camera_frame/current_object_pose` (`PoseStamped`) -- pose in camera frame
- `robot_frame/current_object_pose` (`PoseStamped`) -- pose in robot frame (via `T_RC`)

**Options:** `--serial_number` (15107), `--fps` (40), `--camera_upsidedown`, `--width` (960), `--height` (540), `--debug` (0/1)

## Configuration

### Camera Intrinsics

Camera intrinsics are automatically read from the ZED SDK for live scripts (`record_video.py`, `live_tracking_with_ros.py`). For offline processing (`extract_poses.py`), they are loaded from `cam_K.txt` in the video directory.

### Camera-to-Robot Calibration

The `--calibration` argument accepts a 4x4 homogeneous transform `T_RC` (robot-from-camera) as a `.npy` or `.txt` file. This transform converts poses from camera frame to robot frame: `pose_robot = T_RC @ pose_cam`.

An example calibration file for our lab setup is provided at `calibration/T_RC_example.txt`. **You must replace this with your own calibration for your camera/robot setup.**

### Object Meshes

Object meshes should be `.obj` files with units in **meters**. The mesh origin defines the object coordinate frame for the estimated poses.

## Troubleshooting

- **`unsupported GNU version! gcc versions later than 11 are not supported`**: Make sure you installed GCC 11 via conda (step 2) and set `CC`/`CXX` env vars before compiling PyTorch3D/nvdiffrast.
- **`The detected CUDA version (12.x) mismatches the version that was used to compile PyTorch (11.8)`**: Set `CUDA_HOME=$CONDA_PREFIX` so the build uses the conda-installed CUDA 11.8 toolkit, not the system CUDA.
- **`ModuleNotFoundError: No module named 'torch'` during PyTorch3D build**: Use `--no-build-isolation` flag with pip.
- **`Disabling PyTorch because PyTorch >= 2.1 is required`**: This is a cosmetic warning from `transformers`. SAM mask generation still works correctly with PyTorch 2.0.
