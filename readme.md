# FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects
[[Paper]](https://arxiv.org/abs/2312.08344) [[Website]](https://nvlabs.github.io/FoundationPose/)

# TYLER DOCUMENTATION (September 8, 2024)

NOTE: The purpose of this documentation is NOT to be super precise and detailed, but rather to be a quick reference for how to run the code and how it works.

## EXAMPLE VIDEO

This is an example that demonstrates the reasonable robustness FoundationPose, with the help of the Segment Anything Model 2 (SAM2) model to initialize FoundationPose and reset FoundationPose when the object is lost. If the SAM2 mask (very accurate) is very different from the FoundationPose prediction's mask (less accurate), the tracker will reset.

This video shows FoundationPose working at ~40Hz.

[2024-09-08_SAM2_FP_Robust_compressed.webm](https://github.com/user-attachments/assets/892a984a-571d-4451-bf69-15415800981c)

## INPUTS AND OUTPUTS

### FP ROS NODE

```mermaid
flowchart LR
    subgraph "Inputs"
        A["&lt;rgb_topic&gt;"]
        B["&lt;depth_topic&gt;"]
        C["&lt;cam_intrinsics&gt;"]
        D["/sam2_mask"]
        E["/fp_reset"]
    end
    FP["fp_node"]
    subgraph "Outputs"
        F["/object_pose"]
    end
    A --> FP
    B --> FP
    C --> FP
    D --> FP
    E --> FP
    FP --> F
```

* `<rgb_topic>` is the topic for RGB image data
* `<depth_topic>` is the topic for depth image data
* `<cam_intrinsics>` provides the 3x3 camera intrinsics matrix
* `sam2_mask` is the mask of the object provided by the SAM2 node
* `fp_reset` is a boolean trigger to reset the pose estimation model
* `object_pose` is the estimated 6D pose of the detected object (in camera frame)

### FP EVALUATOR ROS NODE

```mermaid
flowchart LR
    subgraph "Inputs"
        G["&lt;cam_intrinsics&gt;"]
        H["/sam2_mask"]
        I["/object_pose"]
    end
    FPE["fp_evaluator_node"]
    subgraph "Outputs"
        J["/iou"]
        K["/fp_reset"]
        L["/fp_mask"]
    end
    G --> FPE
    H --> FPE
    I --> FPE
    FPE --> J
    FPE --> K
    FPE --> L
```

* `<cam_intrinsics>` provides the 3x3 camera intrinsics matrix
* `sam2_mask` is the mask of the object provided by the SAM2 node
* `object_pose` is the estimated 6D pose from the fp_node
* `iou` is the intersection over union of the SAM2 mask and the mask generated from the predicted pose
* `fp_reset` is 0 in normal operation, but 1 if the IOU is less than a threshold, triggering a reset of the fp_node
* `fp_mask` is the segmentation mask generated from the predicted pose

### Parameters

You should set the following ROS parameters:
```
rosparam set /camera zed  # zed or realsense
rosparam set /mesh_file /path/to/mesh.obj  # Object to be tracked
```

This sets the topics to be used

```
if camera == "zed":
    self.rgb_sub_topic = "/zed/zed_node/rgb/image_rect_color"
    self.depth_sub_topic = "/zed/zed_node/depth/depth_registered"
    self.camera_info_sub_topic = "/zed/zed_node/rgb/camera_info"
elif camera == "realsense":
    self.rgb_sub_topic = "/camera/color/image_raw"
    self.depth_sub_topic = "/camera/aligned_depth_to_color/image_raw"
    self.camera_info_sub_topic = "/camera/color/camera_info"
```

## CHANGES

* Addition of `docker/ros_dockerfile` and `docker/run_ros_container.sh` to add ROS Noetic installation with Robostack (https://robostack.github.io/GettingStarted.html)

* Addition of `fp_ros_node.py`, which listens for RGB images, depth images, and mask images, then publishes poses (make sure to set the input mesh and camera intrinsics)

* Addition of `fp_evaluator_ros_node.py`, which listens for RGB images, depth images, mask images, and poses, then publishes predicted masks and a reset message if predicted mask is poor (make sure to set the input mesh and camera intrinsics)

Note that we keep the `fp_ros_node.py` as simple as possible because we need FoundationPose to run at as high a rate as possible (20-50Hz). If it runs slower than that, it will do very poorly because the tracking method only works well with small pose changes. This is achievable when running at a high rate, but severely degrades if running at a low rate (~5Hz). This is why we make a separate `fp_ros_node.py` for debugging and evaluation (to decide if it needs to reset the pose tracker). Also be careful about other things running on the GPU (e.g. mask generator).

## HOW TO RUN

### Weights

Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

### Docker

We needed to use Docker. Using conda alone did not work, despite lots of effort.

```
# May need to add sudo to all docker commands (if not set up for your user)
cd docker/
docker build --network host -f dockerfile -t foundationpose .
docker build --network host -f ros_dockerfile -t ros_foundationpose .
cd ..

bash docker/run_ros_container.sh
```

If you have Docker permission issues, you may need to do the following:
```
# Source: https://medium.com/@praveenadoni4456/error-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket-at-e68bfab8146a
# start docker daemon
sudo systemctl start docker

# enable docker permissions
sudo groupadd docker
sudo usermod -aG docker ${USER}
sudo chmod 666 /var/run/docker.sock
sudo systemctl restart docker
```


If it's the first time you launch the container, you need to build extensions from inside the container.
```
cd FoundationPose
bash build_all.sh
```

Later you can execute into the container without re-build.
```
docker exec -it ros_foundationpose bash
```

If you close the container, open again with
```
bash docker/run_ros_container.sh
```

### Run on offline data

```
docker exec -it ros_foundationpose bash
python run_demo.py
```

If you get this error:
```
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version GLIBCXX_3.4.29' not found (required by /opt/conda/envs/my/lib/python3.9/site-packages/matplotlib/_c_internal_utils.cpython-39-x86_64-linux-gnu.so)
```

Fix like so:
```
echo $CONDA_PREFIX
/opt/conda/envs/my

find $CONDA_PREFIX -name "*libstdc++.so*"
/opt/conda/envs/my/lib/libstdc++.so
/opt/conda/envs/my/lib/libstdc++.so.6
/opt/conda/envs/my/lib/libstdc++.so.6.0.33

export LD_LIBRARY_PATH=/opt/conda/envs/my/lib:$LD_LIBRARY_PATH  # Replace if yours is different
```

### Run real-time pose estimation with ROS

This is a real-time example that uses 4 components:

1. RGB-D Camera
2. Real-Time SAM2 with smart prompting (initializes pose tracker and resets it if object is lost): https://github.com/tylerlum/segment-anything-2-real-time
3. FoundationPose pose tracking (pose tracking): https://github.com/tylerlum/FoundationPose/
4. FoundationPose evaluation for resetting the tracker (converts pose prediction to a predicted mask, and resets the tracker if the mask is poor compared to the SAM2 mask): https://github.com/tylerlum/FoundationPose/

First run the camera with something like:

```
roslaunch realsense2_camera rs_camera.launch align_depth:=true
roslaunch zed_wrapper zed.launch
```

Check you can see the topics:
```
rostopic list  # See expected topics
```

If you are running across PCs, set the following for each terminal:
```
# Set ROS variables if running across PCs
export ROS_MASTER_URI=http://bohg-ws-5.stanford.edu:11311  # Master machine
export ROS_HOSTNAME=$(hostname)  # This machine (e.g., bohg-ws-19.stanford.edu)
```

In a terminal, run the SAM2 node:
```
python sam2_ros_node.py
```

In another terminal, run the FoundationPose node:
```
docker exec -it ros_foundationpose bash

python fp_ros_node.py
```

In another terminal, run the evaluator node (requires display):
```
docker exec -it ros_foundationpose bash

# Run with visualization (requires a display)
python fp_evaluator_ros_node.py --visualize
```

If you don't have a display, you can run without visualization (need some extra setup because of underlying code):
```
docker exec -it ros_foundationpose bash

apt-get update && apt-get install -y \
    libosmesa6 libosmesa6-dev libgl1-mesa-glx

# Run without visualization
PYOPENGL_PLATFORM=osmesa python fp_evaluator_ros_node.py
```

Sanity check that the camera is working by viewing the RGB-D images and the FoundationPose predicted mask (good to compare with the SAM2 mask):
```
rqt_image_view &
```

You can visualize debug signals /sam2_reset and /sam2_num_mask_pixels with:
```
rqt_plot
```

FoundationPose can lose the object and it struggles to recover without being reset. To do this manually, you can run `rostopic pub /fp_reset std_msgs/Int32 "data: 1"`.

# ORIGINAL DOCUMENTATION

This is the official implementation of our paper to be appeared in CVPR 2024 (Highlight)

Contributors: Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield

We present FoundationPose, a unified foundation model for 6D object pose estimation and tracking, supporting both model-based and model-free setups. Our approach can be instantly applied at test-time to a novel object without fine-tuning, as long as its CAD model is given, or a small number of reference images are captured. We bridge the gap between these two setups with a neural implicit representation that allows for effective novel view synthesis, keeping the downstream pose estimation modules invariant under the same unified framework. Strong generalizability is achieved via large-scale synthetic training, aided by a large language model (LLM), a novel transformer-based architecture, and contrastive learning formulation. Extensive evaluation on multiple public datasets involving challenging scenarios and objects indicate our unified approach outperforms existing methods specialized for each task by a large margin. In addition, it even achieves comparable results to instance-level methods despite the reduced assumptions.


<img src="assets/intro.jpg" width="70%">

**🤖 For ROS version, please check [Isaac ROS Pose Estimation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation), which enjoys TRT fast inference and C++ speed up.**

\
**🥇 No. 1 on the world-wide [BOP leaderboard](https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/core-datasets/) (as of 2024/03) for model-based novel object pose estimation.**
<img src="assets/bop.jpg" width="80%">

## Demos

Robotic Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/aa341004-5a15-4293-b3da-000471fd74ed


AR Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/80e96855-a73c-4bee-bcef-7cba92df55ca


Results on YCB-Video dataset:

https://github.com/NVlabs/FoundationPose/assets/23078192/9b5bedde-755b-44ed-a973-45ec85a10bbe



# Bibtex
```bibtex
@InProceedings{foundationposewen2024,
author        = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
title         = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
booktitle     = {CVPR},
year          = {2024},
}
```

If you find the model-free setup useful, please also consider cite:

```bibtex
@InProceedings{bundlesdfwen2023,
author        = {Bowen Wen and Jonathan Tremblay and Valts Blukis and Stephen Tyree and Thomas M\"{u}ller and Alex Evans and Dieter Fox and Jan Kautz and Stan Birchfield},
title         = {{BundleSDF}: {N}eural 6-{DoF} Tracking and {3D} Reconstruction of Unknown Objects},
booktitle     = {CVPR},
year          = {2023},
}
```

# Data prepare


1) Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

1) [Download demo data](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing) and extract them under the folder `demo_data/`

1) [Optional] Download our large-scale training data: ["FoundationPose Dataset"](https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-?usp=sharing)

1) [Optional] Download our preprocessed reference views [here](https://drive.google.com/drive/folders/1PXXCOJqHXwQTbwPwPbGDN9_vLVe0XpFS?usp=sharing) in order to run model-free few-shot version.

# Env setup option 1: docker (recommended)
  ```
  cd docker/
  docker pull wenbowen123/foundationpose && docker tag wenbowen123/foundationpose foundationpose  # Or to build from scratch: docker build --network host -t foundationpose .
  bash docker/run_container.sh
  ```


If it's the first time you launch the container, you need to build extensions.
```
bash build_all.sh
```

Later you can execute into the container without re-build.
```
docker exec -it foundationpose bash
```

# Env setup option 2: conda (experimental)

- Setup conda environment

```bash
# create conda environment
conda create -n foundationpose python=3.9

# activate conda environment
conda activate foundationpose

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```


# Run model-based demo
The paths have been set in argparse by default. If you need to change the scene, you can pass the args accordingly. By running on the demo data, you should be able to see the robot manipulating the mustard bottle. Pose estimation is conducted on the first frame, then it automatically switches to tracking mode for the rest of the video. The resulting visualizations will be saved to the `debug_dir` specified in the argparse. (Note the first time running could be slower due to online compilation)
```
python run_demo.py
```


<img src="assets/demo.jpg" width="50%">


Feel free to try on other objects (**no need to retrain**) such as driller, by changing the paths in argparse.

<img src="assets/demo_driller.jpg" width="50%">


# Run on public datasets (LINEMOD, YCB-Video)

For this you first need to download LINEMOD dataset and YCB-Video dataset.

To run model-based version on these two datasets respectively, set the paths based on where you download. The results will be saved to `debug` folder
```
python run_linemod.py --linemod_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD --use_reconstructed_mesh 0

python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 0
```

To run model-free few-shot version. You first need to train Neural Object Field. `ref_view_dir` is based on where you download in the above "Data prepare" section. Set the `dataset` flag to your interested dataset.
```
python bundlesdf/run_nerf.py --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16 --dataset ycbv
```

Then run the similar command as the model-based version with some small modifications. Here we are using YCB-Video as example:
```
python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 1 --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16
```

# Troubleshooting


- For more recent GPU such as 4090, refer to [this](https://github.com/NVlabs/FoundationPose/issues/27).

- For setting up on Windows, refer to [this](https://github.com/NVlabs/FoundationPose/issues/148).

- If you are getting unreasonable results, check [this](https://github.com/NVlabs/FoundationPose/issues/44#issuecomment-2048141043)

# Training data download
Our training data include scenes using 3D assets from GSO and Objaverse, rendered with high quality photo-realism and large domain randomization. Each data point includes **RGB, depth, object pose, camera pose, instance segmentation, 2D bounding box**. [[Google Drive]](https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-?usp=sharing).

<img src="assets/train_data_vis.png" width="80%">

- To parse the camera params including extrinsics and intrinsics
  ```
  with open(f'{base_dir}/camera_params/camera_params_000000.json','r') as ff:
    camera_params = json.load(ff)
  world_in_glcam = np.array(camera_params['cameraViewTransform']).reshape(4,4).T
  cam_in_world = np.linalg.inv(world_in_glcam)@glcam_in_cvcam
  world_in_cam = np.linalg.inv(cam_in_world)
  focal_length = camera_params["cameraFocalLength"]
  horiz_aperture = camera_params["cameraAperture"][0]
  vert_aperture = H / W * horiz_aperture
  focal_y = H * focal_length / vert_aperture
  focal_x = W * focal_length / horiz_aperture
  center_y = H * 0.5
  center_x = W * 0.5

  fx, fy, cx, cy = focal_x, focal_y, center_x, center_y
  K = np.eye(3)
  K[0,0] = fx
  K[1,1] = fy
  K[0,2] = cx
  K[1,2] = cy
  ```



# Notes
Due to the legal restrictions of Stable-Diffusion that is trained on LAION dataset, we are not able to release the diffusion-based texture augmented data, nor the pretrained weights using it. We thus release the version without training on diffusion-augmented data. Slight performance degradation is expected.

# Acknowledgement

We would like to thank Jeff Smith for helping with the code release; NVIDIA Isaac Sim and Omniverse team for the support on synthetic data generation; Tianshi Cao for the valuable discussions. Finally, we are also grateful for the positive feebacks and constructive suggestions brought up by reviewers and AC at CVPR.

<img src="assets/cvpr_review.png" width="100%">


# License
The code and data are released under the NVIDIA Source Code License. Copyright © 2024, NVIDIA Corporation. All rights reserved.


# Contact
For questions, please contact [Bowen Wen](https://wenbowen123.github.io/).
