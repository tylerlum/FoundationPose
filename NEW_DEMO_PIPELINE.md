# New Demo Processing Pipeline

This is the high-level workflow we have been using to turn a recorded RGB-D demo into object meshes and FoundationPose pose tracks.

## Expected Input Layout

Each task/demo folder should be read-only input data and should contain:

```text
<task_name>/
  rgb/
  depth/
  cam_K.txt
  IDEA-Research_grounding-dino-base_masks_<object>/
```

For multi-object scenes, use one mask folder per object:

```text
pourwater_1/
  rgb/
  depth/
  cam_K.txt
  IDEA-Research_grounding-dino-base_masks_cup/
  IDEA-Research_grounding-dino-base_masks_bowl/
```

The batch scripts ignore `*tracking_results*` and `*robot_arm*` directories. Mask folder names are converted into object names by stripping the text around `masks`, so `IDEA-Research_grounding-dino-base_masks_binder_clip` becomes `binder_clip`.

If RGB, depth, and masks have different resolutions, the pipeline treats depth and `cam_K.txt` as authoritative. RGB and masks are resized to the depth resolution where the staging/extraction scripts support it.

## Output Layout

Use a separate output root, ideally outside the source dataset folder:

```text
<output_root>/
  mesh_outputs/
    <task_name>/<object>/
      mesh/mesh.obj
      mesh/material.mtl
      mesh/material_0.png
      original_mesh/
      posed_mesh/
      splat.ply
  pose_outputs/
    <task_name>/<object>/poses.json
  logs/
```

Do not write generated files back into the shared input dataset unless there is an explicit reason to do so.

## Step 1: Generate Meshes with SAM3D

Run SAM3D on each task/object using the task RGB, depth, masks, and `cam_K.txt`. The batch script lives in the segmentation/SAM3D repo:

```bash
/home/tylerlum/github_repos/segment-anything-2-real-time/run_sam3_mesh_batch.sh \
  --data-root /path/to/dataset_root \
  --output-root /path/to/output_root/mesh_outputs
```

Useful options:

```bash
--exclude-tasks visionpro_12,visionpro_13
--dry-run
```

The mesh you usually want for FoundationPose is:

```text
mesh_outputs/<task_name>/<object>/mesh/mesh.obj
```

Use `original_mesh/` and `posed_mesh/` only when you are explicitly debugging SAM3D internals.

## Step 2: Sanity-Check RGB-D and Mesh Scale

Before pose tracking, use `visualize.py` to inspect the RGB-D point cloud and mesh scale.

RGB-D only:

```bash
python visualize.py \
  --video_dir /path/to/task
```

Mesh only:

```bash
python visualize.py \
  --video_dir /path/to/task \
  --mesh_path /path/to/output_root/mesh_outputs/<task>/<object>/mesh/mesh.obj
```

If needed for debugging only, pass a temporary scale:

```bash
python visualize.py \
  --video_dir /path/to/task \
  --mesh_path /path/to/mesh.obj \
  --mesh_scale 20
```

`visualize.py` normally expects a `depth/` folder. If you are using a one-off branch or dataset variant with `depth_gt`, `depth_fs`, etc., check the `depth_dir` selection in `visualize.py` before trusting the visualization.

## Step 3: Run FoundationPose Pose Tracking

For one generated mesh:

```bash
python extract_poses.py \
  --video_dir /path/to/task \
  --mesh_path /path/to/output_root/mesh_outputs/<task>/<object>/mesh/mesh.obj \
  --mask_path /path/to/task/IDEA-Research_grounding-dino-base_masks_<object>/mask_00000.png \
  --output_path /path/to/output_root/pose_outputs/<task>/<object>/poses.json \
  --max_initial_candidates 48
```

For a full dataset after meshes are already generated:

```bash
./run_foundationpose_batch.sh \
  --data-root /path/to/dataset_root \
  --mesh-root /path/to/output_root/mesh_outputs \
  --output-root /path/to/output_root/pose_outputs \
  --exclude-tasks visionpro_12,visionpro_13
```

Useful options:

```bash
--debug 1              # write FoundationPose debug visualizations
--max-candidates 48    # lower candidate count to reduce GPU memory
--dry-run              # inspect planned jobs without running
--no-skip-existing     # overwrite existing poses.json files
```

`run_foundationpose_batch.sh` skips objects whose `poses.json` already exists by default, so it can resume a partially completed run.

## Step 4: Inspect Pose Results

Pose-only trajectory check:

```bash
python visualize.py \
  --video_dir /path/to/task \
  --pose_json_path /path/to/output_root/pose_outputs/<task>/<object>/poses.json
```

Mesh plus poses:

```bash
python visualize.py \
  --video_dir /path/to/task \
  --mesh_path /path/to/output_root/mesh_outputs/<task>/<object>/mesh/mesh.obj \
  --pose_json_path /path/to/output_root/pose_outputs/<task>/<object>/poses.json
```

If the overlay looks wrong, check these first:

- Does `cam_K.txt` match the depth image resolution?
- Are RGB and masks resized to the depth resolution?
- Is the mesh metric scale plausible?
- Is the object mask folder matched to the same object as the mesh?
- Did FoundationPose use the intended `mesh/mesh.obj`, not `original_mesh` or `posed_mesh`?

## Full Dataset Wrapper Example

For the `gt_depth_0429/after_process` dataset, this repo has a wrapper:

```bash
./run_gt_depth_0429_after_process_pipeline.sh \
  --output-root /juno/u/tylerlum/2026-05-01_sam3d_foundationpose_gt_depth_0429_after_process \
  --exclude-tasks visionpro_12,visionpro_13
```

It runs SAM3D first, then FoundationPose, and writes:

```text
<output_root>/mesh_outputs
<output_root>/pose_outputs
<output_root>/logs
```

Use `--dry-run` before long batches to confirm which task/object pairs will run.
