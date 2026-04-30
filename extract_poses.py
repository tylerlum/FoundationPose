import os
import sys
import time
import json
import argparse
import numpy as np
import cv2
import trimesh
import nvdiffrast.torch as dr
import imageio
from estimater import *
from generate_mask import generate_binary_mask_box
from scipy.spatial.transform import Rotation


def select_mask_with_sam(rgb):
    """
    Use SAM box-based interactive selection to produce a binary mask.
    The SAM helper expects BGR input; convert from RGB.
    """
    bgr = rgb[..., ::-1]
    mask = generate_binary_mask_box(bgr, polygon_refinement=True)
    if mask is None:
        return None
    mask = (mask > 0).astype(np.uint8)
    return mask


def resize_rgb_to_depth(rgb, depth, frame_name):
    depth_h, depth_w = depth.shape[:2]
    if rgb.shape[:2] == (depth_h, depth_w):
        return rgb
    logging.warning(
        f"RGB/depth size mismatch for {frame_name}: rgb={rgb.shape[:2]} depth={depth.shape[:2]}; resizing RGB to depth size"
    )
    return cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)


def resize_mask_to_depth(mask, depth, mask_path):
    depth_h, depth_w = depth.shape[:2]
    if mask.shape[:2] == (depth_h, depth_w):
        return mask
    logging.warning(
        f"Mask/depth size mismatch for {mask_path}: mask={mask.shape[:2]} depth={depth.shape[:2]}; resizing mask to depth size"
    )
    return cv2.resize(mask, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)


def save_vis_mp4(path, frames_rgb, fps=10):
    if not frames_rgb:
        return
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames_rgb:
            writer.write(frame[..., ::-1])
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser(
        description="Extract 6D object poses from a recorded RGB-D video using FoundationPose."
    )
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing rgb/, depth/, and cam_K.txt')
    parser.add_argument('--mesh_path', type=str, required=True,
                        help='Path to object mesh file (.obj, in meters)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for poses JSON (default: <video_dir>/poses.json)')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to camera-to-robot calibration (.npy). If provided, poses are also saved in robot frame.')
    parser.add_argument('--est_refine_iter', type=int, default=5,
                        help='Refinement iterations for initial registration')
    parser.add_argument('--track_refine_iter', type=int, default=2,
                        help='Refinement iterations per tracking step')
    parser.add_argument('--mask_path', type=str, default=None,
                        help='Path to initial mask image (skips interactive SAM selection)')
    parser.add_argument('--debug', type=int, default=0,
                        help='Debug level (0=off, 1=show vis, 2=save vis)')
    parser.add_argument('--show_window', action='store_true',
                        help='Show an OpenCV preview window when debug visualization is enabled.')
    parser.add_argument('--max_pose_batch', type=int, default=None,
                        help='Maximum number of pose candidates to process at once during registration/refinement. Lower this to reduce peak GPU memory.')
    parser.add_argument('--rotation_grid_min_n_views', type=int, default=40,
                        help='Minimum number of sampled view directions for initial pose registration.')
    parser.add_argument('--rotation_grid_inplane_step', type=int, default=60,
                        help='In-plane rotation step in degrees for initial pose registration.')
    parser.add_argument('--max_initial_candidates', type=int, default=None,
                        help='Optional cap on initial pose candidates after symmetry clustering.')
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(args.video_dir, 'poses.json')
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    debug_dir = os.path.dirname(args.output_path)

    set_logging_format()
    set_seed(0)

    # Load camera-to-robot calibration if provided
    T_RC = None
    if args.calibration:
        cal = np.load(args.calibration, allow_pickle=True)
        try:
            cal_arr = np.atleast_1d(cal)
            T_RC = np.array(cal_arr[0]['agent1']['tcr'])
            if T_RC.shape == (3, 4):
                T_RC = np.vstack([T_RC, [0, 0, 0, 1]])
        except (KeyError, IndexError, TypeError):
            T_RC = np.array(cal)
            if T_RC.shape == (3, 4):
                T_RC = np.vstack([T_RC, [0, 0, 0, 1]])

    # Discover frames
    rgb_dir = os.path.join(args.video_dir, 'rgb')
    depth_dir = os.path.join(args.video_dir, 'depth')
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    assert len(rgb_files) == len(depth_files), \
        f"Mismatch: {len(rgb_files)} RGB vs {len(depth_files)} depth frames"
    n_frames = len(rgb_files)
    logging.info(f"Found {n_frames} frames in {args.video_dir}")

    # Load camera intrinsics
    K = np.loadtxt(os.path.join(args.video_dir, 'cam_K.txt')).reshape(3, 3)

    # Load mesh and initialize FoundationPose
    mesh = trimesh.load(args.mesh_path, process=False)
    scorer = ScorePredictor(max_pose_batch=args.max_pose_batch)
    refiner = PoseRefinePredictor(max_pose_batch=args.max_pose_batch)
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
        mesh=mesh, scorer=scorer, refiner=refiner,
        debug=args.debug, glctx=glctx, debug_dir=debug_dir,
        rotation_grid_min_n_views=args.rotation_grid_min_n_views,
        rotation_grid_inplane_step=args.rotation_grid_inplane_step,
        max_initial_candidates=args.max_initial_candidates,
    )

    poses_cam = []  # 6D poses in camera frame: [x, y, z, qx, qy, qz, qw]
    poses_robot = []  # 6D poses in robot frame (if calibration provided)
    vis_array = []

    show_window = args.debug >= 1 and args.show_window

    if show_window:
        cv2.namedWindow('FoundationPose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FoundationPose', 640, 360)

    try:
        for i in range(n_frames):
            logging.info(f"Frame {i}/{n_frames}")
            rgb = cv2.cvtColor(cv2.imread(os.path.join(rgb_dir, rgb_files[i])), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(os.path.join(depth_dir, depth_files[i]), -1) / 1000.0
            rgb = resize_rgb_to_depth(rgb, depth, rgb_files[i])

            if i == 0:
                # Load pre-existing mask or do interactive SAM selection
                if args.mask_path:
                    mask = cv2.imread(args.mask_path, -1)
                    if mask.ndim == 3:
                        mask = mask[..., 0]
                    mask = (mask > 0).astype(np.uint8)
                    mask = resize_mask_to_depth(mask, depth, args.mask_path)
                else:
                    mask = select_mask_with_sam(rgb)
                if mask is None or mask.sum() == 0:
                    print("Empty ROI selected. Exiting.")
                    return

                print("Registering initial pose...")
                t0 = time.time()
                pose = est.register(
                    K=K, rgb=rgb, depth=depth,
                    ob_mask=mask.astype(bool),
                    iteration=args.est_refine_iter
                )
                print(f"Initial registration done in {time.time() - t0:.3f}s")
            else:
                pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=args.track_refine_iter)

            # Convert pose matrix to [x, y, z, qx, qy, qz, qw]
            xyz = pose[:3, 3].tolist()
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()  # [qx, qy, qz, qw]
            poses_cam.append(xyz + quat)

            if T_RC is not None:
                robot_pose = T_RC @ pose
                xyz_r = robot_pose[:3, 3].tolist()
                quat_r = Rotation.from_matrix(robot_pose[:3, :3]).as_quat().tolist()
                poses_robot.append(xyz_r + quat_r)

            # Visualization
            if args.debug >= 1:
                vis = draw_xyz_axis(rgb, ob_in_cam=pose, scale=0.1, K=K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                vis_array.append(vis)
                if show_window:
                    vis_bgr = vis[..., ::-1]
                    cv2.imshow('FoundationPose', vis_bgr)
                    cv2.waitKey(1)

    finally:
        if show_window:
            cv2.destroyAllWindows()

        # Save poses
        output = {"poses_cam": poses_cam}
        if T_RC is not None:
            output["poses_robot"] = poses_robot
        with open(args.output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved {len(poses_cam)} poses to {args.output_path}")

        # Save visualization video
        if args.debug >= 1 and vis_array:
            vis_path = os.path.join(os.path.dirname(args.output_path), "vis.mp4")
            save_vis_mp4(vis_path, vis_array, fps=10)
            print(f"Saved visualization to {vis_path}")


if __name__ == '__main__':
    main()
