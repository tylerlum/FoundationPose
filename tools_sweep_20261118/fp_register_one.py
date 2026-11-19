"""Register one object in one RGB-D frame with FoundationPose; save T_cam_obj and 2D overlays.

  python fp_register_one.py RGB.png DEPTH.npy|DEPTH_mm.png MASK.png cam_K.txt MESH.obj OUT_DIR
Writes OUT_DIR/pose_cam.txt (4x4), overlay_mesh.png (rendered mesh blended on the image), overlay_box.png
(FoundationPose demo-style 3D box + axes) and overlay_sidebyside.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # third_party/FoundationPose

if __name__ == '__main__':
    import cv2
    import numpy as np
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
    from Utils import nvdiffrast_render, make_mesh_tensors, draw_posed_3d_box, draw_xyz_axis

    rgb_p, depth_p, mask_p, K_p, mesh_p, out = sys.argv[1:7]
    os.makedirs(out, exist_ok=True)
    rgb = cv2.cvtColor(cv2.imread(rgb_p), cv2.COLOR_BGR2RGB)
    depth = np.load(depth_p).astype(np.float32) if depth_p.endswith('.npy') else \
        cv2.imread(depth_p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    depth[~np.isfinite(depth)] = 0
    mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE) > 0
    K = np.loadtxt(K_p).reshape(3, 3)
    mesh = trimesh.load(mesh_p, process=False, force='mesh')
    print(f'mesh {mesh_p}: {len(mesh.vertices)} verts, extents (m) {np.round(mesh.extents, 4)}')

    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug=0, glctx=glctx,
                         debug_dir=os.path.join(out, 'debug'))
    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
    np.savetxt(os.path.join(out, 'pose_cam.txt'), pose)
    print('T_cam_obj:\n', np.round(pose, 4))

    # Rendered mesh (texture / vertex colour) blended over the image, plus the silhouette outline.
    H, W = depth.shape
    color, rdepth, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=torch.as_tensor(pose[None], dtype=torch.float32,
                                         device='cuda'), glctx=glctx, mesh_tensors=make_mesh_tensors(mesh))
    color = (color[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    sil = rdepth[0].cpu().numpy() > 0
    ov = rgb.copy()
    ov[sil] = (0.35 * rgb[sil] + 0.65 * color[sil]).astype(np.uint8)
    cnts, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 1)
    iou = (sil & mask).sum() / max((sil | mask).sum(), 1)
    resid = np.abs(depth[sil & mask & (depth > 0)] - rdepth[0].cpu().numpy()[sil & mask & (depth > 0)])
    print(f'silhouette IoU vs SAM2 mask {iou:.3f}; |depth - render| median {np.median(resid) * 1000:.1f} mm, '
          f'p90 {np.percentile(resid, 90) * 1000:.1f} mm')

    # FoundationPose demo-style box + axes (oriented bounds of the mesh).
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], 0).reshape(2, 3)
    center_pose = pose @ np.linalg.inv(to_origin)
    box = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=center_pose, bbox=bbox)
    box = draw_xyz_axis(box, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)

    cv2.imwrite(os.path.join(out, 'overlay_mesh.png'), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out, 'overlay_box.png'), cv2.cvtColor(box, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out, 'overlay_sidebyside.png'), cv2.cvtColor(np.concatenate([ov, box], 1),
                                                                         cv2.COLOR_RGB2BGR))
    print('wrote', out)
