"""Put a new SAM3D brush mesh into the old red_brush.obj frame (origin = handle centre, +x = handle -> head).

  python align_new_mesh_to_old_frame.py CAPTURE_DIR NEW_MESH OLD_MESH OUT_DIR
CAPTURE_DIR has rgb/00000.png, depth/00000.npy, cam_K.txt and MASK (sam_input/masks/00000.png one level up).
1. FoundationPose-register both meshes on the same frame -> T_C_new, T_C_old.
2. Start T_old_new = inv(T_C_old) @ T_C_new, refine with ICP (new verts -> old surface), rigid, no scale.
3. Write OUT_DIR/red_brush_sam3d.obj (+ texture) in the old frame, T_old_new.txt, report.json, overlays.
"""
import json
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
    from Utils import nvdiffrast_render, make_mesh_tensors

    cap, new_p, old_p, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    rgb = cv2.cvtColor(cv2.imread(f'{cap}/zed/rgb/00000.png'), cv2.COLOR_BGR2RGB)
    depth = np.load(f'{cap}/zed/depth/00000.npy').astype(np.float32)
    depth[~np.isfinite(depth)] = 0
    K = np.loadtxt(f'{cap}/zed/cam_K.txt').reshape(3, 3)
    mask = cv2.imread(f'{cap}/sam_input/masks/00000.png', cv2.IMREAD_GRAYSCALE) > 0

    def load(p):
        m = trimesh.load(p, force='mesh', process=False)
        if isinstance(getattr(m.visual, 'material', None), trimesh.visual.material.PBRMaterial):
            m.visual.material = m.visual.material.to_simple()
        return m

    new, old = load(new_p), load(old_p)
    print('new extents', np.round(new.extents, 4), 'old extents', np.round(old.extents, 4))
    glctx = dr.RasterizeCudaContext()
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()

    def register(m, tag):
        est = FoundationPose(model_pts=m.vertices, model_normals=m.vertex_normals, mesh=m, scorer=scorer,
                             refiner=refiner, debug=0, glctx=glctx, debug_dir=f'{out}/debug_{tag}')
        T = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
        _, rd, _ = nvdiffrast_render(K=K, H=depth.shape[0], W=depth.shape[1], glctx=glctx,
                                     ob_in_cams=torch.as_tensor(T[None], dtype=torch.float32, device='cuda'),
                                     mesh_tensors=make_mesh_tensors(m))
        rd = rd[0].cpu().numpy()
        sil = rd > 0
        ok = sil & mask & (depth > 0)
        iou = float((sil & mask).sum() / max((sil | mask).sum(), 1))
        err = float(np.median(np.abs(rd[ok] - depth[ok]))) if ok.any() else float('nan')
        print(f'{tag}: IoU {iou:.3f}, median |depth - render| {err * 1000:.1f} mm')
        return T, sil, dict(iou=iou, median_depth_err_mm=err * 1000)

    T_C_new, sil_new, q_new = register(new, 'new')
    T_C_old, sil_old, q_old = register(old, 'old')
    T0 = np.linalg.inv(T_C_old) @ T_C_new   # new-mesh coords -> old-mesh coords
    src = new.sample(20000) if len(new.faces) else new.vertices
    T_icp, _, cost = trimesh.registration.icp(src, old, initial=T0, threshold=1e-6, max_iterations=60,
                                              reflection=False, translation=True, scale=False)
    d0 = trimesh.proximity.closest_point(old, trimesh.transform_points(src, T0))[1]
    d1 = trimesh.proximity.closest_point(old, trimesh.transform_points(src, T_icp))[1]
    print(f'new->old surface distance: FP-only median {np.median(d0) * 1000:.1f} mm, after ICP {np.median(d1) * 1000:.1f} mm')
    dR = np.degrees(np.arccos(np.clip((np.trace(T_icp[:3, :3] @ T0[:3, :3].T) - 1) / 2, -1, 1)))
    print(f'ICP moved the FP estimate by {np.linalg.norm(T_icp[:3, 3] - T0[:3, 3]) * 1000:.1f} mm, {dR:.1f} deg')

    aligned = new.copy()
    aligned.apply_transform(T_icp)
    v = aligned.vertices
    print('aligned bounds', np.round(aligned.bounds, 4), '(old', np.round(old.bounds, 4), ')')
    # Convention check: handle end at -x, head (wide) at +x, origin near the handle middle.
    lo, hi = v[v[:, 0] < np.percentile(v[:, 0], 15)], v[v[:, 0] > np.percentile(v[:, 0], 85)]
    head_plus_x = np.ptp(hi[:, 1]) * np.ptp(hi[:, 2]) > np.ptp(lo[:, 1]) * np.ptp(lo[:, 2])
    print('head at +x:', bool(head_plus_x))
    np.savetxt(f'{out}/T_old_new.txt', T_icp)
    aligned.export(f'{out}/red_brush_sam3d.obj')

    # Overlays: old (green) and aligned-new (magenta) silhouettes, both at T_C_old.
    def sil_at(m, T):
        _, rd, _ = nvdiffrast_render(K=K, H=depth.shape[0], W=depth.shape[1], glctx=glctx,
                                     ob_in_cams=torch.as_tensor(T[None], dtype=torch.float32, device='cuda'),
                                     mesh_tensors=make_mesh_tensors(m))
        return rd[0].cpu().numpy() > 0
    ov = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for s, c in [(mask, (255, 200, 0)), (sil_at(old, T_C_old), (0, 255, 0)), (sil_at(aligned, T_C_old), (255, 0, 255))]:
        cs, _ = cv2.findContours(s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cs, -1, c, 2)
    # Draw the shared object frame at T_C_old.
    def proj(p):
        q = K @ (T_C_old[:3, :3] @ p + T_C_old[:3, 3])
        return tuple(int(x) for x in q[:2] / q[2])
    o = proj(np.zeros(3))
    for ax, c in zip(np.eye(3) * 0.08, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.arrowedLine(ov, o, proj(ax), c, 3, tipLength=0.2)
    cv2.imwrite(f'{out}/overlay_old_green_new_magenta.png', ov)
    json.dump(dict(new=q_new, old=q_old, icp_median_mm=float(np.median(d1) * 1000), fp_only_median_mm=float(np.median(d0) * 1000),
                   new_extents=new.extents.tolist(), old_extents=old.extents.tolist(), head_plus_x=bool(head_plus_x),
                   aligned_bounds=aligned.bounds.tolist()), open(f'{out}/report.json', 'w'), indent=2)
    print('wrote', out)
