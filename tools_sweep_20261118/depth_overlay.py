"""Mesh at a pose over the depth image: [observed depth | depth with mesh silhouette | |obs - render| residual]."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # third_party/FoundationPose
if __name__ == '__main__':
    import cv2, numpy as np, torch, trimesh, nvdiffrast.torch as dr
    from Utils import nvdiffrast_render, make_mesh_tensors
    depth_p, K_p, mesh_p, pose_p, out_p = sys.argv[1:6]
    d = np.load(depth_p).astype(np.float32); d[~np.isfinite(d)] = 0
    K = np.loadtxt(K_p); T = np.loadtxt(pose_p); mesh = trimesh.load(mesh_p, force='mesh', process=False)
    H, W = d.shape
    _, rd, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=torch.as_tensor(T[None], dtype=torch.float32, device='cuda'),
                                 glctx=dr.RasterizeCudaContext(), mesh_tensors=make_mesh_tensors(mesh))
    rd = rd[0].cpu().numpy(); sil = rd > 0
    lo, hi = np.percentile(d[d > 0], [2, 98])
    cmap = lambda x: cv2.applyColorMap((np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    a = cmap(d); a[d <= 0] = 0
    b = a.copy(); b[sil] = (0.4 * b[sil] + 0.6 * cmap(rd)[sil]).astype(np.uint8)
    cnts, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(b, cnts, -1, (255, 255, 255), 2)
    res = np.abs(d - rd); ok = sil & (d > 0)
    c = (0.3 * a).astype(np.uint8)
    c[ok] = cv2.applyColorMap((np.clip(res[ok] / 0.03, 0, 1) * 255).astype(np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0]
    cv2.drawContours(c, cnts, -1, (255, 255, 255), 1)
    for img, t in [(a, 'observed depth'), (b, 'mesh render blended + outline'), (c, '|obs - render| (0-30 mm)')]:
        cv2.putText(img, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(out_p, np.concatenate([a, b, c], 1))
    print(f'residual on silhouette: median {np.median(res[ok])*1000:.1f} mm, p90 {np.percentile(res[ok], 90)*1000:.1f} mm; wrote {out_p}')
