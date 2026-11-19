"""Brush (red) and cube (yellow) mesh outlines at a take's logged poses over raw.mp4 frames (z label uses $TRC; meshes from $MESH_DIR).
python mesh_outline_sheet.py TAKE i0 i1 step OUT.png x0 y0 x1 y1"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # third_party/FoundationPose
if __name__ == '__main__':
    import cv2, numpy as np, torch, trimesh, nvdiffrast.torch as dr
    from Utils import nvdiffrast_render, make_mesh_tensors
    d, i0, i1, st, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    x0, y0, x1, y1 = [int(v) for v in sys.argv[6:10]]
    M = os.environ.get('MESH_DIR', '/juno/u/tylerlum/tracking_meshes_20261026')
    meshes = [make_mesh_tensors(trimesh.load(p, force='mesh', process=False)) for p in (f'{M}/red_brush/red_brush.obj', f'{M}/cube_29mm_gold.obj')]
    P = [np.load(f'{d}/poses_object_1.npy'), np.load(f'{d}/poses_object_2.npy')]
    K = np.loadtxt(f'{d}/K.txt'); glctx = dr.RasterizeCudaContext()
    TRC = np.loadtxt(os.environ.get('TRC', '/juno/u/tylerlum/camera_2_calibration_27432424/record_visual_20261118_172707/T_RC.txt'))  # for the z label only
    cap = cv2.VideoCapture(f'{d}/raw.mp4'); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    H, W = fr[0].shape[:2]; s = W / 960; Ks = K.copy()
    if abs(K[0, 2] * 2 - W) > 0.2 * W: Ks[:2] *= s     # K is for 960x540, video may be bigger
    tiles = []
    for i in range(i0, min(i1, len(fr)), st):
        f = fr[i].copy()
        for k, col in ((0, (0, 0, 255)), (1, (0, 255, 255))):
            _, dep, _ = nvdiffrast_render(K=Ks, H=H, W=W, ob_in_cams=torch.as_tensor(P[k][i][None], dtype=torch.float32, device='cuda'),
                                          glctx=glctx, mesh_tensors=meshes[k])
            sil = (dep[0].cpu().numpy() > 0).astype(np.uint8)
            cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(f, cnts, -1, col, max(1, int(2 * s)))
        f = f[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)]; f = cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1])))
        z = (TRC @ P[0][i])[2, 3]
        cv2.rectangle(f, (0, 0), (480, 20), (0, 0, 0), -1)
        cv2.putText(f, f'#{i} t={i/28.3:.2f}s brush z={z:.3f} (red=brush, yellow=cube)', (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(f)
    while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.concatenate([np.concatenate(tiles[k:k + 4], 1) for k in range(0, len(tiles), 4)], 0)); print('wrote', out, 'video', W, H)
