"""SAM2 (our real-time fork, hiera-large, as the tracker uses it) vs SAM3 tracker (transformers, streaming) on a
synthetic take: per-frame time and IoU with the true VISIBLE brush mask (true pose render, unoccluded per exact depth).
python sam_bench.py TAKE [TAKE ...]   (drl env)"""
import sys
import time
sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
import numpy as np  # noqa: E402
import torch  # noqa: E402
import cv2  # noqa: E402
import trimesh  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402
from Utils import nvdiffrast_render, make_mesh_tensors  # noqa: E402
import robust_tool  # noqa: E402

mesh = trimesh.load('/juno/u/tylerlum/tracking_meshes_20261026/red_brush/red_brush.obj', force='mesh', process=False)
mt = make_mesh_tensors(mesh)
glctx = dr.RasterizeCudaContext()


def gt_visible(T, depth_gt, K, tol=0.01):
    _, dep, _ = nvdiffrast_render(K=K, H=540, W=960, ob_in_cams=torch.as_tensor(T[None], dtype=torch.float32,
                                  device='cuda'), glctx=glctx, mesh_tensors=mt)
    dep = dep[0].cpu().numpy()
    return (dep > 0) & (depth_gt >= dep - tol)


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def rich_prompt(m0, n_pos=4, n_neg=6, ring=(12, 30)):
    """Box + n_pos positive points spread inside the mask (farthest-point on the distance-transform core) +
    n_neg negative points on a ring ring[0]..ring[1] px outside it (tells SAM: not the hand / arm next to it)."""
    m = m0.astype(np.uint8)
    box, pt = robust_tool.mask_prompt(m0)
    core = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    cand = np.argwhere(core > 0.5 * core.max())[:, ::-1].astype(np.float32)
    pos = [pt[0]]
    while len(pos) < n_pos and len(cand):
        d = np.min(np.linalg.norm(cand[:, None] - np.array(pos)[None], axis=2), 1)
        pos.append(cand[int(d.argmax())])
    outer = cv2.dilate(m, np.ones((2 * ring[1] + 1,) * 2, np.uint8))
    inner = cv2.dilate(m, np.ones((2 * ring[0] + 1,) * 2, np.uint8))
    rc = np.argwhere((outer > 0) & (inner == 0))[:, ::-1].astype(np.float32)
    rng = np.random.RandomState(0)
    neg = []
    while len(neg) < n_neg and len(rc):
        if not neg:
            neg.append(rc[rng.randint(len(rc))])
            continue
        d = np.min(np.linalg.norm(rc[:, None] - np.array(neg)[None], axis=2), 1)
        neg.append(rc[int(d.argmax())])
    pts = np.array(pos + neg, np.float32)
    lab = np.array([1] * len(pos) + [0] * len(neg), np.int32)
    return box, pts, lab


def run_sam2(frames, m0, size, rich=False):
    sam = robust_tool.SamTool(image_size=size)
    sam.reset(frames[0], m0)
    if rich:
        box, pts, lab = rich_prompt(m0)
        a, b, c = sam._ctx()
        with a, b, c:
            sam.pred.reset_state()
            sam.pred.load_first_frame(frames[0])
            sam.pred.add_new_prompt(frame_idx=0, obj_id=1, bbox=box, points=pts, labels=lab)
    out, ts = [m0], []
    for f in frames[1:]:
        torch.cuda.synchronize(); t = time.time()
        m = sam.track(f)
        torch.cuda.synchronize(); ts.append(time.time() - t)
        out.append(m)
    del sam
    torch.cuda.empty_cache()
    return out, ts


def run_sam3(frames, m0, use_mask=False):
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    dev = 'cuda'
    model = Sam3TrackerVideoModel.from_pretrained('facebook/sam3').to(dev, dtype=torch.bfloat16).eval()
    proc = Sam3TrackerVideoProcessor.from_pretrained('facebook/sam3')
    sess = proc.init_video_session(inference_device=dev, dtype=torch.bfloat16)
    box, pt = robust_tool.mask_prompt(m0)
    out, ts = [], []
    for i, f in enumerate(frames):
        torch.cuda.synchronize(); t = time.time()
        inp = proc(images=f, device=dev, return_tensors='pt')
        if i == 0 and use_mask:
            proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=1,
                                                 input_masks=m0.astype(np.float32), original_size=inp.original_sizes[0])
        elif i == 0:
            proc.add_inputs_to_inference_session(
                inference_session=sess, frame_idx=0, obj_ids=1,
                input_points=[[[list(map(float, np.ravel(pt)[:2]))]]], input_labels=[[[1]]],
                input_boxes=[[list(map(float, np.ravel(box)[:4]))]], original_size=inp.original_sizes[0])
        with torch.inference_mode():
            o = model(inference_session=sess, frame=inp.pixel_values[0])
        m = proc.post_process_masks([o.pred_masks], original_sizes=inp.original_sizes, binarize=True)[0]
        m = m[0, 0].cpu().numpy().astype(bool) if m.ndim == 4 else m[0].cpu().numpy().astype(bool)
        torch.cuda.synchronize()
        if i:
            ts.append(time.time() - t)
        out.append(m)
    del model
    torch.cuda.empty_cache()
    return out, ts


if __name__ == '__main__':
    for take in sys.argv[1:]:
        rd = f'{take}/replay'
        n = len(np.load(f'{take}/gt/T_cam_brush.npy'))
        frames = [cv2.cvtColor(cv2.imread(f'{rd}/rgb/{i:05d}.png'), cv2.COLOR_BGR2RGB) for i in range(n)]
        K = np.loadtxt(f'{rd}/cam_K.txt')
        G = np.load(f'{take}/gt/T_cam_brush.npy')
        gts = [gt_visible(G[i], np.load(f'{rd}/depth_gt/{i:05d}.npy'), K) for i in range(n)]
        m0 = cv2.imread(f'{rd}/masks_1/00000.png', 0) > 0
        res = {}
        import os
        for name, fn in (('sam2_640 (tracker setting)', lambda: run_sam2(frames, m0, 640)),
                         ('sam2_640 +pos/neg points', lambda: run_sam2(frames, m0, 640, rich=True)),
                         ('sam3_tracker box+pt', lambda: run_sam3(frames, m0)),
                         ('sam3_tracker mask prompt', lambda: run_sam3(frames, m0, use_mask=True))):
            masks, ts = fn()
            tiles = []
            for i in range(30, n, max(n // 6, 1))[:6]:
                v = frames[i].copy()
                v[masks[i]] = (0.5 * v[masks[i]] + [0, 120, 0]).astype(np.uint8)
                cs, _ = cv2.findContours(gts[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(v, cs, -1, (255, 0, 255), 2)
                tiles.append(cv2.resize(v, (480, 270)))
            os.makedirs('sam_bench_vis', exist_ok=True)
            cv2.imwrite(f'sam_bench_vis/{take.split("/")[-1]}_{name.split()[0]}_{len(name)}.jpg',
                        cv2.cvtColor(np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:6], 1)], 0),
                                     cv2.COLOR_RGB2BGR))
            ious = np.array([iou(masks[i], gts[i]) for i in range(1, n) if gts[i].sum() > 300])
            bleed = np.array([masks[i].sum() / max(gts[i].sum(), 1) for i in range(1, n) if gts[i].sum() > 300])
            print(f'{take.split("/")[-1]:15s} {name:27s} {np.median(ts) * 1000:6.1f} ms/frame (p90 '
                  f'{np.percentile(ts, 90) * 1000:5.1f})  IoU med {np.median(ious):.2f} p10 {np.percentile(ious, 10):.2f}  '
                  f'frames IoU<0.5: {int((ious < 0.5).sum())}/{len(ious)}  mask/true px med {np.median(bleed):.2f} '
                  f'p90 {np.percentile(bleed, 90):.2f}', flush=True)
