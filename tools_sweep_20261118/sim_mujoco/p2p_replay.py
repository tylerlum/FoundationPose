"""Run Point2Pose (model-based) on one of our replay takes; write poses in the tracker's take layout so score_sim.py
scores it. Runs in the point-to-pose env.

  python p2p_replay.py TAKE_DIR MESH.obj OUT_DIR [--crop 540] [--res 512]
Crop: a fixed square viewport (Point2Pose tracks inside one), centred on the first-frame mask, clamped to the image.
SAM2 prompt: box-free, the rich positive/negative points from the first-frame mask.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path('/home/tylerlum/github_repos/ext_trackers/point-to-pose')
sys.path.insert(0, str(REPO))
for sub in ('LightGlue', 'tapnet', 'segment-anything-2-real-time'):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from point2pose.model_tracking.map_builder import MapBuilder  # noqa: E402,F401
from point2pose.data_types.frame import Frame  # noqa: E402
from point2pose.pipeline.model_tracking_pipeline import ModelTrackingPipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('take'); ap.add_argument('mesh'); ap.add_argument('out')
ap.add_argument('--crop', type=int, default=540); ap.add_argument('--res', type=int, default=512)
ap.add_argument('--max_frames', type=int, default=100000)
a = ap.parse_args()
take, out = Path(a.take).resolve(), Path(a.out).resolve()
rd = take / 'replay'
out.mkdir(parents=True, exist_ok=True)
cfg = OmegaConf.load(REPO / 'configs/realsense/model_tracking.yaml')
cfg.model_tracking.params.mesh_path = a.mesh
cfg.model_tracking.params.map_path = str(out / 'map.pkl')
cfg.model_tracking.params.rebuild_map = True
cfg.model_tracking.params.debug_save_views = False
cfg.model_tracking.params.envmap_path = str(REPO / 'assets/envmaps/wooden_lounge_1k.hdr')
cfg.pipeline.params.debug_dir = str(out / 'debug')
cfg.pipeline.params.debug_level = 0
cfg.input.params.crop_size = a.crop
cfg.input.params.track_res = a.res
import os
os.chdir(REPO)                      # config paths (checkpoints/...) are relative to the repo
cfg.segmenter.params.model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'
cfg.segmenter.params.checkpoint = 'checkpoints/sam2.1/sam2.1_hiera_small.pt'
K_full = np.loadtxt(rd / 'cam_K.txt')
rgbs = sorted((rd / 'rgb').glob('*.png'))[:a.max_frames]
stamps = {l.split()[0]: int(l.split()[1]) for l in open(rd / 'ts_ns.txt') if l.strip()}
m0 = cv2.imread(str(rd / 'masks_1' / '00000.png'), 0) > 0
H, W = m0.shape
ys, xs = np.nonzero(m0)
cx, cy = int(xs.mean()), int(ys.mean())
cs = min(a.crop, H, W)
x0 = int(np.clip(cx - cs // 2, 0, W - cs)); y0 = int(np.clip(cy - cs // 2, 0, H - cs))
s = a.res / float(cs)
K = K_full.copy(); K[0, 2] -= x0; K[1, 2] -= y0; K[:2] *= s


def crop(img, interp):
    return cv2.resize(img[y0:y0 + cs, x0:x0 + cs], (a.res, a.res), interpolation=interp)


def load(i):
    p = rgbs[i]
    rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    d = np.load(rd / 'depth' / (p.stem + '.npy')).astype(np.float32)
    return p.stem, rgb, d


pipe = ModelTrackingPipeline(cfg)
t0 = time.time()
pipe.build_map(np.array([[a.res, 0, a.res / 2], [0, a.res, a.res / 2], [0, 0, 1.0]]), (a.res, a.res))
print(f'[p2p] map built in {time.time() - t0:.1f}s', flush=True)
# SAM2 prompt from the first-frame mask (positive points on the mask core, negatives on a ring outside)
sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
mc = crop(m0.astype(np.uint8), cv2.INTER_NEAREST) > 0
core = cv2.distanceTransform(mc.astype(np.uint8), cv2.DIST_L2, 5)
cand = np.argwhere(core > 0.5 * core.max())[:, ::-1].astype(np.float32)
pos = [cand[len(cand) // 2]]
while len(pos) < 4 and len(cand):
    dd = np.min(np.linalg.norm(cand[:, None] - np.array(pos)[None], axis=2), 1); pos.append(cand[int(dd.argmax())])
ring = (cv2.dilate(mc.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0) & ~(cv2.dilate(mc.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
rc = np.argwhere(ring)[:, ::-1].astype(np.float32)
neg = [rc[0]] if len(rc) else []
while 0 < len(neg) < 6:
    dd = np.min(np.linalg.norm(rc[:, None] - np.array(neg)[None], axis=2), 1); neg.append(rc[int(dd.argmax())])
pts = [list(map(float, p)) for p in pos + neg]
labels = [1] * len(pos) + [0] * len(neg)
stem, rgb, d = load(0)
f0 = Frame(id=0, rgb=crop(rgb, cv2.INTER_LINEAR), depth=crop(d, cv2.INTER_NEAREST), intrinsics=K, depth_factor=1.0,
           timestamp=0.0)
pipe.add_user_points(pts, labels)
print('[p2p] segmenter init', pipe.initialize_segmenter(f0), flush=True)
poses, rows, times = [], [], []
for i in range(len(rgbs)):
    stem, rgb, d = load(i)
    fr = Frame(id=i, rgb=crop(rgb, cv2.INTER_LINEAR), depth=crop(d, cv2.INTER_NEAREST), intrinsics=K, depth_factor=1.0,
               timestamp=i / 30.0)
    t = time.time()
    T = pipe.step(fr)[0]
    times.append(time.time() - t)
    lost = bool(pipe.objects[0].lost)
    poses.append(np.asarray(T, np.float64) if not lost else np.full((4, 4), np.nan))
    rows.append(dict(capture_ns=stamps[stem], object_capture_ns=[stamps[stem], stamps[stem]], lost=lost))
    if i % 30 == 0:
        print(f'[p2p] frame {i} lost {lost} step {times[-1] * 1000:.0f} ms', flush=True)
take_out = out / 'p2p_take'
take_out.mkdir(exist_ok=True)
P = np.asarray(poses)
np.save(take_out / 'poses_object_1.npy', P)
np.save(take_out / 'poses_object_2.npy', np.tile(np.eye(4), (len(P), 1, 1)))   # Point2Pose here tracks object 1 only
with open(take_out / 'frame_log.jsonl', 'w') as fh:
    for r in rows:
        fh.write(json.dumps(r) + '\n')
json.dump(dict(duration_s=float(np.sum(times)), n_frames=len(P), step_ms_median=float(np.median(times) * 1000)),
          open(take_out / 'metadata.json', 'w'))
print(f'[p2p] wrote {take_out}: {len(P)} frames, step median {np.median(times) * 1000:.0f} ms, lost {int(np.mean([r["lost"] for r in rows]) * 100)}%')
