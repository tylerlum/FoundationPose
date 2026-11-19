"""Side-by-side mask videos: SAM2 (old box+1pt prompt) | SAM2 (new pos/neg points) | SAM3 tracker (box+1pt) |
SAM3 tracker (mask prompt). Green = tracker mask, magenta outline = true visible brush (sim only).

  python sam_video.py sim TAKE_DIR OUT.mp4
  python sam_video.py real RAW.mp4 OUT.mp4 [start_s] [dur_s]   (frame-0 mask from SAM3 text "red brush")
"""
import sys
import numpy as np
import cv2
import imageio.v2 as imageio
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import sam_bench as B  # noqa: E402

mode, src, out = sys.argv[1:4]
gts = None
if mode == 'sim':
    rd = f'{src}/replay'
    G = np.load(f'{src}/gt/T_cam_brush.npy')
    n = len(G)
    frames = [cv2.cvtColor(cv2.imread(f'{rd}/rgb/{i:05d}.png'), cv2.COLOR_BGR2RGB) for i in range(n)]
    K = np.loadtxt(f'{rd}/cam_K.txt')
    gts = [B.gt_visible(G[i], np.load(f'{rd}/depth_gt/{i:05d}.npy'), K) for i in range(n)]
    m0 = cv2.imread(f'{rd}/masks_1/00000.png', 0) > 0
    fps = 30
else:
    start = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    dur = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    frames = []
    while len(frames) < int(dur * fps):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
    from text_mask import mask_from_text
    r = mask_from_text(frames[0], 'red brush')
    m0 = np.asarray(r[0] if isinstance(r, tuple) else r).astype(bool)
    print('frame-0 text mask px', int(m0.sum()))

runs = [('SAM2 640, box+1 point (old)', lambda: B.run_sam2(frames, m0, 640)),
        ('SAM2 640, +4 pos / 6 neg points (new)', lambda: B.run_sam2(frames, m0, 640, rich=True)),
        ('SAM3 tracker, box+1 point', lambda: B.run_sam3(frames, m0)),
        ('SAM3 tracker, mask prompt', lambda: B.run_sam3(frames, m0, use_mask=True))]
res = []
for name, fn in runs:
    masks, ts = fn()
    res.append((name, masks, float(np.median(ts) * 1000)))
    print(name, f'{res[-1][2]:.1f} ms/frame', flush=True)

w = imageio.get_writer(out, fps=fps, codec='libx264', quality=7, macro_block_size=1)
for i in range(len(frames)):
    tiles = []
    for name, masks, ms in res:
        v = frames[i].copy()
        m = masks[i]
        v[m] = (0.45 * v[m] + [0, 140, 0]).astype(np.uint8)
        label = f'{name}  {ms:.0f} ms'
        if gts is not None:
            cs, _ = cv2.findContours(gts[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cs, -1, (255, 0, 255), 2)
            label += f'  IoU {B.iou(m, gts[i]):.2f}'
        v = cv2.resize(v, (640, 360))
        cv2.rectangle(v, (0, 0), (640, 26), (0, 0, 0), -1)
        cv2.putText(v, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(v)
    g = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    cv2.putText(g, f'frame {i}', (1180, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    w.append_data(g)
w.close()
print('wrote', out)
