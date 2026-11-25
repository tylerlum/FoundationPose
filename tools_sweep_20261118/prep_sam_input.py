"""Build a SAM3D input dir from one capture frame: SAM3 'marker' mask + plane-fit fix of off-surface depth.
  python prep_sam_input.py CAPTURE_DIR FRAME_IDX OUT_DIR [PROMPT]"""
import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # third_party/FoundationPose
import numpy as np, cv2
from text_mask import mask_from_text_sam3
cap, idx, I = sys.argv[1], int(sys.argv[2]), sys.argv[3]; prompt = sys.argv[4] if len(sys.argv) > 4 else 'marker'
for d in ('rgb', 'depth', 'masks'): os.makedirs(f'{I}/{d}', exist_ok=True)
bgr = cv2.imread(f'{cap}/rgb/{idx:05d}.png'); d = np.load(f'{cap}/depth/{idx:05d}.npy').astype(np.float64)
mk, sc = mask_from_text_sam3(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), prompt); m = mk > 0
e = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
ys, xs = np.nonzero(e); A = np.c_[xs, ys, np.ones_like(xs)]; z = d[ys, xs]; w = (z > 0).astype(float)
for _ in range(5):
    coef = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)[0]; r = z - A @ coef
    s = 1.4826 * np.median(np.abs(r[w > 0])); w = ((np.abs(r) < 3 * max(s, 1e-3)) & (z > 0)).astype(float)
YY, XX = np.mgrid[:d.shape[0], :d.shape[1]]; fit = coef[0] * XX + coef[1] * YY + coef[2]
bad = m & ((np.abs(d - fit) > 0.02) | (d <= 0)); d2 = d.copy(); d2[bad] = fit[bad]
print(f'frame {idx}: mask score {sc:.2f}, {int(m.sum())} px, plane sigma {s*1000:.1f} mm, replaced {int(bad.sum())} px')
cv2.imwrite(f'{I}/rgb/00000.png', bgr); cv2.imwrite(f'{I}/masks/00000.png', m.astype(np.uint8) * 255)
cv2.imwrite(f'{I}/depth/00000.png', np.clip(d2 * 1000, 0, 65535).astype(np.uint16)); shutil.copy(f'{cap}/cam_K.txt', f'{I}/cam_K.txt')
