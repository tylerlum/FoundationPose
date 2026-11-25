"""SAM3 text masks for several prompts on one frame; save each mask and a contact sheet.
  python sam3_prompt_sheet.py RGB.png OUT_DIR ["prompt 1|prompt 2|..."]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # third_party/FoundationPose
import numpy as np, cv2
from text_mask import mask_from_text_sam3
img_p, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rgb = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
prompts = sys.argv[3].split('|') if len(sys.argv) > 3 else ['marker', 'white marker', 'marker pen with gold cap', 'gold cap', 'black marker tip']
tiles = []
for p in prompts:
    m, s = mask_from_text_sam3(rgb, p)
    m = np.zeros(rgb.shape[:2], np.uint8) if m is None else (m > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    box = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
    print(f'{p!r}: score {s:.2f}, {int(m.sum())} px, box {box}')
    cv2.imwrite(f'{out}/mask_{p.replace(" ", "_")}.png', m * 255)
    t = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    t[m > 0] = (0.5 * t[m > 0] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.putText(t, f'{p} {s:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    tiles.append(t[150:450, 190:490])
cv2.imwrite(f'{out}/sheet.png', np.hstack(tiles))
