"""Contact sheet of overlay.mp4 frames around payload validity changes: python payload_sheet.py TAKE OUT.png"""
import sys, json, cv2, numpy as np
d, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f'{d}/frame_log.jsonl')]; v = np.array([r['payload_valid'] for r in rows])
cap = cv2.VideoCapture(f'{d}/overlay.mp4'); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
n = min(len(frames), len(v)); print('video frames', len(frames), 'log rows', len(v))
bad = np.nonzero(~v[:n])[0]
idx = sorted(set(list(np.linspace(0, n - 1, 6).astype(int)) + (list(bad[np.linspace(0, len(bad) - 1, 6).astype(int)]) if len(bad) else [])))
tiles = []
for i in idx:
    r = rows[i]; f = cv2.resize(frames[i], (480, 270))
    txt = f"#{i} {'VALID' if r['payload_valid'] else 'INVALID:' + r['payload_reason']} col{r['color_px']} iou{r['iou']:.2f} surf{r['surface_frac']:.2f} jump{r['jump_m']*100:.1f}cm"
    cv2.rectangle(f, (0, 0), (480, 22), (0, 0, 0), -1)
    cv2.putText(f, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0) if r['payload_valid'] else (0, 0, 255), 1)
    tiles.append(f)
while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.concatenate([np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)], 0)); print('wrote', out)
