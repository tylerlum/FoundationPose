"""Canonical frame for the marker: origin on the axis midway between the cap edge (end of gold) and the start of the
black tip; +x along the axis toward the cap (tip along -x).
  python canonicalize_marker.py RAW_MESH OUT_DIR NAME [REF_MESH_V2]
Used for the 24 Nov marker scans (untaped v2 and taped). Origin = midpoint of the span the untaped marker showed as white.
Writes OUT_DIR/NAME.{obj,glb} (+ texture), T_raw_canonical.txt (p_raw = T @ p_canonical), profile.png, canonical_info.json.
REF_MESH_V2 (optional): old canonical mesh (same convention); reports the axial shift that best matches the radius profile."""
import sys, os, json
import numpy as np, trimesh, cv2
raw_p, out, name = sys.argv[1:4]; ref_p = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(out, exist_ok=True)
def load(p):
    m = trimesh.load(p, force='mesh', process=False)
    if isinstance(m.visual.material, trimesh.visual.material.PBRMaterial): m.visual.material = m.visual.material.to_simple()
    return m
def colored_samples(m, n=80000):
    pts, fid = trimesh.sample.sample_surface(m, n)
    bary = trimesh.triangles.points_to_barycentric(m.triangles[fid], pts)
    uv = (m.visual.uv[m.faces[fid]] * bary[:, :, None]).sum(1)
    img = np.asarray(m.visual.material.image.convert('RGB')); H, W = img.shape[:2]
    px = np.clip((uv[:, 0] % 1) * (W - 1), 0, W - 1).astype(int); py = np.clip((1 - uv[:, 1] % 1) * (H - 1), 0, H - 1).astype(int)
    hsv = cv2.cvtColor(img[py, px][None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0].astype(float)
    return pts, hsv
m = load(raw_p); pts, hsv = colored_samples(m)
gold = (hsv[:, 1] >= 80) & (hsv[:, 0] >= 8) & (hsv[:, 0] <= 35) & (hsv[:, 2] > 80)
white = (hsv[:, 1] < 60) & (hsv[:, 2] > 150); black = hsv[:, 2] < 70
# Axis: PCA, refined as the line through slice centres.
c0 = pts.mean(0); u = np.linalg.svd(pts - c0, full_matrices=False)[2][0]; t = (pts - c0) @ u
if gold[t > np.percentile(t, 90)].mean() < gold[t < np.percentile(t, 10)].mean(): u = -u; t = -t   # +u toward the gold cap
bins = np.linspace(t.min(), t.max(), 41); cen, tc = [], []
for a, b in zip(bins[:-1], bins[1:]):
    s = (t >= a) & (t < b)
    if s.sum() > 50: cen.append((pts[s] - np.outer(t[s], u)).mean(0)); tc.append((a + b) / 2)
coef = np.linalg.lstsq(np.c_[tc, np.ones(len(tc))], np.array(cen), rcond=None)[0]
d = u + coef[0]; d /= np.linalg.norm(d); p0 = coef[1] + c0; t = (pts - p0) @ d
step = 0.002; edges = np.arange(t.min(), t.max() + step, step); prof = []
for a, b in zip(edges[:-1], edges[1:]):
    s = (t >= a) & (t < b); n = max(s.sum(), 1)
    prof.append(((a + b) / 2, white[s].sum() / n, gold[s].sum() / n, black[s].sum() / n, s.sum()))
prof = np.array(prof); ok = prof[:, 4] > 20
# Cap edge: walk from the cap end (max t) toward the tip while gold > 0.15 (gold dips to ~0.5 inside the cap).
i = len(prof) - 1
while i > 0 and (not ok[i] or prof[i, 2] > 0.15): i -= 1
cap_edge = prof[i, 0] + step / 2
# Tip start: walk from the tip end (min t) toward the cap while the slice is not mostly white (black tip).
j = 0
while j < len(prof) - 1 and (not ok[j] or prof[j, 1] < 0.5): j += 1
tip_start = prof[j, 0] - step / 2
tmid = (cap_edge + tip_start) / 2; origin = p0 + tmid * d
x = d; y = np.array([1.0, 0, 0]) - x * x[0]; y /= np.linalg.norm(y); z = np.cross(x, y)
T = np.eye(4); T[:3, :3] = np.c_[x, y, z]; T[:3, 3] = origin
can = m.copy(); can.apply_transform(np.linalg.inv(T))
np.savetxt(f'{out}/T_raw_canonical.txt', T); can.export(f'{out}/{name}.obj'); can.export(f'{out}/{name}.glb')
v = can.vertices; r = np.hypot(v[:, 1], v[:, 2])
info = dict(cap_edge_x_mm=round((cap_edge - tmid) * 1000, 1), tip_start_x_mm=round((tip_start - tmid) * 1000, 1),
            section_length_mm=round((cap_edge - tip_start) * 1000, 1),
            x_range_mm=[round(v[:, 0].min() * 1000, 1), round(v[:, 0].max() * 1000, 1)], extents_m=can.extents.tolist())
def rprof(vv, xs):
    rr = np.hypot(vv[:, 1], vv[:, 2]); outp = []
    for xm in xs:
        s = np.abs(vv[:, 0] - xm) < 0.0015; outp.append(2 * np.percentile(rr[s], 95) if s.sum() > 10 else np.nan)
    return np.array(outp)
for xm in [-60, -45, -30, -15, 0, 15, 30, 45, 60, 80, 100]:
    s = np.abs(v[:, 0] - xm / 1000) < 0.002
    if s.sum(): info[f'diam_mm_at_x{xm:+d}'] = round(float(2 * np.percentile(r[s], 95) * 1000), 1)
if ref_p:
    ref = load(ref_p); xs = np.arange(-0.07, 0.12, 0.001); rn = rprof(v, xs); best = None
    for sh in np.arange(-0.015, 0.0151, 0.0005):
        rr = rprof(ref.vertices, xs + sh); k = np.isfinite(rn) & np.isfinite(rr)
        e = np.mean(np.abs(rn[k] - rr[k])) if k.sum() > 50 else np.inf
        if best is None or e < best[1]: best = (sh, e)
    info['profile_match_shift_mm'] = round(best[0] * 1000, 1); info['profile_match_mean_diam_err_mm'] = round(best[1] * 1000, 2)
    info['ref_x_range_mm'] = [round(ref.vertices[:, 0].min() * 1000, 1), round(ref.vertices[:, 0].max() * 1000, 1)]
json.dump(info, open(f'{out}/canonical_info.json', 'w'), indent=2); print(json.dumps(info, indent=1))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
xs_ = (prof[:, 0] - tmid) * 1000
plt.figure(figsize=(9, 2.6)); plt.plot(xs_, prof[:, 1], 'k-', label='white'); plt.plot(xs_, prof[:, 2], color='orange', label='gold'); plt.plot(xs_, prof[:, 3], 'b-', label='black')
plt.axvline((cap_edge - tmid) * 1000, color='0.5', ls='--'); plt.axvline((tip_start - tmid) * 1000, color='0.5', ls='--'); plt.axvline(0, color='r')
plt.xlabel('canonical x (mm), -x = tip, +x = cap'); plt.ylabel('fraction'); plt.legend(); plt.tight_layout(); plt.savefig(f'{out}/profile.png', dpi=110)
