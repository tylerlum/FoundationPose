import sys, glob, numpy as np
sys.path.insert(0, '.')
from score_sim import score
GEN = ['screw_task','screw_tour_1','screw_tour_2','wrench_task','wrench_tour_1','wrench_tour_2']
def one(take, cfg):
    r = glob.glob(f'runs/{take}_rt/{cfg}/2026*/')
    if not r: return None
    o = score(f'takes/{take}', r[0]); b, c = o[1], o[2]
    return dict(add=b['ok2cm'], adds=b['adds_ok2cm'], lost=100*b['lost']/b['n'], rec=b['end_add_mm'] < 20, wp=c['t_med_mm'], wpr=c['r_med_deg'])
for pre, cam in (('', 'ZED2 front'), ('zed1__', 'ZED1')):
    print(f'===== {cam}')
    for cfg in ('plaincal', 'v4npcal', 'gen', 'genx'):
        for grp, takes in (('screw', GEN[:3]), ('wrench', GEN[3:])):
            rs = [(t, one(pre + t, cfg)) for t in takes]; rs = [(t, x) for t, x in rs if x]
            if not rs: continue
            m = lambda k: np.mean([x[k] for _, x in rs])
            print(f'  {cfg:8s} {grp:6s} ADD<2cm {m("add"):5.1f}%  ADD-S<2cm {m("adds"):5.1f}%  rec {sum(x["rec"] for _, x in rs)}/{len(rs)}  workpiece {m("wp"):4.1f} mm {m("wpr"):4.1f} deg | ' + ' '.join(f'{t.split("_",1)[1]}:{x["add"]:.0f}/{x["adds"]:.0f}' for t, x in rs))
