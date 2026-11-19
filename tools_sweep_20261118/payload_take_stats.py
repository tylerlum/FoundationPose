"""Summarize payload validity in tracker takes: python3 payload_take_stats.py TAKE_DIR..."""
import json, sys, collections
import statistics as st
for d in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f'{d}/frame_log.jsonl')]
    s = [dict(r, valid=r['payload_valid'], reason=r['payload_reason']) for r in rows]
    if not s or not isinstance(s[0], dict) or 'valid' not in s[0]:
        print(d, 'keys:', list(rows[0].keys())[:20]); continue
    v = [x['valid'] for x in s]; reasons = collections.Counter(x.get('reason', '') for x in s if not x['valid'])
    def q(key, sub):
        a = sorted(x[key] for x in sub if isinstance(x.get(key), (int, float)) and x[key] == x[key])
        return f"p10 {a[len(a)//10]:.3g} med {a[len(a)//2]:.3g} p90 {a[9*len(a)//10]:.3g}" if a else '-'
    print(f"{d.split('/')[-1][:15]}: {sum(v)}/{len(v)} valid ({100*sum(v)/len(v):.0f}%); invalid reasons {dict(reasons)}")
    for k in ('color_px', 'iou', 'depth_px', 'surface_frac', 'jump_m'):
        print(f"   {k:12s} valid: {q(k, [x for x in s if x['valid']])}   invalid: {q(k, [x for x in s if not x['valid']])}")
    # longest invalid run
    run = best = 0
    for x in v:
        run = 0 if x else run + 1; best = max(best, run)
    print(f"   longest invalid run {best} frames")
