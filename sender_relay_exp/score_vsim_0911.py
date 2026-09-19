"""R_g = max_{h!=g} sim_X(g,h) vs -|I_g|.  usage: python3 score_vsim.py <tag> <vsim.jsonl> <igc.jsonl> <dataset.json>
Prints per-set Spearman(R_g, -|I_g|) for X in V / K / Kd / E, plus within-case stats; returns pooled arrays via --dump.
"""
import json, sys, statistics as st

tag, vpath, ipath, dsp = sys.argv[1:5]
DS = json.load(open(dsp))
norm = lambda s: ''.join(ch for ch in str(s).lower() if ch.isalnum())

def margin(lp, gold):
    g = next((k for k in lp if norm(k) == norm(gold)), None)
    if g is None: return None
    others = [v for k, v in lp.items() if k != g]
    return lp[g] - max(others) if others else None

def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j+1]] == v[order[i]]: j += 1
        for k in range(i, j+1): r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r

def pearson(a, b):
    if len(a) < 3: return float('nan')
    ma = st.mean(a); mb = st.mean(b)
    num = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    den = (sum((x-ma)**2 for x in a) * sum((y-mb)**2 for y in b)) ** .5
    return num/den if den else 0.0

def spearman(a, b): return pearson(rank(a), rank(b))

IG = {}
for l in open(ipath):
    r = json.loads(l); gold = DS[r['case']]['Answer'].strip(); mf = margin(r['logp']['full'], gold)
    if mf is None: continue
    I = []
    for g in r['roots']:
        mg = margin(r['logp'][f'g{g}'], gold)
        if mg is None: break
        I.append(mf - mg)
    if len(I) == len(r['roots']) and r['case'] not in IG: IG[r['case']] = I

recs = []; seen = set()
for l in open(vpath):
    r = json.loads(l)
    if r['case'] in seen or r['case'] not in IG: continue
    seen.add(r['case'])
    I = IG[r['case']]; G = len(r['roots'])
    if len(I) != G: continue
    d = dict(case=r['case'], I=I, absI=[abs(v) for v in I], size=r['sizes'])
    for key in ('simV', 'simK', 'simKd', 'simE'):
        M = r.get(key)
        d[key] = [max(M[g][h] for h in range(G) if h != g) for g in range(G)] if M else None
    recs.append(d)

n = len(recs)
print(f"=== {tag}: n={n} cases · |I_g| median {st.median(v for x in recs for v in x['absI']):.3f} · groups {3*n}")
print(f"{'redundancy R_g':16s} {'pooled Spearman(R,-|I|)':>24s} {'pooled Pearson':>15s} {'within-case Spearman med':>25s} {'argmax R = argmin |I|':>22s} {'R_g median (range)':>22s}")
for key, label in (('simV', 'persistent V'), ('simK', 'persistent K (raw)'), ('simKd', 'persistent K (pre-RoPE)'), ('simE', 'encoder feature')):
    sub = [x for x in recs if x[key]]
    if not sub: print(f"{label:16s} (none)"); continue
    R = [v for x in sub for v in x[key]]; NI = [-v for x in sub for v in x['absI']]
    wc = [spearman(x[key], [-v for v in x['absI']]) for x in sub]
    agree = sum(x[key].index(max(x[key])) == x['absI'].index(min(x['absI'])) for x in sub)
    print(f"{label:16s} {spearman(R, NI):>+24.3f} {pearson(R, NI):>+15.3f} {st.median(wc):>+25.2f} {agree:>17d}/{len(sub)} (chance {len(sub)/3:.1f}) {st.median(R):>10.3f} ({min(R):.2f}–{max(R):.2f})")
# size confound: R_g and |I_g| vs size
zs = [v for x in recs for v in x['size']]
print(f"size confound: Spearman(size, |I|) {spearman(zs, [v for x in recs for v in x['absI']]):+.3f} · Spearman(size, R_V) {spearman(zs, [v for x in recs for v in x['simV']]):+.3f}")
# size-matched: within each group-size bucket
print("by group size (Spearman(R_V, -|I|) within bucket):")
for s in sorted(set(zs)):
    R = [x['simV'][i] for x in recs for i in range(3) if x['size'][i] == s]; NI = [-x['absI'][i] for x in recs for i in range(3) if x['size'][i] == s]
    print(f"   {s:5d}: n={len(R)} rho {spearman(R, NI):+.3f} · R_V med {st.median(R):.3f} · |I| med {st.median(-v for v in NI):.3f}")
if '--dump' in sys.argv:
    json.dump(recs, open(f'vsim_{tag}.recs.json', 'w'))
