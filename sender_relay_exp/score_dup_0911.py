"""Duplication test: g -> g u g (exact K/V copy of one group appended). Native Answerer.
usage: python3 score_dup.py <tag> <igdup.jsonl> <dataset.json> [igc.jsonl for I_g]
"""
import json, sys, statistics as st

tag, path, dsp = sys.argv[1:4]
igc = sys.argv[4] if len(sys.argv) > 4 else None
DS = json.load(open(dsp))
norm = lambda s: ''.join(ch for ch in str(s).lower() if ch.isalnum())

def margin(lp, gold):
    g = next((k for k in lp if norm(k) == norm(gold)), None)
    if g is None: return None
    others = [v for k, v in lp.items() if k != g]
    return lp[g] - max(others) if others else None

def argmax(lp): return max(lp, key=lp.get)

IG = {}
if igc:
    for l in open(igc):
        r = json.loads(l); gold = DS[r['case']]['Answer'].strip(); mf = margin(r['logp']['full'], gold)
        if mf is None: continue
        I = []
        for g in r['roots']:
            mg = margin(r['logp'][f'g{g}'], gold)
            if mg is None: break
            I.append(mf - mg)
        if len(I) == len(r['roots']): IG[r['case']] = I

recs = []
seen = set()
for l in open(path):
    r = json.loads(l)
    if r['case'] in seen: continue
    seen.add(r['case']); gold = DS[r['case']]['Answer'].strip(); lp = r['logp']
    if 'full' not in lp: continue
    mf = margin(lp['full'], gold); pf = argmax(lp['full'])
    idn = lp.get('dup_none'); ident = max(abs(lp['full'][y] - idn[y]) for y in lp['full']) if idn else None
    sizes = [sum(r['pages'][p] for p in r['groups'][str(g)]) for g in r['roots']]
    dm, flip, dlp_gold, dlp_max = [], [], [], []
    for g in r['roots']:
        d = lp.get(f'dup{g}')
        if d is None: break
        dm.append(margin(d, gold) - mf); flip.append(argmax(d) != pf)
        gk = next(k for k in d if norm(k) == norm(gold))
        dlp_gold.append(d[gk] - lp['full'][gk])
        dlp_max.append(max(abs(d[y] - lp['full'][y]) for y in d))
    if len(dm) != len(r['roots']): continue
    recs.append(dict(case=r['case'], roots=r['roots'], size=sizes, m_full=mf, dm=dm, flip=flip,
                     dlp_gold=dlp_gold, dlp_max=dlp_max, ident=ident, I=IG.get(r['case'])))

n = len(recs)
print(f"=== {tag}: n={n} cases · identity (dup_none vs full) max|Δlogp| {max(x['ident'] for x in recs if x['ident'] is not None):.4f}")
alld = [v for x in recs for v in x['dm']]
print(f"Δmargin(dup g − full): median {st.median(alld):+.3f} nat · |Δ|>0.5: {sum(abs(v)>0.5 for v in alld)}/{len(alld)} · "
      f"|Δ|>1.0: {sum(abs(v)>1.0 for v in alld)} · max|Δ any cand logp| median {st.median(v for x in recs for v in x['dlp_max']):.3f}")
fl = [v for x in recs for v in x['flip']]
print(f"prediction (argmax over candidates) changes under duplication: {sum(fl)}/{len(fl)} group-duplications · cases with any flip {sum(any(x['flip']) for x in recs)}/{n}")
print(f"gold logp shift median {st.median(v for x in recs for v in x['dlp_gold']):+.3f}")
# by group size
bys = {}
for x in recs:
    for s, d, f in zip(x['size'], x['dm'], x['flip']): bys.setdefault(s, []).append((d, f))
print("by duplicated-group size: size  n  Δmargin med  |Δ|>0.5  flips")
for s in sorted(bys):
    v = bys[s]; print(f"  {s:5d} {len(v):3d}  {st.median(d for d, _ in v):+.3f}      {sum(abs(d)>0.5 for d,_ in v):3d}    {sum(f for _, f in v):3d}")
if any(x['I'] for x in recs):
    # sign consistency: does duplicating a useful group (I_g>0) raise the margin?
    pairs = [(i, d) for x in recs if x['I'] for i, d in zip(x['I'], x['dm'])]
    same = sum((i > 0) == (d > 0) for i, d in pairs if abs(i) > 0.5 and abs(d) > 0.5)
    tot = sum(1 for i, d in pairs if abs(i) > 0.5 and abs(d) > 0.5)
    ma = st.mean(i for i, _ in pairs); mb = st.mean(d for _, d in pairs)
    num = sum((i-ma)*(d-mb) for i, d in pairs); den = (sum((i-ma)**2 for i,_ in pairs)*sum((d-mb)**2 for _,d in pairs))**.5
    print(f"vs I_g (removal utility): sign(I_g)==sign(Δmargin_dup) in {same}/{tot} (both |.|>0.5) · pooled r {num/den if den else 0:+.2f}")
print("case  size            Δmargin dup g          flip      I_g")
for x in recs[:14]:
    print(f"{x['case']:4d}  {str(x['size']):15s} {' '.join(f'{v:+.2f}' for v in x['dm']):22s} {''.join('F' if f else '.' for f in x['flip']):5s}  {' '.join(f'{v:+.2f}' for v in x['I']) if x['I'] else ''}")
