"""C2 kill test 1-3: Answerer group scores (native mass / size / LME) vs I_g.
usage: python3 score_aq.py <tag> <igc.jsonl (I_g source)> <igaq.jsonl (aq source)> <dataset.json>
"""
import json, sys, statistics as st, math, random

tag, ig_path, aq_path, dsp = sys.argv[1:5]
DS = json.load(open(dsp))
norm = lambda s: ''.join(ch for ch in str(s).lower() if ch.isalnum())

def margin(lp, gold):
    g = next((k for k in lp if norm(k) == norm(gold)), None)
    if g is None: return None
    others = [v for k, v in lp.items() if k != g]
    return lp[g] - max(others) if others else None

def pearson(a, b):
    ma = st.mean(a); mb = st.mean(b)
    num = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    den = (sum((x-ma)**2 for x in a) * sum((y-mb)**2 for y in b)) ** .5
    return num/den if den else 0.0

IG = {}
for l in open(ig_path):
    r = json.loads(l); gold = DS[r['case']]['Answer'].strip()
    mf = margin(r['logp']['full'], gold)
    if mf is None: continue
    I = []
    for g in r['roots']:
        mg = margin(r['logp'][f'g{g}'], gold)
        if mg is None: break
        I.append(mf - mg)
    if len(I) == len(r['roots']):
        IG[r['case']] = dict(roots=r['roots'], I=I, groups=r['groups'], m_full=mf)

recs = []
seen = set()
for l in open(aq_path):
    r = json.loads(l)
    if r['case'] in seen: continue
    seen.add(r['case'])
    if r['case'] not in IG or 'aq' not in r or 'last' not in r['aq']: continue
    x = IG[r['case']]
    if r['roots'] != x['roots']: continue
    aq = r['aq']
    d = dict(case=r['case'], roots=x['roots'], I=x['I'], size=aq['sizes'],
             mass=aq['last']['mass'], lme=aq['last']['lme'], lse=aq['last']['lse'], M_V=aq['last']['M_V'])
    d['mass_pc'] = [m / s for m, s in zip(d['mass'], d['size'])]     # mass per token (= exp(lme)/Z)
    if 'cont' in aq:
        d['mass_c'] = aq['cont']['mass']; d['lme_c'] = aq['cont']['lme']
    recs.append(d)

n = len(recs)
print(f"=== {tag}: n={n} cases joined (I_g from canonical igsweep, Answerer scores from AQ probe)")
print(f"M_V(last row) median {st.median(x['M_V'] for x in recs):.4f} · sizes {sorted(set(tuple(x['size']) for x in recs))[:3]}")
print(f"I_g: |I_g|>0.5 {sum(abs(v)>0.5 for x in recs for v in x['I'])}/{3*n} · groups differ>0.5: {sum(max(x['I'])-min(x['I'])>0.5 for x in recs)}/{n}")

def report(key, label, subset, target='I'):
    if not subset: return
    agree = sum(x[key].index(max(x[key])) == x[target].index(max(x[target])) for x in subset)
    zs, zI = [], []
    for x in subset:
        for a, b in ((x[key], zs), (x[target], zI)):
            m = st.mean(a); sd = st.pstdev(a) or 1.0
            b.extend((v-m)/sd for v in a)
    picked = [x[target][x[key].index(max(x[key]))] for x in subset]
    regret = [max(x[target]) - p for x, p in zip(subset, picked)]
    print(f"  {label:34s} top1 agree {agree}/{len(subset)} (chance {len(subset)/3:.1f}) · pooled r {pearson(zs, zI):+.2f} · "
          f"I picked med {st.median(picked):+.2f} vs best {st.median(max(x[target]) for x in subset):+.2f} · regret med {st.median(regret):.2f}")

big = [x for x in recs if max(x['I']) - min(x['I']) > 0.5]
for name, subset in (("all", recs), ("groups differ (>0.5 nat)", big)):
    print(f"-- {name} (n={len(subset)}) — target I_g")
    report('mass', '1. native group mass Σα', subset)
    report('size', '2. group size |g|', subset)
    report('lme', '3. LME = LSE − log|g| (Answerer q)', subset)
    report('lse', '   LSE (no size correction)', subset)
    report('mass_pc', '   mass per token', subset)
    if all('mass_c' in x for x in subset):
        report('mass_c', '   mass (continuation rows)', subset)
        report('lme_c', '   LME (continuation rows)', subset)
# size-normalised target
for x in recs: x['Ipc'] = [v / (s / 256) for v, s in zip(x['I'], x['size'])]
print(f"-- target I_g per crop (n={n})")
for key, label in (('mass', '1. native mass'), ('size', '2. size'), ('lme', '3. LME'), ('mass_pc', '   mass per token')):
    report(key, label, recs, target='Ipc')
# how size-driven is native mass / lme?
print("-- structure: corr(size, mass) / corr(size, lme) pooled within-case z")
for key in ('mass', 'lme', 'lse'):
    zs, zk = [], []
    for x in recs:
        for a, b in ((x['size'], zs), (x[key], zk)):
            m = st.mean(a); sd = st.pstdev(a) or 1.0
            b.extend((v-m)/sd for v in a)
    print(f"  size vs {key:5s}: r {pearson(zs, zk):+.2f} · argmax {key}=largest in {sum(x[key].index(max(x[key]))==x['size'].index(max(x['size'])) for x in recs)}/{n}")
print("case  I_g                   mass(%)               lme                   size")
for x in recs[:12]:
    print(f"{x['case']:4d}  {' '.join(f'{v:+.2f}' for v in x['I']):20s}  {' '.join(f'{100*v:.2f}' for v in x['mass']):20s}  {' '.join(f'{v:+.2f}' for v in x['lme']):20s}  {x['size']}")
