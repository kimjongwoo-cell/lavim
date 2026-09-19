"""s_g (latent-query group score) vs I_g (leave-one-group-out gold-margin drop).
usage: python3 score_ig.py <tag> <ig.jsonl> <dataset.json>
"""
import json, sys, statistics as st, random
from itertools import combinations

tag, jl, dsp = sys.argv[1:4]
DS = json.load(open(dsp))
rows = [json.loads(l) for l in open(jl)]
norm = lambda s: ''.join(ch for ch in str(s).lower() if ch.isalnum())

def margin(lp, gold):
    ks = list(lp)
    g = next((k for k in ks if norm(k) == norm(gold)), None)
    if g is None: return None
    others = [v for k, v in lp.items() if k != g]
    return lp[g] - max(others) if others else None

def spearman(a, b):
    n = len(a)
    if n < 2: return None
    ra = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: a[i]))}
    rb = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: b[i]))}
    ma = st.mean(ra.values()); mb = st.mean(rb.values())
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    den = (sum((ra[i]-ma)**2 for i in range(n)) * sum((rb[i]-mb)**2 for i in range(n))) ** .5
    return num/den if den else 0.0

def pearson(a, b):
    n = len(a); ma = st.mean(a); mb = st.mean(b)
    num = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    den = (sum((x-ma)**2 for x in a) * sum((y-mb)**2 for y in b)) ** .5
    return num/den if den else 0.0

# LOO position-profile per token (same n_vis rows only)
by_n = {}
for r in rows: by_n.setdefault(r['n_vis'], []).append(r)

recs = []
for r in rows:
    gold = DS[r['case']]['Answer'].strip()
    mf = margin(r['logp']['full'], gold)
    if mf is None: continue
    roots = r['roots']; S = r['S']
    peers = [o for o in by_n[r['n_vis']] if o['case'] != r['case']]
    prof = [st.mean(o['S'][j] for o in peers) for j in range(r['n_vis'])] if peers else [0.0]*r['n_vis']
    S_loo = [s - p for s, p in zip(S, prof)]
    starts, pages = r['starts'], r['pages']
    def toks(g):
        idx = []
        for p in r['groups'][str(g)]: idx += list(range(starts[p], starts[p]+pages[p]))
        return idx
    I, s_mean, s_max, s_loo_mean, s_loo_max, size = [], [], [], [], [], []
    for g in roots:
        mg = margin(r['logp'][f'g{g}'], gold)
        if mg is None: break
        t = toks(g)
        I.append(mf - mg)
        s_mean.append(st.mean(S[j] for j in t)); s_max.append(max(S[j] for j in t))
        s_loo_mean.append(st.mean(S_loo[j] for j in t)); s_loo_max.append(max(S_loo[j] for j in t))
        size.append(len(t))
    if len(I) != len(roots) or len(roots) < 2: continue
    mn = margin(r['logp']['novis'], gold)
    recs.append(dict(case=r['case'], roots=roots, I=I, s_mean=s_mean, s_max=s_max, s_loo_mean=s_loo_mean,
                     s_loo_max=s_loo_max, size=size, m_full=mf, I_novis=(mf - mn) if mn is not None else None,
                     correct_full=(mf > 0)))

n = len(recs)
print(f"=== {tag}: n={n} cases (roots per case: {sorted(set(len(x['roots']) for x in recs))})")
print(f"m_full median {st.median(x['m_full'] for x in recs):+.2f} · correct(full, teacher-forced) {sum(x['correct_full'] for x in recs)}/{n}")
allI = [v for x in recs for v in x['I']]
print(f"I_g (nat): median {st.median(allI):+.3f}, |I_g|>0.5: {sum(abs(v)>0.5 for v in allI)}/{len(allI)}, I_g<-0.5 (removal helps): {sum(v<-0.5 for v in allI)}, "
      f"max_g I_g per case median {st.median(max(x['I']) for x in recs):+.3f} · I_novis median {st.median(x['I_novis'] for x in recs if x['I_novis'] is not None):+.3f}")
big = [x for x in recs if max(x['I']) - min(x['I']) > 0.5]
print(f"cases where groups differ (max−min I_g > 0.5 nat): {len(big)}/{n}")

def report(key, label, subset, note=""):
    if not subset: return
    agree = sum(subset_i[key].index(max(subset_i[key])) == subset_i['I'].index(max(subset_i['I'])) for subset_i in subset)
    chance = st.mean(1/len(x['roots']) for x in subset)
    sp = [spearman(x[key], x['I']) for x in subset]
    # pooled within-case z
    zs, zI = [], []
    for x in subset:
        for a, b in ((x[key], zs), (x['I'], zI)):
            m = st.mean(a); sd = st.pstdev(a) or 1.0
            b.extend((v-m)/sd for v in a)
    regret = [max(x['I']) - x['I'][x[key].index(max(x[key]))] for x in subset]
    picked = [x['I'][x[key].index(max(x[key]))] for x in subset]
    print(f"  {label:26s} top1 agree {agree}/{len(subset)} (chance {chance*len(subset):.1f}) · Spearman med {st.median(sp):+.2f} · pooled r {pearson(zs, zI):+.2f} · "
          f"I of picked med {st.median(picked):+.2f} vs best {st.median(max(x['I']) for x in subset):+.2f} · regret med {st.median(regret):.2f}{note}")

for name, subset in (("all", recs), ("groups differ", big)):
    print(f"-- {name} (n={len(subset)})")
    report('s_mean', 's_g = mean z (raw)', subset)
    report('s_max', 's_g = max z (raw)', subset)
    report('s_loo_mean', 's_g = mean z (LOO bias−)', subset)
    report('s_loo_max', 's_g = max z (LOO bias−)', subset)
    report('size', 'baseline: largest group', subset)
    # position baseline: always root 0 / random
    if subset:
        first = sum(x['I'].index(max(x['I'])) == 0 for x in subset)
        rng = random.Random(42)
        rnd = st.mean(sum(rng.randrange(len(x['roots'])) == x['I'].index(max(x['I'])) for x in subset) for _ in range(200))
        print(f"  {'baseline: first root / random':26s} top1 agree {first}/{len(subset)} / {rnd:.1f}")
# per-case dump (short)
print("case  roots       I_g                    s_mean(LOO)             pick(s)  best(I)")
for x in recs:
    pk = x['roots'][x['s_loo_mean'].index(max(x['s_loo_mean']))]; bt = x['roots'][x['I'].index(max(x['I']))]
    print(f"{x['case']:4d}  {str(x['roots']):10s}  {' '.join(f'{v:+.2f}' for v in x['I']):22s}  {' '.join(f'{v:+.2f}' for v in x['s_loo_mean']):22s}  {pk:6d}  {bt:6d}")

# --- size-normalized usefulness: I_g per crop (groups are 4/3/1 crops) ---
print("-- I_g per crop (size-normalized) top-1 agree:")
for x in recs: x['Ipc'] = [v/ (s/256) for v, s in zip(x['I'], x['size'])]
sub = [x for x in recs if max(x['Ipc']) - min(x['Ipc']) > 0.3]
for key, label in (('s_mean','mean z raw'),('s_loo_mean','mean z LOO'),('s_max','max z raw'),('s_loo_max','max z LOO'),('size','largest group')):
    a = sum(x[key].index(max(x[key])) == x['Ipc'].index(max(x['Ipc'])) for x in recs)
    b = sum(x[key].index(max(x[key])) == x['Ipc'].index(max(x['Ipc'])) for x in sub)
    print(f"  {label:14s} all {a}/{len(recs)} (chance {len(recs)/3:.1f}) · differ>0.3 {b}/{len(sub)} (chance {len(sub)/3:.1f})")
sm = sum(x['Ipc'].index(max(x['Ipc'])) == x['size'].index(min(x['size'])) for x in recs)
print(f"  smallest group has max I/crop: {sm}/{len(recs)}")
