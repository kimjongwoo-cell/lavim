"""DUP2 kill tests: (2) total-mass vs cardinality split (massfix control), (3) C2 invariance + original-input.
usage: python3 score_dup2.py <tag> <igd2.jsonl> <dataset.json> [igc.jsonl for I_g]
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

def pearson(a, b):
    if len(a) < 3: return float('nan')
    ma = st.mean(a); mb = st.mean(b)
    num = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    den = (sum((x-ma)**2 for x in a) * sum((y-mb)**2 for y in b)) ** .5
    return num/den if den else 0.0

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

recs = []; seen = set()
for l in open(path):
    r = json.loads(l)
    if r['case'] in seen or 'masses' not in r: continue
    seen.add(r['case']); gold = DS[r['case']]['Answer'].strip(); lp = r['logp']; M = r['masses']
    mf = margin(lp['full'], gold); pf = argmax(lp['full'])
    mc2 = margin(lp['c2_full'], gold); pc2 = argmax(lp['c2_full'])
    sizes = [sum(r['pages'][p] for p in r['groups'][str(g)]) for g in r['roots']]
    d = dict(case=r['case'], roots=r['roots'], size=sizes, m_full=mf, m_c2=mc2, p_full=pf, p_c2=pc2,
             correct_full=mf > 0, correct_c2=mc2 > 0, I=IG.get(r['case']),
             mV_full=M['full']['m_V'], rho_full=M['full']['rho_g'], mg_full=M['full']['m_g'],
             mV_c2=M.get('c2_full', {}).get('m_V'))
    d['dm_nat'] = [margin(lp[f'dup{g}'], gold) - mf for g in r['roots']]
    d['dm_mfix'] = [margin(lp[f'mfix_dup{g}'], gold) - mf for g in r['roots']]
    d['dm_c2'] = [margin(lp[f'c2_dup{g}'], gold) - mc2 for g in r['roots']]
    d['fl_nat'] = [argmax(lp[f'dup{g}']) != pf for g in r['roots']]
    d['fl_mfix'] = [argmax(lp[f'mfix_dup{g}']) != pf for g in r['roots']]
    d['fl_c2'] = [argmax(lp[f'c2_dup{g}']) != pc2 for g in r['roots']]
    d['mV_dup'] = [M[f'dup{g}']['m_V'] for g in r['roots']]
    d['rho_dup'] = [M[f'dup{g}']['rho_g'][i] for i, g in enumerate(r['roots'])]      # share of the duplicated group (orig cols + copies)
    d['mV_mfix'] = [M[f'mfix_dup{g}']['m_V'] for g in r['roots']]
    recs.append(d)

n = len(recs)
G = 3 * n
print(f"=== {tag}: n={n} cases (roots per case 3)")
print(f"-- ② total visual mass vs cardinality  [last prompt row, 36-layer·head mean]")
print(f"   m_V native full median {st.median(x['mV_full'] for x in recs):.4f} → after dup (native) median {st.median(v for x in recs for v in x['mV_dup']):.4f} "
      f"(Δm_V median {st.median(v - x['mV_full'] for x in recs for v in x['mV_dup']):+.4f}) · mfix native m_V of the modified forward median {st.median(v for x in recs for v in x['mV_mfix']):.4f}")
rho_full_dup = [x['rho_full'][i] for x in recs for i in range(3)]
print(f"   rho_g of duplicated group: full median {st.median(rho_full_dup):.3f} → dup {st.median(v for x in recs for v in x['rho_dup']):.3f}")
def row(label, dm, fl):
    a = [v for x in recs for v in x[dm]]; f = [v for x in recs for v in x[fl]]
    print(f"   {label:28s} |Δmargin| med {st.median(abs(v) for v in a):.3f} · >0.5: {sum(abs(v)>0.5 for v in a)}/{G} · >1.0: {sum(abs(v)>1.0 for v in a)} · flips {sum(f)}/{G} (cases {sum(any(x[fl]) for x in recs)}/{n})")
row('Native + duplicate', 'dm_nat', 'fl_nat')
row('massfix + duplicate (control)', 'dm_mfix', 'fl_mfix')
row('C2 + duplicate (vs C2 full)', 'dm_c2', 'fl_c2')
a = [v for x in recs for v in x['dm_nat']]; b = [v for x in recs for v in x['dm_mfix']]; c = [v for x in recs for v in x['dm_c2']]
print(f"   ratio med |Δ_mfix|/|Δ_nat| {st.median(abs(y)/max(abs(x),1e-6) for x, y in zip(a, b)):.2f} · |Δ_c2|/|Δ_nat| {st.median(abs(y)/max(abs(x),1e-6) for x, y in zip(a, c)):.2f} · corr(Δ_nat, Δ_mfix) {pearson(a, b):+.2f}")
print(f"-- ③ C2 on ORIGINAL input (no duplication)")
print(f"   margin full med {st.median(x['m_full'] for x in recs):+.3f} vs C2 {st.median(x['m_c2'] for x in recs):+.3f} · Δ(C2−full) med {st.median(x['m_c2']-x['m_full'] for x in recs):+.3f} · "
      f"correct(teacher-forced argmax) full {sum(x['correct_full'] for x in recs)}/{n} vs C2 {sum(x['correct_c2'] for x in recs)}/{n} · argmax changed {sum(x['p_full']!=x['p_c2'] for x in recs)}/{n} · m_V C2 med {st.median(x['mV_c2'] for x in recs if x['mV_c2'] is not None):.4f}")
if any(x['I'] for x in recs):
    pairs = [(i, d) for x in recs if x['I'] for i, d in zip(x['I'], x['dm_nat'])]
    same = sum((i > 0) == (d > 0) for i, d in pairs if abs(i) > 0.5 and abs(d) > 0.5); tot = sum(1 for i, d in pairs if abs(i) > 0.5 and abs(d) > 0.5)
    print(f"   sign(I_g)==sign(Δ_nat) {same}/{tot} · r {pearson([i for i,_ in pairs],[d for _,d in pairs]):+.2f}")
print(f"-- bonus: natural-state branch size ↔ mass (case-wise z, pooled)")
def z(v):
    m = st.mean(v); sd = st.pstdev(v) or 1.0; return [(x-m)/sd for x in v]
zs = [v for x in recs for v in z(x['size'])]
print(f"   corr(size, m_g) {pearson(zs, [v for x in recs for v in z(x['mg_full'])]):+.2f} · corr(size, rho_g) {pearson(zs, [v for x in recs for v in z(x['rho_full'])]):+.2f}"
      + (f" · corr(size, I_g) {pearson([v for x in recs if x['I'] for v in z(x['size'])], [v for x in recs if x['I'] for v in z(x['I'])]):+.2f}" if any(x['I'] for x in recs) else ''))
print(f"   m_g/token by size: " + ' · '.join(f"{s}: {st.median(x['mg_full'][i]/s for x in recs for i in range(3) if x['size'][i]==s)*1e4:.3f}e-4" for s in sorted({v for x in recs for v in x['size']})))
print("case  size            Δnat                  Δmfix                 Δc2                   m_full  m_c2")
for x in recs[:12]:
    print(f"{x['case']:4d}  {str(x['size']):15s} {' '.join(f'{v:+.2f}' for v in x['dm_nat']):20s}  {' '.join(f'{v:+.2f}' for v in x['dm_mfix']):20s}  {' '.join(f'{v:+.2f}' for v in x['dm_c2']):20s}  {x['m_full']:+.2f}  {x['m_c2']:+.2f}")
