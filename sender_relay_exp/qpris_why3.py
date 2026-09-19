#!/usr/bin/env python3
"""GTEx per-class loss and ExpertVQA degenerate outputs, Q-PRIS vs a base run."""
import json, glob, re, sys, collections
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from eval.metrics import expand_letter, acc_of_seq
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
MP = "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda"


def choices_of(item):
    c = item.get("Choice")
    if isinstance(c, str):
        return [s.strip() for s in re.split(r"\n|\|", c) if s.strip()]
    return list(c or [])


def load(d, ds):
    data = json.load(open(f"{MP}/{ds}.json"))
    out = {}
    for f in sorted(glob.glob(f"{R}/{d}/*/result.json")):
        r = json.load(open(f))
        a = r.get("answer")
        a = a.get("answer") if isinstance(a, dict) else a
        idx = r.get("dataset_index")
        ch = choices_of(data[idx]) if isinstance(idx, int) and idx < len(data) else []
        gold = str(r.get("gold_answer", ""))
        out[idx] = (str(a), gold, bool(acc_of_seq(ch, expand_letter(gold, ch),
                                                  expand_letter(str(a), ch))))
    return out


def bacc_by_class(m):
    per = collections.defaultdict(lambda: [0, 0])
    for _, (p, g, hit) in m.items():
        per[g][1] += 1
        per[g][0] += hit
    return per


print("############ GTEx — 클래스별 (base=expq/geom_audit_gtex)")
b = load("expq/geom_audit_gtex", "gtex")
q = load("qpris_full/qt_gtex", "gtex")
pb, pq = bacc_by_class(b), bacc_by_class(q)
print(f"{'class':20s} {'n':>3s} {'base':>7s} {'Q-PRIS':>8s}  변화")
for g in sorted(set(pb) | set(pq)):
    hb, nb = pb.get(g, (0, 0))
    hq, nq = pq.get(g, (0, 0))
    if nb == 0 or nq == 0:
        continue
    d = hq / nq - hb / nb
    if abs(d) > 1e-9:
        print(f"{g:20s} {nb:3d} {hb}/{nb:<5d} {hq}/{nq:<6d} {d * 100:+6.1f}%p")
flip_lose = [i for i in b if i in q and b[i][2] and not q[i][2]]
flip_gain = [i for i in b if i in q and not b[i][2] and q[i][2]]
print(f"\n  base 맞힘→Q-PRIS 틀림 {len(flip_lose)}건 · 반대 {len(flip_gain)}건")

print("\n############ ExpertVQA — 퇴화 출력")
q2 = load("qpris_full/qt_tcga_expert_vqa", "tcga_expert_vqa")
bad = [(i, p) for i, (p, g, h) in q2.items() if len(p) < 8 or p.upper().startswith("CHOICE")]
print(f"  Q-PRIS 짧은/CHOICE 출력 {len(bad)}건: {collections.Counter(p for _, p in bad).most_common()}")
for name in ("expq/evqa10_restage", "expq/strat_evqa10", "expq/pulse_evqa"):
    try:
        bb = load(name, "tcga_expert_vqa")
    except Exception:
        continue
    n = sum(1 for i, (p, g, h) in bb.items() if len(p) < 8 or p.upper().startswith("CHOICE"))
    print(f"  {name:24s} 같은 유형 {n}건")
