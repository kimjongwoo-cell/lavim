#!/usr/bin/env python3
"""Within-dataset: does what Q-PRIS kept track correctness? Plus prediction collapse."""
import json, re, glob, os, sys, collections, statistics as st
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris_full"
MP = "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda"
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from eval.metrics import expand_letter, acc_of_seq

DSS = ("tcga_expert_vqa", "gtex", "tcga", "tcga_slidebench", "panda")
RE_MAG = re.compile(r"kept_by_mag=\{5: (\d+), 20: (\d+)\}")
X5_POOL, X20_POOL = 3 * 256, 5 * 256   # budget 8 = x5 3장 + x20 5장, crop당 256 토큰


def choices_of(item):
    c = item.get("Choice")
    if isinstance(c, str):
        return [s.strip() for s in re.split(r"\n|\|", c) if s.strip()]
    return list(c or [])


for ds in DSS:
    log = f"{K}/qt_{ds}.log"
    mags = [(int(a), int(b)) for a, b in RE_MAG.findall(open(log, errors="ignore").read())]
    res = sorted(glob.glob(f"{K}/qt_{ds}/*/result.json"))
    data = json.load(open(f"{MP}/{ds}.json"))
    ok, bad, preds, hits = [], [], [], 0
    for (x5, x20), f in zip(mags, res):
        r = json.load(open(f))
        a = r.get("answer")
        a = a.get("answer") if isinstance(a, dict) else a
        gold = str(r.get("gold_answer", ""))
        idx = r.get("dataset_index")
        ch = choices_of(data[idx]) if isinstance(idx, int) and idx < len(data) else []
        preds.append(str(a))
        v = acc_of_seq(ch, expand_letter(gold, ch), expand_letter(str(a), ch))
        hit = bool(v)
        hits += hit
        (ok if hit else bad).append(x5 / (x5 + x20))
    c = collections.Counter(preds)
    top, topn = c.most_common(1)[0]
    d = (st.mean(ok) - st.mean(bad)) if ok and bad else float("nan")
    print(f"=== {ds}  n={len(res)}  맞힘 {hits}")
    print(f"    x5 비중 — 맞힌 케이스 {st.mean(ok):.3f}(n={len(ok)}) vs 틀린 케이스 "
          f"{st.mean(bad):.3f}(n={len(bad)})   차이 {d:+.3f}")
    print(f"    유지율 — x5 {st.mean([a for a, b in mags]) / X5_POOL:.1%} · "
          f"x20 {st.mean([b for a, b in mags]) / X20_POOL:.1%}  (예산 비례 25.0%)")
    print(f"    예측 {len(c)}종, 최빈 '{top}' {topn}/{len(res)} ({topn / len(res):.0%})")
