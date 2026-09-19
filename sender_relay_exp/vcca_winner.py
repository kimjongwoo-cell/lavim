#!/usr/bin/env python3
"""VCCA addendum: does the visual contribution actually change WHICH candidate wins?

The main analysis says the visual causal delta is well aligned to the candidate-contrast
subspace and mostly discriminative. That sits oddly next to the accuracy result (removing
96.9% of visual tokens changes almost no answers), so this asks the blunt question the
stored sequence scores can answer exactly:

  winner_full  = argmax_c log p_full(c)      winner_zeroA = argmax_c log p_zeroA(c)
  flip         = the two disagree -> the visual contribution decided the answer
  margin_full  = top1 - top2 of log p_full   (how much the winner has to spare)
  reach        = max_c |Delta S_c - mean(Delta S)| / margin_full
                 the discriminative part of the visual effect measured against the gap it
                 would have to close; reach < 1 means it structurally cannot flip anything.

usage: vcca_winner.py <runs/vcca dir>
"""
import json
import statistics as st
import sys
from pathlib import Path

DSETS = ["gtex", "tcga_expert_vqa", "tcga_slidebench", "tcga", "panda"]
root = Path(sys.argv[1])
out = {}

print("=" * 100)
print("Stage E — does the visual contribution change the winning candidate?")
print("=" * 100)
print(f"{'dataset':16s} {'cases':>5s} {'flips':>7s} {'margin_med':>11s} {'|dS_disc|_med':>14s} "
      f"{'reach_med':>10s} {'reach>1':>8s}")
print("-" * 100)

for ds in DSETS:
    rows = [json.loads(x) for x in (root / f"vcca_{ds}.jsonl").read_text().splitlines() if x.strip()]
    flips, margins, discs, reaches = 0, [], [], []
    for r in rows:
        lf, lz, d_s = r["logp_full"], r["logp_zeroa"], r["dS"]
        if len(lf) < 2:
            continue
        wf = max(range(len(lf)), key=lambda i: lf[i])
        wz = max(range(len(lz)), key=lambda i: lz[i])
        flips += int(wf != wz)
        top2 = sorted(lf, reverse=True)[:2]
        margin = top2[0] - top2[1]
        bar = sum(d_s) / len(d_s)
        disc = max(abs(v - bar) for v in d_s)
        margins.append(margin)
        discs.append(disc)
        reaches.append(disc / margin if margin > 0 else float("inf"))
    med = lambda xs: st.median(xs) if xs else float("nan")
    over = sum(1 for v in reaches if v > 1.0)
    out[ds] = {"cases": len(rows), "flips": flips, "margin_med": med(margins),
               "disc_med": med(discs), "reach_med": med(reaches), "reach_gt1": over}
    print(f"{ds:16s} {len(rows):5d} {flips:4d}/{len(rows):<2d} {med(margins):11.3f} "
          f"{med(discs):14.3f} {med(reaches):10.3f} {over:4d}/{len(rows):<3d}")

tot_c = sum(v["cases"] for v in out.values())
tot_f = sum(v["flips"] for v in out.values())
tot_r = sum(v["reach_gt1"] for v in out.values())
print("-" * 100)
print(f"{'ALL':16s} {tot_c:5d} {tot_f:4d}/{tot_c:<3d} "
      f"{'':11s} {'':14s} {'':10s} {tot_r:4d}/{tot_c:<3d}")
print()
print(f"visual contribution decides the teacher-forced winner in {tot_f}/{tot_c} cases; "
      f"its discriminative part exceeds the top-2 gap in {tot_r}/{tot_c}.")

json.dump(out, open(root / "vcca_winner.json", "w"), ensure_ascii=False, indent=1)
print(f"written {root}/vcca_winner.json")
