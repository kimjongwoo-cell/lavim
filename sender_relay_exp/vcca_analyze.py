#!/usr/bin/env python3
"""Offline analysis for the VCCA diagnosis, following the Notion plan §9 gates.

The one thing the plan insists on (§11) is that a small rho is NOT by itself evidence:
the candidate-contrast subspace has rank r << D, so a direction with no preference at all
already gets rho ~ r/D. Every rho here is therefore reported both raw and as a LIFT over
that chance level, and the primary comparison is the paired one the plan asks for
(visual delta vs the full native state at the same node).

usage: vcca_analyze.py <runs/vcca dir>
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

DSETS = ["gtex", "tcga_expert_vqa", "tcga_slidebench", "tcga", "panda"]
root = Path(sys.argv[1])


def med(xs):
    return st.median(xs) if xs else float("nan")


def load(ds):
    p = root / f"vcca_{ds}.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def hidden_dim(rows):
    for r in rows:
        raw = r.get("first_node_raw") or {}
        if raw.get("post_full"):
            return len(raw["post_full"])
    return None


summary = {}
print("=" * 108)
print("VCCA — per dataset (median over branching nodes unless noted)")
print("=" * 108)
hdr = (f"{'dataset':16s} {'cases':>5s} {'nodes':>6s} {'k':>4s} {'chance':>7s} "
       f"{'rho_post':>9s} {'lift':>6s} {'rhoP_full':>10s} {'liftF':>6s} "
       f"{'d<full':>8s} {'R_disc':>7s} {'R_seq':>7s} {'gate':>7s}")
print(hdr)
print("-" * 108)

for ds in DSETS:
    rows = load(ds)
    dim = hidden_dim(rows)
    nodes = [n for r in rows for n in r["nodes"]]
    chance = [n["rank"] / dim for n in nodes]
    rp = [n["rho_post"] for n in nodes]
    rpf = [n["rho_post_full"] for n in nodes]
    pre = [n["rho_pre"] for n in nodes if "rho_pre" in n]
    pref = [n["rho_pre_full"] for n in nodes if "rho_pre_full" in n]
    lift = [a / b for a, b in zip(rp, chance) if b > 0]
    liftf = [a / b for a, b in zip(rpf, chance) if b > 0]
    lift_pre = [n["rho_pre"] / (n["rank"] / dim) for n in nodes if "rho_pre" in n and n["rank"]]
    d_lt_full = sum(1 for a, b in zip(rp, rpf) if a < b)
    pre_gt_post = sum(1 for n in nodes if "rho_pre" in n and n["rho_pre"] > n["rho_post"])
    n_pre = sum(1 for n in nodes if "rho_pre" in n)
    rd = [n["R_disc"] for n in nodes]
    rs = [r["R_seq"] for r in rows]
    gate = [n["gate_rel"] for n in nodes]

    summary[ds] = {
        "dim": dim, "cases": len(rows), "nodes": len(nodes),
        "k_med": med([n["k"] for n in nodes]), "chance_med": med(chance),
        "rho_post_med": med(rp), "lift_post_med": med(lift),
        "rho_post_full_med": med(rpf), "lift_post_full_med": med(liftf),
        "rho_pre_med": med(pre), "lift_pre_med": med(lift_pre),
        "rho_pre_full_med": med(pref),
        "delta_below_full": d_lt_full, "n_nodes_paired": len(nodes),
        "pre_gt_post": pre_gt_post, "n_pre": n_pre,
        "R_disc_med": med(rd), "R_seq_med": med(rs),
        "R_disc_lt_half": sum(1 for v in rd if v < 0.5),
        "R_seq_lt_half": sum(1 for v in rs if v < 0.5),
        "gate_med": med(gate), "gate_max": max(gate) if gate else float("nan"),
        "_rd": rd, "_rs": rs, "_rp": rp, "_rpf": rpf, "_lift": lift, "_liftf": liftf,
    }
    s = summary[ds]
    print(f"{ds:16s} {s['cases']:5d} {s['nodes']:6d} {s['k_med']:4.0f} {s['chance_med']:7.4f} "
          f"{s['rho_post_med']:9.4f} {s['lift_post_med']:6.2f} {s['rho_post_full_med']:10.4f} "
          f"{s['lift_post_full_med']:6.2f} {d_lt_full:4d}/{len(nodes):<3d} "
          f"{s['R_disc_med']:7.3f} {s['R_seq_med']:7.3f} {s['gate_max']:7.4f}")

print()
print("=" * 108)
print("Control C — final norm localisation (rho_pre vs rho_post, paired per node)")
print("=" * 108)
print(f"{'dataset':16s} {'rho_pre':>9s} {'lift_pre':>9s} {'rho_post':>9s} {'lift_post':>9s} "
      f"{'pre>post':>10s} {'verdict':>28s}")
for ds in DSETS:
    s = summary[ds]
    frac = s["pre_gt_post"] / s["n_pre"] if s["n_pre"] else float("nan")
    verdict = ("norm weakens alignment" if frac > 0.6 else
               "norm strengthens it" if frac < 0.4 else "no consistent change")
    print(f"{ds:16s} {s['rho_pre_med']:9.4f} {s['lift_pre_med']:9.2f} {s['rho_post_med']:9.4f} "
          f"{s['lift_post_med']:9.2f} {s['pre_gt_post']:4d}/{s['n_pre']:<5d} {verdict:>28s}")

print()
print("=" * 108)
print("§9 gates")
print("=" * 108)

def per_ds(fmt):
    return ", ".join(f"{ds}=" + fmt(summary[ds]) for ds in DSETS)


n_rd = sum(1 for ds in DSETS if summary[ds]["R_disc_med"] < 0.5)
n_rs = sum(1 for ds in DSETS if summary[ds]["R_seq_med"] < 0.5)
print(f"1a. median R_disc < 0.5 (visual logit shift mostly candidate-COMMON): {n_rd}/5 datasets")
print("      " + per_ds(lambda s: f"{s['R_disc_med']:.3f}"))
print(f"1b. median R_seq  < 0.5 (sequence level says the same): {n_rs}/5 datasets")
print("      " + per_ds(lambda s: f"{s['R_seq_med']:.3f}"))

n_below = sum(1 for ds in DSETS
              if summary[ds]["delta_below_full"] / summary[ds]["n_nodes_paired"] > 0.5)
print(f"2.  visual delta is contrast-POORER than the full native state in >50% of nodes: "
      f"{n_below}/5 datasets")
print("      " + per_ds(lambda s: f"{s['delta_below_full']}/{s['n_nodes_paired']}"))

n_lift = sum(1 for ds in DSETS if summary[ds]["lift_post_med"] < 1.0)
print(f"2b. visual delta lift over chance < 1 (no preference for the contrast subspace): "
      f"{n_lift}/5 datasets")
print("      " + per_ds(lambda s: f"{s['lift_post_med']:.2f}"))

n_norm = sum(1 for ds in DSETS
             if summary[ds]["n_pre"] and summary[ds]["pre_gt_post"] / summary[ds]["n_pre"] > 0.6)
print(f"4.  rho_pre > rho_post (final norm is the bottleneck): {n_norm}/5 datasets")

gate_ok = all(summary[ds]["gate_max"] < 0.05 for ds in DSETS)
print(f"D.  logit reconstruction gate max rel error < 0.05 in every dataset: {gate_ok}"
      f"   [{', '.join(f'{ds}={summary[ds]['gate_max']:.4f}' for ds in DSETS)}]")

print()
print("=" * 108)
print("pooled hierarchical bootstrap (resample datasets, then nodes/cases within) — 2.5/50/97.5 pct")
print("=" * 108)
random.seed(42)


def boot(key, per_case=False):
    out = []
    for _ in range(2000):
        ds_pick = [random.choice(DSETS) for _ in DSETS]
        vals = []
        for ds in ds_pick:
            pool = summary[ds][key]
            if not pool:
                continue
            vals.extend(random.choice(pool) for _ in range(len(pool)))
        if vals:
            out.append(st.median(vals))
    out.sort()
    lo, mid, hi = out[int(0.025 * len(out))], out[len(out) // 2], out[int(0.975 * len(out))]
    return lo, mid, hi


for label, key in (("R_disc (node)", "_rd"), ("R_seq (case)", "_rs"),
                   ("lift_post  visual delta", "_lift"), ("lift_post  full state", "_liftf")):
    lo, mid, hi = boot(key)
    print(f"  {label:26s} {mid:7.3f}   [{lo:.3f}, {hi:.3f}]")

json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
           for k, v in summary.items()},
          open(root / "vcca_summary.json", "w"), ensure_ascii=False, indent=1)
print(f"\nwritten {root}/vcca_summary.json")
