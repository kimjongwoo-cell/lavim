"""Compare a copy-tree smoke run against runs/ntrs_nav3 on the same dataset indices.

usage: neutral_compare.py <run_name> [<run_name> ...]   (run dirs under the copy tree's runs/)
Prints per-dataset: n, answers identical to base, correct (base -> new), format failures.
"""
import glob
import json
import os
import re
import sys

COPY = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
BASE = "/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/ntrs_nav3"


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def load(pattern):
    out = {}
    for f in glob.glob(pattern):
        r = json.load(open(f))
        m = os.stat(f).st_mtime
        i = r["dataset_index"]
        if i not in out or m > out[i][0]:
            out[i] = (m, r)
    return {i: v[1] for i, v in out.items()}


for run in sys.argv[1:]:
    for ds in ("tcga", "gtex", "tcga_slidebench", "tcga_expert_vqa", "panda"):
        new = load(f"{COPY}/{run}/*/{ds}/gpu*/*/result.json")
        if not new:
            continue
        base = load(f"{BASE}/{ds}/gpu*/*/result.json")
        same = ok_b = ok_n = fmt_b = fmt_n = 0
        changed = []
        for i in sorted(new):
            r = new[i]
            b = base.get(i)
            a_n = r["answer"]["answer"]
            a_b = b["answer"]["answer"] if b else None
            gold = r["gold_answer"]
            choices_norm = None
            # choices are not stored in result.json; use gold match + placeholder heuristics
            def bad(a):
                return (not a) or a == "<no-answer>" or re.search(r"[{}<>\n]|^choice$|^text$", str(a).lower()) is not None
            fmt_n += bad(a_n)
            fmt_b += bad(a_b) if b else 0
            ok_n += norm(a_n) == norm(gold)
            ok_b += (norm(a_b) == norm(gold)) if b else 0
            if b and norm(a_n) == norm(a_b):
                same += 1
            elif b:
                changed.append((i, str(a_b)[:22], str(a_n)[:22], gold[:18]))
        print(f"== {run} {ds}: n={len(new)} same_as_base={same} correct base->new {ok_b}->{ok_n} format_fail base->new {fmt_b}->{fmt_n}")
        for c in changed[:12]:
            print("   changed idx %3d: base=%r -> new=%r (gold %r)" % c)
