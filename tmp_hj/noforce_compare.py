"""Compare noforce_smoke arms against runs/ntrs_nav3 (same cases, same setting).

off arm : answer + raw answerer output must be byte-identical to ntrs_nav3.
on  arm : print gold / ntrs answer / noforce answer / raw tail / answerer seconds.
"""
import glob
import json
import os
import re

COPY = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/noforce_smoke"
BASE = "/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/ntrs_nav3"


def load(root, ds):
    out = {}
    for f in glob.glob(f"{root}/{ds}/gpu*/*/result.json"):
        r = json.load(open(f))
        m = os.stat(f).st_mtime
        i = r["dataset_index"]
        if i not in out or m > out[i][0]:
            out[i] = (m, r)
    return {i: v[1] for i, v in out.items()}


def ans_call(r):
    calls = [c for c in r.get("role_calls", []) if c.get("role") == "answerer"]
    return calls[-1] if calls else {}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


for arm in ("off", "on"):
    for ds in ("tcga", "gtex", "tcga_slidebench"):
        new = load(COPY + "/" + arm, ds)
        if not new:
            continue
        base = load(BASE, ds)
        print(f"===== arm={arm} ds={ds} n={len(new)}")
        same = 0
        for i in sorted(new):
            r, b = new[i], base.get(i)
            a_new = r["answer"]["answer"]
            a_base = b["answer"]["answer"] if b else None
            c_new, c_base = ans_call(r), ans_call(b) if b else {}
            raw_new = str(c_new.get("final_outputs", ""))
            raw_base = str(c_base.get("final_outputs", ""))
            gold = r.get("gold_answer", "")
            if arm == "off":
                ident = (a_new == a_base) and (raw_new == raw_base)
                same += ident
                print(f"  idx {i:3d} identical={ident} ans={a_new!r}")
            else:
                ok_new = norm(a_new) == norm(gold)
                ok_base = norm(a_base) == norm(gold) if b else None
                print(
                    f"  idx {i:3d} gold={gold[:34]!r} | ntrs={str(a_base)[:34]!r} ok={ok_base} "
                    f"| noforce={str(a_new)[:34]!r} ok={ok_new} | "
                    f"sec {float(c_base.get('elapsed_seconds', 0) or 0):5.1f}->{float(c_new.get('elapsed_seconds', 0) or 0):5.1f} "
                    f"repairs={c_new.get('format_repairs')}"
                )
                print(f"           raw: {raw_new[:160]!r}")
        if arm == "off":
            print(f"  off-arm identical {same}/{len(new)}")
