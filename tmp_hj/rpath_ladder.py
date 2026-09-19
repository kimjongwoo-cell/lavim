"""Corrected ladder summary for one dataset: rpath_ladder.py <tag> [kvdir] [wrong_arm] [jsonl-name]
flip = parsed answer field differs; shift = [logP(true ans) - logP(wrong ans)]_patched - same_unpatched
(length bias cancels within a case); pref->wrong = contrast crosses from >0 to <0."""
import json, re, sys
from pathlib import Path
tag = sys.argv[1]
Q = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/expq")
kv = Path(sys.argv[2]) if len(sys.argv) > 2 else (Q / "rpath_out" / ("kv" if tag == "gtex" else f"{tag}/kv"))
wrong_arm = sys.argv[3] if len(sys.argv) > 3 else ("rpath_wrong_gtex" if tag == "gtex" else f"rpath_w_{tag}")
jl = kv / (sys.argv[4] if len(sys.argv) > 4 else "rpath_patch.jsonl")
def ans(s):
    if not isinstance(s, str): return None
    m = re.match(r'\s*"([^"]*)"', s)          # output starts right after json_prefix '{"answer": '
    if m: return m.group(1).strip().rstrip(",")
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', s)
    if m: return m.group(1).strip().rstrip(",")
    m = re.match(r'\s*"?([A-Za-z][^"\n{}]*)', s)
    return (m.group(1).strip().rstrip(",") if m else s.strip())
recs = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
recs = [r for r in recs if not r.get("donor_case")]
_seen = set(); _dd = []
for r in recs:                      # repair-path re-decodes append a 2nd row per case: keep the first
    if r.get("case_index") in _seen: continue
    _seen.add(r.get("case_index")); _dd.append(r)
recs = _dd
_g = [r["geom"]["ok"] for r in recs if isinstance(r.get("geom"), dict)]
if _g: print(f"geometry ok: {sum(_g)}/{len(_g)}")
n = len(recs); m = 10
wrong_flip = [r for r in recs if ans(r["normal"]) != (r.get("wrong_answer") or "").strip()]
print(f"[{tag}] pass T n={n}; wrong-RUN flips (true vs wrong run answer): {len(wrong_flip)}/{n} = {100*len(wrong_flip)/max(n,1):.0f}%")
modes = ["none", "vis"] + [f"R{k}:{m}" for k in range(m, 0, -1)] + [f"vis+R1:{m}", "close", f"R1:{m}+close", f"vis+R1:{m}+close", "vis_drop"]
med = lambda v: sorted(v)[len(v)//2] if v else float("nan")
print(f"{'mode':>16} {'ans flip':>9} {'repro wrong':>12} {'shift med':>10} {'pref->wrong':>12} {'n':>4}")
for mo in modes:
    have = [r for r in recs if f"score_{mo}" in r]
    if not have: continue
    fl = [ans(r.get(f"answer_{mo}")) != ans(r["normal"]) for r in have if f"answer_{mo}" in r]
    rep = [ans(r.get(f"answer_{mo}")) == (r.get("wrong_answer") or "").strip() for r in wrong_flip if f"answer_{mo}" in r]
    sh = []; pw = 0; nn = 0
    for r in have:
        a, b, c, d = r.get("score_normal"), r.get("score_wrong_on_true"), r.get(f"score_{mo}"), r.get(f"score_{mo}_wrong")
        if None in (a, b, c, d) or ans(r["normal"]) == (r.get("wrong_answer") or "").strip(): continue
        base = a - b; pat = c - d; sh.append(pat - base); nn += 1; pw += int(base > 0 and pat < 0)
    flip = f"{100*sum(fl)/len(fl):5.0f}%" if fl else "    -"
    repro = f"{sum(rep)}/{len(rep)}" if rep else "-"
    print(f"{mo:>16} {flip:>9} {repro:>12} {med(sh):+10.2f} {100*pw/max(nn,1):11.0f}% {nn:>4}")
ks = [r["kv_dep_k_per_step"] for r in recs if "kv_dep_k_per_step" in r]
vs = [r["kv_dep_v_per_step"] for r in recs if "kv_dep_v_per_step" in r]
if ks:
    print("KV-level 1-cos  K:", " ".join(f"{med([k[t] for k in ks]):.4f}" for t in range(m)))
    print("                V:", " ".join(f"{med([v[t] for v in vs]):.4f}" for t in range(m)))
vl = [r["kv_dep_visual_lastlayer"] for r in recs if "kv_dep_visual_lastlayer" in r]
if vl: print(f"visual-col K true vs wrong 1-cos (last layer): {med(vl):.4f}")
