import json, glob, os, collections, re, sys
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
groups = collections.defaultdict(list)
for f in glob.glob(f"{R}/**/result.json", recursive=True):
    g = os.path.dirname(os.path.dirname(f))
    groups[g].append(f)
rows = []
for g, fs in groups.items():
    n = pf = digitgap = 0
    for f in fs:
        try:
            d = json.load(open(f))
            rc = [r for r in d["role_calls"] if r["role"] == "answerer"][-1]
            out = rc["final_outputs"][-1]
        except Exception:
            continue
        n += 1
        pre = rc["prompt"]
        txt = '{"answer":' + out
        try:
            json.loads(txt.strip())
        except Exception:
            try:
                json.JSONDecoder().raw_decode(txt.strip())
            except Exception:
                pf += 1
        if re.search(r'"confidence":\s*(\.|0\.\s|\s*[}\n])', out) or re.search(r"(Gleason|grade|Grade)\s{2,}", out): digitgap += 1
    if n: rows.append((os.path.relpath(g, R), n, pf, digitgap, time_ := max(os.path.getmtime(x) for x in fs)))
import time
for r in sorted(rows, key=lambda r: r[4]):
    print(f"{time.strftime('%m-%d %H:%M', time.localtime(r[4]))}  {r[0]:60s} n={r[1]:4d} parse_fail={r[2]:3d} ({100*r[2]/r[1]:4.0f}%) digitgap={r[3]}")
