import json, glob, os, sys, collections, re
for pat in sys.argv[1:]:
    fs = [f for f in glob.glob(pat + "/**/result.json", recursive=True)]
    latest = {}
    for f in fs:
        try: r = json.load(open(f))
        except Exception: continue
        if "panda" not in str(r.get("slide_id", "")): continue
        k = str(r.get("dataset_index"))
        m = os.path.getmtime(f)
        if k not in latest or m > latest[k][0]: latest[k] = (m, r, f)
    rows = []; broken = 0
    for m, r, f in latest.values():
        a = r.get("answer"); p = str(a.get("answer") if isinstance(a, dict) else a).strip()
        rows.append((str(r["gold_answer"]).strip(), p))
        rc = [x for x in r.get("role_calls", []) if x.get("role") == "answerer"]
        if rc:
            out = rc[-1]["final_outputs"][-1] if rc[-1]["final_outputs"] else ""
            try: json.loads(('{"answer":' + out).strip())
            except Exception:
                try: json.JSONDecoder().raw_decode(('{"answer":' + out).strip())
                except Exception: broken += 1
    if not rows: print("EMPTY", pat); continue
    by = collections.defaultdict(list)
    for g, p in rows: by[g].append(p == g)
    b = 100 * sum(sum(v)/len(v) for v in by.values()) / len(by)
    pc = collections.Counter(p if len(p) < 12 else p[:12] + "…" for _, p in rows)
    print(f"{pat.split('/runs/')[-1][:70]:70s} n={len(rows)} exactBACC={b:.2f} JSON깨짐={broken}  예측={dict(pc.most_common(6))}")
