import json, glob, os, sys, collections, re
def cat(out):
    t = '{"answer":' + out
    if re.search(r'"confidence":\s*(\}|$|\s+\}|\.\s*\}|0\.\s+\}|0\.\}|\s+\.)', out.split("}")[0] + "}"): return "confidence 숫자 빠짐"
    if not re.match(r'\s*"[^"]*"\s*,', out): return "답 필드 깨짐"
    if '"rationale"' in out and re.search(r'\w\s{2,}\w|\s{3,}', out.split('"confidence"')[0]): return "rationale 단어 빠짐"
    return "기타"
for pat in sys.argv[1:]:
    latest = {}
    for f in glob.glob(pat + "/**/result.json", recursive=True):
        try: r = json.load(open(f))
        except Exception: continue
        if "panda" not in str(r.get("slide_id", "")): continue
        k = str(r.get("dataset_index")); m = os.path.getmtime(f)
        if k not in latest or m > latest[k][0]: latest[k] = (m, r)
    broken = []
    for k, (m, r) in sorted(latest.items(), key=lambda x: int(x[0])):
        rc = [x for x in r["role_calls"] if x["role"] == "answerer"][-1]
        out = rc["final_outputs"][-1] if rc["final_outputs"] else ""
        t = ('{"answer":' + out).strip()
        try: json.loads(t); continue
        except Exception: pass
        try:
            _, end = json.JSONDecoder().raw_decode(t); continue
        except Exception: pass
        a = r["answer"]
        broken.append((k, r["gold_answer"], a.get("answer"), a.get("confidence"), a.get("rationale", "")[:60], out))
    c = collections.Counter(cat(b[5]) for b in broken)
    print(f"\n##### {pat.split('/runs/')[-1]}  n={len(latest)}  깨짐={len(broken)}  유형={dict(c)}")
    shown = collections.Counter()
    for k, g, a, conf, rat, out in broken:
        ty = cat(out)
        if shown[ty] >= 3: continue
        shown[ty] += 1
        print(f"[idx {k} gold {g}] 유형={ty} | 채점된 답={a!r} conf={conf!r}")
        print("   RAW: {\"answer\":" + out[:260].replace("\n", "\\n"))
