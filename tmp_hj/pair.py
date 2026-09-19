import json, glob, os, sys, re
def load(pat):
    out = {}
    for f in glob.glob(pat + "/**/result.json", recursive=True):
        try: r = json.load(open(f))
        except Exception: continue
        k = int(r["dataset_index"]); m = os.path.getmtime(f)
        rc = [x for x in r["role_calls"] if x["role"] == "answerer"][-1]
        o = rc["final_outputs"][-1] if rc["final_outputs"] else ""
        t = ('{"answer":' + o).strip(); ok = True
        try: json.loads(t)
        except Exception:
            try: json.JSONDecoder().raw_decode(t)
            except Exception: ok = False
        if k not in out or m > out[k][0]:
            out[k] = (m, ok, str(r["answer"].get("answer")).strip(), str(r["gold_answer"]).strip(), o, f)
    return out
A = load(sys.argv[1]); B = load(sys.argv[2])
sh = sorted(set(A) & set(B))
print("eager n=", len(A), "sdpa n=", len(B), "공유=", len(sh))
fa = [k for k in sh if not A[k][1]]; fb = [k for k in sh if not B[k][1]]
print("공유 문항 JSON 깨짐: eager", len(fa), "| sdpa", len(fb), "| 둘 다", len(set(fa) & set(fb)))
print("eager 전체 깨짐", sum(not v[1] for v in A.values()), "/", len(A), "| sdpa 전체 깨짐", sum(not v[1] for v in B.values()), "/", len(B))
same = sum(A[k][2] == B[k][2] for k in sh)
print("같은 답", same, "/", len(sh), "| 정답 eager", sum(A[k][2] == A[k][3] for k in sh), "sdpa", sum(B[k][2] == B[k][3] for k in sh))
for nm, S, L in (("eager만 깨짐", A, set(fa) - set(fb)), ("sdpa만 깨짐", B, set(fb) - set(fa)), ("둘 다", A, set(fa) & set(fb))):
    for k in sorted(L)[:3]:
        print(f"  [{nm} idx {k}] eager: {{\"answer\":{A[k][4][:110]!r}")
        print(f"  {'':{len(nm)+9}} sdpa : {{\"answer\":{B[k][4][:110]!r}")
