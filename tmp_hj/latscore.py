"""Score latmask/latmove/latdepth variants on broken 18 / clean 18 vs baseline (thumbsink runs)."""
import glob, json, os, re, importlib.util
B = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/"
s = importlib.util.spec_from_file_location("am", "/home/users/whddn12316/wsi_latent_0902_2155_hj/eval/answer_match.py")
am = importlib.util.module_from_spec(s); s.loader.exec_module(am)
recs = json.load(open(B + "data_wsivqa/wsivqa.json"))
FOREIGN = re.compile(r"[Ѐ-ӿ가-힣぀-ヿ一-鿿]"); REPEAT = re.compile(r"(\b\S{1,20}\b)(?:[\s,\"']+\1){4,}")
def load(d):
    out = {}
    for f in glob.glob(d + "/**/result.json", recursive=True):
        r = json.load(open(f)); i = int(r["dataset_index"])
        try:
            c = json.load(open(os.path.join(os.path.dirname(f), "answerer_call.json"))); raw = " ".join(c.get("final_outputs") or [])
        except Exception:
            raw = ""
        a = r.get("answer", {}); out[i] = (str(a.get("answer", "") if isinstance(a, dict) else a), raw)
    return out
base = {s: load(B + f"runs/nav4_wsivqa/thumbsink/{s}/run") for s in ("broken", "clean")}
arms = sorted({os.path.basename(d).rsplit("_", 2)[0] + "|" + d for d in glob.glob(B + "runs/nav4_wsivqa/latmask/*") if os.path.isdir(d + "/run")})
rows = {}
for d in glob.glob(B + "runs/nav4_wsivqa/latmask/*"):
    if not os.path.isdir(d + "/run"):
        continue
    name = os.path.basename(d)
    st = "broken" if "_broken" in name else "clean"
    arm = name.replace("_broken", "").replace("_clean", "")
    L = load(d + "/run")
    ok = sum(am.normalize(p) == am.normalize(recs[i]["Answer"]) for i, (p, _) in L.items())
    br = sum(bool(FOREIGN.search(raw) or REPEAT.search(raw)) for p, raw in L.values())
    rows.setdefault(arm, {})[st] = (ok, br, len(L))
for st in ("broken", "clean"):
    L = base[st]
    ok = sum(am.normalize(p) == am.normalize(recs[i]["Answer"]) for i, (p, _) in L.items())
    br = sum(bool(FOREIGN.search(raw) or REPEAT.search(raw)) for p, raw in L.values())
    rows.setdefault("BASELINE", {})[st] = (ok, br, len(L))
for arm in sorted(rows, key=lambda a: (a != "BASELINE", a)):
    r = rows[arm]
    f = lambda k: (f"correct {r[k][0]}/{r[k][2]} garbled {r[k][1]}" + (" (partial)" if r[k][2] < 18 else "")) if k in r else "-"
    print(f"{arm:<16} broken: {f('broken'):<32} clean: {f('clean')}")
