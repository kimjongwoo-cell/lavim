import glob, json, os, re, sys
sys.path.insert(0, ".")
from vision_text_mas.prompts import canonical_open_answer_options
W = "sender_relay_exp/runs/nav4_wsivqa"; recs = json.load(open("sender_relay_exp/data_wsivqa/wsivqa.json"))
FOR = re.compile(r"[Ѐ-ӿ가-힣぀-ヿ一-鿿]"); REP = re.compile(r"(\b\S{1,20}\b)(?:[\s,\"']+\1){4,}")
def load(arm):
    L = {}
    for f in glob.glob(f"{W}/{arm}/wsivqa/gpu*_attempt_*/*/result.json"):
        try:
            r = json.load(open(f)); c = json.load(open(os.path.dirname(f) + "/answerer_call.json"))
        except Exception:
            continue
        i = int(r["dataset_index"]); t = os.path.getmtime(f)
        if i not in L or t > L[i][0]:
            a = r["answer"]; L[i] = (t, str(a.get("answer", "") if isinstance(a, dict) else a), " ".join(c.get("final_outputs") or []))
    return L
b, n, s = load("latplan"), load("navnp"), load("navsel")
ids100 = set(json.load(open(W + "/navnp/navnp100_ids.json")))
bad = []
for i, (t, p, raw) in sorted(b.items()):
    ch = recs[i].get("Choice") or list(canonical_open_answer_options(recs[i]["Question"]))
    if ch and ((p not in ch) or FOR.search(raw) or REP.search(raw)):
        bad.append(i)
print("base garbage:", bad)
for i in bad:
    if i in ids100:
        print(f"#{i} q={recs[i]['Question'][:60]!r} gold={recs[i]['Answer']!r}")
        for nm, d in (("base  ", b), ("navnp ", n), ("navsel", s)):
            print("   ", nm, d[i][2][:120].replace("\n", " "))
