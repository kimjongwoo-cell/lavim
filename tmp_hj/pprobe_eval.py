import glob, json, os, re, sys
sys.path.insert(0, ".")
from vision_text_mas.prompts import canonical_open_answer_options
W = "sender_relay_exp/runs/nav4_wsivqa"; recs = json.load(open("sender_relay_exp/data_wsivqa/wsivqa.json"))
FOR = re.compile(r"[Ѐ-ӿ가-힣぀-ヿ一-鿿]"); REP = re.compile(r"(\b\S{1,20}\b)(?:[\s,\"']+\1){4,}")
IDS = [24, 65, 72, 76, 86, 149, 204, 211, 314, 354, 357, 362, 461, 522, 530, 650, 652, 683]
def load(pat):
    L = {}
    for f in glob.glob(pat):
        try:
            r = json.load(open(f)); c = json.load(open(os.path.dirname(f) + "/answerer_call.json"))
        except Exception:
            continue
        i = int(r["dataset_index"]); t = os.path.getmtime(f)
        if i not in L or t > L[i][0]:
            a = r["answer"]; L[i] = (t, str(a.get("answer", "") if isinstance(a, dict) else a), " ".join(c.get("final_outputs") or []))
    return L
arms = {"base(원런)": load(f"{W}/latplan/wsivqa/gpu*_attempt_*/*/result.json")}
for a in "CAB":
    arms[a] = load(f"{W}/pprobe/{a}/run/*/result.json")
def broken(i, p, raw):
    ch = recs[i].get("Choice") or list(canonical_open_answer_options(recs[i]["Question"]))
    return (p not in ch) or bool(FOR.search(raw)) or bool(REP.search(raw))
for name, L in arms.items():
    b = [i for i in IDS if i in L and broken(i, L[i][1], L[i][2])]
    ok = sum(1 for i in IDS if i in L and L[i][1].strip().lower() == recs[i]["Answer"].strip().lower())
    print(f"{name:10s} n={sum(i in L for i in IDS)} broken={len(b)} correct={ok} broken_ids={b}")
for i in IDS[:6]:
    print(f"#{i}", " | ".join(f"{n}: {L.get(i, (0, '', ''))[2][:45]!r}" for n, L in arms.items()))
def bx(a):
    o = {}
    for f in glob.glob(f"{W}/pprobe/{a}/run/*/result.json"):
        r = json.load(open(f)); o[int(r["dataset_index"])] = set((p["magnification"], p["box"]["x"], p["box"]["y"]) for p in r["patches"])
    return o
Cb = bx("C")
for a in "AB":
    Xb = bx(a); same = [i for i in IDS if Cb.get(i) == Xb.get(i)]
    fixed = [i for i in same if not broken(i, arms[a][i][1], arms[a][i][2])]
    print(f"{a}: identical 12 crops in {len(same)} Qs {same}; of these fixed {len(fixed)} {fixed}")
