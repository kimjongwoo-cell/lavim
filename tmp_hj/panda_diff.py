import json, glob, os, sys, collections
T = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
sys.path.insert(0, T)
from eval.metrics import acc_of_seq, expand_letter
ds = json.load(open("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/panda.json"))
ex = ds[0] if isinstance(ds, list) else ds
if len(sys.argv) > 1 and sys.argv[1] == "keys":
    print(type(ds), len(ds)); print({k: str(v)[:120] for k, v in ex.items()}); sys.exit()
R = T + "/sender_relay_exp/runs/"
dirs = sys.argv[1:]
CH = ["0", "1", "2", "3", "4", "5"]
for d in dirs:
    fs = sorted(glob.glob(R + d + "/*/result.json"))
    rows = []
    for f in fs:
        r = json.load(open(f))
        a = r.get("answer")
        pred = str(a.get("answer") if isinstance(a, dict) else a).strip()
        rows.append((str(r["gold_answer"]).strip(), pred, os.path.basename(os.path.dirname(f))))
    def bacc(fn):
        by = collections.defaultdict(list)
        for g, p, _ in rows: by[g].append(bool(fn(g, p)))
        return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by), {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items())}
    exact = lambda g, p: p == g
    old = lambda g, p: acc_of_seq(CH, g, expand_letter(p, CH))
    be, re_ = bacc(exact); bo, ro = bacc(old)
    preds = collections.Counter(p for _, p, _ in rows)
    print(f"== {d}  n={len(rows)}")
    print(f"   예측분포: {dict(preds.most_common())}")
    print(f"   exact BACC {be:.2f}  클래스별 {re_}")
    print(f"   old   BACC {bo:.2f}  클래스별 {ro}")
    diff = [(g, p) for g, p, _ in rows if bool(exact(g, p)) != bool(old(g, p))]
    print(f"   old만 정답: {len(diff)}  {collections.Counter(diff).most_common(8)}")
