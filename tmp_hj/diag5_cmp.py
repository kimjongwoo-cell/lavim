import json, glob, sys, re
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
def ans(root):
    out = {}
    for f in glob.glob(root + "/**/result.json", recursive=True):
        d = json.load(open(f)); a = d.get("answer")
        out[d.get("dataset_index")] = (a.get("answer") if isinstance(a, dict) else str(a))
    return out
ds = sys.argv[1] if len(sys.argv) > 1 else "gtex"
base = ans(f"{R}/rtask_ntrs50_nav3/{ds}")
gold = {}
try:
    items = json.load(open(f"/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/{ds}.json"))
    items = items if isinstance(items, list) else next(v for v in items.values() if isinstance(v, list))
    gold = {i: it.get("Answer") for i, it in enumerate(items)}
except Exception as e:
    print("gold?", e)
arms = sys.argv[2:] or ["velh_diag", "velh_bind", "velh_handoff", "velh_full", "velh_direct", "velh_shuffle", "tc_diag", "tc_drop", "tc_filler", "tc_space", "tc_drop_lu", "tc_filler_lu"]
print(f"{'idx':>3} {'gold':14s} {'ntrs50':14s} " + " ".join(f"{a:12s}" for a in arms))
tab = {a: ans(f"{R}/rtask_{a}_nav3/diag5_{ds}") for a in arms}
for i in range(5):
    print(f"{i:>3} {str(gold.get(i))[:14]:14s} {str(base.get(i))[:14]:14s} " + " ".join(f"{str(tab[a].get(i, '·'))[:12]:12s}" for a in arms))
for a in arms:
    logs = sorted(glob.glob(f"{R}/rtask_{a}_nav3/diag5_{ds}/*.log"))
    if not logs: continue
    tag = "VELH" if a.startswith("velh") else "TextCut"
    for l in open(logs[-1]):
        if l.startswith(f"[{tag}] case"):
            m = re.search(r"case (\d+) (.*)", l.strip()); print(f"  {a:12s} {m.group(1)} {m.group(2)[:260]}")
