"""Separate the two causes of ident-vs-generated token mismatch:
native teacher-forced forward (bf16 sdpa, no hook) argmax vs the greedy token, and vs ident."""
import glob, json, os, sys
import torch
from safetensors import safe_open
K = sys.argv[1]; M = "/home/users/whddn12316/models/Qwen3-VL-4B-Thinking"
cfg = json.load(open(f"{M}/config.json"))
tie = cfg.get("tie_word_embeddings", cfg.get("text_config", {}).get("tie_word_embeddings"))
W = None
for f in sorted(glob.glob(f"{M}/*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if (k.endswith("lm_head.weight")) or (W is None and tie and k.endswith("embed_tokens.weight")):
                W = s.get_tensor(k).float(); print("using", k, "tie", tie)
W = W.double()
tot = {}
for jf in sorted(glob.glob(f"{K}/nloi_nat_*.jsonl")):
    name = os.path.basename(jf)[5:-6]
    a = b = c = n = 0; dec = [0, 0, 0, 0]
    for l in open(jf):
        r = json.loads(l)
        dp = f"{K}/dump_{name}/nloi_case{r['case']:03d}.pt"
        if not os.path.exists(dp):
            continue
        h = torch.load(dp, weights_only=False)["h"]
        nat = (h["native"].double() @ W.T).argmax(-1).tolist()
        idn = (h["ident"].double() @ W.T).argmax(-1).tolist()
        for k, t in enumerate(r["tok_rows"]):
            g = int(r["gen_tokens"][t]); n += 1
            a += nat[k] == g; b += idn[k] == g; c += nat[k] == idn[k]
            if k == 0:
                dec[0] += 1; dec[1] += nat[k] == g; dec[2] += idn[k] == g; dec[3] += int(idn[k]) == int(r["argmax_ident"][k])
    print(f"{name}: rows={n} native==gen {a} ident==gen {b} native==ident {c} | decision row: n={dec[0]} native==gen {dec[1]} ident==gen {dec[2]} ident_cpu==ident_gpu {dec[3]}")
