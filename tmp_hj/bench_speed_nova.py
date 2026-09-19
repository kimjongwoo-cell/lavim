"""Stage timing: full vision (12 crops x 256 tokens) / NOVA v1 select / LM prefill (base vs 25%) / 10 latent steps / 60 decode steps."""
import os, sys, time, torch
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from transformers import AutoModelForImageTextToText
from memory import nova_select as NV
M = "/home/users/whddn12316/models/Qwen3-VL-4B-Thinking"
dev = "cuda:0"
model = AutoModelForImageTextToText.from_pretrained(M, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
lm = model.model.language_model
def sync(): torch.cuda.synchronize()
def timeit(fn, n=5, warm=2):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        sync(); t = time.time(); fn(); sync(); ts.append(time.time() - t)
    return min(ts)
torch.manual_seed(0)
# 12 crops of 512x512 -> patch16 grid 32x32, merge 2 -> 256 tokens each = 3072
grid = torch.tensor([[1, 32, 32]] * 12, device=dev)
npatch = 12 * 32 * 32
pv = (torch.rand(npatch, 3 * 2 * 16 * 16, device=dev) * 2 - 1).to(torch.bfloat16)
out = {}
with torch.no_grad():
    feats = None
    def vis():
        global feats
        f = model.model.get_image_features(pv, grid, return_dict=True).pooler_output
        feats = torch.cat(list(f), 0) if isinstance(f, (list, tuple)) else f
    out["vision_encoder_12crops"] = timeit(vis)
    print("feats", tuple(feats.shape), flush=True)
    cfg = NV.config()
    pv32 = pv.float()
    out["nova_v1_select(3072->768)"] = timeit(lambda: NV.select_tokens(pv32, grid.cpu(), feats, [256] * 12, 768, cfg), n=5, warm=1)
    D = lm.config.hidden_size
    def prefill(L):
        x = torch.randn(1, L, D, device=dev, dtype=torch.bfloat16)
        return lm(inputs_embeds=x, use_cache=True)
    def steps(L, k):
        o = prefill(L); pkv = o.past_key_values
        def run():
            p = pkv
            for i in range(k):
                y = torch.randn(1, 1, D, device=dev, dtype=torch.bfloat16)
                p = lm(inputs_embeds=y, past_key_values=p, use_cache=True).past_key_values
        return run
    def t1(fn):
        sync(); t = time.time(); r = fn(); sync(); return time.time() - t
    def dec(L):
        p = prefill(L).past_key_values
        sync(); t = time.time()
        for i in range(60):
            y = torch.randn(1, 1, D, device=dev, dtype=torch.bfloat16)
            p = lm(inputs_embeds=y, past_key_values=p, use_cache=True).past_key_values
        sync(); return time.time() - t
    for _ in range(2): prefill(4661); prefill(2357)
    R = {}
    for rep in range(7):
        for name, L in [("base", 3861 + 800), ("nova25", 1557 + 800)]:
            R.setdefault(f"reasoner_prefill_{name}(L={L})", []).append(t1(lambda: prefill(L)))
        for name, L in [("base", 6862), ("nova25", 6862 - 2304)]:
            R.setdefault(f"answerer_prefill_{name}(ctx={L})", []).append(t1(lambda: prefill(L)))
        if rep < 4:
            for name, L in [("base", 6862), ("nova25", 6862 - 2304)]:
                R.setdefault(f"decode_60tok_{name}(ctx={L})", []).append(dec(L))
    for k, v in R.items(): out[k] = min(v)
for k, v in out.items(): print(f"{k:45s} {v:.3f}s (min)")
