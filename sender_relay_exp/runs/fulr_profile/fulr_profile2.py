"""Where does FULR hook time go? Real Qwen3-VL-4B LM, synthetic 7k cache, decode steps timed under: none / noop hook / ref fulr / fast fulr."""
import os, sys, time, types, json
import torch
REPO = "/home/users/whddn12316/wsi_latent_0915_decode_hj"; sys.path.insert(0, REPO)
from transformers import Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from memory import rnlcr as rn
from memory.lu import _call_parts

dev = torch.device("cuda:0")
model = Qwen3VLForConditionalGeneration.from_pretrained("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking", torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
lm = model.model.language_model
L0, M, NVIS = 6800, 10, 3072
torch.manual_seed(0)
def pos(n, start): return torch.arange(start, start + n, device=dev).view(1, 1, -1).expand(3, 1, -1)
cache = DynamicCache()
with torch.no_grad():
    ids = torch.randint(0, 150000, (L0 + M,), device=dev)
    for a in range(0, L0 + M, 2048):
        b = min(a + 2048, L0 + M)
        lm(inputs_embeds=lm.embed_tokens(ids[a:b]).unsqueeze(0), position_ids=pos(b - a, a), past_key_values=cache, use_cache=True)
L = L0 + M
lat = torch.arange(L0, L0 + M, device=dev)
vstar = [layer.values[..., L0:L0 + M, :] * (1 + 0.05 * torch.randn_like(layer.values[..., L0:L0 + M, :])) for layer in cache.layers]
bb = types.SimpleNamespace(lm=lm, device=dev, _rnlcr_R={"latent_cols": lat, "vstar": vstar, "vis_cols": torch.arange(100, 100 + NVIS, device=dev)})
eng = types.SimpleNamespace(_backbone=bb)
print("cache len", L, "layers", len(cache.layers), "kv", tuple(cache.layers[0].keys.shape), flush=True)

STEPS, WARM = 40, 5
def decode(n, start):
    with torch.no_grad():
        for t in range(n):
            x = lm.embed_tokens(torch.randint(0, 150000, (1,), device=dev)).unsqueeze(0)
            lm(inputs_embeds=x, position_ids=pos(1, start + t), past_key_values=cache, use_cache=True)

def run(tag, ctx_factory):
    cache.crop(L)
    with ctx_factory():
        decode(WARM, L)
    cache.crop(L)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with ctx_factory():
        decode(STEPS, L)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    print(f"[TIME] {tag:14s} {1000 * dt / STEPS:8.2f} ms/token", flush=True)
    return dt / STEPS

import contextlib
def none(): return contextlib.nullcontext()
@contextlib.contextmanager
def noop():
    hs = [layer.self_attn.register_forward_hook(lambda m, a, k, o: (_call_parts(a, k), o)[1], with_kwargs=True) for layer in lm.layers]
    try: yield
    finally:
        for h in hs: h.remove()
def fulr(fast, stats="lite"):
    @contextlib.contextmanager
    def f():
        os.environ["VLMAS_RNLCR"] = "fulr"; os.environ["VLMAS_RNLCR_ROWS"] = "gen"; os.environ["VLMAS_RNLCR_STATS"] = stats
        if fast: os.environ["VLMAS_RNLCR_FAST"] = str(fast)
        else: os.environ.pop("VLMAS_RNLCR_FAST", None)
        with rn.terminal(eng, cache, L, case_index=0) as st:
            yield st
        print(f"[STAT] fast={fast} stats={stats} calls={st.get('calls')} rows={st.get('rows')} a_s={st.get('a_s')} cos_rot={st.get('cos_rot')} delta_rel={st.get('delta_rel')} fast_flag={st.get('fast')}", flush=True)
    return f
@contextlib.contextmanager
def flex():
    prev = lm.config._attn_implementation
    lm.config._attn_implementation = "flex_attention"
    try: yield
    finally: lm.config._attn_implementation = prev

res = {}
ARMS = (("none", none), ("noop_hook", noop), ("fulr_ref", fulr(0)), ("fulr_fast1", fulr(1)), ("fulr_fast2", fulr(2)), ("fulr_fast2_nostat", fulr(2, "none")))
for rep in range(2):
    for tag, fac in ARMS:
        res.setdefault(tag, []).append(run(tag, fac))
# equivalence on the real model: next-layer output of one decode step, ref vs fast2 (same cache state)
def one_step(fac):
    cache.crop(L)
    x = lm.embed_tokens(torch.tensor([12345], device=dev)).unsqueeze(0)
    with fac(), torch.no_grad():
        return lm(inputs_embeds=x, position_ids=pos(1, L), past_key_values=cache, use_cache=True).last_hidden_state.float()
h_ref, h_f1, h_f2, h_none = one_step(fulr(0)), one_step(fulr(1)), one_step(fulr(2)), one_step(none)
rel = lambda a, b: float((a - b).norm() / b.norm())
print(f"[EQUIV] last_hidden rel diff: fast1 vs ref {rel(h_f1, h_ref):.3e} | fast2 vs ref {rel(h_f2, h_ref):.3e} | ref vs no-FULR {rel(h_ref, h_none):.3e}", flush=True)
try:
    import signal
    def _to(*a): raise TimeoutError("flex arm timeout")
    signal.signal(signal.SIGALRM, _to); signal.alarm(600)
    res.setdefault("flex_off", []).append(run("flex_off", flex))
    signal.alarm(0)
except Exception as exc:
    print(f"[TIME] flex_off FAILED {exc!r}"[:400], flush=True)
print("[SUMMARY]", json.dumps({k: round(1000 * min(v), 2) for k, v in res.items()}), flush=True)

from torch.profiler import profile, ProfilerActivity
for tag, fac in (("fulr_fast2", fulr(2)), ("fulr_fast2_nostat", fulr(2, "none"))):
    cache.crop(L)
    with fac():
        decode(3, L)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            decode(10, L + 3)
            torch.cuda.synchronize()
    ka = prof.key_averages()
    cpu = sum(e.self_cpu_time_total for e in ka) / 1e3 / 10
    cuda = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in ka) / 1e3 / 10
    nk = sum(e.count for e in ka if e.key.startswith("aten::")) / 10
    launches = sum(e.count for e in ka if "cudaLaunchKernel" in e.key or e.key == "cudaLaunchKernel") / 10
    print(f"\n[PROF] {tag}: self CPU {cpu:.1f} ms/token, self CUDA {cuda:.1f} ms/token, aten calls {nk:.0f}/token, kernel launches {launches:.0f}/token", flush=True)
    print(ka.table(sort_by="self_cpu_time_total", row_limit=14), flush=True)
    sk = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
    print(ka.table(sort_by=sk, row_limit=12), flush=True)
