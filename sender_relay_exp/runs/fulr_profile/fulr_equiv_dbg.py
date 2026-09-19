"""Layer-wise attention-output difference: FULR hook reference vs FAST=2, bf16 and fp32 model."""
import os, sys, types, contextlib, torch
REPO = "/home/users/whddn12316/wsi_latent_0915_decode_hj"; sys.path.insert(0, REPO)
from transformers import Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from memory import rnlcr as rn
dev = torch.device("cuda:0")
for dtype in ([torch.float32] if os.environ.get("DBG_DTYPE") == "fp32" else [torch.bfloat16]):
    model = Qwen3VLForConditionalGeneration.from_pretrained("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking", torch_dtype=dtype, attn_implementation="sdpa").eval()
    lm = model.model.language_model.to(dev)
    del model; import gc; gc.collect()
    L0, M = 2000, 10
    torch.manual_seed(0)
    pos = lambda n, st: torch.arange(st, st + n, device=dev).view(1, 1, -1).expand(3, 1, -1)
    cache = DynamicCache()
    with torch.no_grad():
        ids = torch.randint(0, 150000, (L0 + M,), device=dev)
        lm(inputs_embeds=lm.embed_tokens(ids).unsqueeze(0), position_ids=pos(L0 + M, 0), past_key_values=cache, use_cache=True)
    L = L0 + M; lat = torch.arange(L0, L, device=dev)
    g = torch.Generator(device=dev).manual_seed(1)
    vstar = [l.values[..., L0:L, :] * (1 + 0.05 * torch.randn(l.values[..., L0:L, :].shape, device=dev, generator=g, dtype=l.values.dtype)) for l in cache.layers]
    bb = types.SimpleNamespace(lm=lm, device=dev, _rnlcr_R={"latent_cols": lat, "vstar": vstar, "vis_cols": torch.arange(100, 1100, device=dev)})
    eng = types.SimpleNamespace(_backbone=bb)
    def step(fast):
        cache.crop(L)
        os.environ["VLMAS_RNLCR"] = "fulr"; os.environ["VLMAS_RNLCR_ROWS"] = "gen"; os.environ["VLMAS_RNLCR_STATS"] = "lite"
        if fast: os.environ["VLMAS_RNLCR_FAST"] = str(fast)
        else: os.environ.pop("VLMAS_RNLCR_FAST", None)
        outs, ins = {}, {}
        x = lm.embed_tokens(torch.tensor([12345], device=dev)).unsqueeze(0)
        with torch.no_grad(), rn.terminal(eng, cache, L, case_index=0):
            hs = []
            for li, layer in enumerate(lm.layers):
                hs.append(layer.self_attn.register_forward_hook(lambda m, a, k, o, li=li: outs.__setitem__(li, (o[0] if isinstance(o, tuple) else o).float().clone()), with_kwargs=True))
                hs.append(layer.register_forward_pre_hook(lambda m, a, k, li=li: ins.__setitem__(li, (k.get("hidden_states", a[0] if a else None)).float().clone()), with_kwargs=True))
            h = lm(inputs_embeds=x, position_ids=pos(1, L), past_key_values=cache, use_cache=True).last_hidden_state.float()
            for hh in hs: hh.remove()
        return outs, ins, h
    o_ref, i_ref, h_ref = step(0)
    o_f2, i_f2, h_f2 = step(2)
    os.environ.pop("VLMAS_RNLCR", None)
    _, _, h_none = step(0) if False else (None, None, None)
    rel = lambda a, b: float((a - b).norm() / (b.norm() + 1e-12))
    rows = [(li, rel(i_f2[li], i_ref[li]), rel(o_f2[li], o_ref[li])) for li in range(len(lm.layers))]
    print(f"== {dtype}: final hidden rel diff fast2 vs ref {rel(h_f2, h_ref):.3e}", flush=True)
    print("   layer: input_rel / attn_out_rel ->", " ".join(f"L{li}:{a:.1e}/{b:.1e}" for li, a, b in rows[:6]), "...", " ".join(f"L{li}:{a:.1e}/{b:.1e}" for li, a, b in rows[-3:]), flush=True)
    print("   peak GiB %.1f" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)
