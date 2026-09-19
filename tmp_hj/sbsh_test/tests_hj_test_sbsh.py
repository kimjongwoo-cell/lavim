"""CPU tests for memory/sbsh.py (State-Backed WSI Support Handoff) on a tiny random Qwen3-VL text model.

Run: python3 tests_hj_test_sbsh.py
"""
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel  # noqa: E402

from memory import rsvmh, sbsh  # noqa: E402
from memory.rss_diag import drop_cache_columns, support_columns  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    if not cond:
        raise AssertionError(f"{name} {detail}")
    PASS += 1
    print("ok", name, detail)


torch.manual_seed(0)

# ---------------------------------------------------------------- pure helpers
u = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
groups = [torch.tensor([0, 1]), torch.tensor([4]), torch.tensor([6, 7, 8])]
b = sbsh.support_bias(u, groups, 10)
check("bias values", torch.allclose(b.detach(), torch.tensor([1, 1, 0, 0, -2, 0, 3, 3, 3, 0.0])))
b.sum().backward()
check("bias grad = group sizes", torch.allclose(u.grad, torch.tensor([2.0, 1.0, 3.0])))

row = torch.arange(5, dtype=torch.float32)
m = sbsh.combine_mask(None, row, 1, torch.float32)
check("mask none shape", tuple(m.shape) == (1, 1, 1, 5) and torch.equal(m[0, 0, 0], row))
keep = torch.tensor([[[[True, False, True, True, True, True]]]])
m = sbsh.combine_mask(keep, row, 1, torch.float32)
check("mask bool -> -inf + bias", m[0, 0, 0, 1] < -1e30 and m[0, 0, 0, 2] == 2.0 and m.shape[-1] == 5)
fm = torch.full((1, 1, 1, 7), -1.0)
m = sbsh.combine_mask(fm, row, 1, torch.float32)
check("mask float add + slice", m.shape[-1] == 5 and torch.allclose(m[0, 0, 0], row - 1))

P = sbsh.rademacher(4000, 7, 0, "cpu")
check("rademacher values", set(P.unique().tolist()) == {-1.0, 1.0})
check("rademacher ~isotropic", torch.allclose(P.T @ P / 4000, torch.eye(7), atol=0.06))
check("rank corr", abs(sbsh.rank_corr([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
      and abs(sbsh.rank_corr([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12)

# ---------------------------------------------------------------- tiny model
cfg = Qwen3VLTextConfig(vocab_size=97, hidden_size=48, intermediate_size=96, num_hidden_layers=3,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=12, max_position_embeddings=512,
                        rope_scaling={"rope_type": "default", "mrope_section": [2, 2, 2], "mrope_interleaved": True})
cfg._attn_implementation = "sdpa"
lm = Qwen3VLTextModel(cfg).double().eval()
for p in lm.parameters():
    p.requires_grad_(False)
W = torch.randn(48, 48, dtype=torch.float64) / 48 ** 0.5


class FakeBB:
    def __init__(self, lm):
        self.lm = lm

    def _apply_realign(self, h):
        return torch.tanh(h @ W) * 2.0

    def _text_positions(self, n, start):
        return torch.arange(start, start + n).view(1, -1).expand(3, -1).unsqueeze(1)


bb = FakeBB(lm)
PRE = 40
emb = torch.randn(1, PRE, 48, dtype=torch.float64)
with torch.no_grad():
    o = lm(inputs_embeds=emb, position_ids=bb._text_positions(PRE, 0), past_key_values=DynamicCache(),
           use_cache=True, output_hidden_states=True)
base = o.past_key_values
h0 = o.hidden_states[-1][:, -1:, :]
SC = sbsh.SplitCache(base, PRE)
vis = torch.arange(6, 30)                      # 3 supports of 8 columns
ids = torch.repeat_interleave(torch.arange(3), 8)
pairs = support_columns(vis, ids)
grp = [c for _, c in pairs]
M = 4

# native (no hooks) latent loop, exactly like the backbone
with torch.no_grad():
    c = deepcopy(base)
    h = h0
    for step in range(M):
        oo = lm(inputs_embeds=bb._apply_realign(h), position_ids=bb._text_positions(1, PRE + step),
                past_key_values=c, use_cache=True, output_hidden_states=True)
        c = oo.past_key_values
        h = oo.hidden_states[-1][:, -1:, :]
    z_native = h[0, 0].double()

with torch.no_grad():
    z0 = sbsh.replay_latents(bb, SC, h0, PRE, M, grp, torch.zeros(3, dtype=torch.float64))
check("zero gate replay == native", torch.allclose(z0.double(), z_native.float().double(), atol=1e-5),
      f"max {float((z0.double() - z_native).abs().max()):.2e}")
check("base cache untouched", base.get_seq_length() == PRE)

# very negative gate on one support == deleting its columns from the cache
with torch.no_grad():
    zg = sbsh.replay_latents(bb, SC, h0, PRE, M, grp, torch.tensor([0.0, -1e9, 0.0], dtype=torch.float64))
    cd = deepcopy(base)
    drop_cache_columns(cd, grp[1])
    h = h0
    for step in range(M):
        oo = lm(inputs_embeds=bb._apply_realign(h), position_ids=bb._text_positions(1, PRE + step),
                past_key_values=cd, use_cache=True, output_hidden_states=True)
        cd = oo.past_key_values
        h = oo.hidden_states[-1][:, -1:, :]
check("gate -inf == column deletion", torch.allclose(zg.double(), h[0, 0].float().double(), atol=1e-4),
      f"max {float((zg.double() - h[0, 0].double()).abs().max()):.2e}")
check("gate changes state", float((zg.double() - z0.double()).norm()) > 1e-3)


def zbar_of(uv):
    with torch.no_grad():
        z = sbsh.replay_latents(bb, SC, h0, PRE, M, grp, uv).double()
    return z / z.norm()


# Jacobian columns by central difference. RMSNorm casts to float32 internally, so the step must stay
# well above float32 round-off (checked: eps 0.03-0.1 agree with autograd to 3-4 digits, 1e-5 does not)
eps = 3e-2
J = []
for g in range(3):
    e = torch.zeros(3, dtype=torch.float64)
    e[g] = eps
    J.append((zbar_of(e) - zbar_of(-e)) / (2 * eps))
S_exact = torch.stack([j.pow(2).sum() for j in J])

# autograd gradient of r . zbar matches the FD directional derivative
with torch.enable_grad():
    uu = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    z = sbsh.replay_latents(bb, SC, h0, PRE, M, grp, uu).double()
    zb = z / z.norm()
    r = torch.randn(zb.shape, dtype=torch.float64)
    (gr,) = torch.autograd.grad((r * zb).sum(), uu)
fd_dir = torch.stack([(r * J[g]).sum() for g in range(3)])
check("autograd grad == FD directional", torch.allclose(gr, fd_dir, rtol=5e-3, atol=1e-6),
      f"{gr.tolist()} vs {fd_dir.tolist()}")

# Hutchinson with many probes approaches the exact Jacobian energy
with torch.enable_grad():
    uu = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    z = sbsh.replay_latents(bb, SC, h0, PRE, M, grp, uu).double()
    zb = z / z.norm()
    probes = sbsh.rademacher(3000, int(zb.numel()), 1, "cpu").double()
    S_h = sbsh.hutchinson_energy(zb, uu, probes)
rel = (S_h - S_exact).abs() / S_exact
check("hutchinson ~ exact energy", bool((rel < 0.08).all()), f"rel {rel.tolist()} exact {S_exact.tolist()}")


# ---------------------------------------------------------------- compute() end to end on the fake probe
class Out:
    pass


out = Out()
with torch.no_grad():
    c = deepcopy(base)
    h = h0
    for step in range(M):
        oo = lm(inputs_embeds=bb._apply_realign(h), position_ids=bb._text_positions(1, PRE + step),
                past_key_values=c, use_cache=True, output_hidden_states=True)
        c = oo.past_key_values
        h = oo.hidden_states[-1][:, -1:, :]
out.past_key_values = c
out.hidden_states = oo.hidden_states
probe = SimpleNamespace(bb=bb, image_ids=ids, note=None, h0=h0, vis=vis, m=M, pre_len=PRE, cursor=PRE,
                        proxy={"supports": [0, 1, 2], "mass_sum_heads": [3.0, 1.0, 2.0], "s_hat": [1.0, 2.0, 3.0]})
os.environ["VLMAS_SBSH_EST"] = "both"
os.environ["VLMAS_SBSH_K"] = "400"
os.environ["VLMAS_SBSH_EPS"] = "3e-2"
rec = sbsh.compute(probe, out)
check("compute no error", "error" not in rec and "skip" not in rec, str(rec.get("error") or rec.get("skip")))
check("compute floor", rec["floor_cos"] < 1e-8, f"{rec['floor_cos']:.2e}")
check("compute fd == exact", torch.allclose(torch.tensor(rec["S_fd"], dtype=torch.float64), S_exact, rtol=1e-3),
      f"{rec['S_fd']} vs {S_exact.tolist()}")
check("compute hutchinson ~ exact", bool(((torch.tensor(rec["S"], dtype=torch.float64) - S_exact).abs() / S_exact < 0.3).all()),
      f"{rec['S']} vs {S_exact.tolist()}")
check("compute live cache untouched", out.past_key_values.get_seq_length() == PRE + M)
check("compute rho fields", rec.get("rho_S_mass") is not None and rec.get("rho_hutch_fd") is not None)

# gated attention over SplitCache == library SDPA on the concatenated K/V (GQA, one query)
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

att = lm.layers[0].self_attn
kk = torch.randn(1, 2, 9, 12, dtype=torch.float64)
vv = torch.randn(1, 2, 9, 12, dtype=torch.float64)
qq = torch.randn(1, 4, 1, 12, dtype=torch.float64)
fake = SimpleNamespace(layers=[SimpleNamespace(keys=kk[..., :8, :], values=vv[..., :8, :]) for _ in lm.layers])
sc = sbsh.SplitCache(fake, 8)
sc.update(kk[..., 8:, :], vv[..., 8:, :], 0)
ref, _ = sdpa_attention_forward(att, qq, kk, vv, None, scaling=att.scaling)
got, _ = sbsh.gated_attention_fn(torch.zeros(1, dtype=torch.float64), [torch.tensor([2, 3])], sc)(att, qq, None, None, None, scaling=att.scaling)
check("split gated attention == sdpa (GQA)", torch.allclose(got, ref, atol=1e-10))
keep = torch.ones(1, 1, 1, 9, dtype=torch.bool)
keep[..., 3] = False
ref_m, _ = sdpa_attention_forward(att, qq, kk, vv, keep, scaling=att.scaling)
got_m, _ = sbsh.gated_attention_fn(torch.zeros(1, dtype=torch.float64), [torch.tensor([2])], sc)(att, qq, None, None, keep, scaling=att.scaling)
check("split gated attention honours bool mask", torch.allclose(got_m, ref_m, atol=1e-10))
check("split cache lengths", sc.get_seq_length(0) == 9 and sc.get_mask_sizes(1, 0) == (10, 0) and sc.get_seq_length(1) == 8)
before = dict(ALL_ATTENTION_FUNCTIONS._local_mapping)
with sbsh.support_gate(lm, grp, torch.zeros(3, dtype=torch.float64), SC):
    check("gate swaps sdpa", ALL_ATTENTION_FUNCTIONS["sdpa"] is not sdpa_attention_forward)
check("registry restored", ALL_ATTENTION_FUNCTIONS._local_mapping == before and ALL_ATTENTION_FUNCTIONS["sdpa"] is sdpa_attention_forward)

# memory: tensors autograd keeps for backward must not grow with (prefix length x layers x steps)
def saved_extra_bytes(pre):
    e = torch.randn(1, pre, 48, dtype=torch.float64)
    with torch.no_grad():
        oo = lm(inputs_embeds=e, position_ids=bb._text_positions(pre, 0), past_key_values=DynamicCache(),
                use_cache=True, output_hidden_states=True)
    sc = sbsh.SplitCache(oo.past_key_values, pre)
    hh = oo.hidden_states[-1][:, -1:, :]
    saved = {}

    def pack(t):
        saved[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t

    shared = {kv.untyped_storage().data_ptr() for pair in sc.prefix for kv in pair}
    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        uu = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        sbsh.replay_latents(bb, sc, hh, pre, M, grp, uu)
    per_layer_kv = sum(kv.untyped_storage().nbytes() for kv in sc.prefix[0])
    return sum(v for k, v in saved.items() if k not in shared), per_layer_kv


small, kv_small = saved_extra_bytes(40)
big, kv_big = saved_extra_bytes(800)
growth = big - small
copy_cost = len(lm.layers) * M * (kv_big - kv_small)          # what per-step K/V copies would add
check("saved-for-backward growth << per-step K/V copies", growth < 0.25 * copy_cost,
      f"growth {growth} bytes vs K/V-copy cost {copy_cost}")

# ---------------------------------------------------------------- env switch
for k in ("VLMAS_SBSH", "VLMAS_KV_RESTAGE_ORDER"):
    os.environ.pop(k, None)
check("disabled by default", not sbsh.enabled())
os.environ["VLMAS_KV_RESTAGE_ORDER"] = "state_backed"
check("enabled by order", sbsh.enabled())
os.environ.pop("VLMAS_KV_RESTAGE_ORDER")
os.environ["VLMAS_SBSH"] = "1"
check("enabled by flag", sbsh.enabled())
os.environ.pop("VLMAS_SBSH")

# ---------------------------------------------------------------- rsvmh plan with SBSH scores
ctx = SimpleNamespace(done=True, note=None, vis=torch.arange(100, 124), image_ids=torch.repeat_interleave(torch.arange(3), 8),
                      proxy={"supports": [0, 1, 2], "size": [8, 8, 8], "mag": [5, 20, 20],
                             "s_hat": [0.3, 0.2, 0.1], "mass_sum_heads": [1.0, 2.0, 3.0]},
                      sbsh={"supports": [0, 1, 2], "S": [5.0, 1.0, 3.0]})
fbb = SimpleNamespace(_rss_ctx=ctx)
cols = torch.arange(100, 124)
perm, counts, note = rsvmh.plan(fbb, cols, "state_backed")
check("state_backed order far->near = ascending S", perm is not None and perm[:8].tolist() == list(range(8, 16))
      and perm[-8:].tolist() == list(range(0, 8)), note)
perm, counts, note = rsvmh.plan(fbb, cols, "state_backed_rev")
check("state_backed_rev reverses", perm[:8].tolist() == list(range(0, 8)) and perm[-8:].tolist() == list(range(8, 16)), note)
ctx.sbsh = {"skip": "x"}
perm, counts, note = rsvmh.plan(fbb, cols, "state_backed")
check("missing SBSH falls back", perm is None and "no SBSH" in note, note)
ctx.sbsh = {"supports": [0, 1, 2], "S": [5.0, 1.0, 3.0]}
perm, _, _ = rsvmh.plan(fbb, cols, "sensitivity")
check("sensitivity unchanged (s_hat)", perm[:8].tolist() == list(range(16, 24)))
check("orders include state modes", "state_backed" in rsvmh.ORDERS and "state_backed_rev" in rsvmh.ORDERS)

print(f"{PASS}/{PASS} passed")
