"""Unit tests for memory/rss_diag.py (Efficient Role-Support Sensitivity). Run: python tests_hj_test_rss.py"""
import math
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory import rss_diag as R  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")


torch.manual_seed(0)

# 1. closed-form deletion == explicit masked softmax (fp64), incl. a heavy group
H, L, D = 4, 30, 8
scores = torch.randn(H, L, dtype=torch.float64) * 2
scores[:, 5:9] += 4.0  # heavy support
V = torch.randn(H, L, D, dtype=torch.float64)
alpha = torch.softmax(scores, dim=-1)
groups = [torch.arange(0, 5), torch.arange(5, 9), torch.arange(9, 20), torch.tensor([25, 27])]
o, mass, delta = R.closed_form_deletion(alpha, V, groups)
check("o equals alpha@V", torch.allclose(o, torch.einsum("hl,hld->hd", alpha, V)))
for gi, cols in enumerate(groups):
    explicit = R.masked_row_output(scores, V, cols)
    check(f"deletion identity g{gi}", torch.allclose(o - delta[gi], explicit, atol=1e-10),
          f"max {float((o - delta[gi] - explicit).abs().max())}")
    check(f"mass g{gi}", torch.allclose(mass[gi], alpha[:, cols].sum(-1)))
check("heavy group mass largest", int(mass.mean(-1).argmax()) == 1)

# 2. redundant support (value equals o direction mix) gives small delta
V2 = V.clone()
o2 = torch.einsum("hl,hld->hd", alpha, V2)
V2[:, 20:25, :] = o2.unsqueeze(1)  # values equal to the full message
o2b, m2, d2 = R.closed_form_deletion(alpha, V2, [torch.arange(20, 25)])
# c = m * o_before; o changes slightly because V changed, recompute exact expectation
explicit2 = R.masked_row_output(scores, V2, torch.arange(20, 25))
check("redundant support identity", torch.allclose(o2b - d2[0], explicit2, atol=1e-10))

# 3. project_heads == nn.Linear on concat (with and without bias)
for bias in (False, True):
    lin = nn.Linear(H * D, 12, bias=bias).double()
    x = torch.randn(3, H, D, dtype=torch.float64)
    ref = lin(x.reshape(3, H * D)).float()
    got = R.project_heads(lin, x)
    check(f"project_heads bias={bias}", torch.allclose(got, ref, atol=1e-5))

# 4. JS divergence
p = torch.softmax(torch.randn(50), -1)
q = torch.softmax(torch.randn(50), -1)
check("js self zero", R.js_divergence(p, p) < 1e-12)
check("js symmetric", abs(R.js_divergence(p, q) - R.js_divergence(q, p)) < 1e-12)
check("js bound ln2", R.js_divergence(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) <= math.log(2) + 1e-9)
check("js disjoint = ln2", abs(R.js_divergence(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) - math.log(2)) < 1e-9)

# 5. cosine distance
a = torch.randn(16)
check("cos dist self", abs(R.cosine_distance(a, a)) < 1e-12)
check("cos dist opposite", abs(R.cosine_distance(a, -a) - 2.0) < 1e-12)

# 6. support columns / image ids
cols = torch.tensor([100, 101, 102, 150, 151, 200])
ids = torch.tensor([0, 0, 0, 2, 2, 5])
sc = R.support_columns(cols, ids)
check("support ids", [i for i, _ in sc] == [0, 2, 5])
check("support cols", [c.tolist() for _, c in sc] == [[100, 101, 102], [150, 151], [200]])
check("ids from counts", R.image_ids_from_counts([2, 0, 3]).tolist() == [0, 0, 2, 2, 2])


# 7. drop_cache_columns
class FakeLayer:
    def __init__(self, n):
        self.keys = torch.arange(n, dtype=torch.float32).view(1, 1, n, 1).repeat(1, 2, 1, 3)
        self.values = -self.keys.clone()


class FakeCache:
    def __init__(self, n, nl=3):
        self.layers = [FakeLayer(n) for _ in range(nl)]


c = FakeCache(10)
R.drop_cache_columns(c, torch.tensor([2, 3, 7]))
check("drop columns order", c.layers[1].keys[0, 0, :, 0].tolist() == [0, 1, 4, 5, 6, 8, 9])
check("drop columns values", c.layers[2].values[0, 1, :, 2].tolist() == [0, -1, -4, -5, -6, -8, -9])
c2 = FakeCache(5)
R.drop_cache_columns(c2, torch.empty(0, dtype=torch.long))
check("drop none", c2.layers[0].keys.shape[-2] == 5)


# 8. close flag detection
class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        n = 2 + (2 if text.startswith("</think>") else 0)
        return {"input_ids": list(range(n))}


check("close flag False", R.close_flag_for_length(FakeTok(), 2) is False)
check("close flag True", R.close_flag_for_length(FakeTok(), 4) is True)
check("close flag None", R.close_flag_for_length(FakeTok(), 3) is None)

# 9. proxy layer default
import os  # noqa: E402

os.environ.pop("VLMAS_RSS_LAYER", None)
check("proxy layer 36 -> 27", R.proxy_layer(36) == 27)
os.environ["VLMAS_RSS_LAYER"] = "20"
check("proxy layer env", R.proxy_layer(36) == 20)
os.environ.pop("VLMAS_RSS_LAYER", None)


# 10. full hook on a fake GQA attention module vs brute force
class FakeAttn(nn.Module):
    def __init__(self, hidden=24, hq=4, hkv=2, hd=6):
        super().__init__()
        self.head_dim = hd
        self.num_key_value_groups = hq // hkv
        self.scaling = hd ** -0.5
        self.q_proj = nn.Linear(hidden, hq * hd, bias=False)
        self.q_norm = nn.Identity()
        self.o_proj = nn.Linear(hq * hd, hidden, bias=False)


hidden, hq, hkv, hd, Lc = 24, 4, 2, 6, 40
attn = FakeAttn(hidden, hq, hkv, hd)
layer_idx = 27
layers = [types.SimpleNamespace(self_attn=attn if i == layer_idx else None) for i in range(36)]
bb = types.SimpleNamespace(
    lm=types.SimpleNamespace(layers=layers),
    model=types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2))),
    _prefill_prune_survivor_image_ids=torch.tensor([0] * 6 + [1] * 5 + [3] * 7),
    _prune_image_magnifications=(5, 5, 20, 20),
)
vis = torch.arange(10, 28)
probe = R.ReasonerProbe(bb, cur_vis=vis, cursor=100, cur_len=39, last_hidden=torch.randn(1, 1, hidden), m=10, thw=None)
check("probe id source", probe.id_source == "prefill_survivors")
keys = torch.randn(1, hkv, Lc, hd)
values = torch.randn(1, hkv, Lc, hd)
kv = types.SimpleNamespace(layers=[types.SimpleNamespace(keys=keys, values=values) if i == layer_idx else None for i in range(36)])
h_in = torch.randn(1, 1, hidden)
cos = torch.ones(1, 1, hd)
sin = torch.zeros(1, 1, hd)
with torch.no_grad():
    q = attn.q_proj(h_in).view(1, 1, hq, hd).transpose(1, 2)[0, :, 0, :]                 # [Hq, D]
    K = keys[0].repeat_interleave(2, dim=0)
    Vv = values[0].repeat_interleave(2, dim=0)
    sc_ = (q.unsqueeze(1) @ K.transpose(-2, -1))[:, 0, :] * attn.scaling                   # [Hq, L]
    a_ = torch.softmax(sc_, -1)
    o_ = torch.einsum("hl,hld->hd", a_, Vv)
    out_row = attn.o_proj(o_.reshape(1, hq * hd))
    output = (out_row.view(1, 1, hidden), None)
    probe.before_step(9)
    probe._hook(attn, (), {"hidden_states": h_in, "position_embeddings": (cos, sin), "past_key_values": kv}, output)
    if probe.handle is not None:
        probe.handle.remove()
px = probe.proxy
check("hook produced proxy", px is not None, str(probe.note))
if px is not None:
    check("hook recon", px["recon_rel_err"] < 1e-5, px["recon_rel_err"])
    check("hook identity", px["identity_max_rel_err"] < 1e-5, px["identity_max_rel_err"])
    check("hook supports", px["supports"] == [0, 1, 3] and px["size"] == [6, 5, 7])
    check("hook mags", px["mag"] == [5, 5, 20])
    brute = []
    for img, cols in R.support_columns(vis, bb._prefill_prune_survivor_image_ids):
        s2 = sc_.clone()
        s2[:, cols] = float("-inf")
        o_minus = torch.einsum("hl,hld->hd", torch.softmax(s2, -1), Vv)
        brute.append(float(attn.o_proj((o_ - o_minus).reshape(1, hq * hd)).norm()))
    check("hook s_hat brute force", all(abs(x - y) < 1e-4 * max(1.0, y) for x, y in zip(px["s_hat"], brute)),
          f"{px['s_hat']} vs {brute}")
    lme_ref = [float((torch.logsumexp(sc_[:, cols], -1) - math.log(len(cols))).mean())
               for _, cols in R.support_columns(vis, bb._prefill_prune_survivor_image_ids)]
    check("hook qk_lme", all(abs(x - y) < 1e-4 for x, y in zip(px["qk_lme"], lme_ref)), f"{px['qk_lme']} vs {lme_ref}")
    check("hook mass sum", all(abs(x - float(a_[:, cols].sum())) < 1e-5 for x, (_, cols) in
                               zip(px["mass_sum_heads"], R.support_columns(vis, bb._prefill_prune_survivor_image_ids))))

# 11. probe falls back to grid counts / unavailable
bb2 = types.SimpleNamespace(lm=bb.lm, model=bb.model, _prefill_prune_survivor_image_ids=None, _prune_image_magnifications=())
p2 = R.ReasonerProbe(bb2, cur_vis=torch.arange(8), cursor=0, cur_len=8, last_hidden=torch.zeros(1, 1, hidden), m=10,
                     thw=torch.tensor([[1, 4, 4], [1, 4, 4]]))
check("grid fallback ids", p2.id_source == "grid_thw" and p2.image_ids.tolist() == [0] * 4 + [1] * 4)
p3 = R.ReasonerProbe(bb2, cur_vis=torch.arange(7), cursor=0, cur_len=7, last_hidden=torch.zeros(1, 1, hidden), m=10,
                     thw=torch.tensor([[1, 4, 4], [1, 4, 4]]))
check("unavailable ids", p3.image_ids is None and p3.id_source == "unavailable")
p3.before_step(9)
check("no hook without ids", p3.handle is None)

# 12. enabled flag
os.environ.pop("VLMAS_RSS", None)
check("disabled by default", not R.enabled())
os.environ["VLMAS_RSS"] = "/tmp/x.jsonl"
check("enabled with env", R.enabled())
os.environ.pop("VLMAS_RSS", None)

print(f"\n{PASS}/{PASS + FAIL} passed")
sys.exit(1 if FAIL else 0)
