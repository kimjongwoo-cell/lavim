"""Unit tests for memory/rsvmh.py and the RSVMH branch of backbone restage_visual_kv. Run: python tests_hj_test_rsvmh.py"""
import os
import sys
import types

import torch

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory import rsvmh as M  # noqa: E402
from memory.paging import page_anchor_positions  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")


# token permutation / page order
perm, counts = M.token_permutation([2, 0, 1], [3, 2, 4])
check("perm pages contiguous", perm == [5, 6, 7, 8, 0, 1, 2, 3, 4] and counts == [4, 3, 2], str((perm, counts)))
check("order ascending", M.page_order([0.5, 0.1, 0.9], 3, "sensitivity") == [1, 0, 2])
check("order ties stable", M.page_order([0.2, 0.2, 0.1], 3, "mass") == [2, 0, 1])
check("shuffle deterministic", M.page_order(None, 6, "page_shuffle", seed=7) == M.page_order(None, 6, "page_shuffle", seed=7))
check("shuffle is permutation", sorted(M.page_order(None, 6, "page_shuffle", seed=3)) == list(range(6)))
proxy = {"supports": [0, 1, 2], "size": [3, 2, 4], "s_hat": [0.5, 0.1, 0.9], "mass_sum_heads": [3.0, 1.0, 2.0],
         "mag": [5, 20, 20]}
check("scores sensitivity", M.page_scores(proxy, "sensitivity") == [0.5, 0.1, 0.9])
check("scores rev", M.page_scores(proxy, "sensitivity_rev") == [-0.5, -0.1, -0.9])
check("scores mass", M.page_scores(proxy, "mass") == [3.0, 1.0, 2.0])
check("scores size", M.page_scores(proxy, "size") == [3.0, 2.0, 4.0])
check("enabled orders", M.enabled_order("sensitivity") and not M.enabled_order("original") and not M.enabled_order(""))

# plan against a fake probe context
cols = torch.tensor([100, 101, 102, 150, 151, 200, 201, 202, 203])
ctx = types.SimpleNamespace(done=True, proxy=proxy, note=None, vis=cols.clone(),
                            image_ids=torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2]), pre_len=900)
bb = types.SimpleNamespace(_rss_ctx=ctx)
p, c, note = M.plan(bb, cols, "sensitivity")
check("plan sensitivity perm", p is not None and p.tolist() == [3, 4, 0, 1, 2, 5, 6, 7, 8] and c == [2, 3, 4], str((p, c, note)))
check("plan note far->near", "pages(far->near)=[1, 0, 2]" in note, note)
p, c, _ = M.plan(bb, cols, "sensitivity_rev")
check("plan rev perm", p.tolist() == [5, 6, 7, 8, 0, 1, 2, 3, 4] and c == [4, 3, 2])
p, c, _ = M.plan(bb, cols, "mass")
check("plan mass perm", p.tolist() == [3, 4, 5, 6, 7, 8, 0, 1, 2] and c == [2, 4, 3])
p, c, _ = M.plan(bb, cols[:-1], "sensitivity")
check("plan mismatch skip", p is None)
bb_none = types.SimpleNamespace(_rss_ctx=None)
p, c, note = M.plan(bb_none, cols, "sensitivity")
check("plan no ctx skip", p is None and "no reasoner proxy" in note)

# page anchors realize order positionally (per-page rigid blocks with MROPE=1)
old = torch.tensor([[7] * 9, [0, 0, 1, 3, 3, 0, 1, 1, 2], [0, 1, 0, 3, 4, 0, 0, 1, 1]])
os.environ["VLMAS_KV_RESTAGE_MROPE"] = "1"
permuted = old.index_select(1, torch.tensor([3, 4, 0, 1, 2, 5, 6, 7, 8]))
new, nxt = page_anchor_positions(permuted, [2, 3, 4], 1000)
check("anchors page1 min at cursor", int(new[:, 0:2].min()) == 1000)
check("anchors ordered pages", int(new[:, 0:2].max()) < int(new[:, 2:5].min()) <= int(new[:, 2:5].max()) < int(new[:, 5:].min()))
check("anchors keep in-page geometry", torch.equal(new[1:, 2:5] - new[1:, 2:5].min(dim=1, keepdim=True).values,
                                                   permuted[1:, 2:5] - permuted[1:, 2:5].min(dim=1, keepdim=True).values))
os.environ.pop("VLMAS_KV_RESTAGE_MROPE", None)
new1, nxt1 = page_anchor_positions(permuted, [2, 3, 4], 1000)
check("1-D anchors follow permuted order", new1[0].tolist() == list(range(1000, 1009)) and nxt1 == 1009)

# integration: backbone.restage_visual_kv non-park path with RSVMH order (identity rotary)
from backbone.qwen3vl import Qwen3VLBackbone  # noqa: E402


class FakeRotary:
    def __call__(self, x, position_ids):
        n = position_ids.shape[-1]
        d = x.shape[-1]
        return torch.ones(1, n, d), torch.zeros(1, n, d)


L = 12
keys = torch.arange(L, dtype=torch.float32).view(1, 1, L, 1).repeat(1, 2, 1, 4)
vals = -keys.clone()
layer = types.SimpleNamespace(keys=keys.clone(), values=vals.clone())
cache = types.SimpleNamespace(layers=[layer])
vis_cols = torch.tensor([2, 3, 4, 5, 6, 7])
ctx2 = types.SimpleNamespace(done=True, note=None, vis=vis_cols.clone(), image_ids=torch.tensor([0, 0, 1, 1, 1, 2]),
                             pre_len=10, proxy={"supports": [0, 1, 2], "size": [2, 3, 1], "s_hat": [0.9, 0.1, 0.5],
                                                "mass_sum_heads": [1.0, 2.0, 3.0], "mag": [5, 20, 20]})
fake = types.SimpleNamespace(device=torch.device("cpu"), lm=types.SimpleNamespace(rotary_emb=FakeRotary()),
                             _rss_ctx=ctx2, _reanchor_positions=Qwen3VLBackbone._reanchor_positions)
old_positions = torch.tensor([[0] * 6, [0, 1, 0, 1, 2, 0], [0, 0, 1, 1, 1, 0]])
os.environ["VLMAS_KV_RESTAGE_ORDER"] = "sensitivity"
os.environ["VLMAS_KV_RESTAGE_MROPE"] = "1"
try:
    cursor, moved = Qwen3VLBackbone.restage_visual_kv(fake, cache, vis_cols, old_positions, 500)
finally:
    os.environ.pop("VLMAS_KV_RESTAGE_ORDER", None)
    os.environ.pop("VLMAS_KV_RESTAGE_MROPE", None)
tail = cache.layers[0].keys[0, 0, :, 0].tolist()
check("restage moved all", moved == 6 and len(tail) == 12)
check("restage non-visual prefix kept", tail[:6] == [0, 1, 8, 9, 10, 11], str(tail))
check("restage sensitivity order at tail", tail[6:] == [4, 5, 6, 7, 2, 3], str(tail))
check("restage values paired", cache.layers[0].values[0, 0, 6:, 0].tolist() == [-4, -5, -6, -7, -2, -3])
check("restage cursor advanced", cursor > 500, str(cursor))

# default path untouched: no order env -> native column order at tail
layer_b = types.SimpleNamespace(keys=keys.clone(), values=vals.clone())
cache_b = types.SimpleNamespace(layers=[layer_b])
cur_b, mv_b = Qwen3VLBackbone.restage_visual_kv(fake, cache_b, vis_cols, old_positions, 500)
check("restage default order", cache_b.layers[0].keys[0, 0, 6:, 0].tolist() == [2, 3, 4, 5, 6, 7] and cur_b == 506)

print(f"\n{PASS}/{PASS + FAIL} passed")
sys.exit(1 if FAIL else 0)
