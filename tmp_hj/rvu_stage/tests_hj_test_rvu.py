"""Unit tests: memory/rvu_ablation.py + memory/group_attn.py mode 'dropfix'.
Run: python tests_hj_test_rvu.py"""
import os
import sys
import types
from copy import deepcopy

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import group_attn as ga  # noqa: E402
from memory import rvu_ablation as rvu  # noqa: E402


def test_overlap_frac():
    assert rvu.overlap_frac((0, 0, 10, 10), (20, 0, 10, 10)) == 0.0
    assert rvu.overlap_frac((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0          # touching edge
    assert abs(rvu.overlap_frac((0, 0, 10, 10), (5, 0, 10, 10)) - 0.5) < 1e-9
    assert abs(rvu.overlap_frac((0, 0, 40, 40), (10, 10, 10, 10)) - 1.0) < 1e-9  # contained
    assert rvu.overlap_frac(None, (0, 0, 1, 1)) == 0.0


def _layout():
    # crops 0-2 x5, 3-7 x20; x20 crops 3 and 4 near-duplicates, 7 orthogonal,
    # crop 5 overlaps crop 3 physically (must not count as its neighbour)
    mags = [5, 5, 5, 20, 20, 20, 20, 20]
    boxes = [(0, 0, 4000, 4000), (5000, 0, 4000, 4000), (10000, 0, 4000, 4000),
             (0, 0, 1000, 1000), (2000, 0, 1000, 1000), (500, 0, 1000, 1000),
             (4000, 0, 1000, 1000), (6000, 0, 1000, 1000)]
    torch.manual_seed(0)
    base = torch.randn(8, 16)
    base[4] = base[3] + 0.01 * torch.randn(16)
    base[5] = base[3] + 0.001 * torch.randn(16)      # identical morphology but same place
    base[0] = base[3]                                # cross-scale twin of crop 3
    return mags, boxes, F.normalize(base, dim=-1)


def test_repetition_scores_rules():
    mags, boxes, emb = _layout()
    rep = rvu.repetition_scores(emb, mags, boxes, thr=0.0)
    v = rep["valid"]
    assert not v[3][5] and not v[5][3], "overlapping same-scale pair must be excluded"
    assert not v[3][0] and not v[0][3], "cross-scale pair must be excluded"
    assert not v[3][3]
    assert v[3][4] and v[3][6]
    sim = rep["sim"]
    assert abs(rep["R"][3] - max(sim[3][j] for j in (4, 6, 7))) < 1e-6
    assert abs(rep["R"][0] - max(sim[0][1], sim[0][2])) < 1e-6
    # threshold 1.0 re-admits the overlapping pair
    rep2 = rvu.repetition_scores(emb, mags, boxes, thr=1.0)
    assert rep2["valid"][3][5]


def test_select_arms():
    mags, boxes, emb = _layout()
    rep = rvu.repetition_scores(emb, mags, boxes, thr=0.0)
    sel = rvu.select_arms(rep["R"], rep["meanc"], mags, seed=42)
    s20 = sel["20"]
    assert s20["primary"] and s20["n_same_mag"] == 5 and len(s20["comparable"]) == 5
    assert s20["rep"] in (3, 4)
    R = rep["R"]
    assert R[s20["uniq"]] == min(R[p] for p in s20["comparable"])
    assert s20["rand"] not in (s20["rep"], s20["uniq"]) and s20["rand"] in s20["comparable"]
    assert rvu.select_arms(rep["R"], rep["meanc"], mags, seed=42) == sel      # deterministic
    s5 = sel["5"]
    assert s5["primary"] and len(s5["comparable"]) == 3
    assert len({s5["rep"], s5["uniq"], s5["rand"]}) == 3


def test_select_ties_and_small_scale():
    # three crops, pair (0,1) most similar -> both have the same R; the tie is
    # broken by mean similarity, then by the lower index
    R = [0.9, 0.9, 0.5]
    meanc = [0.70, 0.75, 0.45]
    sel = rvu.select_arms(R, meanc, [20, 20, 20], seed=1)["20"]
    assert sel["rep"] == 1 and sel["uniq"] == 2 and sel["rand"] == 0
    sel = rvu.select_arms([0.9, 0.9, 0.5], [0.7, 0.7, 0.45], [20, 20, 20], seed=1)["20"]
    assert sel["rep"] == 0
    # fewer than three comparable crops -> not primary, no random arm
    sel = rvu.select_arms([0.8, 0.8, None], [0.8, 0.8, None], [5, 5, 5], seed=3)["5"]
    assert not sel["primary"] and sel["comparable"] == [0, 1] and sel["rand"] is None


def test_dropfix_alpha_mass_and_shares():
    torch.manual_seed(1)
    L = 12
    scores = torch.randn(1, 2, 3, L)
    scores[..., 0, 11] = float("-inf")               # a causally hidden column
    vmask = torch.zeros(L, dtype=torch.bool); vmask[2:9] = True
    dmask = torch.zeros(L, dtype=torch.bool); dmask[4:6] = True
    ref = torch.rand(1, 2, 3) * 0.5 + 0.1
    new, mv_post = ga.dropfix_alpha(scores, vmask, dmask, ref)
    assert torch.allclose(new.sum(-1), torch.ones(1, 2, 3), atol=1e-6)
    assert torch.allclose(mv_post, ref, atol=1e-6)
    assert float(new[..., dmask].abs().max()) == 0.0
    a_d = torch.softmax(scores.masked_fill(dmask, float("-inf")), dim=-1)
    rem = vmask & ~dmask
    r1 = new[..., rem] / new[..., rem].sum(-1, keepdim=True)
    r0 = a_d[..., rem] / a_d[..., rem].sum(-1, keepdim=True)
    assert torch.allclose(r1, r0, atol=1e-6), "within-visual shares must be kept"
    nv = ~vmask
    c1 = new[..., nv] / new[..., nv].sum(-1, keepdim=True)
    c0 = a_d[..., nv] / a_d[..., nv].sum(-1, keepdim=True)
    assert torch.allclose(c1, c0, atol=1e-6), "non-visual shares must be kept"
    assert float(new[..., 0, 11].abs().max()) == 0.0


def test_dropfix_alpha_identity():
    torch.manual_seed(2)
    L = 10
    scores = torch.randn(1, 4, 5, L)
    vmask = torch.zeros(L, dtype=torch.bool); vmask[1:7] = True
    dmask = torch.zeros(L, dtype=torch.bool)
    alpha = torch.softmax(scores, dim=-1)
    new, mv_post = ga.dropfix_alpha(scores, vmask, dmask, alpha[..., vmask].sum(-1))
    assert torch.allclose(new, alpha, atol=1e-6)


def _tiny_qwen3vl():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    cfg = Qwen3VLTextConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, max_position_embeddings=256,
                            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3],
                                          "mrope_interleaved": True})
    cfg._attn_implementation = "eager"
    torch.manual_seed(3)
    return Qwen3VLTextModel(cfg).float().eval()


def test_hook_on_tiny_qwen3vl():
    try:
        from transformers import DynamicCache
        model = _tiny_qwen3vl()
    except Exception as exc:  # pragma: no cover - version dependent
        print(f"  SKIP tiny model ({type(exc).__name__}: {exc})")
        return
    bb = types.SimpleNamespace(lm=model)
    torch.manual_seed(4)
    prefix = torch.randn(1, 20, 64)
    new = torch.randn(1, 5, 64)
    cache0 = DynamicCache()
    with torch.no_grad():
        model(inputs_embeds=prefix, past_key_values=cache0, use_cache=True)
    vis = torch.arange(3, 15)
    groups = [torch.arange(3, 9), torch.arange(9, 15)]

    def fwd(ctx):
        c = deepcopy(cache0)
        with torch.no_grad(), ctx as rec:
            out = model(inputs_embeds=new, past_key_values=c, use_cache=True).last_hidden_state
        return out, rec

    import contextlib
    native, _ = fwd(contextlib.nullcontext())
    cap, r0 = fwd(ga.install_group_attention(bb, groups, vis, "capture"))
    if r0["calls"] == 0:
        print("  SKIP tiny model (hook kwargs not seen in this transformers version)")
        return
    assert torch.equal(cap, native), "capture must not change the output"
    ref = r0["mv"]
    ident, ri = fwd(ga.install_group_attention(bb, groups, vis, "dropfix", mv_ref=ref,
                                               drop_cols=torch.zeros(0, dtype=torch.long)))
    assert ri["fallback"] == 0 and ri["calls"] == 2
    assert float((ident - native).abs().max()) < 1e-4, float((ident - native).abs().max())
    drop, rd = fwd(ga.install_group_attention(bb, groups, vis, "dropfix", mv_ref=ref, drop_cols=groups[1]))
    assert rd["fallback"] == 0 and rd["dev_mv"] < 1e-5 and rd["dev_row"] < 1e-5
    assert float((drop - native).abs().max()) > 1e-4, "removal must change the output"
    post = ga.summarize_post_masses(rd, 4)
    pre = ga.summarize_masses(r0, 4)
    assert abs(post["m_V"] - pre["m_V"]) < 1e-5 and post["m_g"][1] == 0.0
    bad = {li: v[:, :3] for li, v in ref.items()}
    out_bad, rb = fwd(ga.install_group_attention(bb, groups, vis, "dropfix", mv_ref=bad, drop_cols=groups[1]))
    assert rb["fallback"] == 2 and torch.equal(out_bad, native)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    n = 0
    for t in tests:
        t()
        n += 1
        print(f"ok {t.__name__}")
    print(f"{n}/{len(tests)} passed")
