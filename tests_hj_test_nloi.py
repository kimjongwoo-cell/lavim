"""Unit tests for memory/nloi_diag.py (Non-local Observation Integration).
Run: python tests_hj_test_nloi.py"""
import contextlib
import math
import os
import sys
import types
from copy import deepcopy

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import nloi_diag as nd  # noqa: E402


def test_pair_metrics_additive_and_superadditive():
    torch.manual_seed(0)
    h_f = torch.randn(16, dtype=torch.float64)
    d_i = torch.randn(16, dtype=torch.float64)
    d_j = torch.randn(16, dtype=torch.float64)
    m = nd.pair_metrics(h_f, h_f - d_i, h_f - d_j, h_f - d_i - d_j)
    assert m["psi"] < 1e-12 and m["psi_rel"] < 1e-12
    assert abs(m["d_ij"] - float((d_i + d_j).norm())) < 1e-12
    assert abs(m["C"] - (m["d_ij"] - max(m["d_i"], m["d_j"]))) < 1e-12
    extra = torch.randn(16, dtype=torch.float64)
    m2 = nd.pair_metrics(h_f, h_f - d_i, h_f - d_j, h_f - d_i - d_j - extra)
    assert abs(m2["psi"] - float(extra.norm())) < 1e-12
    # redundant pair: removing both moves the state no further than removing one
    m3 = nd.pair_metrics(h_f, h_f - d_i, h_f - d_i, h_f - d_i)
    assert abs(m3["C"]) < 1e-12 and abs(m3["cos_ij"] - 1.0) < 1e-12
    assert abs(m3["psi"] - float(d_i.norm())) < 1e-12          # Psi = -Delta_i
    z = torch.zeros(16, dtype=torch.float64)
    m4 = nd.pair_metrics(h_f, h_f, h_f, h_f)
    assert m4["psi_rel"] is None and m4["C_rel"] is None and m4["cos_ij"] is None
    del z


def test_pair_geometry_and_pairs():
    boxes = [(0, 0, 100, 100), (1000, 0, 100, 100), (25, 25, 25, 25)]
    mags = [5, 5, 20]
    parents = [-1, -1, 0]
    g01 = nd.pair_geometry(0, 1, boxes, mags, parents)
    assert abs(g01["dist"] - 1000.0) < 1e-9 and g01["same_mag"] and not g01["parent_child"]
    g02 = nd.pair_geometry(0, 2, boxes, mags, parents)
    assert g02["parent_child"] and not g02["same_mag"] and g02["overlap"] == 1.0
    assert abs(g02["dist"] - math.dist((50, 50), (37.5, 37.5))) < 1e-9
    assert nd.all_pairs(8) == [(i, j) for i in range(8) for j in range(i + 1, 8)]
    assert len(nd.all_pairs(8)) == 28


def test_js_rows():
    p = torch.log(torch.tensor([[0.5, 0.5], [1.0, 1e-300]], dtype=torch.float64))
    q = torch.log(torch.tensor([[0.5, 0.5], [1e-300, 1.0]], dtype=torch.float64))
    js = nd.js_rows(p, q)
    assert float(js[0]) < 1e-12 and abs(float(js[1]) - math.log(2)) < 1e-9


def _tiny_qwen3vl(n_layers=2):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    cfg = Qwen3VLTextConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, max_position_embeddings=256,
                            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3],
                                          "mrope_interleaved": True})
    cfg._attn_implementation = "eager"
    torch.manual_seed(3)
    return Qwen3VLTextModel(cfg).float().eval()


def test_value_cut_on_tiny_qwen3vl():
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
    g0, g1 = torch.arange(3, 9), torch.arange(9, 15)

    attn0 = {}

    def fwd(ctx):
        """last hidden state + layer-0 attention output (seen AFTER the cut hook)."""
        c = deepcopy(cache0)
        with torch.no_grad(), ctx as rec:
            # registered after the cut hook, so it sees the modified attention output
            h = model.layers[0].self_attn.register_forward_hook(
                lambda m, a, o: attn0.__setitem__("o", (o[0] if isinstance(o, tuple) else o).clone()))
            try:
                out = model(inputs_embeds=new, past_key_values=c, use_cache=True).last_hidden_state
            finally:
                h.remove()
        return out, attn0["o"], rec

    native, a_nat, _ = fwd(contextlib.nullcontext())
    ident, a_id, ri = fwd(nd.install_value_cut(bb, torch.zeros(0, dtype=torch.long),
                                               mass_groups=[g0, g1], mass_rows=torch.tensor([0, 4])))
    if ri["calls"] == 0:
        print("  SKIP tiny model (hook kwargs not seen in this transformers version)")
        return
    assert ri["calls"] == 2
    assert float((ident - native).abs().max()) < 1e-4, float((ident - native).abs().max())
    assert ri["mass"].shape == (2, 2) and ri["n_layers_mass"] == 2
    assert float(ri["mass"].min()) > 0 and float((ri["mass"] / 2).max()) < 1.0

    _, a_0, _ = fwd(nd.install_value_cut(bb, g0))
    _, a_1, _ = fwd(nd.install_value_cut(bb, g1))
    out01, a_01, _ = fwd(nd.install_value_cut(bb, torch.cat([g0, g1])))
    # the removal operator is exactly additive inside one attention layer
    lin = (a_id - a_0) + (a_id - a_1) - (a_id - a_01)
    assert float(lin.abs().max()) < 1e-5, float(lin.abs().max())
    assert float((a_id - a_0).abs().max()) > 1e-4, "removal must change the attention output"
    # ...so any pair interaction at the output comes from later layers / MLP / norm
    out0, _, _ = fwd(nd.install_value_cut(bb, g0))
    out1, _, _ = fwd(nd.install_value_cut(bb, g1))
    m = nd.pair_metrics(ident[0, -1].double(), out0[0, -1].double(), out1[0, -1].double(),
                        out01[0, -1].double())
    assert m["psi"] > 1e-6, "a 2-layer MLP model should show some non-additivity"

    # the value cut of ALL listed columns keeps the native softmax (mass is spent, not read):
    # same result as memory/visual_cut 'zero' restricted to those columns
    from memory import visual_cut as vc
    old = {k: os.environ.get(k) for k in ("VLMAS_VCUT", "VLMAS_VCUT_STAGE", "VLMAS_VCUT_LAYERS")}
    os.environ["VLMAS_VCUT"] = "zero"; os.environ["VLMAS_VCUT_STAGE"] = "A"
    os.environ.pop("VLMAS_VCUT_LAYERS", None)
    try:
        ref, _, _ = fwd(vc.install_visual_cut(bb, torch.cat([g0, g1]), stage="A"))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert float((ref - out01).abs().max()) < 1e-5, float((ref - out01).abs().max())

    # Stage B: cutting from layer 1 only leaves layer 0's attention output untouched
    outL1, a_L1, _ = fwd(nd.install_value_cut(bb, torch.cat([g0, g1]), from_layer=1))
    assert float((a_L1 - a_id).abs().max()) < 1e-6
    assert float((outL1 - ident).abs().max()) > 1e-5 and float((outL1 - out01).abs().max()) > 1e-5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
