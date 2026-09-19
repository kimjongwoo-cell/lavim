"""Unit tests for the GLVR core math (no model, CPU only)."""
from __future__ import annotations

import torch

import glvr_core as G


def setup(seed: int = 0, H: int = 4, N: int = 40, D: int = 8, m: int = 10):
    torch.manual_seed(seed)
    logits = torch.randn(H, N) * 2.0
    values = torch.randn(H, N, D)
    sink = torch.tensor([0])
    vis = torch.arange(5, 25)
    latent = torch.arange(25, 25 + m)
    hc = torch.cat([torch.arange(1, 5), latent])          # inherited non-visual + latent
    prompt = torch.arange(25 + m, N)
    return logits, values, sink, vis, latent, hc, prompt


def test_masses_sum_to_one():
    logits, values, sink, vis, latent, hc, prompt = setup()
    a = G.attn_weights(logits)
    total = 0.0
    for idx in (sink, vis, hc, prompt):
        rho, _ = G.group_mass_message(a, values, idx)
        total = total + rho
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), total


def test_group_message_matches_direct_sum():
    logits, values, _, vis, _, _, _ = setup(seed=1)
    a = G.attn_weights(logits)
    rho, m = G.group_mass_message(a, values, vis)
    ref = torch.einsum("hk,hkd->hd", a[:, vis], values[:, vis, :].float())
    assert torch.allclose(m, ref, atol=1e-6)
    assert torch.allclose(rho, a[:, vis].sum(-1), atol=1e-6)


def test_step_distribution_normalized():
    logits, values, _, _, latent, _, _ = setup(seed=2)
    a = G.attn_weights(logits)
    rho, g = G.step_distribution(a, latent)
    assert torch.allclose(g.sum(-1), torch.ones_like(rho), atol=1e-6)
    assert torch.allclose(rho, a[:, latent].sum(-1), atol=1e-6)


def test_step_distribution_zero_mass_falls_back_uniform():
    H, N, D, m = 2, 30, 4, 10
    a = torch.zeros(H, N)
    a[:, 0] = 1.0
    latent = torch.arange(10, 10 + m)
    rho, g = G.step_distribution(a, latent)
    assert torch.allclose(rho, torch.zeros(H))
    assert torch.allclose(g, torch.full((H, m), 1.0 / m))


def test_conditional_read_is_message_over_mass():
    logits, values, _, vis, _, _, _ = setup(seed=3)
    rho, u = G.conditional_visual_read(logits, values, vis)
    a = G.attn_weights(logits)
    _, m = G.group_mass_message(a, values, vis)
    assert torch.allclose(u, m / rho.unsqueeze(-1), atol=1e-6)
    # u is a convex combination of visual values, so it stays inside their span/range
    lo = values[:, vis, :].min(dim=1).values
    hi = values[:, vis, :].max(dim=1).values
    assert bool(((u >= lo - 1e-5) & (u <= hi + 1e-5)).all())


def test_lambda_zero_is_identity_and_one_replaces_donor():
    logits, values, _, vis, latent, hc, _ = setup(seed=4)
    a = G.attn_weights(logits)
    o_nat = G.native_head_output(a, values)
    rho_B, m_B = G.group_mass_message(a, values, hc)
    _, g = G.step_distribution(a, latent)
    U_V = torch.randn(a.shape[0], latent.numel(), values.shape[-1])
    u_relay = G.relay_message(g, U_V)
    o0 = G.apply_relay(o_nat, rho_B, m_B, u_relay, 0.0)
    assert torch.allclose(o0, o_nat.float(), atol=1e-6)
    o1 = G.apply_relay(o_nat, rho_B, m_B, u_relay, 1.0)
    assert torch.allclose(o1, o_nat.float() - m_B + rho_B.unsqueeze(-1) * u_relay, atol=1e-6)


def test_apply_relay_matches_token_level_weights():
    """apply_relay must equal rebuilding the attention weights explicitly."""
    logits, values, _, vis, latent, hc, _ = setup(seed=5)
    a = G.attn_weights(logits)
    o_nat = G.native_head_output(a, values)
    rho_B, m_B = G.group_mass_message(a, values, hc)
    _, g = G.step_distribution(a, latent)
    # build U_V from an explicit per-latent conditional distribution over visual columns
    P = torch.softmax(torch.randn(a.shape[0], latent.numel(), vis.numel()), dim=-1)
    U_V = torch.einsum("hmk,hkd->hmd", P, values[:, vis, :].float())
    u_relay = G.relay_message(g, U_V)
    gamma = torch.einsum("hm,hmk->hk", g, P)            # mixed conditional over visual columns
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        o_fast = G.apply_relay(o_nat, rho_B, m_B, u_relay, lam)
        w = G.relay_weight_view(a, hc, vis, gamma, lam)
        o_ref = G.native_head_output(w, values)
        assert torch.allclose(o_fast, o_ref, atol=1e-5), (lam, (o_fast - o_ref).abs().max())


def test_total_mass_preserved_under_relay_view():
    logits, values, _, vis, latent, hc, _ = setup(seed=6)
    a = G.attn_weights(logits)
    _, g = G.step_distribution(a, latent)
    P = torch.softmax(torch.randn(a.shape[0], latent.numel(), vis.numel()), dim=-1)
    gamma = torch.einsum("hm,hmk->hk", g, P)
    for lam in (0.25, 1.0):
        w = G.relay_weight_view(a, hc, vis, gamma, lam)
        assert torch.allclose(w.sum(-1), torch.ones(a.shape[0]), atol=1e-5)


def test_sink_and_prompt_untouched():
    logits, values, sink, vis, latent, hc, prompt = setup(seed=7)
    a = G.attn_weights(logits)
    _, g = G.step_distribution(a, latent)
    P = torch.softmax(torch.randn(a.shape[0], latent.numel(), vis.numel()), dim=-1)
    gamma = torch.einsum("hm,hmk->hk", g, P)
    w = G.relay_weight_view(a, hc, vis, gamma, 1.0)
    assert torch.allclose(w[:, sink], a[:, sink], atol=1e-7)
    assert torch.allclose(w[:, prompt], a[:, prompt], atol=1e-7)


def test_donor_is_disjoint_from_preserved_groups():
    _, _, sink, vis, latent, hc, prompt = setup(seed=8)
    for other in (sink, vis, prompt):
        inter = set(hc.tolist()) & set(other.tolist())
        assert not inter, inter
    assert set(latent.tolist()) <= set(hc.tolist())


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    print("all good" if not fails else f"{fails} failed")
    raise SystemExit(1 if fails else 0)
