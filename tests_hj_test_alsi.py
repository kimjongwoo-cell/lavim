"""Unit tests for memory/alsi_diag.py (Answerer Latent Source Interference).
Run: python tests_hj_test_alsi.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import alsi_diag as A  # noqa: E402


def test_cosine_matches_the_definition():
    a = torch.tensor([3.0, 4.0], dtype=torch.float64)
    b = torch.tensor([4.0, 3.0], dtype=torch.float64)
    assert abs(A.cosine(a, a) - 1.0) < 1e-12
    assert abs(A.cosine(a, -a) + 1.0) < 1e-12
    assert abs(A.cosine(a, b) - 24.0 / 25.0) < 1e-12
    assert abs(A.cosine(a, torch.tensor([-4.0, 3.0], dtype=torch.float64))) < 1e-12


def test_cancellation_is_zero_unless_the_sources_oppose():
    d_v = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert A.cancellation(d_v, torch.tensor([2.0, 0.0], dtype=torch.float64)) == 0.0   # aligned
    assert A.cancellation(d_v, torch.tensor([0.0, 5.0], dtype=torch.float64)) == 0.0   # orthogonal
    k = A.cancellation(d_v, torch.tensor([-0.3, 9.0], dtype=torch.float64))
    assert abs(k - 0.3) < 1e-9          # only the opposing component along V counts


def test_cancellation_scales_with_the_visual_norm():
    """kappa is a fraction of the visual direction, so doubling d_V halves it."""
    n = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    k1 = A.cancellation(torch.tensor([1.0, 0.0], dtype=torch.float64), n)
    k2 = A.cancellation(torch.tensor([2.0, 0.0], dtype=torch.float64), n)
    assert abs(k1 - 1.0) < 1e-9 and abs(k2 - 0.5) < 1e-9


def test_source_columns_partition_is_exhaustive_and_disjoint():
    vis = torch.tensor([2, 3, 4], dtype=torch.long)
    lat = torch.tensor([7, 8], dtype=torch.long)
    g = A.source_columns(n_total=20, vis_cols=vis, lat_cols=lat, gen_start=12, row_abs=15)
    allc = torch.cat([g[s] for s in A.SOURCES]).sort().values
    assert torch.equal(allc, torch.arange(16))                     # everything up to the row
    for i, s in enumerate(A.SOURCES):
        for s2 in A.SOURCES[i + 1:]:
            assert len(set(g[s].tolist()) & set(g[s2].tolist())) == 0
    assert g["V"].tolist() == [2, 3, 4] and g["R"].tolist() == [7, 8]
    assert g["Y"].tolist() == [12, 13, 14, 15]
    assert 12 not in g["Q"].tolist() and 2 not in g["Q"].tolist()


def test_source_columns_never_looks_past_the_query_row():
    vis = torch.tensor([2, 30], dtype=torch.long)
    g = A.source_columns(n_total=40, vis_cols=vis, lat_cols=None, gen_start=20, row_abs=5)
    assert g["V"].tolist() == [2] and g["Y"].numel() == 0
    assert int(torch.cat([g[s] for s in A.SOURCES]).max()) == 5


def test_first_answer_token_has_an_empty_prefix_source():
    g = A.source_columns(n_total=30, vis_cols=torch.tensor([1]), lat_cols=torch.tensor([2]),
                         gen_start=10, row_abs=10)
    assert g["Y"].tolist() == [10]          # the query row itself is the only generated col
    g0 = A.source_columns(n_total=30, vis_cols=torch.tensor([1]), lat_cols=torch.tensor([2]),
                          gen_start=10, row_abs=9)
    assert g0["Y"].numel() == 0             # before generation starts, Y is empty


def test_finite_difference_propagation_recovers_a_linear_suffix():
    """With f(h) = W h + b the symmetric difference must return W m exactly, and the two
    epsilon scales must agree — that is the section-8B linearity check."""
    torch.manual_seed(0)
    d_in, d_out = 6, 5
    W = torch.randn(d_out, d_in, dtype=torch.float32)
    b = torch.randn(d_out, dtype=torch.float32)

    def fake_suffix(backbone, cache, h, pe, layer_index, row_pos):
        return (W @ h.float() + b).double()

    real, A.suffix_forward = A.suffix_forward, fake_suffix
    try:
        h0 = torch.randn(d_in)
        ms = [torch.randn(d_in) for _ in range(4)]
        out = A.propagate(None, None, h0, None, ms, 0, 0, [1.0, 0.5])
    finally:
        A.suffix_forward = real
    for e in (1.0, 0.5):
        for i, m in enumerate(ms):
            assert torch.allclose(out[e][i], (W @ m).double(), atol=1e-5), (e, i)
    assert torch.allclose(out[1.0], out[0.5], atol=1e-5)


def test_propagated_sources_add_up_like_the_messages_do():
    """m_attn = sum_s m_s and J is linear, so dbar must inherit the same identity."""
    torch.manual_seed(1)
    W = torch.randn(4, 7, dtype=torch.float32)

    def fake_suffix(backbone, cache, h, pe, layer_index, row_pos):
        return (W @ h.float()).double()

    real, A.suffix_forward = A.suffix_forward, fake_suffix
    try:
        ms = [torch.randn(7) for _ in range(4)]
        out = A.propagate(None, None, torch.zeros(7), None, ms, 0, 0, [1.0])[1.0]
        total = A.propagate(None, None, torch.zeros(7), None, [sum(ms)], 0, 0, [1.0])[1.0][0]
    finally:
        A.suffix_forward = real
    assert torch.allclose(out.sum(0), total, atol=1e-5)


def test_rmsnorm_jvp_matches_a_finite_difference():
    torch.manual_seed(2)
    h = torch.randn(32, dtype=torch.float64) * 2
    m = torch.randn(32, dtype=torch.float64) * 0.1
    w = torch.rand(32, dtype=torch.float64) + 0.5

    def f(x):
        return w * x / torch.sqrt((x * x).mean() + 1e-6)

    e = 1e-6
    fd = (f(h + e * m) - f(h - e * m)) / (2 * e)
    assert torch.allclose(A.rmsnorm_jvp(h, m, w), fd, atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
