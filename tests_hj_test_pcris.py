"""Unit tests: Provenance-Conditioned Residual Information Selection (VLMAS_WSI_CONSOL_MODE=pcris).

Spec = Notion PCRIS page, section 8 (Korean paper-facing draft) with sections 3-5:
  h_hat_i = h_i / ||h_i||;  U_o = orthonormal basis of the parent (or ancestor) states (QR/SVD)
  r_i = (I - U_o U_o^T) h_hat_i  for a fine observation,  r_i = h_hat_i for a root observation
  F(S) = log det(I + sum_{i in S} r_i r_i^T),  |S| <= B,  greedy j* = argmax F(S+j) - F(S).
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from memory.pcris_select import logdet_greedy, logdet_value, pcris_keep_mask, provenance_residuals

COUNTS = (4, 4, 4, 4)
PARENTS = (-1, -1, 0, 1)        # obs 2 <- obs 0, obs 3 <- obs 1
MAGS = (5, 5, 20, 20)


def _z(seed=0, d=12):
    return torch.randn(sum(COUNTS), d, generator=torch.Generator().manual_seed(seed))


def _hat(z):
    return torch.nn.functional.normalize(z.double(), dim=-1)


def brute_F(R, keep):
    R = R.double()
    d = R.shape[1]
    M = torch.eye(d, dtype=torch.float64)
    for i in range(R.shape[0]):
        if bool(keep[i]):
            M = M + torch.outer(R[i], R[i])
    return float(torch.logdet(M))


def test_residual_orthogonal_to_parent_span_and_roots_unchanged():
    z = _z(1)
    R, _ = provenance_residuals(z, COUNTS, PARENTS)
    H = _hat(z)
    assert float((H[0:4] @ R[8:12].double().T).abs().max()) < 1e-5      # obs 2 residual _|_ parent obs 0
    assert float((H[4:8] @ R[12:16].double().T).abs().max()) < 1e-5     # obs 3 residual _|_ parent obs 1
    assert float((R[:8].double() - H[:8]).abs().max()) < 1e-6           # roots: r = h_hat


def test_fine_token_inside_parent_span_has_zero_residual():
    z = _z(2)
    z[8] = 2.0 * z[0] - 3.0 * z[1] + 0.5 * z[3]
    R, _ = provenance_residuals(z, COUNTS, PARENTS)
    assert float(R[8].norm()) < 1e-5


def test_fine_token_outside_parent_span_keeps_unit_residual():
    z = torch.zeros(16, 12)
    z[0:4, 0:4] = torch.eye(4) + 0.1                                 # parent obs 0 spans dims 0-3
    z[4:8, 4:8] = torch.eye(4)
    z[8:12, 8] = 1.0
    z[8:12, 9:12] = torch.rand(4, 3, generator=torch.Generator().manual_seed(0))
    z[12:16, 0] = 1.0
    R, _ = provenance_residuals(z, COUNTS, PARENTS)
    assert abs(float(R[8].norm()) - 1.0) < 1e-5                         # dims 8-11 are outside span(obs 0)


def test_ancestor_context_conditions_on_whole_chain():
    counts, parents = (3, 3, 3), (-1, 0, 1)                             # x5 -> x10 -> x20
    z = torch.randn(9, 10, generator=torch.Generator().manual_seed(3))
    H = _hat(z)
    Ra, _ = provenance_residuals(z, counts, parents, context="ancestors")
    Rp, _ = provenance_residuals(z, counts, parents, context="parent")
    assert float((H[0:6] @ Ra[6:9].double().T).abs().max()) < 1e-5     # _|_ parent AND grandparent
    assert float((H[3:6] @ Rp[6:9].double().T).abs().max()) < 1e-5     # _|_ parent
    assert float((H[0:3] @ Rp[6:9].double().T).abs().max()) > 1e-3     # not conditioned on grandparent


def test_cond_none_is_plain_normalized_states():
    z = _z(4)
    R, _ = provenance_residuals(z, COUNTS, PARENTS, cond="none")
    assert float((R.double() - _hat(z)).abs().max()) < 1e-6


def test_wrong_parent_changes_residuals():
    z = _z(5)
    Rt, it = provenance_residuals(z, COUNTS, PARENTS)
    Rw, iw = provenance_residuals(z, COUNTS, PARENTS, cond="wrong")
    assert tuple(iw["parents"]) == (-1, -1, 1, 0)
    assert float((Rt[8:] - Rw[8:]).abs().max()) > 1e-3


def test_logdet_value_matches_direct():
    R = torch.randn(10, 6, generator=torch.Generator().manual_seed(6))
    random.seed(6)
    for _ in range(10):
        keep = torch.tensor([random.random() < 0.5 for _ in range(10)])
        assert abs(logdet_value(R, keep) - brute_F(R, keep)) < 1e-6


def test_greedy_takes_argmax_marginal_gain_each_step():
    R = torch.randn(12, 5, generator=torch.Generator().manual_seed(7)) * torch.linspace(0.2, 1.5, 12)[:, None]
    keep, order, gains = logdet_greedy(R, 7)
    assert len(order) == 7 and int(keep.sum()) == 7
    cur = torch.zeros(12, dtype=torch.bool)
    for step, j in enumerate(order):
        base = brute_F(R, cur)
        cand = []
        for k in range(12):
            if not cur[k]:
                k2 = cur.clone(); k2[k] = True
                cand.append((brute_F(R, k2) - base, k))
        best = max(g for g, _ in cand)
        mine = [g for g, k in cand if k == j][0]
        assert abs(mine - best) < 1e-6
        assert abs(float(gains[step]) - mine) < 1e-5
        cur[j] = True


def test_monotone_submodular():
    R = torch.randn(14, 6, generator=torch.Generator().manual_seed(8))
    random.seed(8)
    for _ in range(40):
        A = torch.tensor([random.random() < 0.3 for _ in range(14)])
        Bm = A | torch.tensor([random.random() < 0.3 for _ in range(14)])
        free = [k for k in range(14) if not Bm[k]]
        if not free:
            continue
        x = random.choice(free)
        ax = A.clone(); ax[x] = True
        bx = Bm.clone(); bx[x] = True
        da = logdet_value(R, ax) - logdet_value(R, A)
        db = logdet_value(R, bx) - logdet_value(R, Bm)
        assert logdet_value(R, Bm) >= logdet_value(R, A) - 1e-9 and da >= db - 1e-9 and db >= -1e-9


def test_duplicate_direction_gains_less_than_new_direction():
    R = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.2]])
    keep, order, gains = logdet_greedy(R, 2)
    assert set(order) == {0, 2} or set(order) == {1, 2}     # never the duplicate pair
    assert order[0] in (0, 1, 2)


def test_exact_budget_and_budget_ge_n():
    z = _z(9)
    for B in (1, 5, 11):
        assert int(pcris_keep_mask(z, COUNTS, PARENTS, B).sum()) == B
    assert bool(pcris_keep_mask(z, COUNTS, PARENTS, 99).all())


def test_rank_tolerance_drops_duplicate_parent_rows():
    z = _z(10)
    z[1] = z[0] * 3.0
    z[2] = z[0] - z[3]
    _, info = provenance_residuals(z, COUNTS, PARENTS)
    assert info["ranks"][2] == 2 and info["ranks"][3] == 4 and info["ranks"][0] == 0


def test_keep_mask_info_and_determinism():
    z = _z(11)
    k1, info = pcris_keep_mask(z, COUNTS, PARENTS, 6, magnifications=MAGS, return_info=True)
    k2 = pcris_keep_mask(z, COUNTS, PARENTS, 6, magnifications=MAGS)
    assert torch.equal(k1, k2)
    for key in ("cond", "context", "parents", "ranks", "F", "resid_norm_fine", "kept_by_mag", "per_crop", "order"):
        assert key in info, key
    assert sum(info["kept_by_mag"].values()) == 6


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = 0
    for name, fn in tests:
        try:
            fn()
            ok += 1
            print(f"PASS {name}")
        except Exception as e:  # report every failure, keep going
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
