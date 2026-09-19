"""Unit tests: Provenance-Conditioned Submodular Information Selection (VLMAS_WSI_CONSOL_MODE=pcsi).

Spec = Notion PCSI page, section 10 (Korean paper-facing draft):
  F(S)  = sum_o I_f(S & V_o; Q | P_o),   P_o = V_pi(o)  (empty if o has no parent)
  I_f(A; Q | P) = sum_{i in V_o} [ min( max_{j in A} s_ij , max_{q in Q} s_iq ) - max_{p in P} s_ip ]_+
  s_ij = [cos(h_i, h_j)]_+ ,  greedy j* = argmax F(S+j) - F(S),  |S| <= B (c_i = 1).
"""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from memory.pcsi_select import (
    cos_plus, flcmi_value, locate_question_rows, pcsi_keep_mask,
    pcsi_marginal_gains, wrong_parents)


def _offsets(counts):
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + c)
    return offs


def brute_F(z, q, counts, parents, keep, cond="true"):
    """Direct nested-loop evaluation of the section-10 objective."""
    offs = _offsets(counts)
    zn = torch.nn.functional.normalize(z.double(), dim=-1)
    qn = torch.nn.functional.normalize(q.double(), dim=-1) if q is not None else None
    s = lambda a, b: max(0.0, float(zn[a] @ zn[b]))
    par = list(parents)
    if cond == "none":
        par = [-1] * len(counts)
    elif cond == "wrong":
        par = list(wrong_parents(parents))
    total = 0.0
    for o in range(len(counts)):
        Vo = range(offs[o], offs[o + 1])
        A = [j for j in Vo if bool(keep[j])]
        P = range(offs[par[o]], offs[par[o] + 1]) if par[o] >= 0 else []
        for i in Vo:
            cov = max([s(i, j) for j in A], default=0.0)
            rq = (max(max(0.0, float(zn[i] @ qn[k])) for k in range(qn.shape[0]))
                  if qn is not None else 1.0)
            rp = max([s(i, p) for p in P], default=0.0)
            total += max(min(cov, rq) - rp, 0.0)
    return total


# 3 coarse observations (no parent) + 3 fine observations with actual parents 0, 1, 0
COUNTS = (4, 4, 4, 4, 4, 4)
PARENTS = (-1, -1, -1, 0, 1, 0)
MAGS = (5, 5, 5, 20, 20, 20)


def _instance(seed=0, d=8, nq=3):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(sum(COUNTS), d, generator=g)
    q = z[[13, 17, 22]] + 0.3 * torch.randn(nq, d, generator=g)   # question close to some fine tokens
    return z, q


def test_cos_plus_clamps_negative():
    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    b = torch.tensor([[-3.0, 0.0], [1.0, 1.0]])
    s = cos_plus(a, b)
    assert float(s[0, 0]) == 0.0                       # opposite -> clamped
    assert abs(float(s[0, 1]) - 2 ** -0.5) < 1e-6
    assert abs(float(cos_plus(a, a)[1, 1]) - 1.0) < 1e-6


def test_flcmi_matches_bruteforce_formula():
    z, q = _instance(1)
    random.seed(1)
    for _ in range(10):
        keep = torch.tensor([random.random() < 0.4 for _ in range(sum(COUNTS))])
        for cond in ("true", "none", "wrong"):
            v = flcmi_value(z, q, COUNTS, PARENTS, keep, cond=cond)
            assert abs(v - brute_F(z, q, COUNTS, PARENTS, keep, cond)) < 1e-5, cond


def test_parentless_observation_has_no_parent_term():
    z, q = _instance(2)
    keep = torch.zeros(sum(COUNTS), dtype=torch.bool)
    keep[:4] = True                                    # whole coarse observation 0
    zn = torch.nn.functional.normalize(z, dim=-1)
    rq = (zn[:4] @ torch.nn.functional.normalize(q, dim=-1).T).clamp(min=0).max(1).values
    # observation 0 fully selected -> coverage 1 -> term = min(1, R_Q) = R_Q, no parent subtraction
    only0 = flcmi_value(z[:4], q, COUNTS[:1], (-1,), keep[:4])
    assert abs(only0 - float(rq.sum())) < 1e-5


def test_coverage_is_within_observation():
    z, q = _instance(3)
    z = z.clone()
    z[12] = z[0]                                        # fine token 12 identical to coarse token 0
    keep = torch.zeros(sum(COUNTS), dtype=torch.bool)
    base = flcmi_value(z, q, COUNTS, PARENTS, keep, cond="none")
    keep[0] = True
    # selecting coarse token 0 changes only observation 0's term; the identical fine token in
    # observation 3 stays uncovered (A_o is per observation)
    fine_before = brute_F(z[12:16], q, (4,), (-1,), torch.zeros(4, dtype=torch.bool))
    gain = flcmi_value(z, q, COUNTS, PARENTS, keep, cond="none") - base
    own = flcmi_value(z[:4], q, (4,), (-1,), keep[:4])
    assert abs(gain - own) < 1e-5 and fine_before == 0.0


def test_marginal_gains_match_bruteforce():
    z, q = _instance(4)
    keep = torch.zeros(sum(COUNTS), dtype=torch.bool)
    keep[[1, 14, 20]] = True
    g = pcsi_marginal_gains(z, q, COUNTS, PARENTS, keep)
    base = brute_F(z, q, COUNTS, PARENTS, keep)
    for j in range(sum(COUNTS)):
        if keep[j]:
            assert g[j] == float("-inf")
            continue
        k2 = keep.clone(); k2[j] = True
        assert abs(float(g[j]) - (brute_F(z, q, COUNTS, PARENTS, k2) - base)) < 1e-5


def test_greedy_takes_argmax_gain_each_step():
    z, q = _instance(5)
    B = 8
    keep, info = pcsi_keep_mask(z, q, COUNTS, PARENTS, B, magnifications=MAGS, return_info=True)
    order = info["order"]
    assert len(order) == B and int(keep.sum()) == B
    cur = torch.zeros(sum(COUNTS), dtype=torch.bool)
    checked = 0
    for j in order[: info["saturated_at"] or B]:
        base = brute_F(z, q, COUNTS, PARENTS, cur)
        gains = []
        for k in range(sum(COUNTS)):
            if cur[k]:
                continue
            k2 = cur.clone(); k2[k] = True
            gains.append((brute_F(z, q, COUNTS, PARENTS, k2) - base, k))
        best = max(g for g, _ in gains)
        mine = [g for g, k in gains if k == j][0]
        assert abs(mine - best) < 1e-6
        cur[j] = True
        checked += 1
    assert checked >= 3


def test_exact_budget_and_budget_ge_n():
    z, q = _instance(6)
    for B in (1, 5, 13):
        assert int(pcsi_keep_mask(z, q, COUNTS, PARENTS, B).sum()) == B
    assert bool(pcsi_keep_mask(z, q, COUNTS, PARENTS, 999).all())


def test_parent_explained_fine_token_gets_no_gain():
    z, q = _instance(7)
    z = z.clone()
    z[12:16] = z[0:4]                                   # fine obs 3 duplicates its parent obs 0
    q = z[[12]].clone()                                 # question points at that duplicated content
    keep = torch.zeros(sum(COUNTS), dtype=torch.bool)
    g = pcsi_marginal_gains(z, q, COUNTS, PARENTS, keep)
    assert float(g[12:16].abs().max()) < 1e-6           # parent already explains it
    g_none = pcsi_marginal_gains(z, q, COUNTS, PARENTS, keep, cond="none")
    assert float(g_none[12]) > 0.5                      # without conditioning it is valuable


def test_irrelevant_token_gets_zero_gain():
    z = torch.zeros(8, 4)
    z[:4, 0] = 1.0                                      # obs 0: direction e0
    z[4:, 1] = 1.0                                      # obs 1: direction e1
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])            # question = e0
    g = pcsi_marginal_gains(z, q, (4, 4), (-1, -1), torch.zeros(8, dtype=torch.bool))
    assert float(g[4:].abs().max()) == 0.0              # R_Q = 0 ceiling, however much it covers
    assert float(g[0]) > 3.9                            # covers its 4 identical tokens up to R_Q = 1


def test_monotone_submodular():
    z, q = _instance(8)
    random.seed(8)
    n = sum(COUNTS)
    for _ in range(40):
        A = torch.tensor([random.random() < 0.3 for _ in range(n)])
        Bm = A | torch.tensor([random.random() < 0.3 for _ in range(n)])
        x = random.choice([k for k in range(n) if not Bm[k]] or [None])
        if x is None:
            continue
        fa = flcmi_value(z, q, COUNTS, PARENTS, A)
        fb = flcmi_value(z, q, COUNTS, PARENTS, Bm)
        ax = A.clone(); ax[x] = True
        bx = Bm.clone(); bx[x] = True
        da = flcmi_value(z, q, COUNTS, PARENTS, ax) - fa
        db = flcmi_value(z, q, COUNTS, PARENTS, bx) - fb
        assert fb >= fa - 1e-6 and da >= db - 1e-6 and db >= -1e-6


def test_wrong_parents_derange_within_level():
    wp = wrong_parents((-1, -1, -1, 0, 1, 0, 1, 0), (5, 5, 5, 20, 20, 20, 20, 20))
    assert wp[:3] == (-1, -1, -1)
    assert all(w != t and w in (0, 1, 2) for w, t in zip(wp[3:], (0, 1, 0, 1, 0)))
    z, q = _instance(9)
    keep = torch.ones(sum(COUNTS), dtype=torch.bool)
    assert abs(flcmi_value(z, q, COUNTS, PARENTS, keep)
               - flcmi_value(z, q, COUNTS, PARENTS, keep, cond="wrong")) > 1e-4


def test_multilevel_uses_direct_parent_only():
    counts = (3, 3, 3)
    parents = (-1, 0, 1)                                # x5 -> x10 -> x20 chain
    g = torch.Generator().manual_seed(10)
    z = torch.randn(9, 6, generator=g)
    q = torch.randn(2, 6, generator=g)
    keep = torch.tensor([1, 0, 1, 0, 1, 1, 1, 0, 0], dtype=torch.bool)
    assert abs(flcmi_value(z, q, counts, parents, keep)
               - brute_F(z, q, counts, parents, keep)) < 1e-5


def test_locate_question_rows():
    pieces = ["<|vision_end|>", "Question", " stem", ":", " What", " organ", " type", " is",
              " shown", "?\n", "The", " evidence", " plan"]
    assert locate_question_rows(pieces) == [4, 5, 6, 7, 8, 9]
    assert locate_question_rows(["The", " plan", "\n"]) == []


def test_missing_question_uses_unit_ceiling():
    z, _ = _instance(11)
    keep, info = pcsi_keep_mask(z, None, COUNTS, PARENTS, 6, return_info=True)
    assert info["q"] == "none" and int(keep.sum()) == 6
    k1 = torch.zeros(sum(COUNTS), dtype=torch.bool); k1[0] = True
    assert abs(flcmi_value(z, None, COUNTS, PARENTS, k1)
               - brute_F(z, None, COUNTS, PARENTS, k1)) < 1e-5


def test_saturation_then_lexicographic_fill():
    z, _ = _instance(12)
    z = z.clone()
    z[12:16] = z[0:4]; z[16:20] = z[4:8]; z[20:24] = z[0:4]   # every fine obs = its parent
    q = z[[0, 4]].clone()
    B = 14
    keep, info = pcsi_keep_mask(z, q, COUNTS, PARENTS, B, magnifications=MAGS, return_info=True)
    sat = info["saturated_at"]
    assert sat is not None and sat < B and int(keep.sum()) == B
    assert sum(info["fill"].values()) == B - sat
    # PCSI stage never picks a parent-explained fine token
    assert all(j < 12 for j in info["order"][:sat])
    keep_n, info_n = pcsi_keep_mask(z, q, COUNTS, PARENTS, B, fill="none", return_info=True)
    assert int(keep_n.sum()) == info_n["saturated_at"] == sat


def test_deterministic():
    z, q = _instance(13)
    a = pcsi_keep_mask(z, q, COUNTS, PARENTS, 9)
    b = pcsi_keep_mask(z, q, COUNTS, PARENTS, 9)
    assert torch.equal(a, b)


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
