"""Tests for memory/ssda_select.py (C1 #14.8). Run: python3 tests_hj_test_ssda.py"""
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from memory import ssda_select as ss  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    if not cond:
        raise AssertionError(f"{name} {detail}")
    PASS += 1
    print("ok", name, detail)


rng = np.random.default_rng(0)

# water filling == brute-force best allocation of top eigenvalues
spectra = [np.sort(rng.random(5))[::-1] * s for s in (1.0, 0.3, 2.0)]
for B in (1, 4, 7, 11):
    k, tau = ss.water_fill(spectra, B, [5, 5, 5])
    best = max((c for c in itertools.product(range(6), repeat=3) if sum(c) == B),
               key=lambda c: sum(spectra[g][:c[g]].sum() for g in range(3)))
    check(f"water_fill optimal B={B}", abs(sum(spectra[g][:k[g]].sum() for g in range(3))
                                          - sum(spectra[g][:best[g]].sum() for g in range(3))) < 1e-12 and sum(k) == B)
k, _ = ss.water_fill([np.array([9., 8., 7.]), np.array([1., .5])], 4, [2, 2])
check("water_fill cap", k == [2, 2])

# weighted spectrum == eigenvalues of the weighted centred covariance
z = rng.normal(size=(30, 6))
z /= np.linalg.norm(z, axis=1, keepdims=True)
p = rng.random(30)
p /= p.sum()
mu = p @ z
C = sum(p[i] * np.outer(z[i] - mu, z[i] - mu) for i in range(30))
check("weighted spectrum", np.allclose(ss.weighted_spectrum(z, p), np.sort(np.linalg.eigvalsh(C))[::-1], atol=1e-12))
check("uniform p = plain covariance", np.allclose(ss.weighted_spectrum(z, np.full(30, 1 / 30)).sum(),
                                                  ((z - z.mean(0)) ** 2).sum() / 30))

# facility location greedy == brute-force greedy
def fl_value(S):
    s = (1 + z @ z.T) / 2
    return float((p * s[:, S].max(axis=1)).sum()) if S else 0.0


order = ss.facility_location(z, p, 5)
S = []
ok = True
for t in range(5):
    j = max((j for j in range(30) if j not in S), key=lambda j: fl_value(S + [j]) - fl_value(S))
    ok &= j == order[t]
    S.append(j)
check("facility location greedy", ok, str(order))
check("facility location k=0", ss.facility_location(z, p, 0) == [])

# residual weights
b = rng.normal(size=16)
b -= b.mean()
a = [np.exp(2.0 * b + 0.01 * rng.normal(size=16)) for _ in range(3)]
ps, gamma = ss.residual_weights(a, [b, b, b])
check("gamma recovers template strength", abs(gamma - 2.0) < 0.05, f"{gamma:.3f}")
check("p sums to 1 per support", all(abs(x.sum() - 1) < 1e-12 for x in ps))
raw_ratio = max(float(x.max() / x.min()) for x in a)
res_ratio = max(float(x.max() / x.min()) for x in ps)
check("residual much flatter than raw", res_ratio < 2.0 and raw_ratio > 50, f"raw {raw_ratio:.1f} residual {res_ratio:.2f}")
ps0, g0 = ss.residual_weights([np.exp(-b)], [b])
check("negative alignment -> gamma 0", g0 == 0.0)

# template resampling
t16 = ss.centred_log(rng.random(256) * 5)
check("resample identity", np.allclose(ss.resample_template(t16, (16, 16), (16, 16)), t16))
up = ss.resample_template(t16, (16, 16), (32, 32))
check("resample shape + centred", up.size == 1024 and abs(up.mean()) < 1e-12)
check("resample keeps relative pattern", np.corrcoef(up.reshape(32, 32)[::2, ::2].ravel(), t16)[0, 1] > 0.8)

# end-to-end: homogeneous support gets fewer slots than heterogeneous one
N = 256
D = 512                                         # d > N so every support has full rank N-1 (d=64 caps rank at 64)
base = rng.normal(size=D)
homo = base + 0.02 * rng.normal(size=(N, D))
hetero = rng.normal(size=(N, D))
mixed = np.vstack([base + 0.02 * rng.normal(size=(N // 2, D)), rng.normal(size=(N // 2, D))])
Z = np.vstack([homo, hetero, mixed])
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
A = np.ones(3 * N)
b16 = np.zeros(256)
keep, info = ss.select(Z, A, [N, N, N], [(16, 16)] * 3, b16, 192)
check("select budget", len(keep) == 192 and sum(info["k"]) == 192)
check("select per-support counts", [int(((keep >= g * N) & (keep < (g + 1) * N)).sum()) for g in range(3)] == info["k"])
check("homogeneous < mixed < heterogeneous", info["k"][0] < info["k"][2] < info["k"][1], str(info["k"]))
check("select unique", len(set(keep.tolist())) == len(keep))

# real GTEx arrays (remote tree only): residual weights equal the #14.6 w_page of vit_ntrs_gtex
ARR = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/vit_ntrs_gtex/arrays/gtex_00_arrays.npz")
NULL = Path(__file__).resolve().parent / "memory" / "ssda_null_block23.npz"
if ARR.exists() and NULL.exists():
    arr = np.load(ARR)
    counts = arr["counts"].tolist()
    offs = np.cumsum([0] + counts)
    bb16 = ss.load_null_b16(NULL)
    ps, g = ss.residual_weights([arr["a_raw"][offs[i]:offs[i + 1]] for i in range(len(counts))], [bb16] * len(counts))
    check("real: p == #14.6 w_page", np.allclose(np.concatenate(ps), arr["w_page"], atol=1e-9), f"gamma {g:.3f} vs {float(arr['gamma']):.3f}")
    Zr = arr["X"].astype(np.float64)
    Zr /= np.linalg.norm(Zr, axis=1, keepdims=True)
    keep, info = ss.select(Zr, arr["a_raw"], counts, [(16, 16)] * len(counts), bb16, 768)
    check("real: budget 768", len(keep) == 768, str(info["k"]))
else:
    print("skip real-array checks (not on remote tree)")

# #14.6 NTRS global facility location
zz = rng.normal(size=(3 * 12, 8))
zz /= np.linalg.norm(zz, axis=1, keepdims=True)
pp = [rng.random(12) for _ in range(3)]
pp = [x / x.sum() for x in pp]


def gfl_value(S):
    tot = 0.0
    for g in range(3):
        Sg = [j - 12 * g for j in S if 12 * g <= j < 12 * (g + 1)]
        if Sg:
            sg = (1 + zz[12 * g:12 * (g + 1)] @ zz[12 * g:12 * (g + 1)].T) / 2
            tot += float((pp[g] * sg[:, Sg].max(axis=1)).sum())
    return tot


order = ss.global_facility_location(zz, pp, [12, 12, 12], 7)
S = []
ok = True
for t in range(7):
    j = max((j for j in range(36) if j not in S), key=lambda j: gfl_value(S + [j]) - gfl_value(S))
    ok &= j == order[t]
    S.append(j)
check("global facility location greedy", ok, str(order))
keep, info = ss.select_ntrs(Z, A, [N, N, N], [(16, 16)] * 3, b16, 192)
check("ntrs budget + counts", len(keep) == 192 and sum(info["k"]) == 192 and len(set(keep.tolist())) == 192, str(info["k"]))
if ARR.exists() and NULL.exists():
    names = [str(n) for n in arr["sel_names"]]
    ref = np.asarray(arr[f"sel_{[i for i, n in enumerate(names) if n.startswith('#14.6 residual + support')][0]}"])
    keep, info = ss.select_ntrs(Zr, arr["a_raw"], counts, [(16, 16)] * len(counts), bb16, int(round(0.25 * sum(counts))))
    ov = len(set(keep.tolist()) & set(ref.tolist())) / len(ref)
    check("real: ntrs == offline #14.6 selection", ov > 0.98, f"overlap {ov:.3f} k={info['k']}")

# dispatch from qasc_select
import memory.qasc_select as qs  # noqa: E402

called = {}
orig = ss.select_vision_features
ss.select_vision_features = lambda bb, pv, thw: called.setdefault("ssda", True) or ("feats", "keep")
for mode in ("ssda", "ntrs"):
    called.clear()
    os.environ["VLMAS_C1_SELECT"] = mode
    try:
        out = qs.select_vision_features(object(), None, None)
    finally:
        os.environ.pop("VLMAS_C1_SELECT")
    check(f"qasc dispatches {mode}", called.get("ssda") is True)
ss.select_vision_features = orig

print(f"{PASS}/{PASS} passed")
