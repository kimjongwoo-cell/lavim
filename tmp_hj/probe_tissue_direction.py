"""Is tissue vs background linearly separable in the C1-boundary representation?

Scalar readouts measured so far (||phi||^2 = 0.656, hidden norm = 0.642, a_raw = 0.582) are
magnitudes. This asks whether a DIRECTION separates them. Ridge-regularised Fisher direction,
5-fold by CASE (never train and test on the same slide), AUC on held-out cases only.

The label is the audit's colour heuristic (HSV S >= 26 per 32x32 px), so a probe that works is
evidence about tissue-vs-white, not about diagnostic content.
"""
import json, sys
from pathlib import Path
import numpy as np

ARR = Path("/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vit_ntrs_gtex/arrays")
BG, TIS = 0.10, 0.90


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    a = np.concatenate([pos, neg]); r = np.empty_like(a)
    o = np.argsort(a, kind="mergesort"); s = a[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


cases = sorted(ARR.glob("*_arrays.npz"))
Z, Y, C = [], [], []
for ci, f in enumerate(cases):
    d = np.load(f, allow_pickle=True)
    X = d["X"].astype(np.float64)
    z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    t = d["tissue"].astype(float)
    m = (t <= BG) | (t >= TIS)
    Z.append(z[m]); Y.append((t[m] >= TIS).astype(int)); C.append(np.full(int(m.sum()), ci))
Z = np.vstack(Z); Y = np.concatenate(Y); C = np.concatenate(C)
n_case = len(cases)
print(f"tokens {Z.shape[0]} (tissue {int(Y.sum())} / background {int((1 - Y).sum())}) "
      f"dim {Z.shape[1]} cases {n_case}")

folds = [np.arange(n_case)[i::5] for i in range(5)]
for lam in (1e-1, 1e-2, 1e-3):
    fisher, meandiff = [], []
    for te in folds:
        tr = ~np.isin(C, te); tem = np.isin(C, te)
        Ztr, Ytr = Z[tr], Y[tr]
        mu1, mu0 = Ztr[Ytr == 1].mean(0), Ztr[Ytr == 0].mean(0)
        dmu = mu1 - mu0
        Zc = np.vstack([Ztr[Ytr == 1] - mu1, Ztr[Ytr == 0] - mu0])
        S = (Zc.T @ Zc) / len(Zc)
        w = np.linalg.solve(S + lam * np.trace(S) / S.shape[0] * np.eye(S.shape[0]), dmu)
        sc = Z[tem] @ w; y = Y[tem]
        fisher.append(auc(sc[y == 1], sc[y == 0]))
        sc2 = Z[tem] @ dmu
        meandiff.append(auc(sc2[y == 1], sc2[y == 0]))
    print(f"lambda={lam:<6} Fisher direction AUC {np.mean(fisher):.3f} "
          f"(folds {[round(x, 3) for x in fisher]})   mean-difference AUC {np.mean(meandiff):.3f}")
# scalar references on the same token subset
nrm = np.linalg.norm(Z, axis=1)
print(f"reference: unit-norm z magnitude AUC {auc(nrm[Y == 1], nrm[Y == 0]):.3f} (must be ~0.5)")
