"""Unsupervised version: can the content-free templates themselves give the DIRECTION?

NR-PSRD used the null states as a subspace to remove, then read the leftover magnitude (AUC 0.656).
Here the same null states define a contrast direction instead. No tissue label is used to build any
direction; the label is only for scoring.

  w_null   = mean(case tokens) - mean(null template states)      [uses no labels]
  w_nullwh = whitened version, (S + lam I)^-1 w_null             [S from case tokens only]
  score_i  = z_i . w / ||w||
Held out by case for the whitened variant (S and the means come from the other 16 cases).
"""
from pathlib import Path
import numpy as np

ARR = Path("/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vit_ntrs_gtex/arrays")
NULLC = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/vit_nrpsrd_gtex/null_states.npz")
BG, TIS = 0.10, 0.90


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    a = np.concatenate([pos, neg]); r = np.empty_like(a)
    o = np.argsort(a, kind="mergesort"); s = a[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


nu = np.load(NULLC)["nu"].astype(np.float64)
nu = nu / (np.linalg.norm(nu, axis=1, keepdims=True) + 1e-8)
mu_null = nu.mean(0)

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
folds = [np.arange(n_case)[i::5] for i in range(5)]

# plain direction, no whitening, no folds needed (mu_case from all tokens, still label-free)
w = Z.mean(0) - mu_null
sc = Z @ w
print(f"null-contrast direction (label-free, no whitening)      AUC {auc(sc[Y == 1], sc[Y == 0]):.3f}")

for lam in (1e-1, 1e-2, 1e-3):
    got = []
    for te in folds:
        tr = ~np.isin(C, te); tem = np.isin(C, te)
        Ztr = Z[tr]
        mu = Ztr.mean(0)
        wv = mu - mu_null
        Zc = Ztr - mu
        S = (Zc.T @ Zc) / len(Zc)
        ww = np.linalg.solve(S + lam * np.trace(S) / S.shape[0] * np.eye(S.shape[0]), wv)
        s2 = Z[tem] @ ww; y = Y[tem]
        got.append(auc(s2[y == 1], s2[y == 0]))
    print(f"null-contrast whitened, held-out by case, lam={lam:<6} AUC {np.mean(got):.3f} "
          f"(folds {[round(x, 3) for x in got]})")
