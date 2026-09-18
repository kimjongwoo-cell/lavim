"""Latent-Provenance Visual Handoff (LPVH) -- C2 candidate 11.9 (opt-in; off = no hooks).

Notion: C1 / C2 Method Evolution -> "Latent-Provenance Visual Handoff"
(3dbd771b-c9e2-81b5-8c98-e7db3e072db0). The Reasoner already read the C1-retained WSI supports
when it produced its latent KV; keep that native visual access trace as provenance and compose it
with the Answerer's native read of the latent KV, so the Answerer's visual mass is re-allocated
between physical supports along the two-hop path Answerer -> Reasoner latent -> WSI support.

Reasoner stage (reasoner_probe, hooks on every decoder layer during the Reasoner call):
  latent step t = a single-row call (T == 1) of every layer; its column c_t = L - 1 at that call.
  A_{t,g}^{(l,h)} = sum_{j in G_g} alpha^{R,(l,h)}_{t,j}      (native row attention, recomputed
  A_{t,g}         = mean_{l,h} A_{t,g}^{(l,h)}                   from q/K of the call)      [page 4]
  Stored on the backbone as bb._lpvh = {"A": [T, P], "latent_cols": [T], "vis_cols", "groups", ...}.
  No KV token is added; A is metadata.

Answerer stage (install_lpvh, hooks on decoder layers; row s, layer l, head h, native q/K/V):
  alpha      = softmax over ALL columns (native);   rho_V = sum_{j in V} alpha_j
  nu_{g,j}   = alpha_j / sum_{k in G_g} alpha_k                       (native within-support)
  b_{s,t}    = alpha_{s,c_t}                                          (Answerer -> latent)
  pi_g       = sum_t b_{s,t} A_{t,g},   pibar = pi / sum_r pi_r        (two-hop provenance) [page 5]
  alpha~_{g,j} = rho_V * pibar_g * nu_{g,j}                                             [page 6]
  o~ = o - sum_{j in V} alpha_j v_j + sum_{j in V} alpha~_j v_j ; then o_proj.
  sum_{g,j} alpha~ = rho_V exactly; if sum_r pi_r == 0 the native allocation is kept; a support
  with zero native mass receives its share uniformly over its columns.

Implementation choices (the page leaves them open; all reported in the log line):
  * support g: `obs` (default) = one C1-retained physical crop per support (page 3, G~_g = G_g cap
    S_C1); `branch` = C1 top-level 5x branch (memory.csis_select); `shuffled` = obs sizes with
    membership permuted (falsifier). j -> g(j) comes from the backbone's visual provenance
    (_visual_meta, column order) and is used only if its columns equal the Reasoner's visual
    columns AND the terminal's (engine._rpath_visual_cols); otherwise SKIP with a note.
  * A reduction: mean over all layers and heads of the native trajectory (page 4, "layer search
    없이"). Per-layer A is kept in the probe for analysis only (A_layers [L, T, P]).
  * composition layers/heads: b is per (l, h) of the Answerer call; A is the (l,h)-mean, exactly
    as page 5 writes pi^{(l,h)} = sum_t b^{(l,h)} A.
  * rows: `gen` (default) = last prompt row (predicts the first answer token) + every generated
    row; `all` adds every Answerer prompt-prefill row.
  * layers: `all` (default) or `a-b`.

Env
  VLMAS_LPVH=1|identity           identity = same recompute with pibar := native support share
                                  (output must equal native up to bf16: GPU smoke 09-15 max|diff| 0.5 =
                                  1 bf16 ULP at |x|~64-128, answers == base; gate on maxrel_identity)
  VLMAS_LPVH_SUPPORT=obs|branch|shuffled
  VLMAS_LPVH_PROV=native|shuffled|uniform|reasoner_mass
                                  falsifier controls for the provenance matrix A:
                                  shuffled = support axis permuted (seed = case index);
                                  uniform  = A_{t,g} = 1/P (pure cardinality re-normalisation);
                                  reasoner_mass = A_{t,:} := mean_t A (Reasoner visual mass only,
                                  the Answerer->latent weighting b drops out)
  VLMAS_LPVH_ROWS=gen|all
  VLMAS_LPVH_LAYERS=all|a-b
  VLMAS_LPVH_LOG=1                per-call detail
"""
from __future__ import annotations

import contextlib
import os
import random

import torch

EPS = 1e-12


# --------------------------------------------------------------------------- config


def enabled() -> bool:
    return os.environ.get("VLMAS_LPVH", "").strip() in ("1", "identity")


def config() -> dict:
    return {
        "mode": os.environ.get("VLMAS_LPVH", "").strip(),
        "support": os.environ.get("VLMAS_LPVH_SUPPORT", "obs").strip() or "obs",
        "prov": os.environ.get("VLMAS_LPVH_PROV", "native").strip() or "native",
        "rows": os.environ.get("VLMAS_LPVH_ROWS", "gen").strip() or "gen",
        "layers": os.environ.get("VLMAS_LPVH_LAYERS", "all").strip() or "all",
        "log": os.environ.get("VLMAS_LPVH_LOG", "") == "1",
    }


def layer_window(spec: str, n_layers: int) -> tuple[int, int]:
    if spec in ("", "all"):
        return 0, n_layers - 1
    a, b = (int(x) for x in spec.split("-")) if "-" in spec else (int(spec), int(spec))
    lo, hi = max(0, min(a, b)), min(n_layers - 1, max(a, b))
    if lo > hi:
        raise ValueError(f"empty LPVH layer window {spec!r} for {n_layers} layers")
    return lo, hi


# --------------------------------------------------------------------------- pure


def support_groups(obs_ids, parents, mode: str = "obs", seed: int = 0) -> list[list[int]]:
    """Index lists into the retained visual columns, one per support with >= 1 column.

    obs  g(j) = obs_ids[j] (one physical crop per support, page 3)
    branch  g(j) = top-level 5x branch (memory.csis_select)
    shuffled  obs sizes, membership permuted over the columns (falsifier)
    """
    if mode not in ("obs", "branch", "shuffled"):
        raise ValueError(f"support must be obs|branch|shuffled, got {mode!r}")
    obs = [int(o) for o in (obs_ids.tolist() if hasattr(obs_ids, "tolist") else obs_ids)]
    if mode == "branch":
        from memory.csis_select import drop_self_parents, support_branches
        par, _ = drop_self_parents(list(parents))
        br = support_branches(par)
        key = [br[o] if 0 <= o < len(br) else o for o in obs]
    else:
        key = obs
    groups: dict[int, list[int]] = {}
    for j, g in enumerate(key):
        groups.setdefault(g, []).append(j)
    out = [groups[g] for g in sorted(groups)]
    if mode == "shuffled":
        order = list(range(len(obs)))
        random.Random(int(seed)).shuffle(order)
        shuf, pos = [], 0
        for g in out:
            shuf.append(sorted(order[pos:pos + len(g)]))
            pos += len(g)
        out = shuf
    return out


def group_ids(groups, n_vis: int) -> torch.Tensor:
    """[n_vis] long: support id of every visual column (-1 = in no group)."""
    gid = torch.full((n_vis,), -1, dtype=torch.long)
    for g, cols in enumerate(groups):
        gid[torch.as_tensor(cols, dtype=torch.long)] = g
    return gid


def support_mass(a_vis: torch.Tensor, gid: torch.Tensor, P: int) -> torch.Tensor:
    """[..., n_vis] -> [..., P]: sum of the visual attention of every support."""
    out = a_vis.new_zeros(a_vis.shape[:-1] + (P,))
    return out.index_add(-1, gid.to(a_vis.device), a_vis)


def compose(b: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """pi = b A : b [..., T] (Answerer -> latent attention), A [T, P] -> [..., P]."""
    return torch.matmul(b, A.to(dtype=b.dtype, device=b.device))


def redistribute(a_vis: torch.Tensor, pibar: torch.Tensor, gid: torch.Tensor, sizes: torch.Tensor,
                 pi_sum: torch.Tensor | None = None) -> torch.Tensor:
    """alpha~ over the visual columns (page 6): rho_V * pibar_g * nu_{g,j}.

    a_vis [..., n_vis] native visual attention; pibar [..., P] support distribution;
    gid [n_vis]; sizes [P] (columns per support); pi_sum [..., 1] or None -- rows whose
    composed mass is 0 keep their native allocation. Total visual mass is conserved exactly.
    """
    P = int(sizes.numel())
    gid = gid.to(a_vis.device)
    mass = support_mass(a_vis, gid, P)                                    # [..., P]
    rho = mass.sum(-1, keepdim=True)                                      # [..., 1]
    target = rho * pibar                                                  # [..., P]
    ratio = torch.where(mass > 0, target / mass.clamp_min(EPS), torch.zeros_like(mass))
    scaled = a_vis * ratio[..., gid]                                      # native nu, new support mass
    uniform = (target / sizes.to(a_vis.dtype).to(a_vis.device))[..., gid]  # support with 0 native mass
    empty = (mass <= 0)[..., gid]
    new = torch.where(empty, uniform, scaled)
    if pi_sum is not None:
        new = torch.where(pi_sum > 0, new, a_vis)
    return new


def provenance_control(A: torch.Tensor, mode: str, seed: int = 0) -> torch.Tensor:
    """Falsifier transforms of the provenance matrix A [T, P]."""
    if mode == "native":
        return A
    if mode == "shuffled":
        perm = list(range(int(A.shape[1])))
        random.Random(int(seed)).shuffle(perm)
        return A[:, torch.as_tensor(perm, dtype=torch.long, device=A.device)]
    if mode == "uniform":
        return torch.full_like(A, 1.0 / max(int(A.shape[1]), 1))
    if mode == "reasoner_mass":
        return A.mean(0, keepdim=True).expand_as(A).clone()
    raise ValueError(f"prov must be native|shuffled|uniform|reasoner_mass, got {mode!r}")


def spearman(x, y) -> float:
    """Rank correlation of two 1-D sequences (average ranks for ties)."""
    import numpy as np
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v)); r[order] = np.arange(len(v))
        # average ties
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- shared attention math


def _row_attention(module, hidden, pe, keys, values, sel):
    """Native attention of the selected query rows over ALL cached columns (fp32).

    Returns alpha [1, H, Ts, L], V [1, H, L, D] (kv-repeated), o [1, H, Ts, D] (native context).
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

    L = int(keys.shape[2])
    T = int(hidden.shape[1])
    Hd = module.head_dim
    G = module.num_key_value_groups
    cos, sin = pe
    h_sel = hidden.index_select(1, sel)
    Ts = int(sel.numel())
    q = module.q_norm(module.q_proj(h_sel).view(1, Ts, -1, Hd)).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos.index_select(1, sel), sin.index_select(1, sel))
    K = repeat_kv(keys, G).float()
    V = repeat_kv(values, G).float()
    scores = torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling             # [1,H,Ts,L]
    if T > 1:
        past = L - T
        col = torch.arange(L, device=keys.device)[None, None, None, :]
        row = (past + sel)[None, None, :, None]
        scores = scores.masked_fill(col > row, float("-inf"))
    alpha = torch.softmax(scores, dim=-1)
    o = torch.matmul(alpha, V)
    return alpha, V, o


def _call_parts(args, kwargs):
    hidden = kwargs.get("hidden_states", args[0] if args else None)
    pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
    cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
    return hidden, pe, cache


# --------------------------------------------------------------------------- Reasoner probe


class ReasonerProvenance:
    """Collects A [T, P] over the Reasoner's latent steps; stored as bb._lpvh at exit."""

    def __init__(self, bb, m: int, support: str, seed: int, log: bool):
        self.bb, self.m, self.support, self.seed, self.log = bb, int(m), support, int(seed), log
        self.n_layers = len(bb.lm.layers)
        self.groups = None
        self.gid = None
        self.vis = None
        self.note = ""
        self.rows = []          # per step: [P] mean over layers/heads
        self.rows_layers = []   # per step: [L, P] mean over heads
        self.latent_cols = []
        self._acc = None
        self._acc_layers = None
        self._n_calls = 0

    def _init_groups(self, L: int):
        meta = getattr(self.bb, "_visual_meta", None)
        if meta is None:
            self.note = "no visual provenance (" + str(getattr(self.bb, "_visual_meta_note", "")) + ")"
            return False
        vis = meta["abs_cols"].long()
        if int(vis.numel()) == 0 or int(vis.max()) >= L:
            self.note = f"visual columns out of range (max {int(vis.max()) if vis.numel() else -1} >= {L})"
            return False
        self.groups = support_groups(meta["obs_id"], meta["parents"], self.support, self.seed)
        self.gid = group_ids(self.groups, int(vis.numel()))
        self.vis = vis
        return True

    def _hook(self, li):
        def post(module, args, kwargs, output):
            hidden, pe, cache = _call_parts(args, kwargs)
            if hidden is None or pe is None or cache is None or int(hidden.shape[1]) != 1:
                return output                                   # prefill rows: not a latent step
            if li == 0:
                if len(self.rows) >= self.m:
                    return output                               # extra single-row calls after the m steps
                keys = cache.layers[0].keys
                L = int(keys.shape[2])
                if self.groups is None and not self._init_groups(L):
                    return output
                self.latent_cols.append(L - 1)
                self._acc = torch.zeros(len(self.groups), dtype=torch.float64)
                self._acc_layers = torch.zeros(self.n_layers, len(self.groups), dtype=torch.float64)
                self._n_calls = 0
            if self._acc is None or self.groups is None or len(self.rows) >= self.m:
                return output
            keys = cache.layers[li].keys
            values = cache.layers[li].values
            sel = torch.tensor([0], device=hidden.device)
            with torch.no_grad():
                alpha, _, _ = _row_attention(module, hidden, pe, keys, values, sel)   # [1,H,1,L]
                a_vis = alpha[0, :, 0, :][:, self.vis.to(alpha.device)]               # [H, n_vis]
                mass = support_mass(a_vis, self.gid, len(self.groups)).mean(0)        # [P] mean heads
            m_cpu = mass.detach().double().cpu()
            self._acc += m_cpu
            self._acc_layers[li] = m_cpu
            self._n_calls += 1
            if li == self.n_layers - 1:
                self.rows.append(self._acc / self.n_layers)
                self.rows_layers.append(self._acc_layers.clone())
                self._acc = None
            return output
        return post

    def finish(self):
        T = len(self.rows)
        if T == 0:
            self.bb._lpvh = None
            self.bb._lpvh_note = self.note or "no latent step observed"
            return
        A = torch.stack(self.rows).float()                       # [T, P]
        self.bb._lpvh = {
            "A": A, "A_layers": torch.stack(self.rows_layers).float().permute(1, 0, 2),  # [L,T,P]
            "latent_cols": torch.tensor(self.latent_cols, dtype=torch.long),
            "vis_cols": self.vis.clone(), "groups": self.groups,
            "sizes": torch.tensor([len(g) for g in self.groups], dtype=torch.long),
            "support": self.support, "m": self.m,
        }
        self.bb._lpvh_note = self.note
        if self.log:
            print(f"[LPVH:R] steps={T}/{self.m} supports={len(self.groups)} "
                  f"latent_cols={self.latent_cols[0]}..{self.latent_cols[-1]} "
                  f"A_sum_per_step={[round(float(v), 4) for v in A.sum(1)]} "
                  f"A_mean_per_support={[round(float(v), 4) for v in A.mean(0)]}", flush=True)


@contextlib.contextmanager
def reasoner_probe(bb, m: int, stage: str = "reasoner", seed: int = 0):
    """Wrap the Reasoner's prefill+latent call. Off / other stage = nullcontext."""
    if not enabled() or stage != "reasoner" or int(m) <= 0:
        yield None
        return
    cfg = config()
    bb._lpvh = None
    bb._lpvh_note = "probe installed, no step yet"
    probe = ReasonerProvenance(bb, m, cfg["support"], seed, cfg["log"])
    handles = [layer.self_attn.register_forward_hook(probe._hook(li), with_kwargs=True)
               for li, layer in enumerate(bb.lm.layers)]
    try:
        yield probe
    finally:
        for h in handles:
            h.remove()
        probe.finish()


# --------------------------------------------------------------------------- Answerer hook


def engine_state(engine, prov_mode: str) -> dict:
    """{"ok", "note", "A", "latent_cols", "vis", "gid", "sizes", "groups"} from bb._lpvh + engine bookkeeping."""
    bb = engine._backbone
    st = getattr(bb, "_lpvh", None)
    if st is None:
        return {"ok": False, "note": "no Reasoner provenance (" + str(getattr(bb, "_lpvh_note", "")) + ")"}
    cols = getattr(engine, "_rpath_visual_cols", None)
    if cols is None or not torch.equal(cols.cpu().long(), st["vis_cols"].cpu().long()):
        return {"ok": False, "note": f"terminal visual columns != Reasoner provenance columns "
                                     f"({None if cols is None else int(cols.numel())} vs {int(st['vis_cols'].numel())})"}
    lat = getattr(engine, "_rpath_latent_cols", None)
    note = ""
    if lat is not None and not torch.equal(lat.cpu().long(), st["latent_cols"].cpu().long()):
        note = (f"latent cols differ from engine bookkeeping (probe {st['latent_cols'].tolist()[:3]}.. "
                f"vs engine {lat.tolist()[:3]}..); using probe")
    seed = 42 + int(getattr(engine, "_rpath_case_index", 0) or 0)
    A = provenance_control(st["A"], prov_mode, seed)
    return {"ok": True, "note": note or f"support={st['support']} T={int(A.shape[0])} P={int(A.shape[1])}",
            "A": A, "latent_cols": st["latent_cols"], "vis": st["vis_cols"],
            "gid": group_ids(st["groups"], int(st["vis_cols"].numel())), "sizes": st["sizes"],
            "groups": st["groups"], "A_native": st["A"]}


@contextlib.contextmanager
def install_lpvh(backbone, A, latent_cols, vis_cols, gid, sizes, *, layers=(0, 0), rows="gen",
                 identity=False, log=False):
    """Hooks on decoder layers layers[0]..layers[1]; yields a stats dict."""
    lo, hi = layers
    dev = backbone.device
    A = A.to(device=dev, dtype=torch.float32)
    lat = latent_cols.to(device=dev, dtype=torch.long)
    vis = vis_cols.to(device=dev, dtype=torch.long)
    gid_d = gid.to(dev)
    sizes_d = sizes.to(dev)
    P = int(sizes.numel())
    stats = {"calls": 0, "rows": 0, "rho_sum": 0.0, "b_sum": 0.0, "zero_pi_rows": 0, "l1_sum": 0.0,
             "pibar_sum": torch.zeros(P, dtype=torch.float64), "share_sum": torch.zeros(P, dtype=torch.float64),
             "maxdiff_identity": 0.0, "maxrel_identity": 0.0, "n_rowhead": 0}
    handles = []

    def make_post(li):
        def post(module, args, kwargs, output):
            hidden, pe, cache = _call_parts(args, kwargs)
            if hidden is None or pe is None or cache is None:
                return output
            attn_out = output[0] if isinstance(output, tuple) else output
            keys = cache.layers[li].keys
            values = cache.layers[li].values
            L = int(keys.shape[2])
            T = int(hidden.shape[1])
            if int(vis.numel()) == 0 or int(vis.max()) >= L or int(lat.max()) >= L:
                return output
            sel = torch.arange(T, device=hidden.device) if rows == "all" else torch.tensor([T - 1], device=hidden.device)
            with torch.no_grad():
                alpha, V, o = _row_attention(module, hidden, pe, keys, values, sel)    # [1,H,Ts,L]
                a_vis = alpha[..., vis]                                               # [1,H,Ts,n_vis]
                V_vis = V[:, :, vis, :]
                mass = support_mass(a_vis, gid_d, P)                                  # [1,H,Ts,P]
                rho = mass.sum(-1, keepdim=True)
                share = mass / rho.clamp_min(EPS)
                if identity:
                    pibar, pi_sum = share, None
                else:
                    b = alpha[..., lat]                                               # [1,H,Ts,T]
                    pi = compose(b, A)                                                # [1,H,Ts,P]
                    pi_sum = pi.sum(-1, keepdim=True)
                    pibar = torch.where(pi_sum > 0, pi / pi_sum.clamp_min(EPS), share)
                    stats["b_sum"] += float(b.sum(-1).mean())
                    stats["zero_pi_rows"] += int((pi_sum <= 0).sum())
                    stats["l1_sum"] += float((pibar - share).abs().sum(-1).mean())
                    stats["pibar_sum"] += pibar.double().mean(dim=(0, 1, 2)).cpu()
                    stats["share_sum"] += share.double().mean(dim=(0, 1, 2)).cpu()
                    stats["n_rowhead"] += 1
                a_new = redistribute(a_vis, pibar, gid_d, sizes_d, pi_sum)
                o_new = o + torch.matmul(a_new - a_vis, V_vis)
                stats["rho_sum"] += float(rho.mean())
                Ts = int(sel.numel())
                flat = o_new.transpose(1, 2).reshape(1, Ts, -1)
                upd = module.o_proj(flat.to(module.o_proj.weight.dtype))
                if identity:
                    _nat = attn_out.index_select(1, sel).float()
                    _d = float((upd.float() - _nat).abs().max())
                    stats["maxdiff_identity"] = max(stats["maxdiff_identity"], _d)
                    stats["maxrel_identity"] = max(stats["maxrel_identity"], _d / max(float(_nat.abs().max()), 1e-6))
                new = attn_out.clone()
                new[:, sel, :] = upd.to(attn_out.dtype)
            stats["calls"] += 1
            stats["rows"] += Ts
            if log:
                print(f"[LPVH:A] L{li} T={T} rho_V={float(rho.mean()):.5f} "
                      f"pibar={[round(float(v), 3) for v in pibar.mean(dim=(0, 1, 2))]}", flush=True)
            return (new, *output[1:]) if isinstance(output, tuple) else new
        return post

    for li, layer in enumerate(backbone.lm.layers):
        if lo <= li <= hi:
            handles.append(layer.self_attn.register_forward_hook(make_post(li), with_kwargs=True))
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()


def summary(stats: dict, state: dict | None = None) -> str:
    c = max(stats["calls"], 1)
    n = max(stats["n_rowhead"], 1)
    s = (f"calls={stats['calls']} rows={stats['rows']} mean_rho_V={stats['rho_sum'] / c:.5f} "
         f"mean_b(A->latent)={stats['b_sum'] / n:.5f} zero_pi_rows={stats['zero_pi_rows']} "
         f"mean_L1(pibar,share)={stats['l1_sum'] / n:.4f} maxdiff_identity={stats['maxdiff_identity']:.3e} "
         f"maxrel_identity={stats['maxrel_identity']:.3e}")
    if state is not None and stats["n_rowhead"] > 0:
        pibar = (stats["pibar_sum"] / n).tolist()
        share = (stats["share_sum"] / n).tolist()
        sizes = state["sizes"].tolist()
        a_r = state["A_native"].sum(0).tolist()          # Reasoner visual mass per support
        s += (f" | pibar={[round(v, 3) for v in pibar]} share={[round(v, 3) for v in share]} "
              f"sizes={sizes} A_R={[round(v, 4) for v in a_r]}"
              f" | spearman(pibar,share)={spearman(pibar, share):.3f} "
              f"spearman(pibar,size)={spearman(pibar, sizes):.3f} spearman(pibar,A_R)={spearman(pibar, a_r):.3f} "
              f"spearman(share,size)={spearman(share, sizes):.3f}")
    return s


__all__ = ["enabled", "config", "layer_window", "support_groups", "group_ids", "support_mass", "compose",
           "redistribute", "provenance_control", "spearman", "ReasonerProvenance", "reasoner_probe",
           "engine_state", "install_lpvh", "summary"]
