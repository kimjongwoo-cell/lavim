"""Support-Resolved Visual Write Handoff (SRVW) -- C2 candidate 11.9 (opt-in; off = no hooks).

Notion: C1 / C2 Method Evolution -> "Support-Resolved Visual Write Handoff"
(3dbd771b-c9e2-81b5-8c98-e7db3e072db0, revised 09-15 09:53 from Latent-Provenance Visual Handoff).
Not attention mass: the value-bearing visual message alpha*V that each physical support writes.

Reasoner stage (reasoner_probe, post-hooks on every decoder layer during the Reasoner call):
  latent step t = a single-row call (T == 1); native row attention recomputed from q/K/V of the call
  c_{t,g}^{(l,h)} = sum_{j in G_g} alpha^{R,(l,h)}_{t,j} V_j^{(l,h)}                       [page 4]
  C_g^{(l)}       = W_O^{(l)} [ concat_h sum_t c_{t,g}^{(l,h)} ]          -> bb._srvw["C"] [L, P, d]
  (online accumulation over the m latent steps; no KV token, no extra forward)

Answerer stage (install_srvw, post-hooks on decoder layers):
  boundary row s0 of the first Answerer call at layer l (VLMAS_SRVW_BOUNDARY: last = last prompt row,
  i.e. the row that predicts the first answer token; first = first row of the Answerer segment)
  M_g^{(l)} = W_O^{(l)} [ concat_h sum_{j in G_g} alpha^{A,(l,h)}_{s0,j} V_j^{(l,h)} ]    [page 5]
  a_g^{(l)} = [ M_g . C_g / ||M_g||^2 ]_+                    (non-negative least squares) [page 6]
  every modified row s >= s0 (same call) and every later row, per head:
  alpha~_{s,g,j} = rho_V a_g alpha_{s,g,j} / sum_r a_r sum_{k in G_r} alpha_{s,r,k}         [page 7]
  o~ = o - sum_{j in V} alpha_j v_j + sum_{j in V} alpha~_j v_j ; then o_proj.
  Total visual mass rho_V and within-support ratios are exact; all a = 0 or a zero denominator
  -> native allocation for that row/head.

Implementation choices the page leaves open (all printed in the log line):
  * support g = one C1-retained physical crop (_visual_meta obs_id, column order); used only if the
    columns equal the terminal's engine._rpath_visual_cols (else SKIP).
  * s0 default `last`. M is read at layer l from the row's actual input at that layer (upstream layers
    of the same row are already handed off when s0 is inside the modified range) -- no extra forward.
  * a is fixed per layer for the whole Answerer call (page 7).

Implementation (VLMAS_SRVW_IMPL)
  recompute (default, the 09-15 full runs): per hooked call the selected rows' attention is recomputed in fp32
    over the kv-repeated cache and the attention output of those rows is replaced by o_proj(o~).
  light: no kv repeat and no fp32 copy of the cache -- query heads are folded into their KV group
    (q [1,H,Ts,D] -> [1,Hkv,G*Ts,D]) and scored against the cache in its own dtype; softmax in fp32 on the
    [1,H,Ts,L] logits only; only the visual columns' V are gathered; the output is native + o_proj-weight
    applied to (alpha~ - alpha) V_vis, so identity (a = 1) returns the native output bit-exactly.

Env
  VLMAS_SRVW=1|identity          identity = same recompute with a_g := 1 (output == native up to fp; path check)
  VLMAS_SRVW_IMPL=recompute|light
  VLMAS_SRVW_BOUNDARY=last|first
  VLMAS_SRVW_LAYERS=all|a-b
  VLMAS_SRVW_LOG=1               per-layer detail
"""
from __future__ import annotations

import contextlib
import os

import torch
import torch.nn.functional as F

EPS = 1e-12


# --------------------------------------------------------------------------- config


def enabled() -> bool:
    return os.environ.get("VLMAS_SRVW", "").strip() in ("1", "identity")


def config() -> dict:
    boundary = os.environ.get("VLMAS_SRVW_BOUNDARY", "last").strip() or "last"
    if boundary not in ("last", "first"):
        raise ValueError(f"VLMAS_SRVW_BOUNDARY must be last|first, got {boundary!r}")
    impl = os.environ.get("VLMAS_SRVW_IMPL", "recompute").strip() or "recompute"
    if impl not in ("recompute", "light"):
        raise ValueError(f"VLMAS_SRVW_IMPL must be recompute|light, got {impl!r}")
    return {
        "impl": impl,
        "mode": os.environ.get("VLMAS_SRVW", "").strip(),
        "boundary": boundary,
        "layers": os.environ.get("VLMAS_SRVW_LAYERS", "all").strip() or "all",
        "log": os.environ.get("VLMAS_SRVW_LOG", "") == "1",
    }


def layer_window(spec: str, n_layers: int) -> tuple[int, int]:
    if spec in ("", "all"):
        return 0, n_layers - 1
    a, b = (int(x) for x in spec.split("-")) if "-" in spec else (int(spec), int(spec))
    lo, hi = max(0, min(a, b)), min(n_layers - 1, max(a, b))
    if lo > hi:
        raise ValueError(f"empty SRVW layer window {spec!r} for {n_layers} layers")
    return lo, hi


# --------------------------------------------------------------------------- pure


def obs_groups(obs_ids) -> list[list[int]]:
    """Index lists into the retained visual columns, one per physical crop (page 3), ordered by crop id."""
    groups: dict[int, list[int]] = {}
    for j, g in enumerate(int(o) for o in (obs_ids.tolist() if hasattr(obs_ids, "tolist") else obs_ids)):
        groups.setdefault(g, []).append(j)
    return [groups[g] for g in sorted(groups)]


def group_ids(groups, n_vis: int) -> torch.Tensor:
    gid = torch.full((n_vis,), -1, dtype=torch.long)
    for g, cols in enumerate(groups):
        gid[torch.as_tensor(cols, dtype=torch.long)] = g
    return gid


def support_messages(a_vis: torch.Tensor, V_vis: torch.Tensor, gid: torch.Tensor, P: int) -> torch.Tensor:
    """a_vis [H, n_vis], V_vis [H, n_vis, D] -> [H, P, D]: sum_{j in G_g} alpha_j V_j per head."""
    out = V_vis.new_zeros(V_vis.shape[0], P, V_vis.shape[-1])
    return out.index_add(1, gid.to(V_vis.device), a_vis.unsqueeze(-1) * V_vis)


def project_out(msg: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[H, P, D] -> [P, d]: W_O [concat_h msg] (o_proj has no bias in Qwen3-VL; weight only, page 4-5)."""
    H, P, D = msg.shape
    return F.linear(msg.permute(1, 0, 2).reshape(P, H * D).to(weight.dtype), weight)


def nnls_coeff(M: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """a_g = [M_g . C_g / ||M_g||^2]_+ for [P, d] -> [P]; ||M_g|| = 0 -> 0 (page 6)."""
    M = M.double(); C = C.double()
    den = (M * M).sum(-1)
    num = (M * C).sum(-1)
    return torch.where(den > 0, (num / den.clamp_min(EPS)).clamp_min(0.0), torch.zeros_like(den))


def reweight(a_vis: torch.Tensor, coeff: torch.Tensor, gid: torch.Tensor) -> torch.Tensor:
    """alpha~ over visual columns (page 7). a_vis [..., n_vis], coeff [P] -> [..., n_vis].

    rho_V a_g alpha_j / sum_r a_r m_r ; rows whose denominator is 0 keep the native allocation.
    """
    P = int(coeff.numel())
    gid = gid.to(a_vis.device)
    c = coeff.to(device=a_vis.device, dtype=a_vis.dtype)
    mass = a_vis.new_zeros(a_vis.shape[:-1] + (P,)).index_add(-1, gid, a_vis)      # [..., P]
    rho = mass.sum(-1, keepdim=True)
    den = (mass * c).sum(-1, keepdim=True)
    factor = torch.where(den > 0, rho * c / den.clamp_min(EPS), torch.ones_like(mass))  # [..., P]
    return a_vis * factor[..., gid]


# --------------------------------------------------------------------------- shared attention math


def _row_attention(module, hidden, pe, keys, values, sel):
    """Native attention of the selected query rows over ALL cached columns (fp32).

    Returns alpha [1, H, Ts, L], V [1, H, L, D] (kv-repeated), o [1, H, Ts, D] (native context).
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

    L = int(keys.shape[2])
    T = int(hidden.shape[1])
    Hd = module.head_dim
    cos, sin = pe
    h_sel = hidden.index_select(1, sel)
    Ts = int(sel.numel())
    q = module.q_norm(module.q_proj(h_sel).view(1, Ts, -1, Hd)).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos.index_select(1, sel), sin.index_select(1, sel))
    K = repeat_kv(keys, module.num_key_value_groups).float()
    V = repeat_kv(values, module.num_key_value_groups).float()
    scores = torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling
    if T > 1:
        col = torch.arange(L, device=keys.device)[None, None, None, :]
        row = ((L - T) + sel)[None, None, :, None]
        scores = scores.masked_fill(col > row, float("-inf"))
    alpha = torch.softmax(scores, dim=-1)
    return alpha, V, torch.matmul(alpha, V)


def _light_attention(module, hidden, pe, keys, sel):
    """Native attention of the selected rows over ALL cached columns without repeating/upcasting the cache.

    Query heads are folded into their KV group so the logits are one matmul against keys [1,Hkv,L,D] in the
    cache dtype; softmax runs in fp32 on [1,H,Ts,L]. Returns alpha [1, H, Ts, L] (fp32).
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    Hkv, L, D = int(keys.shape[1]), int(keys.shape[2]), module.head_dim
    G = module.num_key_value_groups
    T = int(hidden.shape[1])
    cos, sin = pe
    Ts = int(sel.numel())
    q = module.q_norm(module.q_proj(hidden.index_select(1, sel)).view(1, Ts, -1, D)).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos.index_select(1, sel), sin.index_select(1, sel))
    qf = q.reshape(1, Hkv, G, Ts, D).reshape(1, Hkv, G * Ts, D).to(keys.dtype)          # head h = kv*G + g
    scores = torch.matmul(qf, keys.transpose(-2, -1)).float() * module.scaling          # [1,Hkv,G*Ts,L]
    scores = scores.view(1, Hkv, G, Ts, L).reshape(1, Hkv * G, Ts, L)
    if T > 1:
        col = torch.arange(L, device=keys.device)[None, None, None, :]
        row = ((L - T) + sel)[None, None, :, None]
        scores = scores.masked_fill(col > row, float("-inf"))
    return torch.softmax(scores, dim=-1)


def _gqa_support_messages(a_vis: torch.Tensor, V_vis_kv: torch.Tensor, gid: torch.Tensor, P: int) -> torch.Tensor:
    """a_vis [H, n] (fp32), V_vis_kv [Hkv, n, D] (cache dtype) -> [H, P, D] = sum_{j in G_g} alpha_j V_j, no kv repeat."""
    H, n = a_vis.shape
    Hkv, D = int(V_vis_kv.shape[0]), int(V_vis_kv.shape[-1])
    G = H // Hkv
    onehot = torch.zeros(P, n, dtype=a_vis.dtype, device=a_vis.device)
    onehot[gid.to(a_vis.device), torch.arange(n, device=a_vis.device)] = 1.0
    w = (a_vis[:, None, :] * onehot[None]).reshape(Hkv, G * P, n)                      # [Hkv, G*P, n]
    return torch.matmul(w, V_vis_kv.float()).reshape(Hkv, G, P, D).reshape(H, P, D)


def _gqa_delta_out(delta: torch.Tensor, V_vis_kv: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """delta [1, H, Ts, n] (alpha~ - alpha over visual columns) -> [1, Ts, d] = W_O [concat_h delta V]."""
    _, H, Ts, n = delta.shape
    Hkv, D = int(V_vis_kv.shape[0]), int(V_vis_kv.shape[-1])
    G = H // Hkv
    o = torch.matmul(delta.reshape(Hkv, G * Ts, n), V_vis_kv.float())                   # [Hkv, G*Ts, D]
    o = o.reshape(Hkv, G, Ts, D).reshape(H, Ts, D).transpose(0, 1).reshape(1, Ts, H * D)
    return F.linear(o, weight.float())


def _call_parts(args, kwargs):
    hidden = kwargs.get("hidden_states", args[0] if args else None)
    pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
    cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
    return hidden, pe, cache


# --------------------------------------------------------------------------- Reasoner probe


class ReasonerVisualWrite:
    """Accumulates sum_t c_{t,g}^{(l,h)} over the Reasoner's latent steps; bb._srvw at exit."""

    def __init__(self, bb, m: int, log: bool, impl: str = "recompute"):
        self.bb, self.m, self.log, self.impl = bb, int(m), log, impl
        self.layers = bb.lm.layers
        self.groups = self.gid = self.vis = None
        self.acc = [None] * len(self.layers)      # [H, P, D] per layer
        self.steps = 0
        self.in_step = False
        self.note = ""

    def _init_groups(self, L: int) -> bool:
        meta = getattr(self.bb, "_visual_meta", None)
        if meta is None:
            self.note = "no visual provenance (" + str(getattr(self.bb, "_visual_meta_note", "")) + ")"
            return False
        vis = meta["abs_cols"].long()
        if int(vis.numel()) == 0 or int(vis.max()) >= L:
            self.note = f"visual columns out of range (max {int(vis.max()) if vis.numel() else -1} >= {L})"
            return False
        self.groups = obs_groups(meta["obs_id"])
        self.gid = group_ids(self.groups, int(vis.numel()))
        self.vis = vis
        return True

    def _hook(self, li):
        def post(module, args, kwargs, output):
            hidden, pe, cache = _call_parts(args, kwargs)
            if hidden is None or pe is None or cache is None or int(hidden.shape[1]) != 1:
                return output                                   # prefill rows: not a latent step
            if li == 0:
                self.in_step = False
                if self.steps >= self.m:
                    return output
                L = int(cache.layers[0].keys.shape[2])
                if self.groups is None and not self._init_groups(L):
                    return output
                self.in_step = True
                self.steps += 1
            if not self.in_step:
                return output
            with torch.no_grad():
                keys, values = cache.layers[li].keys, cache.layers[li].values
                sel0 = torch.tensor([0], device=hidden.device)
                vis = self.vis.to(keys.device)
                if self.impl == "light":
                    alpha = _light_attention(module, hidden, pe, keys, sel0)
                    msg = _gqa_support_messages(alpha[0, :, 0, vis], values[0, :, vis, :], self.gid, len(self.groups))
                else:
                    alpha, V, _ = _row_attention(module, hidden, pe, keys, values, sel0)
                    msg = support_messages(alpha[0, :, 0, vis], V[0, :, vis, :], self.gid, len(self.groups))
            self.acc[li] = msg if self.acc[li] is None else self.acc[li] + msg
            return output
        return post

    def finish(self):
        if self.steps == 0 or any(a is None for a in self.acc):
            self.bb._srvw = None
            self.bb._srvw_note = self.note or f"latent steps observed {self.steps}"
            return
        with torch.no_grad():
            C = torch.stack([project_out(self.acc[li], layer.self_attn.o_proj.weight.float())
                             for li, layer in enumerate(self.layers)])            # [L, P, d]
        self.bb._srvw = {"C": C, "vis_cols": self.vis.clone(), "groups": self.groups,
                         "sizes": torch.tensor([len(g) for g in self.groups], dtype=torch.long), "steps": self.steps}
        self.bb._srvw_note = self.note
        if self.log:
            print(f"[SRVW:R] steps={self.steps}/{self.m} supports={len(self.groups)} "
                  f"|C| per support (mean over layers)={[round(float(v), 3) for v in C.norm(dim=-1).mean(0)]}",
                  flush=True)


@contextlib.contextmanager
def reasoner_probe(bb, m: int, stage: str = "reasoner"):
    """Wrap the Reasoner's prefill+latent call. Off / other stage = nullcontext."""
    if not enabled() or stage != "reasoner" or int(m) <= 0:
        yield None
        return
    bb._srvw = None
    bb._srvw_note = "probe installed, no step yet"
    cfg = config()
    probe = ReasonerVisualWrite(bb, m, cfg["log"], cfg["impl"])
    handles = [layer.self_attn.register_forward_hook(probe._hook(li), with_kwargs=True)
               for li, layer in enumerate(bb.lm.layers)]
    try:
        yield probe
    finally:
        for h in handles:
            h.remove()
        probe.finish()


# --------------------------------------------------------------------------- Answerer hook


def engine_state(engine) -> dict:
    """{"ok", "note", "C", "vis", "gid", "sizes", "groups"} from bb._srvw + the terminal's visual columns."""
    bb = engine._backbone
    st = getattr(bb, "_srvw", None)
    if st is None:
        return {"ok": False, "note": "no Reasoner visual write (" + str(getattr(bb, "_srvw_note", "")) + ")"}
    cols = getattr(engine, "_rpath_visual_cols", None)
    if cols is None or not torch.equal(cols.cpu().long(), st["vis_cols"].cpu().long()):
        return {"ok": False, "note": f"terminal visual columns != Reasoner provenance columns "
                                     f"({None if cols is None else int(cols.numel())} vs {int(st['vis_cols'].numel())})"}
    if int(st["C"].shape[0]) != len(bb.lm.layers):
        return {"ok": False, "note": f"C layers {int(st['C'].shape[0])} != {len(bb.lm.layers)}"}
    return {"ok": True, "note": f"P={len(st['groups'])} steps={st['steps']}", "C": st["C"], "vis": st["vis_cols"],
            "gid": group_ids(st["groups"], int(st["vis_cols"].numel())), "sizes": st["sizes"], "groups": st["groups"]}


@contextlib.contextmanager
def install_srvw(backbone, C, vis_cols, gid, sizes, *, layers=(0, 0), boundary="last", identity=False, log=False,
                 impl=None):
    """Post-hooks on decoder layers layers[0]..layers[1] for one Answerer call; yields a stats dict.
    impl None = VLMAS_SRVW_IMPL (engine call sites do not pass it)."""
    impl = impl or config()["impl"]
    lo, hi = layers
    dev = backbone.device
    vis = vis_cols.to(device=dev, dtype=torch.long)
    gid_d = gid.to(dev)
    P = int(sizes.numel())
    coeff = {}                                                   # layer -> a [P] (fixed for the call)
    stats = {"calls": 0, "rows": 0, "boundary_T": None, "zero_a_layers": 0, "a_share_sum": torch.zeros(P, dtype=torch.float64),
             "native_share_sum": torch.zeros(P, dtype=torch.float64), "l1_sum": 0.0, "n_l1": 0, "cos_sum": 0.0,
             "n_layers": 0, "maxdiff_identity": 0.0, "a": {}, "impl": impl}
    handles = []

    def make_post(li):
        C_l = C[li].to(device=dev, dtype=torch.float32)

        def post(module, args, kwargs, output):
            hidden, pe, cache = _call_parts(args, kwargs)
            if hidden is None or pe is None or cache is None:
                return output
            attn_out = output[0] if isinstance(output, tuple) else output
            keys, values = cache.layers[li].keys, cache.layers[li].values
            L, T = int(keys.shape[2]), int(hidden.shape[1])
            if int(vis.numel()) == 0 or int(vis.max()) >= L:
                return output
            with torch.no_grad():
                if li not in coeff:
                    s0 = T - 1 if boundary == "last" else 0
                    stats["boundary_T"] = T if stats["boundary_T"] is None else stats["boundary_T"]
                    s0t = torch.tensor([s0], device=hidden.device)
                    if impl == "light":
                        ab = _light_attention(module, hidden, pe, keys, s0t)
                        msg = _gqa_support_messages(ab[0, :, 0, vis], values[0, :, vis, :], gid_d, P)
                    else:
                        ab, Vb, _ = _row_attention(module, hidden, pe, keys, values, s0t)
                        msg = support_messages(ab[0, :, 0, vis], Vb[0, :, vis, :], gid_d, P)
                    M = project_out(msg, module.o_proj.weight.float())             # [P, d]
                    a = torch.ones(P, dtype=torch.float64, device=dev) if identity else nnls_coeff(M, C_l)
                    coeff[li] = a
                    stats["a"][li] = a.detach().cpu()
                    stats["n_layers"] += 1
                    stats["zero_a_layers"] += int(float(a.sum()) <= 0)
                    if float(a.sum()) > 0:
                        stats["a_share_sum"] += (a / a.sum()).cpu()
                    stats["cos_sum"] += float(F.cosine_similarity(M.double(), C_l.double(), dim=-1).mean())
                    sel = torch.arange(s0, T, device=hidden.device)
                else:
                    sel = torch.arange(T, device=hidden.device)
                a = coeff[li]
                Ts = int(sel.numel())
                if impl == "light":
                    alpha = _light_attention(module, hidden, pe, keys, sel)
                    a_vis = alpha[..., vis]
                    new = reweight(a_vis, a, gid_d)
                    upd = (attn_out.index_select(1, sel).float()
                           + _gqa_delta_out(new - a_vis, values[0, :, vis, :], module.o_proj.weight))
                else:
                    alpha, V, o = _row_attention(module, hidden, pe, keys, values, sel)
                    a_vis = alpha[..., vis]
                    new = reweight(a_vis, a, gid_d)
                    o_new = o + torch.matmul(new - a_vis, V[:, :, vis, :])
                    upd = module.o_proj(o_new.transpose(1, 2).reshape(1, Ts, -1).to(module.o_proj.weight.dtype))
                if identity:
                    stats["maxdiff_identity"] = max(stats["maxdiff_identity"], float(
                        (upd.float() - attn_out.index_select(1, sel).float()).abs().max()))
                    if impl == "light":
                        upd = attn_out.index_select(1, sel)             # a = 1: native output, delta only measured
                else:
                    m_old = a_vis.new_zeros(a_vis.shape[:-1] + (P,)).index_add(-1, gid_d, a_vis)
                    m_new = new.new_zeros(new.shape[:-1] + (P,)).index_add(-1, gid_d, new)
                    rho = m_old.sum(-1, keepdim=True).clamp_min(EPS)
                    stats["l1_sum"] += float(((m_new - m_old) / rho).abs().sum(-1).mean())
                    stats["native_share_sum"] += (m_old / rho).double().mean(dim=(0, 1, 2)).cpu()
                    stats["n_l1"] += 1
                out = attn_out.clone()
                out[:, sel, :] = upd.to(attn_out.dtype)
            stats["calls"] += 1
            stats["rows"] += Ts
            if log:
                print(f"[SRVW:A] L{li} T={T} rows={Ts} a={[round(float(v), 3) for v in a]}", flush=True)
            return (out, *output[1:]) if isinstance(output, tuple) else out
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
    nl = max(stats["n_layers"], 1)
    s = (f"impl={stats.get('impl')} calls={stats['calls']} rows={stats['rows']} boundary_T={stats['boundary_T']} layers={stats['n_layers']} "
         f"zero_a_layers={stats['zero_a_layers']} mean_cos(M,C)={stats['cos_sum'] / nl:.4f} "
         f"maxdiff_identity={stats['maxdiff_identity']:.3e}")
    if stats["n_l1"]:
        s += (f" mean_L1(support share new-native)={stats['l1_sum'] / stats['n_l1']:.4f}"
              f" | a_share={[round(v, 3) for v in (stats['a_share_sum'] / nl).tolist()]}"
              f" native_share={[round(v, 3) for v in (stats['native_share_sum'] / stats['n_l1']).tolist()]}")
    if state is not None:
        s += f" sizes={state['sizes'].tolist()}"
    return s


__all__ = ["enabled", "config", "layer_window", "obs_groups", "group_ids", "support_messages", "project_out",
           "nnls_coeff", "reweight", "ReasonerVisualWrite", "reasoner_probe", "engine_state", "install_srvw", "summary"]
