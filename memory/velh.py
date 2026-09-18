"""Visual-Evidence-Bound Latent Handoff (VELH), C2 #14 -- Notion 3ddd771b-c9e2-8176 (Candidate).

Opt-in.  VLMAS_VELH unset = no hook, no cache change, byte-identical pipeline.

Sender = Reasoner, consumer = Answerer (main C2 boundary).  Everything acts on ONE cached column,
the Reasoner endpoint e_R = its last latent step (architecture-defined completion state), and is
applied IN PLACE on the cache the Answerer consumes, after the Reasoner turn is closed and before
the Answerer prompt is prefilled.  No column is appended or duplicated; cardinality is preserved.

Question anchor (page 2): the last token of the ORIGINAL user question inside the Reasoner prompt
(I_Q located as in memory/lu.py).  Its pre-RoPE query q_norm(q_proj(h)) IS the canonical
q_bar_Q = R(p_Q)^-1 q_Q, captured per layer during the Reasoner prefill.  Visual keys are
canonicalised from the cache with the model's own cos/sin rows: K_bar_j = R(p_j)^-1 K_j.

Step 1 -- question-conditioned evidence binding (page 3), per layer l and head h:
  beta_{Q,j} = softmax_{j in V}( q_bar_Q . K_bar_{V,j} / sqrt(d) )       (retained visual slots only)
  m_Q        = sum_j beta_{Q,j} V_{V,j}
  GQA: beta / m are formed per query head with its kv group's keys / values and m is averaged over
  the group's query heads -> one m_Q per kv head (the page has one head axis; this is the choice).
  u_Q = m_Q / |m_Q|;   V~_R = V_R + [ u_Q.m_Q - u_Q.V_R ]_+ u_Q      (minimum-change projection)
  Keys untouched.
Step 2 -- receiver-boundary handoff (page 4): K^_R = R(p_A^-) R(p_R)^-1 K_R, p_A^- = the coordinate
  just before the Answerer's first logical position (= position_cursor - 1, as RELH), V = V~_R.
  Written in place, so EVERY Answerer row reads the rebound key (RELH's default rebinds only the
  decision rows through a hook; the formula is identical).
Direct-Visual control (page 6): the endpoint slot gets K_D = R(p_A^-) R(p_Q)^-1 K_Q (the anchor's
  cached key rephased) and V_D = m_Q; the native endpoint K/V are discarded.
Question-shuffle control (page 7): Step 1's read uses the PREVIOUS case's anchor query (full arm
  otherwise unchanged); the first case of a process has no donor and is skipped (logged).

Measurements (page 8), every arm incl. diag: constraint activation rate and gap over (layer, kv
head), |dV|/|V_R|, |m_Q| vs |V_R|; Answerer attention mass on the endpoint column, on the native
anchor column and on all visual columns (fp32 recompute in hooks, rows = last prompt row + every
generated row, mean over layers/heads/rows).  Written to VLMAS_VELH_LOG (jsonl) and printed.

Env
  VLMAS_VELH=diag|bind|handoff|full|direct|shuffle
  VLMAS_VELH_LOG=<jsonl>
"""
from __future__ import annotations

import contextlib
import json
import os
import time

import torch

from memory.lu import _call_parts, _kvlen, locate_question, question_rows_alpha
from memory.restage import delta_cos_sin, rerotate_keys, rotate_half

MODES = ("diag", "bind", "handoff", "full", "direct", "shuffle")


def mode() -> str:
    v = os.environ.get("VLMAS_VELH", "").strip().lower()
    return v if v in MODES else ""


def enabled() -> bool:
    return mode() != ""


def config() -> dict:
    return {"mode": mode(), "log": os.environ.get("VLMAS_VELH_LOG", "").strip()}


# --------------------------------------------------------------------------- pure


def derotate(K: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """R(p)^-1 K for cached (post-RoPE) keys K [..., n, D] with the cos/sin rows [n, D] of p (fp32)."""
    Kf = K.float()
    return Kf * cos.float() - rotate_half(Kf) * sin.float()


def rephase(K: torch.Tensor, cos_old, sin_old, cos_new, sin_new) -> torch.Tensor:
    """R(p_new) R(p_old)^-1 K for cached keys K [..., n, D]; cos/sin rows [n, D] (fp32)."""
    cd, sd = delta_cos_sin(cos_old.float(), sin_old.float(), cos_new.float(), sin_new.float())
    Kf = K.float()
    return Kf * cd + rotate_half(Kf) * sd


def question_read(q_pre: torch.Tensor, Kbar: torch.Tensor, V: torch.Tensor, scaling: float,
                  groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """q_pre [Hq,D] canonical anchor query; Kbar/V [Hk,n,D] canonical visual keys / values.

    Returns (m_kv [Hk,D] = group-mean of sum_j beta_j V_j, beta [Hq,n]).
    """
    Hq, D = q_pre.shape
    Hk = Kbar.shape[0]
    Kq = Kbar.float().repeat_interleave(groups, dim=0)                    # [Hq,n,D]
    Vq = V.float().repeat_interleave(groups, dim=0)
    scores = torch.einsum("hd,hnd->hn", q_pre.float(), Kq) * float(scaling)
    beta = torch.softmax(scores, dim=-1)                                   # [Hq,n]
    m = torch.einsum("hn,hnd->hd", beta, Vq)                               # [Hq,D]
    return m.view(Hk, groups, D).mean(1), beta


def bind(V_R: torch.Tensor, m: torch.Tensor, eps: float = 1e-12):
    """Minimum-change projection: V~ = V + [|m| - u.V]_+ u, u = m/|m|.  Rows = kv heads.

    Returns (V~ [Hk,D] fp32, gap [Hk] = |m| - u.V (positive = constraint active), u.V [Hk], |m| [Hk]).
    """
    Vf, mf = V_R.float(), m.float()
    mn = mf.norm(dim=-1)                                                   # [Hk]
    u = mf / mn.clamp_min(eps)[:, None]
    proj = (u * Vf).sum(-1)
    gap = mn - proj
    Vt = Vf + torch.clamp(gap, min=0)[:, None] * u
    return Vt, gap, proj, mn


def text_cos_sin(bb, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin rows [D] of a text position (all three MRoPE axes equal), from the model's own tables."""
    dev = bb.device
    p = torch.full((3, 1, 1), int(pos), dtype=torch.long, device=dev)
    sample = bb.lm.layers[0].self_attn.q_proj.weight.new_zeros(1, 1, 1, 1)
    cos, sin = bb.lm.rotary_emb(sample, p)
    return cos.reshape(-1).float(), sin.reshape(-1).float()


# --------------------------------------------------------------------------- Reasoner capture


class ReasonerCapture:
    """Per Reasoner call: anchor row, its pre-RoPE query per layer, the prefill cos/sin rows, latent bookkeeping."""

    def __init__(self, bb, m: int, targets):
        self.bb, self.m = bb, int(m)
        self.targets = list(targets or [])
        self.n_layers = len(bb.lm.layers)
        self.past0 = 0
        self.T = 0
        self.rows = None
        self.anchor = None                        # anchor row relative to the prefill call
        self.q_pre: list = [None] * self.n_layers  # per layer [Hq, D] fp32
        self.cos = None                           # [T, D] fp32 prefill cos rows
        self.sin = None
        self.lat_len: list[int] = []
        self.lat_pos: list[int] = []
        self.note = ""
        self.how = ""
        self.in_prefill = False

    def lm_pre(self, module, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        kv = kwargs.get("past_key_values")
        pos = kwargs.get("position_ids")
        self.in_prefill = False
        if emb is None or emb.dim() != 3:
            return None
        T = int(emb.shape[1])
        if T > 1:
            if self.rows is None and not self.note:
                self.past0 = _kvlen(kv)
                self.T = T
                loc, note, how = locate_question(self.bb, emb[0].detach(), self.targets)
                if loc is None:
                    self.note = note
                    return None
                self.rows = (int(loc[0]), int(loc[1]))
                self.anchor = int(loc[1]) - 1
                self.how = how
                self.in_prefill = True
            return None
        if len(self.lat_len) < self.m and kv is not None and pos is not None:
            self.lat_len.append(_kvlen(kv))
            flat = pos.reshape(-1)
            self.lat_pos.append(int(flat[0]) if bool((flat == flat[0]).all()) else -1)
        return None

    def attn_post(self, li: int):
        def post(module, args, kwargs, output):
            if not self.in_prefill or self.anchor is None:
                return output
            hidden, pe, _cache = _call_parts(args, kwargs)
            if hidden is None or pe is None or int(hidden.shape[1]) == 1:
                return output
            with torch.no_grad():
                if li == 0:
                    cos, sin = pe
                    self.cos = cos[0].detach().float().clone()
                    self.sin = sin[0].detach().float().clone()
                h = hidden[:, self.anchor:self.anchor + 1]
                q = module.q_norm(module.q_proj(h).view(1, 1, -1, module.head_dim))[0, 0]     # [Hq, D]
                self.q_pre[li] = q.detach().float().clone()
            return output
        return post


@contextlib.contextmanager
def reasoner_probe(bb, m: int, stage: str, targets=None):
    if not enabled() or stage != "reasoner" or int(m) <= 0:
        yield None
        return
    cap = ReasonerCapture(bb, m, targets)
    bb._velh_cap = None
    handles = [bb.lm.register_forward_pre_hook(cap.lm_pre, with_kwargs=True)]
    handles += [layer.self_attn.register_forward_hook(cap.attn_post(li), with_kwargs=True)
                for li, layer in enumerate(bb.lm.layers)]
    try:
        yield cap
    finally:
        for h in handles:
            h.remove()
        bb._velh_cap = cap


def record_reasoner(engine, result: dict, *, m: int, case_index: int = -1) -> dict | None:
    """Right after the Reasoner append: freeze what the terminal step needs (bb._velh_R); keep the previous case's."""
    if not enabled():
        return None
    bb = engine._backbone
    cap = getattr(bb, "_velh_cap", None)
    bb._velh_cap = None
    bb._velh_prev = getattr(bb, "_velh_R", None)          # donor for the question-shuffle control
    bb._velh_R = None
    tag = f"[VELH:R] case {case_index}"

    def skip(why):
        bb._velh_R = {"skip": why, "case": int(case_index)}
        print(f"{tag}: SKIP ({why})", flush=True)
        return None

    if cap is None:
        return skip("no Reasoner capture")
    if cap.anchor is None:
        return skip(cap.note or "question anchor not located")
    m = int(m)
    if len(cap.lat_len) != m:
        return skip(f"latent steps observed {len(cap.lat_len)}/{m}")
    L0 = int(cap.lat_len[0])
    if cap.lat_len != list(range(L0, L0 + m)):
        return skip("latent cache lengths not contiguous")
    c0 = int(cap.lat_pos[0])
    if c0 < 0 or cap.lat_pos != list(range(c0, c0 + m)):
        return skip("latent positions not text-contiguous")
    if int(result["past_len"]) != L0 + m or int(result["pos_cursor"]) != c0 + m:
        return skip("result past_len / pos_cursor disagree with the latent bookkeeping (relays?)")
    if any(q is None for q in cap.q_pre) or cap.cos is None:
        return skip("anchor query / cos-sin rows not captured on every layer")
    vis = result.get("vis_cols")
    if not isinstance(vis, torch.Tensor) or int(vis.numel()) == 0:
        return skip("no visual columns")
    vis = vis.detach().cpu().long()
    if int(vis.min()) < cap.past0 or int(vis.max()) >= cap.past0 + cap.T:
        return skip("visual columns outside this Reasoner prefill")
    rel = vis - cap.past0
    R = {
        "case": int(case_index), "ep_col": L0 + m - 1, "ep_pos": c0 + m - 1,
        "anchor_col": cap.past0 + cap.anchor, "anchor_cos": cap.cos[cap.anchor].clone(), "anchor_sin": cap.sin[cap.anchor].clone(),
        "vis_cols": vis, "vis_cos": cap.cos[rel.to(cap.cos.device)].clone(), "vis_sin": cap.sin[rel.to(cap.sin.device)].clone(),
        "q_pre": [q.clone() for q in cap.q_pre], "q_rows": cap.rows, "how": cap.how, "L0": L0, "m": m,
    }
    bb._velh_R = R
    print(f"{tag} ep_col={R['ep_col']} ep_pos={R['ep_pos']} anchor_col={R['anchor_col']} "
          f"q_rows={cap.rows} n_vis={int(vis.numel())} how={cap.how}", flush=True)
    return R


# --------------------------------------------------------------------------- terminal


def _g(x) -> float:
    return float(f"{float(x):.5g}")


@contextlib.contextmanager
def terminal(engine, cache, position_cursor: int, *, case_index: int = -1):
    """In-place VELH on the Answerer's cache + attention-mass measurement during its call; yields stats."""
    cfg = config()
    md = cfg["mode"]
    bb = engine._backbone
    stats = {"case": int(case_index), "mode": md, "applied": False, "calls": 0, "rows": 0,
             "mass_ep": 0.0, "mass_anchor": 0.0, "mass_vis": 0.0}
    t0 = time.time()
    R = getattr(bb, "_velh_R", None)
    if R is None or "skip" in R:
        stats["skip"] = "no Reasoner record" if R is None else R["skip"]
        yield stats
        _finish(cfg, stats, t0)
        return
    L = _kvlen(cache)
    ep, anchor = int(R["ep_col"]), int(R["anchor_col"])
    if ep >= L or anchor >= L:
        stats["skip"] = f"columns ep={ep} anchor={anchor} beyond cache {L}"
        yield stats
        _finish(cfg, stats, t0)
        return
    pA = int(position_cursor) - 1
    stats.update({"ep_col": ep, "ep_pos": int(R["ep_pos"]), "anchor_col": anchor, "p_A": pA,
                  "shift": pA - int(R["ep_pos"]), "n_vis": int(R["vis_cols"].numel())})
    dev = bb.device
    vis = R["vis_cols"].to(dev)
    q_src = R["q_pre"]
    if md == "shuffle":
        P = getattr(bb, "_velh_prev", None)
        if P is None or "skip" in P or P.get("case") == R.get("case"):
            stats["skip"] = "no donor case for question-shuffle"
            yield stats
            _finish(cfg, stats, t0)
            return
        q_src = P["q_pre"]
        stats["donor_case"] = int(P["case"])
    cosA, sinA = text_cos_sin(bb, pA)
    cosR, sinR = text_cos_sin(bb, int(R["ep_pos"]))
    gaps, projs, mnorms, vnorms, dv, act = [], [], [], [], [], []
    with torch.no_grad():
        for li, layer in enumerate(bb.lm.layers):
            attn = layer.self_attn
            keys, values = cache.layers[li].keys, cache.layers[li].values
            kdt = keys.dtype
            Kv = keys[0, :, vis, :]                                               # [Hk, n, D]
            Vv = values[0, :, vis, :]
            Kbar = derotate(Kv, R["vis_cos"].to(dev), R["vis_sin"].to(dev))
            m_kv, _beta = question_read(q_src[li].to(dev), Kbar, Vv, attn.scaling, attn.num_key_value_groups)
            V_R = values[0, :, ep, :]                                             # [Hk, D]
            K_R = keys[0, :, ep, :]
            Vt, gap, proj, mn = bind(V_R, m_kv)
            gaps.append(gap.mean().item()); projs.append(proj.mean().item()); mnorms.append(mn.mean().item())
            vnorms.append(V_R.float().norm(dim=-1).mean().item())
            act.append(float((gap > 0).float().mean()))
            dv.append(float((Vt - V_R.float()).norm() / V_R.float().norm().clamp_min(1e-12)))
            if md in ("bind", "full", "shuffle"):
                values[0, :, ep, :] = Vt.to(values.dtype)
            if md in ("handoff", "full", "shuffle"):
                keys[0, :, ep, :] = rephase(K_R, cosR, sinR, cosA, sinA).to(kdt)
            if md == "direct":
                K_Q = keys[0, :, anchor, :]
                keys[0, :, ep, :] = rephase(K_Q, R["anchor_cos"].to(dev), R["anchor_sin"].to(dev), cosA, sinA).to(kdt)
                values[0, :, ep, :] = m_kv.to(values.dtype)
    stats.update({"applied": md != "diag", "gap_mean": _g(sum(gaps) / len(gaps)),
                  "act_rate": _g(sum(act) / len(act)), "uV_mean": _g(sum(projs) / len(projs)),
                  "m_norm": _g(sum(mnorms) / len(mnorms)), "vR_norm": _g(sum(vnorms) / len(vnorms)),
                  "dv_rel": _g(sum(dv) / len(dv)), "act_by_layer": [_g(a) for a in act]})

    # -- Answerer attention mass on the endpoint / anchor / visual columns (rows = gen)
    acc = {"ep": 0.0, "anchor": 0.0, "vis": 0.0, "n": 0}

    def make_post(li):
        def post(module, args, kwargs, output):
            hidden, pe, c = _call_parts(args, kwargs)
            if hidden is None or pe is None or c is None:
                return output
            T = int(hidden.shape[1])
            rows = torch.tensor([T - 1], device=hidden.device)
            with torch.no_grad():
                a = question_rows_alpha(module, hidden, pe, c.layers[li].keys, rows)[0]      # [L]
                acc["ep"] += float(a[ep]); acc["anchor"] += float(a[anchor]); acc["vis"] += float(a[vis].sum())
                acc["n"] += 1
            return output
        return post

    handles = [layer.self_attn.register_forward_hook(make_post(li), with_kwargs=True)
               for li, layer in enumerate(bb.lm.layers)]
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()
        n = max(acc["n"], 1)
        stats.update({"calls": acc["n"], "rows": acc["n"] // max(len(bb.lm.layers), 1),
                      "mass_ep": _g(acc["ep"] / n), "mass_anchor": _g(acc["anchor"] / n), "mass_vis": _g(acc["vis"] / n)})
        _finish(cfg, stats, t0)


def _finish(cfg, stats, t0):
    stats["sec"] = round(time.time() - t0, 2)
    if cfg["log"]:
        with open(cfg["log"], "a") as fh:
            fh.write(json.dumps(stats) + "\n")


def summary(stats: dict) -> str:
    if "skip" in stats:
        return f"mode={stats['mode']} SKIP ({stats['skip']})"
    return (f"mode={stats['mode']} ep={stats['ep_col']}@{stats['ep_pos']}->{stats['p_A']} (shift {stats['shift']}) "
            f"anchor={stats['anchor_col']} n_vis={stats['n_vis']} act_rate={stats['act_rate']:.3f} gap={stats['gap_mean']:.4g} "
            f"uV={stats['uV_mean']:.4g} |m|={stats['m_norm']:.4g} |V_R|={stats['vR_norm']:.4g} dv_rel={stats['dv_rel']:.3g} "
            f"mass ep={stats['mass_ep']:.3e} anchor={stats['mass_anchor']:.3e} vis={stats['mass_vis']:.4f} "
            f"rows={stats['rows']} sec={stats['sec']}" + (f" donor={stats['donor_case']}" if "donor_case" in stats else ""))
