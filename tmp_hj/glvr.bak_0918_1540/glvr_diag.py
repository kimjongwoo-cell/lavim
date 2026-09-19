"""GLVR canonical-A diagnostic (tmp_hj/glvr).  Original pipeline files untouched.

Runs three measurements per case and writes one jsonl line:

  A. provenance masses at the terminal Answerer-prompt row, per layer:
       sink (column 0) / H_V (all roles' visual KV) / H_C (inherited non-visual,
       incl. latent) / P_A (Answerer prompt).
  B. the grounded latent trajectory's conditional visual reads u_{V,k} captured
     during the one grounding replay: pairwise cosine / JSD of the routing rows,
     and the distance between the relayed visual distribution and the Answerer's
     own native visual distribution.
  C. a lambda sweep of the source-local relay at the terminal row:
       o' = o_nat + lambda * (rho_B * u_relay - m_B),      B = H_C
     reporting the answer, the candidate margins and |o'-o|/|o| per lambda.

Env
  VLMAS_GLVR=<out.jsonl>     enable, append one record per case
  VLMAS_GLVR_LAMBDAS=0,0.25,0.5,0.75,1
  VLMAS_GLVR_DONOR=hc|hc_nolatent|latent      donor domain for the sweep (default hc)
  VLMAS_GLVR_GEN=0           skip re-decoding (scores only)
Requires the grounding replay to run, i.e. VLMAS_RNLCR=ground (or fulr).
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time

import torch

# --------------------------------------------------------------------------- config


def _env(name: str, default: str = "") -> str:
    """Environment lookup where an empty value counts as unset."""
    val = os.environ.get(name, "").strip()
    return val or default


def enabled() -> bool:
    return bool(_env("VLMAS_GLVR"))


def lambdas() -> list[float]:
    raw = _env("VLMAS_GLVR_LAMBDAS", "0,0.25,0.5,0.75,1")
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or [0.0, 1.0]


# --------------------------------------------------------------------------- groups


def _as_cols(t) -> torch.Tensor:
    if t is None:
        return torch.empty(0, dtype=torch.long)
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().long().reshape(-1)
    return torch.tensor(list(t), dtype=torch.long)


def _minus(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if b.numel() == 0:
        return a
    mask = torch.ones(a.numel(), dtype=torch.bool)
    lookup = set(b.tolist())
    for i, v in enumerate(a.tolist()):
        if v in lookup:
            mask[i] = False
    return a[mask]


def provenance_groups(*, role_visual: dict, role_latent: dict, base_len: int,
                      sink_col: int = 0) -> dict[str, torch.Tensor]:
    """Split the inherited prefix [0, base_len) into sink / H_V / H_C, plus latent."""
    vis = [c[c < base_len] for c in (_as_cols(v) for v in (role_visual or {}).values())]
    lat = [c[c < base_len] for c in (_as_cols(v) for v in (role_latent or {}).values())]
    H_V = torch.unique(torch.cat(vis)) if vis else torch.empty(0, dtype=torch.long)
    LAT = torch.unique(torch.cat(lat)) if lat else torch.empty(0, dtype=torch.long)
    allc = torch.arange(base_len, dtype=torch.long)
    sink = torch.tensor([sink_col], dtype=torch.long)
    H_C = _minus(_minus(allc, H_V), sink)
    return {"sink": sink, "H_V": H_V, "H_C": H_C, "latent": LAT,
            "H_C_nolatent": _minus(H_C, LAT)}


def donor_cols(groups: dict[str, torch.Tensor]) -> torch.Tensor:
    kind = _env("VLMAS_GLVR_DONOR", "hc")
    return {"hc": groups["H_C"], "hc_nolatent": groups["H_C_nolatent"],
            "latent": groups["latent"]}.get(kind, groups["H_C"])


# --------------------------------------------------------------------------- math


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1)


def _bc(p: torch.Tensor, q: torch.Tensor) -> float:
    """Bhattacharyya coefficient of two distributions over the same support."""
    return float((p.clamp_min(0).sqrt() * q.clamp_min(0).sqrt()).sum())


def _jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    m = 0.5 * (p + q)
    def _kl(a, b):
        a = a.clamp_min(eps); b = b.clamp_min(eps)
        return float((a * (a.log() - b.log())).sum())
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def trajectory_stats(P: torch.Tensor, U: torch.Tensor) -> dict:
    """P [m, Nv] routing rows (layer/head averaged), U [m, D] conditional reads."""
    m = P.shape[0]
    cos, bc, jsd = [], [], []
    for i in range(m):
        for j in range(i + 1, m):
            cos.append(float(torch.nn.functional.cosine_similarity(U[i], U[j], dim=0)))
            bc.append(_bc(P[i], P[j]))
            jsd.append(_jsd(P[i], P[j]))
    sv = torch.linalg.svdvals(U.float())
    p = (sv / sv.sum().clamp_min(1e-12))
    eff_rank = float(torch.exp(-(p * (p.clamp_min(1e-12)).log()).sum()))
    return {"m": int(m), "cos_mean": float(sum(cos) / len(cos)) if cos else None,
            "cos_min": float(min(cos)) if cos else None,
            "bc_mean": float(sum(bc) / len(bc)) if bc else None,
            "jsd_mean": float(sum(jsd) / len(jsd)) if jsd else None,
            "eff_rank": eff_rank}


# --------------------------------------------------------------------------- per-head alpha


def rows_alpha_per_head(module, hidden, pe, keys, lo: int, hi: int) -> torch.Tensor:
    """Native attention of rows [lo, hi) over all cached columns, per query head -> [H, Ts, L].

    Same math as memory.lu.question_rows_alpha, which averages over heads; GLVR needs the
    per-head rows because the relay is head-local.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

    L = int(keys.shape[2])
    T = int(hidden.shape[1])
    Hd = module.head_dim
    G = module.num_key_value_groups
    cos, sin = pe
    rows = torch.arange(int(lo), int(hi), device=hidden.device)
    h = hidden.index_select(1, rows)
    Ts = int(rows.numel())
    q = module.q_norm(module.q_proj(h).view(1, Ts, -1, Hd)).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos.index_select(1, rows), sin.index_select(1, rows))
    K = repeat_kv(keys, G)
    scores = torch.matmul(q.float(), K.float().transpose(-2, -1)) * module.scaling   # [1,H,Ts,L]
    past = L - T
    col = torch.arange(L, device=keys.device)[None, None, None, :]
    row = (past + rows)[None, None, :, None]
    scores = scores.masked_fill(col > row, float("-inf"))
    return torch.softmax(scores, dim=-1)[0]                                          # [H, Ts, L]


# --------------------------------------------------------------------------- replay capture


class ReplayCapture:
    """Captures each grounded latent row's visual routing during the replay forward.

    Installed around the m-row replay: for every decoder layer it recomputes the
    row logits against the cached keys (SDPA does not expose probabilities), keeps
    the visual columns only and reduces them to (rho_V, u_V) per query head.
    """

    def __init__(self, bb, m: int, vis_cols: torch.Tensor, prefix_len: int):
        self.bb = bb
        self.m = int(m)
        self.vis = vis_cols
        self.prefix_len = int(prefix_len)
        self.rho = []            # per layer: [H, m]
        self.u = []              # per layer: [H, m, D]
        self.route = []          # per layer: [H, m, Nv] (kept only if small)
        self.n_layers = 0
        self.note = None

    def attn_post(self, li: int):
        def post(module, args, kwargs, output):
            from memory.lu import _call_parts
            hidden, pe, cache = _call_parts(args, kwargs)
            if hidden is None or pe is None or cache is None:
                return output
            if int(hidden.shape[1]) != self.m:
                return output
            try:
                with torch.no_grad():
                    keys = cache.layers[li].keys
                    vals = cache.layers[li].values
                    a = rows_alpha_per_head(module, hidden, pe, keys, 0, self.m).float()  # [H, m, N]
                    vis = self.vis.to(a.device)
                    av = a[..., vis]                                # [H, m, Nv]
                    rho = av.sum(-1)                                # [H, m]
                    V = vals[0] if vals.dim() == 4 else vals        # [Hkv, N, D]
                    Vv = V[..., vis, :].float()                     # [Hkv, Nv, D]
                    if Vv.shape[0] != av.shape[0]:
                        rep = av.shape[0] // max(1, Vv.shape[0])
                        Vv = Vv.repeat_interleave(rep, dim=0)
                    mv = torch.einsum("hmk,hkd->hmd", av, Vv)       # [H, m, D]
                    u = mv / rho.clamp_min(1e-12).unsqueeze(-1)
                    self.rho.append(rho.cpu())
                    self.u.append(u.cpu())
                    self.route.append((av / rho.clamp_min(1e-12).unsqueeze(-1)).mean(0).cpu())
                    self.n_layers += 1
            except Exception as exc:  # noqa: BLE001 - diagnostics never break the run
                self.note = f"{type(exc).__name__}: {exc}"
            return output
        return post


@contextlib.contextmanager
def capture_replay(bb, m: int, vis_cols: torch.Tensor, prefix_len: int):
    cap = ReplayCapture(bb, m, vis_cols, prefix_len)
    handles = [layer.self_attn.register_forward_hook(cap.attn_post(li), with_kwargs=True)
               for li, layer in enumerate(bb.lm.layers)]
    try:
        yield cap
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- terminal row


class TerminalRelay:
    """Measures the terminal Answerer row and optionally applies the relay.

    The delta is head-local and pre-o_proj; o_proj is linear and has no bias in
    this backbone, so applying it to the delta and adding the result to the layer
    output is exact.
    """

    def __init__(self, bb, groups: dict, U_V: list, lam: float = 0.0,
                 latent_cols: torch.Tensor | None = None, n_prompt: int = 0):
        self.bb = bb
        self.groups = groups
        self.U_V = U_V                    # per layer [H, m, D] or None
        self.lam = float(lam)
        self.latent = latent_cols if latent_cols is not None else groups["latent"]
        self.n_prompt = int(n_prompt)
        self.stats = []                   # per layer dict
        self.rows_applied = []
        self.note = None

    def _row_start(self, rows: int) -> int:
        """First row of this call that the relay applies to.

        The decision row is the last prompt row; every row after it (continuation
        rows when scoring, generated rows when decoding) also reads the relayed
        message, mirroring FULR's rows=gen.
        """
        if self.n_prompt and rows >= self.n_prompt:
            return max(0, self.n_prompt - 1)
        return 0

    def attn_post(self, li: int):
        def post(module, args, kwargs, output):
            from memory.lu import _call_parts
            hidden, pe, cache = _call_parts(args, kwargs)
            if hidden is None or pe is None or cache is None:
                return output
            rows = int(hidden.shape[1])
            r0 = self._row_start(rows)
            try:
                with torch.no_grad():
                    keys = cache.layers[li].keys
                    vals = cache.layers[li].values
                    A = rows_alpha_per_head(module, hidden, pe, keys, r0, rows).float()  # [H, R, N]
                    a = A[:, 0, :]              # decision row (first applied row)
                    V = vals[0] if vals.dim() == 4 else vals
                    Vd = V.float()
                    if Vd.shape[0] != a.shape[0]:
                        rep = a.shape[0] // max(1, Vd.shape[0])
                        Vd = Vd.repeat_interleave(rep, dim=0)
                    rec = {"layer": li}
                    for name in ("sink", "H_V", "H_C", "latent", "H_C_nolatent"):
                        idx = self.groups[name].to(a.device)
                        rec[f"rho_{name}"] = float(a[:, idx].sum(-1).mean()) if idx.numel() else 0.0
                    self.stats.append(rec)
                    if self.lam and self.U_V and li < len(self.U_V) and self.U_V[li] is not None:
                        donor = donor_cols(self.groups).to(a.device)
                        lat = self.latent.to(a.device)
                        U = self.U_V[li].to(a.device).float()            # [H, m, D]
                        if U.shape[0] != a.shape[0]:
                            rep = a.shape[0] // max(1, U.shape[0])
                            U = U.repeat_interleave(rep, dim=0)
                        # every applied row gets its own weights, not just the decision row
                        rho_B_all = A[:, :, donor].sum(-1)                        # [H, R]
                        m_B_all = torch.einsum("hrk,hkd->hrd", A[:, :, donor], Vd[:, donor, :])
                        al_all = A[:, :, lat]
                        g_all = al_all / al_all.sum(-1, keepdim=True).clamp_min(1e-12)
                        u_all = torch.einsum("hrm,hmd->hrd", g_all, U)            # [H, R, D]
                        delta = self.lam * (rho_B_all.unsqueeze(-1) * u_all - m_B_all)  # [H, R, D]
                        out = output[0] if isinstance(output, tuple) else output
                        R = delta.shape[1]
                        flat = delta.permute(1, 0, 2).reshape(1, R, -1).to(out.dtype)
                        add = module.o_proj(flat)
                        rec["delta_rel"] = float(add.norm() / out[:, r0:, :].norm().clamp_min(1e-9))
                        rec["rows"] = int(R)
                        out[:, r0:, :] = out[:, r0:, :] + add
                        self.rows_applied.append(int(R))
                        if isinstance(output, tuple):
                            output = (out,) + tuple(output[1:])
                        else:
                            output = out
            except Exception as exc:  # noqa: BLE001
                self.note = f"{type(exc).__name__}: {exc}"
            return output
        return post


@contextlib.contextmanager
def terminal_relay(bb, groups: dict, U_V: list, lam: float = 0.0, n_prompt: int = 0):
    tr = TerminalRelay(bb, groups, U_V, lam, n_prompt=n_prompt)
    handles = [layer.self_attn.register_forward_hook(tr.attn_post(li), with_kwargs=True)
               for li, layer in enumerate(bb.lm.layers)]
    try:
        yield tr
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- record


def summarize_layers(stats: list[dict], layers: tuple[int, ...] = (18, 27, 35)) -> dict:
    by = {int(s["layer"]): s for s in stats}
    out = {}
    for li in layers:
        s = by.get(li)
        if s:
            out[str(li)] = {k: round(v, 5) for k, v in s.items() if k.startswith("rho_")}
    keys = [k for k in (stats[0] if stats else {}) if k.startswith("rho_")]
    out["mean"] = {k: round(float(sum(s[k] for s in stats) / len(stats)), 5) for k in keys} if stats else {}
    return out


def write(rec: dict) -> None:
    path = _env("VLMAS_GLVR")
    if not path:
        return
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- entry points


def on_reasoner(engine, result: dict, *, m: int, case_index: int) -> None:
    """Install the replay capture around memory.rnlcr.apply_reasoner.

    Called by the sitecustomize wrapper; stores the per-layer conditional visual
    reads on the backbone as ``_glvr`` so the Answerer-side entry can use them.
    """
    bb = engine._backbone
    vis = _as_cols(result.get("vis_cols"))
    L1 = int(result.get("past_len") or 0)
    L0 = max(0, L1 - int(m))
    vis = vis[vis < L0]
    bb._glvr = {"case": int(case_index), "m": int(m), "n_vis": int(vis.numel()),
                "vis": vis, "lat": torch.arange(L0, L1, dtype=torch.long), "pre_len": L0,
                "U": None, "rho": None, "route": None, "note": None}
    if vis.numel() == 0:
        bb._glvr["note"] = "no visual columns"
        return bb._glvr
    return bb._glvr


def finish_reasoner(bb, cap) -> None:
    """Reduce a finished ReplayCapture into the ``_glvr`` record."""
    rec = getattr(bb, "_glvr", None)
    if rec is None or cap is None:
        return
    rec["n_layers"] = int(cap.n_layers)
    rec["note"] = cap.note
    if cap.u:
        rec["U"] = cap.u                                   # list per layer [H, m, D]
        rec["rho"] = torch.stack([r.mean(0) for r in cap.rho])      # [L, m]
        rec["route"] = torch.stack(cap.route)                        # [L, m, Nv]


def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        case_index: int, max_new_tokens: int = 512, **_ignored) -> None:
    """Answerer-boundary diagnostic: provenance masses, trajectory stats, lambda sweep."""
    if not enabled():
        return
    import traceback
    from copy import deepcopy

    from vision_text_mas.latent_terminal import (_assistant_prompt, generate_terminal_json,
                                                 score_terminal_continuation)

    t0 = time.time()
    bb = engine._backbone
    base_len = int(bb._kv_len(cache))
    tok = bb.processor.tokenizer
    prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt,
                               user_prompt=user_prompt, json_prefix=json_prefix)
    n_prompt = int(tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])
    gl = getattr(bb, "_glvr", None) or {}
    role_visual = getattr(engine, "_alsi_role_visual_cols", {}) or {}
    role_latent = getattr(engine, "_alsi_role_latent_cols", {}) or {}
    source = "role_registry"
    if not role_visual and gl.get("vis") is not None and gl["vis"].numel():
        # the registry is empty in this arm; use the columns the grounding replay saw
        role_visual = {"replay": gl["vis"]}
        role_latent = {"replay": gl.get("lat", torch.empty(0, dtype=torch.long))}
        source = "replay_bookkeeping"
    groups = provenance_groups(role_visual=role_visual, role_latent=role_latent,
                               base_len=base_len)
    U_V = gl.get("U")
    rec = {"case": int(case_index), "base_len": base_len, "n_prompt": n_prompt,
           "donor": _env("VLMAS_GLVR_DONOR", "hc"), "cols_from": source,
           "n_cols": {k: int(v.numel()) for k, v in groups.items()},
           "replay": {"n_layers": gl.get("n_layers"), "n_vis": gl.get("n_vis"),
                      "note": gl.get("note")}}
    try:
        # --- B. trajectory statistics from the grounding replay
        if gl.get("route") is not None:
            P = gl["route"].mean(0)                          # [m, Nv] layer mean
            Umean = torch.stack([u.mean(0) for u in U_V]).mean(0) if U_V else None
            if Umean is not None:
                rec["trajectory"] = trajectory_stats(P, Umean)
            rec["rho_V_replay"] = {"per_layer_mean": [round(float(x), 5) for x in gl["rho"].mean(1)][:40],
                                   "mean": round(float(gl["rho"].mean()), 5)}
        # --- A + C. terminal row masses and the lambda sweep
        cands = [str(c).strip() for c in (getattr(engine, "_ver_candidates", None) or ()) if str(c).strip()]
        tmpl = _env("VLMAS_GLVR_CONT_TMPL", ' "{y}",')
        do_gen = _env("VLMAS_GLVR_GEN", "1") != "0"
        maxnew = min(int(max_new_tokens), int(_env("VLMAS_GLVR_MAXNEW", "192")))
        rec["conditions"] = []
        for lam in lambdas():
            row = {"lambda": lam}
            c = deepcopy(cache)
            try:
                with terminal_relay(bb, groups, U_V or [], lam, n_prompt=n_prompt) as tr:
                    if do_gen:
                        row["text"] = generate_terminal_json(
                            backbone=bb, cache=c, position_cursor=position_cursor,
                            system_prompt=system_prompt, user_prompt=user_prompt,
                            json_prefix=json_prefix, max_new_tokens=maxnew,
                            temperature=0.0, top_p=1.0, do_sample=False)
                    if lam == lambdas()[0]:
                        rec["mass"] = summarize_layers(tr.stats)
                    deltas = [s["delta_rel"] for s in tr.stats if "delta_rel" in s]
                    if deltas:
                        row["delta_rel_mean"] = round(float(sum(deltas) / len(deltas)), 6)
                    row["note"] = tr.note
            finally:
                del c
            if cands:
                scores = {}
                c = deepcopy(cache)
                try:
                    with terminal_relay(bb, groups, U_V or [], lam, n_prompt=n_prompt):
                        for y in cands:
                            lp, _n = score_terminal_continuation(
                                backbone=bb, cache=c, position_cursor=position_cursor,
                                system_prompt=system_prompt, user_prompt=user_prompt,
                                json_prefix=json_prefix, continuation=tmpl.format(y=y))
                            scores[y] = float(lp)
                            c.crop(base_len)
                finally:
                    del c
                top = max(scores, key=scores.get)
                rest = sorted((v for k, v in scores.items() if k != top), reverse=True)
                row["argmax"] = top
                row["margin"] = round(float(scores[top] - rest[0]), 4) if rest else None
            rec["conditions"].append(row)
    except Exception as exc:  # noqa: BLE001 - diagnostics never break the case
        rec["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    rec["sec"] = round(time.time() - t0, 2)
    write(rec)
    mass = (rec.get("mass") or {}).get("mean", {})
    conds = " ".join(f"{r['lambda']}:{r.get('argmax') or r.get('text', '')[:12]}"
                     for r in rec.get("conditions", []))
    print(f"[GLVR] case {case_index} rho_HC={mass.get('rho_H_C')} rho_HV={mass.get('rho_H_V')} "
          f"sink={mass.get('rho_sink')} traj_cos={(rec.get('trajectory') or {}).get('cos_mean')} "
          f"{conds} sec={rec['sec']}", flush=True)
