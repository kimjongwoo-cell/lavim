"""Latent Unsilencing (LU), Phase 1 -- Notion "[Planned] Latent Unsilencing -- Reasoner hidden-state rescue"
(3ddd771b-c9e2-81e1), following arXiv 2605.02735 ("Visual Latents Know More Than They Say").

Opt-in.  VLMAS_LU unset or "base" = no hook, no extra forward, byte-identical pipeline.

What is optimised
  Z = the Reasoner's m native latent INPUT embeddings e_1..e_m (the realigned vectors the LM consumed
  at its latent steps = result["latent_trajectory"]), jointly, with every model weight and every
  earlier KV column frozen.  Post-norm hidden states and K/V are never optimised directly.  After
  the optimisation the cache is cropped to the pre-latent length L0 and the m rows are replayed ONCE
  (one causal m-row forward) so the Reasoner latent KV is regenerated consistently; the turn close
  and the Answerer then run unchanged on that cache.

Stage I  (modes s1, s12) -- query-guided contrastive latent-visual alignment (paper Eq. 2-4)
  relevance  s_n = mean_{q in I_Q} A_{q,n}: native attention of the ORIGINAL question rows I_Q of the
             Reasoner prompt to the currently visible visual columns n, mean over ALL layers and heads
             (recomputed in fp32 from q/K inside hooks during the Reasoner prefill; I_Q is located by
             recovering the prefill token ids from the input embeddings (exact table lookup, visual rows
             -> <|image_pad|>) and running the backbone's _locate_role_spans, whose decode/offset
             fallback survives BPE merges such as "?\n").
  sets       columns ranked by s_n descending; latent k gets the non-overlapping chunks
             P_k = rank_desc[k*pos : (k+1)*pos]   and   N_k = rank_asc[k*neg : (k+1)*neg].
  space      v_n = projected visual input embedding (the merger-output row that entered the LM),
             i.e. the same input space as e_k.  sim = cosine.
  loss       L = -mean_k log( beta * sum_{v in P_k} exp(sim/tau) / sum_{v in P_k u N_k} exp(sim/tau) ),
             beta = (pos+neg)/pos.  Adam for N_sft steps (lr = VLMAS_LU_S1_LR x per-dim RMS of Z_0).
Stage II (mode s12) -- confidence-progression reward (paper Eq. 5-6)
  E_k   = top-delta entropy at latent row k: post-norm hidden -> lm_head -> softmax (fp32) -> keep the
          delta largest probabilities, renormalise, Shannon entropy.
  R     = mean_{k<K} max(0, E_k - E_{k+1})
  NES   H_{i+1} = H_i + (alpha / sigma_i^2) * R(H_i + eps_i) * eps_i,  eps_i ~ N(0, sigma_i^2 I),
          sigma_i = sigma_0 * decay^i, N_rl steps.  The best-reward H seen (H_0, every perturbed
          candidate, the final H) is retained.  No gold label and no candidate restriction anywhere.
  Units: sigma_0 = VLMAS_LU_NES_SIGMA x rms(Z), alpha = VLMAS_LU_NES_ALPHA x rms(Z)^2, so that
          alpha / sigma_i^2 = ALPHA / (SIGMA decay^i)^2 in per-dim-RMS units.

Env
  VLMAS_LU=base|s1|s12|replay      replay = Z untouched, crop + replay only (numerical identity check)
  VLMAS_LU_POS=2  VLMAS_LU_NEG=4  VLMAS_LU_S1_STEPS=5  VLMAS_LU_S1_LR=0.05  VLMAS_LU_TAU=0.1
  VLMAS_LU_TOPK=20  VLMAS_LU_S2_STEPS=15  VLMAS_LU_NES_SIGMA=0.1  VLMAS_LU_NES_DECAY=0.9
  VLMAS_LU_NES_ALPHA=0.01  VLMAS_LU_LOG=<jsonl>  (per-case record)
Implementation choices the paper leaves open (all reported in the log): tau, delta, Adam lr, NES
sigma/alpha/decay, embedding-match location of I_Q, no norm constraint on Z.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time

import torch

MODES = ("s1", "s12", "replay")


# --------------------------------------------------------------------------- config


def mode() -> str:
    v = os.environ.get("VLMAS_LU", "").strip().lower()
    return v if v in MODES else ""


def enabled() -> bool:
    return mode() != ""


def config() -> dict:
    e = os.environ.get
    return {
        "mode": mode(),
        "pos": int(e("VLMAS_LU_POS", "2")), "neg": int(e("VLMAS_LU_NEG", "4")),
        "s1_steps": int(e("VLMAS_LU_S1_STEPS", "5")), "s1_lr": float(e("VLMAS_LU_S1_LR", "0.05")),
        "tau": float(e("VLMAS_LU_TAU", "0.1")),
        "topk": int(e("VLMAS_LU_TOPK", "20")), "s2_steps": int(e("VLMAS_LU_S2_STEPS", "15")),
        "sigma": float(e("VLMAS_LU_NES_SIGMA", "0.1")), "decay": float(e("VLMAS_LU_NES_DECAY", "0.9")),
        "alpha": float(e("VLMAS_LU_NES_ALPHA", "0.01")),
        "log": e("VLMAS_LU_LOG", "").strip(),
    }


# --------------------------------------------------------------------------- pure


def assign_sets(score: torch.Tensor, K: int, pos: int, neg: int) -> tuple[torch.Tensor, torch.Tensor]:
    """score [n] -> (P [K,pos], N [K,neg]) column indices: non-overlapping chunks from the top / bottom."""
    n = int(score.numel())
    if n < K * (pos + neg):
        raise ValueError(f"need >= K*(pos+neg) = {K * (pos + neg)} visual columns, have {n}")
    order = torch.argsort(score.float(), descending=True, stable=True)
    P = order[: K * pos].view(K, pos)
    N = order.flip(0)[: K * neg].view(K, neg)
    return P, N


def cosine_sim(Z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """[K,d] x [n,d] -> [K,n] cosine similarities (fp32)."""
    zn = torch.nn.functional.normalize(Z.float(), dim=-1)
    vn = torch.nn.functional.normalize(V.float(), dim=-1)
    return zn @ vn.T


def contrastive_loss(Z, V, P, N, tau: float) -> torch.Tensor:
    """Paper Eq. 4 with beta = (pos+neg)/pos; mean over the K latents."""
    S = cosine_sim(Z, V) / float(tau)
    sp = S.gather(1, P.to(S.device))
    sn = S.gather(1, N.to(S.device))
    beta = (int(P.shape[1]) + int(N.shape[1])) / int(P.shape[1])
    num = math.log(beta) + torch.logsumexp(sp, dim=1)
    den = torch.logsumexp(torch.cat([sp, sn], dim=1), dim=1)
    return -(num - den).mean()


def stage1(Z0, V, P, N, steps: int, lr_rel: float, tau: float):
    """Adam on Z (fp32) for `steps` steps; returns (Z, losses[steps+1]) with the final loss appended."""
    with torch.enable_grad():
        Z = Z0.detach().float().clone().requires_grad_(True)
        rms = float(Z0.float().pow(2).mean().sqrt())
        opt = torch.optim.Adam([Z], lr=lr_rel * rms)
        losses = []
        for _ in range(int(steps)):
            opt.zero_grad()
            loss = contrastive_loss(Z, V, P, N, tau)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        losses.append(float(contrastive_loss(Z, V, P, N, tau)))
    return Z.detach(), losses


def top_delta_entropy(logits: torch.Tensor, delta: int) -> torch.Tensor:
    """[..., vocab] -> [...]: entropy of the renormalised top-delta probabilities (nats)."""
    p = torch.softmax(logits.float(), dim=-1)
    top = p.topk(min(int(delta), int(p.shape[-1])), dim=-1).values
    pt = top / top.sum(-1, keepdim=True)
    return -(pt * pt.clamp_min(1e-30).log()).sum(-1)


def reward(E: torch.Tensor) -> torch.Tensor:
    """Paper Eq. 5: mean_k max(0, E_k - E_{k+1}) over consecutive latent rows."""
    if int(E.numel()) < 2:
        return E.new_zeros(())
    return torch.clamp(E[:-1] - E[1:], min=0).mean()


def nes(Z0, eval_fn, steps: int, sigma_rel: float, decay: float, alpha_rel: float, gen):
    """Paper Eq. 6 (single Gaussian perturbation per step, geometric sigma decay), best-reward retained.

    Units: rms = per-dim RMS of Z0; sigma_i = sigma_rel*decay^i*rms; alpha = alpha_rel*rms^2, so the
    update H += (alpha/sigma_i^2) R eps == (alpha_rel / s_i^2) R eps with s_i the relative sigma.
    """
    rms = float(Z0.float().pow(2).mean().sqrt())
    H = Z0.detach().float().clone()
    R0 = float(eval_fn(H))
    best_R, best_H, best_it = R0, H.clone(), -1
    hist = [R0]
    for i in range(int(steps)):
        s = float(sigma_rel) * (float(decay) ** i)
        eps = torch.randn(H.shape, generator=gen, device=H.device, dtype=H.dtype) * (s * rms)
        Ht = H + eps
        R = float(eval_fn(Ht))
        hist.append(R)
        if R > best_R:
            best_R, best_H, best_it = R, Ht.clone(), i
        H = H + (float(alpha_rel) / (s * s)) * R * eps
    Rf = float(eval_fn(H))
    hist.append(Rf)
    if Rf > best_R:
        best_R, best_H, best_it = Rf, H.clone(), int(steps)
    return best_H, {"R0": R0, "hist": hist, "best": best_R, "best_iter": best_it}


def locate_rows(E: torch.Tensor, targets) -> tuple[int, int] | None:
    """First exact match of any target embedding sequence [Lq,d] as contiguous rows of E [T,d]."""
    T = int(E.shape[0])
    for tgt in targets:
        Lq = int(tgt.shape[0])
        if Lq == 0 or Lq > T:
            continue
        cand = (E == tgt[0].to(E.dtype)).all(-1).nonzero(as_tuple=True)[0]
        for i in cand.tolist():
            if i + Lq <= T and bool((E[i:i + Lq] == tgt.to(E.dtype)).all()):
                return i, i + Lq
    return None


def recover_ids(X: torch.Tensor, E: torch.Tensor, row_chunk: int = 1024, vocab_chunk: int = 32768) -> torch.Tensor:
    """[T,d] input embeddings -> [T] token ids (-1 where no table row is bit-identical, e.g. visual rows).

    Nearest table row by squared distance (fp32, chunked over rows and vocabulary), then exact check.
    """
    T = int(X.shape[0])
    V = int(E.shape[0])
    ids = torch.full((T,), -1, dtype=torch.long, device=X.device)
    for r0 in range(0, T, row_chunk):
        x = X[r0:r0 + row_chunk].float()
        best = torch.full((int(x.shape[0]),), float("-inf"), device=X.device)
        arg = torch.zeros(int(x.shape[0]), dtype=torch.long, device=X.device)
        for v0 in range(0, V, vocab_chunk):
            e = E[v0:v0 + vocab_chunk].float()
            sc = x @ e.T - 0.5 * e.pow(2).sum(-1)[None, :]          # = -0.5*||x-e||^2 + const
            m, a = sc.max(-1)
            upd = m > best
            best = torch.where(upd, m, best)
            arg = torch.where(upd, a + v0, arg)
        ok = (E[arg].to(X.dtype) == X[r0:r0 + row_chunk]).all(-1)
        ids[r0:r0 + row_chunk] = torch.where(ok, arg, torch.full_like(arg, -1))
    return ids


def locate_ids(id_list: list, targets, tok) -> list:
    """Fallback span search (backbone-free): contiguous id-subsequence match of ' '+t / '\n'+t / t."""
    spans = []
    n = len(id_list)
    for t in targets:
        for v in (" " + t, "\n" + t, t):
            try:
                seq = tok.encode(v, add_special_tokens=False)
            except Exception:
                seq = []
            L = len(seq)
            if L == 0 or L > n:
                continue
            hit = next((i for i in range(0, n - L + 1) if id_list[i:i + L] == seq), None)
            if hit is not None:
                spans.append((hit, hit + L))
                break
    return spans


def locate_question(bb, X: torch.Tensor, targets) -> tuple[tuple[int, int] | None, str, str]:
    """(span, note, how): rows of the shortest matching target inside prefill embeddings X [T,d].

    Token ids are recovered from the embeddings (exact table lookup; visual rows -> <|image_pad|>)
    and the backbone's _locate_role_spans (decode/offset fallback, survives BPE merges such as
    "?\n") finds the span; without a backbone locator a plain id-subsequence match is used.
    """
    targets = [str(t).strip() for t in (targets or []) if t is not None and str(t).strip()]
    if not targets:
        return None, "no question target for this role", ""
    tok = bb.processor.tokenizer
    with torch.no_grad():
        ids = recover_ids(X, bb.lm.embed_tokens.weight.detach())
    n_text = int((ids >= 0).sum())
    try:
        pad = int(tok.convert_tokens_to_ids("<|image_pad|>"))
    except Exception:
        pad = -1
    if pad is None or pad < 0:
        pad = 0
    ids_full = torch.where(ids < 0, torch.full_like(ids, pad), ids)
    loc_fn = getattr(bb, "_locate_role_spans", None)
    if callable(loc_fn):
        spans = loc_fn(ids_full.cpu(), 0, int(ids_full.numel()), list(targets))
        how = "backbone"
    else:
        spans = locate_ids(ids_full.tolist(), list(targets), tok)
        how = "fallback"
    spans = [(int(a), int(b)) for a, b in spans if int(b) > int(a)]
    if not spans:
        return None, f"question rows not found in the Reasoner prefill ({how}, text rows {n_text}/{int(ids.numel())})", how
    a, b = min(spans, key=lambda sp: sp[1] - sp[0])
    return (a, b), "", f"{how}:{len(spans)}"


# --------------------------------------------------------------------------- attention capture


def _call_parts(args, kwargs):
    hidden = kwargs.get("hidden_states", args[0] if args else None)
    pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
    cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
    return hidden, pe, cache


def _kvlen(kv) -> int:
    if kv is None:
        return 0
    try:
        layers = getattr(kv, "layers", None)
        if layers is not None and len(layers) and getattr(layers[0], "keys", None) is not None:
            return int(layers[0].keys.shape[-2])
        if hasattr(kv, "get_seq_length"):
            return int(kv.get_seq_length())
    except Exception:
        pass
    return 0


def question_rows_alpha(module, hidden, pe, keys, rows) -> torch.Tensor:
    """Native attention (fp32) of the selected prefill rows over ALL cached columns, mean over heads -> [Ts, L]."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

    L = int(keys.shape[2])
    T = int(hidden.shape[1])
    Hd = module.head_dim
    G = module.num_key_value_groups
    cos, sin = pe
    rows = rows.to(hidden.device)
    h = hidden.index_select(1, rows)
    Ts = int(rows.numel())
    q = module.q_norm(module.q_proj(h).view(1, Ts, -1, Hd)).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos.index_select(1, rows), sin.index_select(1, rows))
    K = repeat_kv(keys, G)
    scores = torch.matmul(q.float(), K.float().transpose(-2, -1)) * module.scaling      # [1,H,Ts,L]
    past = L - T
    col = torch.arange(L, device=keys.device)[None, None, None, :]
    row = (past + rows)[None, None, :, None]
    scores = scores.masked_fill(col > row, float("-inf"))
    return torch.softmax(scores, dim=-1)[0].mean(0)                                       # [Ts, L]


class ReasonerCapture:
    """One Reasoner call: prefill input embeddings, question rows, mean attention, latent-step bookkeeping."""

    def __init__(self, bb, m: int, targets):
        self.bb, self.m = bb, int(m)
        self.targets = [str(t).strip() for t in (targets or []) if t is not None and str(t).strip()]
        self.n_layers = len(bb.lm.layers)
        self.embeds0 = None          # [T, d] input embeddings of the prefill call
        self.past0 = 0               # cache length before the prefill call
        self.rows = None             # question rows (relative to the prefill call)
        self.acc = None              # [Ts, past0+T] sum over layers of head-mean attention
        self.n_acc = 0
        self.lat_len: list[int] = []  # cache length before each latent step
        self.lat_pos: list[int] = []  # text position of each latent step
        self.note = ""
        self.in_prefill = False
        self.matched = ""

    def _locate_question(self, X):
        """Rows of the (shortest matching) question target inside the prefill embeddings X [T,d]."""
        loc, note, how = locate_question(self.bb, X, self.targets)
        if loc is None:
            self.note = note
            return None
        self.matched = how
        return loc

    def lm_pre(self, module, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        kv = kwargs.get("past_key_values")
        pos = kwargs.get("position_ids")
        self.in_prefill = False
        if emb is None or emb.dim() != 3:
            return None
        T = int(emb.shape[1])
        if T > 1:
            if self.embeds0 is None:                 # the FIRST multi-row call of this role = its prefill
                self.embeds0 = emb[0].detach().clone()
                self.past0 = _kvlen(kv)
                loc = self._locate_question(self.embeds0)
                if loc is None:
                    self.note = self.note or "question rows not found in the Reasoner prefill"
                    return None
                self.rows = torch.arange(loc[0], loc[1], device=emb.device)
                self.acc = torch.zeros(int(self.rows.numel()), self.past0 + T, dtype=torch.float32, device=emb.device)
                self.in_prefill = True
            return None
        if len(self.lat_len) < self.m and kv is not None and pos is not None:
            self.lat_len.append(_kvlen(kv))
            flat = pos.reshape(-1)
            self.lat_pos.append(int(flat[0]) if bool((flat == flat[0]).all()) else -1)
        return None

    def attn_post(self, li: int):
        def post(module, args, kwargs, output):
            if not self.in_prefill or self.rows is None:
                return output
            hidden, pe, cache = _call_parts(args, kwargs)
            if hidden is None or pe is None or cache is None or int(hidden.shape[1]) == 1:
                return output
            keys = cache.layers[li].keys
            with torch.no_grad():
                a = question_rows_alpha(module, hidden, pe, keys, self.rows)
            if tuple(a.shape) == tuple(self.acc.shape):
                self.acc += a
                self.n_acc += 1
            return output
        return post


@contextlib.contextmanager
def reasoner_probe(bb, m: int, stage: str, targets=None):
    """Hooks for one Reasoner call; off / other stage = nothing installed.  Result: bb._lu_cap."""
    if not enabled() or stage != "reasoner" or int(m) <= 0:
        yield None
        return
    cap = ReasonerCapture(bb, m, targets)
    bb._lu_cap = None
    handles = [bb.lm.register_forward_pre_hook(cap.lm_pre, with_kwargs=True)]
    handles += [layer.self_attn.register_forward_hook(cap.attn_post(li), with_kwargs=True)
                for li, layer in enumerate(bb.lm.layers)]
    try:
        yield cap
    finally:
        for h in handles:
            h.remove()
        bb._lu_cap = cap


# --------------------------------------------------------------------------- engine entry


def _g(x) -> float:
    return float(f"{float(x):.5g}")


def _write(cfg, rec, t0):
    rec["sec"] = round(time.time() - t0, 2)
    if cfg["log"]:
        with open(cfg["log"], "a") as fh:
            fh.write(json.dumps(rec) + "\n")


@torch.no_grad()
def apply(engine, result: dict, *, m: int, case_index: int = -1):
    """Optimise Z, crop the cache to L0 and replay the m rows once (in place on result["past_key_values"])."""
    cfg = config()
    md = cfg["mode"]
    if not md:
        return None
    bb = engine._backbone
    t0 = time.time()
    tag = f"[LU] case {case_index}"
    rec = {"case": int(case_index), "mode": md, "m": int(m), "cfg": {k: v for k, v in cfg.items() if k != "log"}}
    cap = getattr(bb, "_lu_cap", None)
    bb._lu_cap = None

    def skip(why):
        rec["skip"] = why
        _write(cfg, rec, t0)
        print(f"{tag}: SKIP ({why})", flush=True)
        return rec

    if cap is None:
        return skip("no Reasoner capture (probe not installed?)")
    if cap.rows is None:
        return skip(cap.note or "question rows not located")
    m = int(m)
    if len(cap.lat_len) != m:
        return skip(f"latent steps observed {len(cap.lat_len)}/{m}")
    L0 = int(cap.lat_len[0])
    if cap.lat_len != list(range(L0, L0 + m)):
        return skip(f"latent cache lengths not contiguous {cap.lat_len}")
    c0 = int(cap.lat_pos[0])
    if c0 < 0 or cap.lat_pos != list(range(c0, c0 + m)):
        return skip(f"latent positions not text-contiguous {cap.lat_pos}")
    cache = result["past_key_values"]
    if _kvlen(cache) != L0 + m:
        return skip(f"cache length {_kvlen(cache)} != L0+m {L0 + m} (relays / eviction?)")
    if int(result["pos_cursor"]) != c0 + m:
        return skip(f"pos_cursor {int(result['pos_cursor'])} != first latent position {c0} + m")
    vis = result.get("vis_cols")
    if not isinstance(vis, torch.Tensor) or int(vis.numel()) == 0:
        return skip("no visual columns")
    vis = vis.detach().cpu().long()
    if int(vis.min()) < cap.past0 or int(vis.max()) >= L0:
        return skip(f"visual columns [{int(vis.min())},{int(vis.max())}] outside this prefill [{cap.past0},{L0})")
    if cap.n_acc != cap.n_layers:
        return skip(f"attention captured on {cap.n_acc}/{cap.n_layers} layers")
    traj = result.get("latent_trajectory")
    if not traj or len(traj) != m:
        return skip("latent_trajectory missing or wrong length")
    K_, pos, neg = m, cfg["pos"], cfg["neg"]
    if int(vis.numel()) < K_ * (pos + neg):
        return skip(f"n_vis {int(vis.numel())} < m*(pos+neg) {K_ * (pos + neg)}")

    dev = bb.device
    zdtype = traj[0].dtype
    Z0 = torch.stack([z.to(dev) for z in traj]).float()                       # [m, d]
    vis_dev = vis.to(dev)
    V = cap.embeds0[vis_dev - cap.past0]                                       # [n_vis, d] projected visual inputs
    s = (cap.acc / cap.n_acc)[:, vis_dev].mean(0)                              # [n_vis] relevance
    P, N = assign_sets(s, K_, pos, neg)
    S0 = cosine_sim(Z0, V)
    rec.update({
        "L0": L0, "cursor0": c0, "n_vis": int(vis.numel()), "q_rows": int(cap.rows.numel()),
        "q_match": cap.matched, "layers": cap.n_layers,
        "rel_top": [_g(v) for v in s[P.flatten()].tolist()[:6]], "rel_bottom": [_g(v) for v in s[N.flatten()].tolist()[:6]],
        "rel_sum": _g(s.sum()), "rel_mean": _g(s.mean()),
        "cos_pos_native": _g(S0.gather(1, P).mean()), "cos_neg_native": _g(S0.gather(1, N).mean()),
        "z_norm_native": _g(Z0.norm(dim=-1).mean()),
    })

    # native latent K/V snapshot (diagnostic: how far the regenerated rows are from the native ones)
    nat = [(l.keys[..., L0:L0 + m, :].detach().clone(), l.values[..., L0:L0 + m, :].detach().clone())
           for l in cache.layers]
    positions = bb._text_positions(m, start=c0)
    lm_head = bb.model.lm_head
    wd = lm_head.weight.dtype

    def replay(Z):
        cache.crop(L0)
        o = bb.lm(inputs_embeds=Z.to(zdtype).unsqueeze(0), position_ids=positions, past_key_values=cache,
                  use_cache=True, output_hidden_states=True)
        return o.hidden_states[-1][0]                                          # [m, d] post-norm

    def entropies(Z):
        return top_delta_entropy(lm_head(replay(Z).to(wd)).float(), cfg["topk"])

    n_fwd = 0
    E_nat = entropies(Z0)
    n_fwd += 1
    rec["E_native"] = [_g(v) for v in E_nat.tolist()]
    rec["R_native"] = _g(reward(E_nat))
    Z = Z0
    if md in ("s1", "s12"):
        Z, losses = stage1(Z0, V, P, N, cfg["s1_steps"], cfg["s1_lr"], cfg["tau"])
        S1 = cosine_sim(Z, V)
        rec.update({"s1_loss": [_g(v) for v in losses],
                    "cos_pos_s1": _g(S1.gather(1, P).mean()), "cos_neg_s1": _g(S1.gather(1, N).mean()),
                    "dz_rel_s1": _g((Z - Z0).norm() / Z0.norm().clamp_min(1e-12))})
    if md == "s12":
        E_s1 = entropies(Z)
        n_fwd += 1
        rec["E_s1"] = [_g(v) for v in E_s1.tolist()]
        rec["R_s1"] = _g(reward(E_s1))
        gen = torch.Generator(device=dev).manual_seed(42 + int(case_index))
        Z_s1 = Z

        def ev(H):
            nonlocal n_fwd
            n_fwd += 1
            return reward(entropies(H))

        Z, st = nes(Z_s1, ev, cfg["s2_steps"], cfg["sigma"], cfg["decay"], cfg["alpha"], gen)
        rec["s2"] = {"R0": _g(st["R0"]), "best": _g(st["best"]), "best_iter": st["best_iter"],
                     "hist": [_g(v) for v in st["hist"]]}
        rec["dz_rel_s2"] = _g((Z - Z_s1).norm() / Z_s1.norm().clamp_min(1e-12))

    # final replay: this is the KV the turn close and the Answerer will see
    hF = replay(Z)
    n_fwd += 1
    E_fin = top_delta_entropy(lm_head(hF.to(wd)).float(), cfg["topk"])
    if _kvlen(cache) != L0 + m:
        raise RuntimeError(f"LU replay left the cache at {_kvlen(cache)} != {L0 + m}")
    kmax = vmax = 0.0
    knum = kden = vnum = vden = 0.0
    for (k0, v0), l in zip(nat, cache.layers):
        k1 = l.keys[..., L0:L0 + m, :]
        v1 = l.values[..., L0:L0 + m, :]
        kmax = max(kmax, float((k1.float() - k0.float()).abs().max()))
        vmax = max(vmax, float((v1.float() - v0.float()).abs().max()))
        knum += float((k1.float() - k0.float()).pow(2).sum()); kden += float(k0.float().pow(2).sum())
        vnum += float((v1.float() - v0.float()).pow(2).sum()); vden += float(v0.float().pow(2).sum())
    Zb = Z.to(zdtype)
    result["latent_trajectory"] = [z.detach().cpu() for z in Zb]
    rec.update({
        "E_final": [_g(v) for v in E_fin.tolist()], "R_final": _g(reward(E_fin)),
        "dz_rel_total": _g((Zb.float() - Z0).norm() / Z0.norm().clamp_min(1e-12)),
        "z_norm_final": _g(Zb.float().norm(dim=-1).mean()),
        "kv_diff": {"k_max": _g(kmax), "v_max": _g(vmax),
                    "k_rel": _g(math.sqrt(knum / max(kden, 1e-30))), "v_rel": _g(math.sqrt(vnum / max(vden, 1e-30)))},
        "n_forward": n_fwd,
    })
    _write(cfg, rec, t0)
    s1txt = (f" s1 loss {rec['s1_loss'][0]:.4g}->{rec['s1_loss'][-1]:.4g} cos+ {rec['cos_pos_native']:.3f}->{rec['cos_pos_s1']:.3f}"
             f" cos- {rec['cos_neg_native']:.3f}->{rec['cos_neg_s1']:.3f} dz {rec['dz_rel_s1']:.3g}" if "s1_loss" in rec else "")
    s2txt = (f" s2 R {rec['s2']['R0']:.4g}->best {rec['s2']['best']:.4g}@{rec['s2']['best_iter']} dz {rec['dz_rel_s2']:.3g}"
             if "s2" in rec else "")
    print(f"{tag} mode={md} m={m} L0={L0} n_vis={int(vis.numel())} q_rows={rec['q_rows']} "
          f"R_native {rec['R_native']:.4g} -> R_final {rec['R_final']:.4g}{s1txt}{s2txt} "
          f"kv_rel k {rec['kv_diff']['k_rel']:.3g} v {rec['kv_diff']['v_rel']:.3g} fwd={n_fwd} sec={rec['sec']}", flush=True)
    return rec
