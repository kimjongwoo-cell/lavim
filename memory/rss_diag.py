"""Efficient Role-Support Sensitivity — closed-form deletion proxy vs causal LOO.

Notion Experiment Log "[C2 구조 검증] Efficient Role-Support Sensitivity — closed-form
deletion proxy vs causal LOO" (C2 candidate RSVMH). Measurement only; env-gated.

    VLMAS_RSS=<jsonl>        enable (unset = no-op everywhere)
    VLMAS_RSS_LAYER=<int>    proxy layer (default floor(3L/4) = 27 for 36 layers)
    VLMAS_RSS_NO_LOO=1       Stage A: proxy + identity checks only (skip LOO replays)

Reasoner stage (backbone, last latent step only): one explicit attention row at layer
L* gives, per physical support G_g (one retained crop) and head h,

    m_{g,h} = sum_{j in G_g} a_{h,j},   c_{g,h} = sum_{j in G_g} a_{h,j} v_{h,j},
    Delta_{g,h} = (c_{g,h} - m_{g,h} o_h) / (1 - m_{g,h}),
    S_hat_g = || W_O concat_h Delta_{g,h} ||_2 .

Identity checks recorded with the proxy: (i) analytic o^{-g} vs the explicit masked
softmax output of the same row (fp32), (ii) W_O concat_h o_h vs the module's own output
for that row (sdpa bf16), which validates the query/key reconstruction.

Answerer boundary (engine, after the normal decode, validation only): from the cache
cropped to the pre-latent length, drop G_g's visual K/V, replay the m Reasoner latent
steps from the same starting hidden and positions, close the turn with the same
terminator, and read the Answerer's first answer-content decision row.

    S^R_g = 1 - cos(z_R, z_R^{-g}),   S^A_g = JS(p_A, p_A^{-g}).

A no-deletion replay is recorded as the numerical floor.
"""
from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy

import torch
import torch.nn.functional as F

try:  # transformers >= 4.57 layout (remote venv: 5.8.1)
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
except Exception:  # pragma: no cover - unit tests do not need these
    apply_rotary_pos_emb = None
    repeat_kv = None


def enabled() -> bool:
    return bool(os.environ.get("VLMAS_RSS", "").strip())


def proxy_layer(n_layers: int) -> int:
    raw = os.environ.get("VLMAS_RSS_LAYER", "").strip()
    if raw:
        return int(raw)
    return (3 * int(n_layers)) // 4


# --------------------------------------------------------------------------- pure math


def closed_form_deletion(alpha: torch.Tensor, values: torch.Tensor, groups):
    """alpha [H, L] row softmax, values [H, L, D], groups: list of LongTensor column sets.

    Returns (o [H, D], mass [G, H], delta [G, H, D]) with
    delta_g = (c_g - m_g o) / (1 - m_g), i.e. o - o^{-g} after renormalizing the rest.
    """
    o = torch.einsum("hl,hld->hd", alpha, values)
    masses, deltas = [], []
    for cols in groups:
        a = alpha[:, cols]
        m = a.sum(dim=-1)                                            # [H]
        c = torch.einsum("hl,hld->hd", a, values[:, cols, :])        # [H, D]
        denom = (1.0 - m).clamp_min(1e-12).unsqueeze(-1)
        deltas.append((c - m.unsqueeze(-1) * o) / denom)
        masses.append(m)
    if not groups:
        return o, alpha.new_zeros((0, alpha.shape[0])), values.new_zeros((0,) + tuple(o.shape))
    return o, torch.stack(masses), torch.stack(deltas)


def masked_row_output(scores: torch.Tensor, values: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """Explicit local attention output with columns `cols` removed from the softmax. scores [H, L]."""
    s = scores.clone()
    s[:, cols] = float("-inf")
    return torch.einsum("hl,hld->hd", torch.softmax(s, dim=-1), values)


def project_heads(o_proj, x: torch.Tensor) -> torch.Tensor:
    """W_O concat_h x_h for x [..., H, D] -> [..., hidden], computed in fp32."""
    flat = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1]).float()
    weight = o_proj.weight.float()
    bias = None if getattr(o_proj, "bias", None) is None else o_proj.bias.float()
    return F.linear(flat.to(weight.device), weight, bias)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.double().clamp_min(0)
    q = q.double().clamp_min(0)
    p = p / p.sum().clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    mid = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * (a[mask].clamp_min(eps).log() - b[mask].clamp_min(eps).log())).sum())

    return max(0.0, 0.5 * kl(p, mid) + 0.5 * kl(q, mid))


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    return float(1.0 - F.cosine_similarity(a, b, dim=0))


def support_columns(cols: torch.Tensor, image_ids: torch.Tensor):
    """Split absolute visual columns into per-image column sets (image order ascending)."""
    ids = image_ids.to(cols.device).long()
    return [(int(i), cols[ids == i]) for i in torch.unique(ids, sorted=True).tolist()]


def image_ids_from_counts(counts) -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(len(counts)), torch.tensor([int(c) for c in counts], dtype=torch.long))


def drop_cache_columns(cache, cols: torch.Tensor) -> None:
    """In place: remove absolute columns `cols` from every layer's K/V (order of the rest kept)."""
    if cols is None or int(cols.numel()) == 0:
        return
    for layer in cache.layers:
        length = int(layer.keys.shape[-2])
        keep = torch.ones(length, dtype=torch.bool, device=layer.keys.device)
        keep[cols.to(layer.keys.device).long()] = False
        idx = keep.nonzero(as_tuple=True)[0]
        layer.keys = layer.keys.index_select(-2, idx)
        layer.values = layer.values.index_select(-2, idx)


def close_flag_for_length(tokenizer, n_close: int):
    """Which close_assistant_turn terminator produced `n_close` cache columns (True = with </think>)."""
    lengths = {}
    for flag in (False, True):
        text = ("</think>\n" if flag else "") + "<|im_end|>\n"
        lengths[flag] = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    matches = [flag for flag, n in lengths.items() if n == int(n_close)]
    return matches[0] if len(matches) == 1 else None


# --------------------------------------------------------------------------- Reasoner probe


class ReasonerProbe:
    """Lives on the backbone for one Reasoner latent block (created by `begin`)."""

    def __init__(self, bb, *, cur_vis, cursor, cur_len, last_hidden, m, thw):
        self.bb = bb
        self.m = int(m)
        self.cursor = int(cursor)
        self.pre_len = int(cur_len)
        self.h0 = last_hidden.detach().clone()
        self.vis = cur_vis.detach().clone().long()
        n_layers = len(bb.lm.layers)
        self.layer = proxy_layer(n_layers)
        self.handle = None
        self.proxy = None
        self.sbsh = None
        self.z = None
        self.done = False
        self.note = None
        ids = getattr(bb, "_prefill_prune_survivor_image_ids", None)
        if isinstance(ids, torch.Tensor) and int(ids.numel()) == int(self.vis.numel()):
            self.image_ids = ids.detach().cpu().long()
            self.id_source = "prefill_survivors"
        else:
            merge = int(getattr(getattr(bb.model.config, "vision_config", None), "spatial_merge_size", 2)) ** 2
            counts = [int(t * h * w) // merge for t, h, w in thw.tolist()] if thw is not None else []
            if sum(counts) == int(self.vis.numel()):
                self.image_ids = image_ids_from_counts(counts)
                self.id_source = "grid_thw"
            else:
                self.image_ids = None
                self.id_source = "unavailable"
                self.note = f"image ids unavailable (n_vis={int(self.vis.numel())}, grid sum={sum(counts)})"
        self.mags = tuple(int(x) for x in (getattr(bb, "_prune_image_magnifications", ()) or ()))

    def before_step(self, step: int) -> None:
        if step != self.m - 1 or self.image_ids is None:
            return
        attn = self.bb.lm.layers[self.layer].self_attn
        self.handle = attn.register_forward_hook(self._hook, with_kwargs=True)

    def after_step(self, step: int, output, cur_vis) -> None:
        if step != self.m - 1:
            return
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.z = output.hidden_states[-1][0, -1].detach().float().clone()
        if int(cur_vis.numel()) != int(self.vis.numel()):
            self.note = "visual columns changed during latent steps (eviction) — LOO invalid"
        self.done = True
        from memory import sbsh as _sbsh
        if _sbsh.enabled():
            # State-Backed WSI Support Handoff: support-gate Jacobian energy of the final Reasoner state
            self.sbsh = _sbsh.compute(self, output)

    @torch.no_grad()
    def _hook(self, module, args, kwargs, output):
        t0 = time.time()
        try:
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            kv = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or pe is None or kv is None:
                self.note = "hook inputs missing"
                return output
            keys = kv.layers[self.layer].keys
            values = kv.layers[self.layer].values
            L = int(keys.shape[-2])
            Hd = module.head_dim
            cos, sin = pe
            row = int(hidden.shape[1]) - 1
            cos_r = cos[..., row:row + 1, :] if cos.dim() >= 3 else cos
            sin_r = sin[..., row:row + 1, :] if sin.dim() >= 3 else sin
            q = module.q_norm(module.q_proj(hidden[:, row:row + 1, :]).view(1, 1, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos_r, sin_r)
            K = repeat_kv(keys, module.num_key_value_groups).float()[0]      # [H, L, D]
            V = repeat_kv(values, module.num_key_value_groups).float()[0]
            scores = torch.matmul(q.float()[0], K.transpose(-2, -1))[:, 0, :] * module.scaling   # [H, L]
            alpha = torch.softmax(scores, dim=-1)
            vis = self.vis.to(keys.device)
            groups = support_columns(vis, self.image_ids)
            o, mass, delta = closed_form_deletion(alpha, V, [c for _, c in groups])
            s_hat = project_heads(module.o_proj, delta).norm(dim=-1)                   # [G]
            out_row = (output[0] if isinstance(output, tuple) else output)[0, row].float()
            recon = project_heads(module.o_proj, o)
            recon_rel = float((recon - out_row).norm() / out_row.norm().clamp_min(1e-12))
            ident = 0.0
            for (_, cols), m_g, d_g in zip(groups, mass, delta):
                explicit = masked_row_output(scores, V, cols)
                analytic = o - d_g
                err = ((explicit - analytic).norm(dim=-1) / explicit.norm(dim=-1).clamp_min(1e-12)).max()
                ident = max(ident, float(err))
            # latent-query similarity baselines (past negative reference): per support, head-mean of the
            # log-mean-exp and of the mean of the same row's post-RoPE scaled dot products
            qk_lme, qk_mean = [], []
            for _, cols in groups:
                sc = scores[:, cols]
                qk_lme.append(float((torch.logsumexp(sc, dim=-1) - math.log(int(cols.numel()))).mean()))
                qk_mean.append(float(sc.mean()))
            vis_mask = torch.zeros(L, dtype=torch.bool, device=alpha.device)
            vis_mask[vis] = True
            if alpha.is_cuda:
                torch.cuda.synchronize()
            self.proxy = {
                "layer": self.layer, "L": L, "heads": int(alpha.shape[0]),
                "supports": [int(i) for i, _ in groups],
                "size": [int(c.numel()) for _, c in groups],
                "mag": [self.mags[i] if i < len(self.mags) else None for i, _ in groups],
                "s_hat": [float(x) for x in s_hat],
                "mass_sum_heads": [float(x) for x in mass.sum(dim=-1)],
                "mass_mean_heads": [float(x) for x in mass.mean(dim=-1)],
                "max_head_mass": [float(x) for x in mass.max(dim=-1).values],
                "qk_lme": qk_lme, "qk_mean": qk_mean,
                "rho_v": float(alpha[:, vis_mask].sum(dim=-1).mean()),
                "o_norm": float(recon.norm()),
                "identity_max_rel_err": ident,
                "recon_rel_err": recon_rel,
                "proxy_sec": round(time.time() - t0, 4),
            }
        except Exception as exc:  # measurement must never break the pipeline
            self.note = f"proxy failed: {type(exc).__name__}: {exc}"
        return output


def begin(bb, **kwargs):
    old = getattr(bb, "_rss_ctx", None)
    if old is not None and getattr(old, "handle", None) is not None:
        old.handle.remove()
    ctx = ReasonerProbe(bb, **kwargs)
    bb._rss_ctx = ctx
    return ctx


# --------------------------------------------------------------------------- Answerer-side LOO


@torch.no_grad()
def run(engine, *, cache, probe_base_len, position_cursor, system_prompt, user_prompt, json_prefix,
        case_index: int, normal_output=None) -> None:
    out_path = os.environ.get("VLMAS_RSS", "").strip()
    if not out_path:
        return
    from memory.vgain_diag import _forward_logits, candidate_lead
    from vision_text_mas.latent_terminal import _assistant_prompt

    bb = engine._backbone
    ctx = getattr(bb, "_rss_ctx", None)
    bb._rss_ctx = None
    t0 = time.time()
    rec = {"case": int(case_index), "slide": getattr(engine, "_rpath_slide_id", None)}
    try:
        if ctx is None or not ctx.done or ctx.proxy is None:
            rec["skip"] = "no reasoner proxy" if ctx is None else (ctx.note or "proxy missing")
            return
        rec.update({"proxy": ctx.proxy, "note": ctx.note, "id_source": ctx.id_source, "m": ctx.m,
                    "pre_len": ctx.pre_len, "cursor": ctx.cursor, "n_vis": int(ctx.vis.numel()),
                    "sbsh": getattr(ctx, "sbsh", None)})
        if os.environ.get("VLMAS_KV_RESTAGE", "") == "1":
            # terminal restage moved the visual columns: the Answerer-side replays below would be invalid
            rec["skip"] = "restaged cache (proxy/SBSH recorded only)"
            return
        tok = bb.processor.tokenizer
        n_close = int(probe_base_len) - (ctx.pre_len + ctx.m)
        flag = close_flag_for_length(tok, n_close)
        rec["n_close"] = n_close
        rec["close_thinking"] = flag
        if flag is None or int(position_cursor) != ctx.cursor + ctx.m + n_close:
            rec["skip"] = (f"cache layout mismatch: n_close={n_close} flag={flag} "
                           f"cursor={position_cursor} expected={ctx.cursor + ctx.m + n_close}")
            return
        prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt, user_prompt=user_prompt,
                                   json_prefix=json_prefix)
        prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
        lead = candidate_lead(normal_output)
        lead_ids = tok(lead, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
        n = int(prompt_ids.shape[1])
        rec["n_prompt"] = n
        rec["lead"] = lead

        def decision(c):
            if n > 1:
                _forward_logits(bb, c, prompt_ids[:, :-1], position_cursor)
            seq = torch.cat([prompt_ids[:, -1:], lead_ids], dim=1)
            return torch.softmax(_forward_logits(bb, c, seq, position_cursor + n - 1)[-1].float(), dim=-1)

        native_cache = deepcopy(cache)
        native_cache.crop(int(probe_base_len))
        p_nat = decision(native_cache)
        del native_cache
        base = deepcopy(cache)
        base.crop(ctx.pre_len)
        top_nat = int(p_nat.argmax())
        rec["native_top"] = top_nat
        rec["native_top_text"] = tok.decode([top_nat])

        def replay(drop):
            c = deepcopy(base)
            try:
                drop_cache_columns(c, drop)
                h = ctx.h0
                for step in range(ctx.m):
                    le = bb._apply_realign(h)
                    o = bb.lm(inputs_embeds=le, position_ids=bb._text_positions(1, ctx.cursor + step),
                              past_key_values=c, use_cache=True, output_hidden_states=True)
                    c = o.past_key_values
                    h = o.hidden_states[-1][:, -1:, :]
                z = h[0, 0].float()
                c, _, _ = bb.close_assistant_turn(c, ctx.cursor + ctx.m, close_thinking=flag)
                p = decision(c)
            finally:
                del c
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return z, p

        if os.environ.get("VLMAS_RSS_NO_LOO", "") == "1":
            z_rep, p_rep = replay(None)
            rec["floor"] = {"S_R": cosine_distance(ctx.z, z_rep), "S_A": js_divergence(p_nat, p_rep),
                            "top_same": int(p_rep.argmax()) == top_nat}
            return
        t_rep = time.time()
        z_rep, p_rep = replay(None)
        rec["floor"] = {"S_R": cosine_distance(ctx.z, z_rep), "S_A": js_divergence(p_nat, p_rep),
                        "top_same": int(p_rep.argmax()) == top_nat, "sec": round(time.time() - t_rep, 2)}
        vis = ctx.vis.to(bb.device)
        loo = []
        for img, cols in support_columns(vis, ctx.image_ids):
            z_g, p_g = replay(cols)
            loo.append({
                "support": int(img),
                "S_R": cosine_distance(ctx.z, z_g), "S_R_vs_replay": cosine_distance(z_rep, z_g),
                "S_A": js_divergence(p_nat, p_g), "S_A_vs_replay": js_divergence(p_rep, p_g),
                "top": int(p_g.argmax()), "top_flip": int(p_g.argmax()) != top_nat,
                "logp_native_top": float(torch.log(p_g[top_nat].clamp_min(1e-30))),
            })
        rec["loo"] = loo
        rec["logp_native_top_native"] = float(torch.log(p_nat[top_nat].clamp_min(1e-30)))
        del base
    except Exception as exc:  # measurement must never break the pipeline
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        rec["sec"] = round(time.time() - t0, 1)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        px = rec.get("proxy") or {}
        print(f"[RSS] case {case_index} supports={len(px.get('s_hat', []))} "
              f"ident={px.get('identity_max_rel_err')} recon={px.get('recon_rel_err')} "
              f"floor={rec.get('floor')} skip={rec.get('skip')} error={rec.get('error')} sec={rec['sec']}",
              flush=True)


__all__ = ["enabled", "proxy_layer", "closed_form_deletion", "masked_row_output", "project_heads",
           "js_divergence", "cosine_distance", "support_columns", "image_ids_from_counts",
           "drop_cache_columns", "close_flag_for_length", "ReasonerProbe", "begin", "run"]
