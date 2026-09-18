"""Per-step visual-access decomposition for the Reasoner (no extra inference).

At every Reasoner latent step t we read the attention logits the model already
computes and split the visual group's mass into the two terms that produce it:

    logit M_V(t) = B_comp(t) + B_card(t)

    M_V(t)    = Z_V / (Z_V + Z_C)                          visual group mass
    B_card(t) = log|V| - log|C_t|                           cardinality term
    B_comp(t) = [log Z_V - log|V|] - [log Z_C - log|C_t|]   compatibility gap

with Z_V the sum of exp(logit) over the visual columns and Z_C the same over
the visible non-visual columns. The identity is exact by construction (the
|V|/|C_t| factors cancel), so it doubles as an extraction check: if the
recorded logit(M_V) and B_comp+B_card disagree, the readout is wrong rather
than the model.

Reading the split tells C2 where to act. A large negative B_comp beside a small
positive B_card means the visual tokens are simply not compatible with the
reasoning query, and re-weighting group sizes cannot fix that; only changing
what is compared can. The reverse pattern would say the opposite.

Env:
    VLMAS_ACCESS_DIAG=<dir>   enable; one JSON per case written into <dir>
    VLMAS_ACCESS_DIAG_TERMINAL=1   additionally capture the Answerer terminal
                              decode (prompt-prefill last query row + every
                              generated token) -> access_XXXX_terminal.json
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb, repeat_kv,
)


def output_dir() -> str:
    return os.environ.get("VLMAS_ACCESS_DIAG", "").strip()


def enabled() -> bool:
    return bool(output_dir())


def terminal_enabled() -> bool:
    return enabled() and os.environ.get("VLMAS_ACCESS_DIAG_TERMINAL", "").strip() == "1"


@contextlib.contextmanager
def capture_access(backbone, stage: str = "reasoner", visual_cols=None):
    """Record the decomposition for every single-token attention call.

    stage="terminal" (Answerer decode): the prompt prefill is multi-token, so
    its LAST query row (the position that emits the first answer token, which
    sees the whole cache) is recorded too; each generated token is then a
    single-row call exactly like a latent step. visual_cols overrides the
    column set (the Reasoner stage reads backbone._aiva_visual_cols).
    """
    if not enabled():
        yield None
        return
    steps: list[dict] = []
    handles: list[object] = []
    layer_count = len(backbone.lm.layers)

    def make_hook(layer_index: int):
        def hook(module, args, kwargs, _output):
            cols = visual_cols if visual_cols is not None else getattr(
                backbone, "_aiva_visual_cols", None)
            if cols is None or int(cols.numel()) == 0:
                return
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get(
                "position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get(
                "past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or cache is None or position_embeddings is None:
                return
            if int(hidden.shape[1]) != 1:
                if stage != "terminal":
                    return  # latent steps only: one query row, whole cache visible
                # terminal prompt prefill: keep the last query row (causal ->
                # it sees the entire cache incl. the prompt itself)
                hidden = hidden[:, -1:, :]
                cos, sin = position_embeddings
                position_embeddings = (cos[:, -1:, :], sin[:, -1:, :]) if cos.dim() == 3 else (cos[..., -1:, :], sin[..., -1:, :])
            keys = cache.layers[layer_index].keys
            kv_len = int(keys.shape[2])
            visual = cols.to(device=keys.device, dtype=torch.long)
            visual = visual[visual < kv_len]
            if int(visual.numel()) == 0 or int(visual.numel()) >= kv_len:
                return
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
            cos, sin = position_embeddings
            query, _ = apply_rotary_pos_emb(
                query, torch.zeros_like(query), cos, sin)
            expanded = repeat_kv(keys, module.num_key_value_groups)
            scores = (
                torch.matmul(query.float(), expanded.float().transpose(-2, -1))
                * module.scaling
            )[0, :, 0, :]                                   # [H_q, kv]
            mask = torch.zeros(kv_len, dtype=torch.bool, device=scores.device)
            mask[visual] = True
            log_zv = torch.logsumexp(scores[:, mask], dim=-1)
            log_zc = torch.logsumexp(scores[:, ~mask], dim=-1)
            # Level 2 -- read contribution. Attention mass alone ignores what
            # the values carry, so also measure the visual share of the actual
            # attention-block output: ||sum_{j in V} a_j v_j W_O|| against the
            # full ||sum_j a_j v_j W_O||.
            alpha = torch.softmax(scores, dim=-1)                # [H_q, kv]
            values = repeat_kv(
                cache.layers[layer_index].values, module.num_key_value_groups
            )[0].float()                                          # [H_q, kv, D]
            ctx_all = torch.einsum("hk,hkd->hd", alpha, values)
            ctx_vis = torch.einsum(
                "hk,hkd->hd", alpha * mask.unsqueeze(0), values)
            weight = module.o_proj.weight.float()
            out_all = torch.matmul(ctx_all.reshape(1, -1), weight.t())
            out_vis = torch.matmul(ctx_vis.reshape(1, -1), weight.t())
            c_all = float(out_all.norm())
            c_vis = float(out_vis.norm())
            n_v = float(int(visual.numel()))
            n_c = float(kv_len - int(visual.numel()))
            b_card = math.log(n_v) - math.log(n_c)
            b_comp = (log_zv - math.log(n_v)) - (log_zc - math.log(n_c))
            logit_mv = log_zv - log_zc
            steps.append({
                "layer": layer_index,
                "m_v": float(torch.sigmoid(logit_mv).mean()),
                "logit_m_v": float(logit_mv.mean()),
                "b_comp": float(b_comp.mean()),
                "b_card": b_card,
                "c_vis": c_vis,
                "c_all": c_all,
                "c_ratio": c_vis / c_all if c_all > 0 else 0.0,
                "n_v": int(n_v),
                "n_c": int(n_c),
            })

        return hook

    for index in range(layer_count):
        handles.append(
            backbone.lm.layers[index].self_attn.register_forward_hook(
                make_hook(index), with_kwargs=True))
    try:
        yield steps
    finally:
        for handle in handles:
            handle.remove()
        _dump(backbone, steps, stage)


def _dump(backbone, records: list[dict], stage: str = "reasoner") -> None:
    if not records:
        return
    per_step: list[list[dict]] = []
    for record in records:
        if record["layer"] == 0 or not per_step:
            per_step.append([])
        per_step[-1].append(record)
    summary = []
    for step_index, group in enumerate(per_step, start=1):
        if not group:
            continue

        def mean(key: str) -> float:
            return sum(item[key] for item in group) / len(group)

        row = {
            "step": step_index,
            "m_v": round(mean("m_v"), 6),
            "b_comp": round(mean("b_comp"), 4),
            "b_card": round(mean("b_card"), 4),
            "logit_m_v": round(mean("logit_m_v"), 4),
            "c_vis": round(mean("c_vis"), 4),
            "c_all": round(mean("c_all"), 4),
            "c_ratio": round(mean("c_ratio"), 6),
            "n_v": group[0]["n_v"],
            "n_c": group[0]["n_c"],
            "layers": len(group),
        }
        row["identity_gap"] = round(
            row["logit_m_v"] - (row["b_comp"] + row["b_card"]), 8)
        row["per_layer_m_v"] = [round(item["m_v"], 6) for item in group]
        summary.append(row)
    target = Path(output_dir())
    target.mkdir(parents=True, exist_ok=True)
    if stage == "terminal":
        # same case counter as the Reasoner file written just before it
        index = getattr(backbone, "_access_diag_index", 1) - 1
        path = target / f"access_{index:04d}_terminal.json"
    else:
        index = getattr(backbone, "_access_diag_index", 0)
        backbone._access_diag_index = index + 1
        path = target / f"access_{index:04d}.json"
    _ = path.write_text(json.dumps({"stage": stage, "steps": summary}, indent=2), encoding="utf-8")
    worst = max((abs(row["identity_gap"]) for row in summary), default=0.0)
    print(f"[AccessDiag] {stage} case {index}: {len(summary)} steps, "
          f"M_V {summary[0]['m_v']:.4f}->{summary[-1]['m_v']:.4f}, "
          f"C_V/C {summary[0]['c_ratio']:.4f}->{summary[-1]['c_ratio']:.4f}, "
          f"identity max|gap|={worst:.2e}", flush=True)


__all__ = ["capture_access", "enabled", "terminal_enabled", "output_dir"]
