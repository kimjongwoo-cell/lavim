"""C2 -- Visual Evidence Ratio Decoding (opt-in, VLMAS_ANSWERER_VER=1).

At the Answerer, split the cache into M_V (the crop K/V columns written by the
Reasoner prefill) and M_R (everything else). For every answer candidate y,
teacher-force the same continuation under the full cache and under a copy with
M_V removed, and pick

    y_hat = argmax_y  log p(y | M_V, M_R, Q) - log p(y | M_R, Q)

i.e. the answer most supported *because* the visual KV exists. The normal
decode still runs (rationale/format come from it); only the answer field is
replaced. Length bias cancels inside the ratio (same continuation both sides).

Env:
    VLMAS_ANSWERER_VER=1        enable
    VLMAS_ANSWERER_VER_OUT=<f>  optional JSONL with per-candidate scores
"""
from __future__ import annotations

import json
import os
import re
from copy import deepcopy

import torch


def enabled() -> bool:
    return os.environ.get("VLMAS_ANSWERER_VER", "").strip() == "1"


def drop_columns(backbone, cache, cols):
    """Deep-copy `cache` without the given absolute columns (RoPE is baked into
    the remaining keys, so positions stay valid; same as the KVDrop path)."""
    copy = deepcopy(cache)
    total = int(backbone._kv_len(copy))
    cols = cols.to(device=backbone.device, dtype=torch.long)
    cols = cols[cols < total]
    keep = torch.ones(total, dtype=torch.bool, device=backbone.device)
    keep[cols] = False
    keep_idx = keep.nonzero(as_tuple=True)[0]
    for layer in copy.layers:
        layer.keys = layer.keys.index_select(2, keep_idx)
        layer.values = layer.values.index_select(2, keep_idx)
    return copy


def visual_evidence_ratio(engine, *, cache, position_cursor, system_prompt,
                          user_prompt, json_prefix, candidates, visual_cols,
                          normal_output: str, case_index) -> str:
    """Return `normal_output` with its answer field replaced by the VER argmax."""
    from vision_text_mas.latent_terminal import score_terminal_continuation

    backbone = engine._backbone
    cands = [str(c).strip() for c in candidates if str(c).strip()]
    if not cands or visual_cols is None or int(visual_cols.numel()) == 0:
        print(f"[VER] case {case_index}: skipped (candidates={len(cands)}, "
              f"vis={0 if visual_cols is None else int(visual_cols.numel())})", flush=True)
        return normal_output
    score_only = os.environ.get("VLMAS_VER_SCORE_ONLY", "").strip() == "1"
    novis = None if score_only else drop_columns(backbone, cache, visual_cols)

    def score(source, y):
        copy = deepcopy(source)
        try:
            total, n = score_terminal_continuation(
                backbone=backbone, cache=copy, position_cursor=position_cursor,
                system_prompt=system_prompt, user_prompt=user_prompt,
                json_prefix=json_prefix, continuation=f'"{y}"')
            return float(total), int(n)
        finally:
            del copy

    rows = []
    for y in cands:
        s_full, n = score(cache, y)
        s_nov = float("nan") if score_only else score(novis, y)[0]
        rows.append({"y": y, "log_p_full": s_full, "log_p_novis": s_nov,
                     "s_v": s_full - s_nov, "tokens": n})
    if novis is not None:
        del novis
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    best = max(rows, key=lambda r: (r["log_p_full"] if score_only else r["s_v"]))
    best_full = max(rows, key=lambda r: r["log_p_full"])
    m = re.match(r'\s*"([^"]*)"', normal_output or "")
    normal_answer = m.group(1).strip() if m else ""
    out = os.environ.get("VLMAS_ANSWERER_VER_OUT", "").strip()
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"case_index": case_index, "normal": normal_answer,
                                 "ver": best["y"], "full_argmax": best_full["y"],
                                 "rows": rows}, ensure_ascii=False) + "\n")
    print(f"[VER] case {case_index}: normal={normal_answer!r} full_argmax={best_full['y']!r} "
          f"ver={best['y']!r} (S_V {best['s_v']:+.2f}; {len(cands)} cands)", flush=True)
    if score_only or not m:
        return normal_output
    # replace only the answer value; rationale/confidence stay as generated
    return normal_output[:m.start(1)] + best["y"] + normal_output[m.end(1):]


__all__ = ["enabled", "drop_columns", "visual_evidence_ratio"]
