"""Answerer text-context cut (Text-cut) -- receiver competition diagnostic (0917, user request).

Opt-in.  VLMAS_TEXTCUT unset = no hook, no cache change, byte-identical pipeline.

Model, latent steps and every visual / latent KV column stay exactly as they are.  Only the OLD PROMPT
TEXT of earlier roles (default: evidence_planner + navigator; their system/user prompt tokens and their
turn-close tokens) is changed on the cache the Answerer consumes, after the Reasoner turn is closed
and before the Answerer prompt is prefilled:

  drop    the columns are physically removed from the cache (index_select on every layer).  RoPE is
          baked into K, so the surviving columns keep their coordinates and the Answerer's positions
          (position_cursor) are unchanged: this equals masking those columns for every Answerer row
          and also shrinks the cache (real KV saving).
  filler  same number of columns, meaningless content: each maximal run of cut columns is recomputed
          from filler tokens at the ORIGINAL positions on top of the cache prefix before the run
          (K/V of filler tokens, later columns untouched).  VLMAS_TEXTCUT_FILLER=shuffle (the run's own
          token ids, seeded permutation: same length, same unigram content, meaning destroyed) |
          space (one neutral token repeated).  Separates "softmax denominator / length" from
          "meaningful text keys pulling the Answerer query".
  diag    nothing changed; measurement only.

Measurement (every mode): Answerer attention mass (fp32 recompute in hooks, rows = last prompt row +
each generated row, mean over layers/heads/rows) on: each role's prompt text / visual / latent /
close columns, the Answerer's own prompt, the sink column 0; plus the number of columns cut.
Written to VLMAS_TEXTCUT_LOG (jsonl) and printed as [TextCut] case ...

Bookkeeping (no engine registry needed): a pre-hook on bb.lm during every role's append records, for
each multi-row call, the token ids recovered from the input embeddings (visual rows -> -1) and the
position ids; record_stage() then labels the role's columns as prompt text / visual / latent / close.
Only the cumulative transports (full cache kept, turn closed) are supported.

Combining: Text-cut acts at the terminal on absolute columns; do not combine with terminal hooks that
index columns (VELH, PSAR, restage...).  LU (Reasoner-side) combines freely.

Env
  VLMAS_TEXTCUT=diag|drop|filler   VLMAS_TEXTCUT_ROLES=evidence_planner,navigator
  VLMAS_TEXTCUT_FILLER=shuffle|space   VLMAS_TEXTCUT_LOG=<jsonl>
"""
from __future__ import annotations

import contextlib
import json
import os
import time

import torch

from memory.lu import _call_parts, _kvlen, question_rows_alpha, recover_ids

MODES = ("diag", "drop", "filler")


def mode() -> str:
    v = os.environ.get("VLMAS_TEXTCUT", "").strip().lower()
    return v if v in MODES else ""


def enabled() -> bool:
    return mode() != ""


def config() -> dict:
    roles = [r.strip() for r in os.environ.get("VLMAS_TEXTCUT_ROLES", "evidence_planner,navigator").split(",") if r.strip()]
    return {"mode": mode(), "roles": roles,
            "filler": (os.environ.get("VLMAS_TEXTCUT_FILLER", "shuffle").strip().lower() or "shuffle"),
            "log": os.environ.get("VLMAS_TEXTCUT_LOG", "").strip()}


# --------------------------------------------------------------------------- pure


def runs_of(cols: list[int]) -> list[tuple[int, int]]:
    """Sorted column list -> maximal contiguous [start, end) runs."""
    out: list[tuple[int, int]] = []
    for c in sorted(set(int(x) for x in cols)):
        if out and out[-1][1] == c:
            out[-1] = (out[-1][0], c + 1)
        else:
            out.append((c, c + 1))
    return out


def remap_after_drop(cols: torch.Tensor, cut: torch.Tensor) -> torch.Tensor:
    """New indices of `cols` (must not be cut) after the sorted `cut` columns are removed."""
    cut_sorted = torch.sort(cut.long()).values
    shift = torch.searchsorted(cut_sorted, cols.long(), right=True)
    return cols.long() - shift


def drop_columns(cache, cut: torch.Tensor) -> None:
    """In place: remove absolute columns `cut` from every layer's K/V (order of the rest kept)."""
    if cut is None or int(cut.numel()) == 0:
        return
    for layer in cache.layers:
        n = int(layer.keys.shape[-2])
        keep = torch.ones(n, dtype=torch.bool, device=layer.keys.device)
        keep[cut.to(layer.keys.device).long()] = False
        idx = keep.nonzero(as_tuple=True)[0]
        layer.keys = layer.keys.index_select(-2, idx)
        layer.values = layer.values.index_select(-2, idx)


def filler_ids(ids: list[int], kind: str, seed: int, space_id: int) -> list[int]:
    if kind == "space":
        return [int(space_id)] * len(ids)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(ids), generator=g).tolist()
    return [int(ids[i]) for i in perm]


# --------------------------------------------------------------------------- stage probe


class StageCapture:
    """Multi-row lm calls of one role's append: (start column, ids [-1 = visual], positions [3, T])."""

    def __init__(self, bb):
        self.bb = bb
        self.calls: list[dict] = []

    def lm_pre(self, module, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        kv = kwargs.get("past_key_values")
        pos = kwargs.get("position_ids")
        if emb is None or emb.dim() != 3 or int(emb.shape[1]) <= 1 or pos is None:
            return None
        with torch.no_grad():
            ids = recover_ids(emb[0].detach(), self.bb.lm.embed_tokens.weight.detach()).cpu()
        self.calls.append({"start": _kvlen(kv), "ids": ids, "pos": pos.detach().reshape(3, -1).cpu().clone()})
        return None


@contextlib.contextmanager
def stage_probe(bb, stage: str):
    if not enabled():
        yield None
        return
    cap = StageCapture(bb)
    bb._tc_cap = None
    h = bb.lm.register_forward_pre_hook(cap.lm_pre, with_kwargs=True)
    try:
        yield cap
    finally:
        h.remove()
        bb._tc_cap = cap


def reset(engine) -> None:
    """Forget the previous case's blocks (call at the first append of a case, i.e. when the cache is None)."""
    engine._backbone._tc_blocks = []


def record_stage(engine, *, stage: str, start: int, past_len: int, latent_steps: int, closed_length: int,
                 vis_cols, close_thinking: bool, pos_cursor: int = -1) -> None:
    """Label the columns [start, closed_length) this role appended (called after its turn close)."""
    if not enabled():
        return
    bb = engine._backbone
    cap = getattr(bb, "_tc_cap", None)
    bb._tc_cap = None
    blocks = list(getattr(bb, "_tc_blocks", []) or [])
    if start == 0:
        blocks = []
    tok = bb.processor.tokenizer
    m = int(latent_steps)
    L0 = int(past_len) - m
    vis = set(int(v) for v in (vis_cols.tolist() if isinstance(vis_cols, torch.Tensor) else []))
    text_ids: dict[int, int] = {}
    text_pos: dict[int, list[int]] = {}
    note = ""
    if cap is None or not cap.calls:
        note = "no prefill capture (text-only branch?)"
    else:
        for c in cap.calls:
            s = int(c["start"])
            for r in range(int(c["ids"].numel())):
                col = s + r
                if col < L0 and col >= int(start) and int(c["ids"][r]) >= 0 and col not in vis:
                    text_ids[col] = int(c["ids"][r])
                    text_pos[col] = [int(c["pos"][a, r]) for a in range(3)]
    close_text = ("</think>\n" if close_thinking else "") + "<|im_end|>\n"
    close_ids = tok.encode(close_text, add_special_tokens=False) if closed_length > past_len else []
    close_cols = list(range(int(past_len), int(closed_length)))
    if len(close_ids) != len(close_cols):
        close_ids = [-1] * len(close_cols)
    # the turn-close tokens are text tokens at positions pos_cursor .. (all three MRoPE axes equal)
    close_pos = ({c: [int(pos_cursor) + k] * 3 for k, c in enumerate(close_cols)} if int(pos_cursor) >= 0 else {})
    blocks.append({
        "stage": str(stage), "start": int(start), "end": int(closed_length), "L0": L0, "m": m,
        "text": sorted(text_ids), "text_ids": text_ids, "text_pos": text_pos,
        "visual": sorted(vis), "latent": list(range(L0, int(past_len))),
        "close": close_cols, "close_ids": close_ids, "close_pos": close_pos, "note": note,
    })
    bb._tc_blocks = blocks
    print(f"[TextCut:R] {stage} cols [{start},{closed_length}) text={len(text_ids)} visual={len(vis)} "
          f"latent={m} close={len(close_cols)}{(' ' + note) if note else ''}", flush=True)


# --------------------------------------------------------------------------- terminal


def _g(x) -> float:
    return float(f"{float(x):.5g}")


def _finish(cfg, stats, t0):
    stats["sec"] = round(time.time() - t0, 2)
    if cfg["log"]:
        with open(cfg["log"], "a") as fh:
            fh.write(json.dumps(stats) + "\n")


def _filler_forward(bb, cache, s: int, e: int, ids: list[int], pos: torch.Tensor) -> None:
    """Recompute columns [s, e) of `cache` from filler token ids at positions pos [3, e-s], prefix = cache[:s]."""
    from transformers.cache_utils import DynamicCache
    tmp = DynamicCache()
    for li, layer in enumerate(cache.layers):
        tmp.update(layer.keys[..., :s, :], layer.values[..., :s, :], li)
    emb = bb.lm.embed_tokens(torch.tensor(ids, device=bb.device)).unsqueeze(0)
    out = bb.lm(inputs_embeds=emb, position_ids=pos.to(bb.device).unsqueeze(1), past_key_values=tmp, use_cache=True)
    new = out.past_key_values
    for li, layer in enumerate(cache.layers):
        layer.keys[..., s:e, :] = new.layers[li].keys[..., s:e, :].to(layer.keys.dtype)
        layer.values[..., s:e, :] = new.layers[li].values[..., s:e, :].to(layer.values.dtype)


@contextlib.contextmanager
def terminal(engine, cache, position_cursor: int, *, case_index: int = -1):
    """Apply the cut in place on the Answerer's cache and measure attention mass per source during its call."""
    cfg = config()
    md = cfg["mode"]
    bb = engine._backbone
    t0 = time.time()
    stats = {"case": int(case_index), "mode": md, "roles": cfg["roles"], "filler": cfg["filler"] if md == "filler" else None,
             "applied": False, "n_cut": 0, "calls": 0}
    blocks = list(getattr(bb, "_tc_blocks", []) or [])
    L = _kvlen(cache)
    if not blocks or any(int(b["end"]) > L for b in blocks):
        stats["skip"] = "no block bookkeeping for this case" if not blocks else "blocks beyond the cache"
        yield stats
        _finish(cfg, stats, t0)
        return
    base_len = L
    cut_cols: list[int] = []
    cut_blocks = []
    for b in blocks:
        if b["stage"] in cfg["roles"]:
            if b["note"]:
                stats.setdefault("notes", []).append(f"{b['stage']}: {b['note']}")
            cut_cols += list(b["text"]) + list(b["close"])
            cut_blocks.append(b)
    cut = torch.tensor(sorted(set(cut_cols)), dtype=torch.long)
    stats["n_cut"] = int(cut.numel())
    stats["cut_by_role"] = {b["stage"]: len(b["text"]) + len(b["close"]) for b in cut_blocks}
    stats["base_len"] = base_len

    # -- apply
    with torch.no_grad():
        if md == "filler" and int(cut.numel()):
            ids_of = {}
            pos_of = {}
            for b in cut_blocks:
                ids_of.update({int(c): int(i) for c, i in b["text_ids"].items()})
                pos_of.update({int(c): p for c, p in b["text_pos"].items()})
                ids_of.update({int(c): int(i) for c, i in zip(b["close"], b["close_ids"])})
                pos_of.update({int(c): p for c, p in b.get("close_pos", {}).items()})
            tok = bb.processor.tokenizer
            try:
                space_id = int(tok.encode(" ", add_special_tokens=False)[0])
            except Exception:
                space_id = 220
            runs = runs_of(cut.tolist())
            n_fwd = 0
            for (s, e) in runs:
                ids = [ids_of.get(c, -1) for c in range(s, e)]
                pos_rows = []
                known = [c for c in range(s, e) if c in pos_of]
                if not known:
                    stats.setdefault("notes", []).append(f"run [{s},{e}) has no positions; skipped")
                    continue
                p0 = pos_of[known[0]]
                for k, c in enumerate(range(s, e)):
                    pos_rows.append(pos_of.get(c, [p0[0] + (c - known[0]), p0[1] + (c - known[0]), p0[2] + (c - known[0])]))
                ids = [i if i >= 0 else space_id for i in ids]
                fids = filler_ids(ids, cfg["filler"], seed=42 + int(case_index) * 1000 + s, space_id=space_id)
                pos = torch.tensor(pos_rows, dtype=torch.long).T.contiguous()          # [3, n]
                _filler_forward(bb, cache, s, e, fids, pos)
                n_fwd += 1
            stats["filler_runs"] = n_fwd
            stats["applied"] = True
        elif md == "drop" and int(cut.numel()):
            drop_columns(cache, cut)
            stats["applied"] = True
            stats["len_after"] = _kvlen(cache)

    # -- measurement groups (absolute columns after the cut)
    def mapped(cols):
        t = torch.tensor(sorted(set(int(c) for c in cols)), dtype=torch.long)
        if md == "drop" and int(cut.numel()) and int(t.numel()):
            keep = ~torch.isin(t, cut)
            t = remap_after_drop(t[keep], cut)
        return t.to(bb.device)

    groups: dict[str, torch.Tensor] = {}
    for b in blocks:
        st = b["stage"]
        groups[f"{st}.text"] = mapped(b["text"] + b["close"])
        groups[f"{st}.visual"] = mapped(b["visual"])
        groups[f"{st}.latent"] = mapped(b["latent"])
    L_after = _kvlen(cache)
    groups["sink"] = torch.tensor([0], device=bb.device)
    acc = {k: 0.0 for k in groups}
    acc["answerer_prompt"] = 0.0
    acc["n"] = 0

    def make_post(li):
        def post(module, args, kwargs, output):
            hidden, pe, c = _call_parts(args, kwargs)
            if hidden is None or pe is None or c is None:
                return output
            T = int(hidden.shape[1])
            rows = torch.tensor([T - 1], device=hidden.device)
            with torch.no_grad():
                a = question_rows_alpha(module, hidden, pe, c.layers[li].keys, rows)[0]      # [L_now]
                for k, idx in groups.items():
                    if int(idx.numel()):
                        acc[k] += float(a[idx].sum())
                acc["answerer_prompt"] += float(a[L_after:].sum())
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
        stats["calls"] = acc["n"]
        stats["mass"] = {k: _g(v / n) for k, v in acc.items() if k != "n"}
        _finish(cfg, stats, t0)


def summary(stats: dict) -> str:
    if "skip" in stats:
        return f"mode={stats['mode']} SKIP ({stats['skip']})"
    ms = stats.get("mass", {})
    keys = sorted(k for k in ms if k.endswith(".text"))
    txt = " ".join(f"{k.split('.')[0][:3]}.text={ms[k]:.3f}" for k in keys)
    vis = sum(v for k, v in ms.items() if k.endswith(".visual"))
    lat = sum(v for k, v in ms.items() if k.endswith(".latent"))
    return (f"mode={stats['mode']} roles={','.join(stats['roles'])} cut={stats['n_cut']} by_role={stats.get('cut_by_role')} "
            f"base_len={stats.get('base_len')}" + (f"->{stats['len_after']}" if "len_after" in stats else "") +
            f" mass: {txt} visual={vis:.3f} latent={lat:.4f} answerer_prompt={ms.get('answerer_prompt', 0):.3f} "
            f"sink={ms.get('sink', 0):.3f} calls={stats['calls']} sec={stats['sec']}" +
            (f" notes={stats['notes']}" if stats.get("notes") else ""))
