"""Single-Pass Dual-Address Visual Routing (method draft 0910, Component 2).

Each retained WSI observation (crop page) exposes ONE of two positional
addresses to the terminal Answerer -- its native multimodal position, or a
receiver-proximal virtual band adjacent to the answer boundary -- chosen per
decoder layer from the Answerer's OWN boundary query. No extra forward, no
extra token, no counterfactual evaluation, no re-encoding.

    mu_{p,N}^l = mean of the page's native (post-RoPE) keys          [H_kv, D]
    mu_{p,R}^l = R(Delta_p) mu_{p,N}^l   (rigid MRoPE translation; exact,
                 computed as the mean of the translated block)
    G_{p,a}^l  = (1/H) sum_h  q_b^{l,h} . mu_{p,a}^{l,h} / sqrt(d_h)
    a_p^{l*}   = argmax_a G_{p,a}^l ;  keys of page p at layer l are rotated
                 to the band iff a = R. Values never change.

q_b is the last Answerer-prompt token's query, captured POST-RoPE inside the
prompt prefill (the model's own q_norm(q_proj(h)) rotated by that forward's
own cos/sin tables), so G is exactly the mean native query-key logit of the
page. Under GQA the query heads each KV head serves are group-averaged, which
reduces (1/H) sum_h exactly.

Resolution happens once, in the last layer's hook of the prefill forward,
after every layer's boundary query is known; the resolved keys are what all
subsequent generation steps attend to. (The boundary token's own prefill
attention precedes resolution -- the single approximation, noted here.)

Env: VLMAS_KV_ROUTE=1 on the BASE layout (no park/restage: the native visual
columns must still sit where the Reasoner prefill put them). Off =
byte-identical. VLMAS_KV_ROUTE_LOG=1 prints per-layer counts.
"""
from __future__ import annotations

import math
import os

import torch

from memory.aavm import reduce_query_heads
from memory.restage import delta_cos_sin, rerotate_keys


def _page_spans(kept_counts):
    spans, offset = [], 0
    for count in kept_counts:
        spans.append((offset, offset + count))
        offset += count
    return spans


def band_coordinates(vis_pos: torch.Tensor, kept_counts, band_start: int):
    """Compact virtual band: pages placed consecutively in their ORIGINAL
    order, each rigidly translated so its per-axis minimum lands on the
    running cursor. Returns coords [3, n] aligned with vis_pos."""
    blocks = []
    cursor = int(band_start)
    for start, end in _page_spans(kept_counts):
        page = vis_pos[:, start:end]
        base = page.min(dim=1, keepdim=True).values
        moved = page - base + cursor
        blocks.append(moved)
        cursor = int(moved.max().item()) + 1
    return torch.cat(blocks, dim=1), cursor


def band_span(vis_pos: torch.Tensor, kept_counts) -> int:
    _, end = band_coordinates(vis_pos, kept_counts, 0)
    return int(end)


class SinglePassRouter:
    """Attach around the terminal generate; resolves addresses once."""

    def __init__(self, backbone, cache, vis_cols, vis_pos, kept_counts, b_pos):
        self.backbone = backbone
        self.cache = cache
        self.cols = vis_cols.to(device=backbone.device, dtype=torch.long)
        self.pos = vis_pos.to(device=backbone.device, dtype=torch.long)
        self.spans = _page_spans(kept_counts)
        self.kept_counts = list(kept_counts)
        self.b_pos = int(b_pos)
        self.captured: dict[int, torch.Tensor] = {}
        self.handles = []
        self.resolved = False
        self.decisions: list[list[str]] = []

    @classmethod
    def build(cls, backbone, cache, past_len: int, b_pos: int):
        cols = getattr(backbone, "_route_vis_cols", None)
        pos = getattr(backbone, "_route_vis_pos", None)
        pages = getattr(backbone, "_route_pages", None)
        if cols is None or pos is None or not pages:
            print("[KVRoute] SKIP: no visual bookkeeping", flush=True)
            return None
        if int(cols.numel()) != sum(pages) or int(pos.shape[-1]) != int(cols.numel()):
            print(f"[KVRoute] SKIP: cols {int(cols.numel())} vs pages "
                  f"{sum(pages)} vs pos {int(pos.shape[-1])}", flush=True)
            return None
        if int(cols.max()) >= past_len:
            print(f"[KVRoute] SKIP: visual col {int(cols.max())} >= past_len "
                  f"{past_len} (layout moved?)", flush=True)
            return None
        if os.environ.get("VLMAS_KV_PARK", "") == "1" or \
                os.environ.get("VLMAS_KV_RESTAGE", "") == "1":
            print("[KVRoute] SKIP: park/restage active (base layout required)",
                  flush=True)
            return None
        return cls(backbone, cache, cols, pos, pages, b_pos)

    # ── hooks ────────────────────────────────────────────────────────────
    def attach(self):
        layers = self.backbone.lm.layers
        last = len(layers) - 1
        for index, layer in enumerate(layers):
            self.handles.append(layer.self_attn.register_forward_hook(
                self._make_hook(index, index == last), with_kwargs=True))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self, layer_index: int, is_last: bool):
        def hook(module, args, kwargs, _output):
            if self.resolved:
                return
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings",
                            args[1] if len(args) > 1 else None)
            if hidden is None or pe is None or int(hidden.shape[1]) < 1:
                return
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            q_pre = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
            q_last = q_pre[:, :, -1:, :].float()              # [1, H_q, 1, D]
            cos, sin = pe
            q_post = rerotate_keys(q_last, cos[:, -1:].float(), sin[:, -1:].float())
            u_q = q_post[0, :, 0, :]                          # [H_q, D]
            h_kv = int(module.k_proj.out_features) // int(module.head_dim)
            self.captured[layer_index] = reduce_query_heads(u_q, h_kv)
            if is_last:
                self._resolve()
        return hook

    # ── resolution ───────────────────────────────────────────────────────
    @torch.no_grad()
    def _resolve(self):
        self.resolved = True
        backbone, cache = self.backbone, self.cache
        span = band_span(self.pos, self.kept_counts)
        band_start = max(0, self.b_pos - span)
        band, _ = band_coordinates(self.pos, self.kept_counts, band_start)
        sample = cache.layers[0].keys
        cos_old, sin_old = backbone.lm.rotary_emb(sample, self.pos.unsqueeze(1))
        cos_new, sin_new = backbone.lm.rotary_emb(sample, band.unsqueeze(1))
        cos_d, sin_d = delta_cos_sin(cos_old.float(), sin_old.float(),
                                     cos_new.float(), sin_new.float())
        log = os.environ.get("VLMAS_KV_ROUTE_LOG", "") == "1"
        # Position-corrected margin (env VLMAS_KV_ROUTE_TAU=<float>, unset =
        # raw argmax as specified): the raw G_R - G_N is dominated by RoPE
        # distance decay (same content, nearer position wins), so subtract the
        # position-only gain measured on a content-free key -- the global mean
        # of ALL visual keys -- translated by the same page delta. A page moves
        # only if its content-specific residual exceeds tau.
        tau_env = os.environ.get("VLMAS_KV_ROUTE_TAU", "").strip()
        tau = float(tau_env) if tau_env else None
        margins: list[float] = []
        share_mode = os.environ.get("VLMAS_KV_ROUTE_MODE", "").strip() == "share"
        lambdas: list[float] = []
        per_layer_R = []
        for layer_index, layer in enumerate(cache.layers):
            query = self.captured.get(layer_index)
            if query is None:
                per_layer_R.append(0)
                continue
            keys = layer.keys
            q = query.to(device=keys.device, dtype=torch.float32)  # [H_kv, D]
            dim = q.shape[-1]
            mu_all = keys.index_select(2, self.cols).float().mean(dim=2)  # [1,H,D]
            g_all = float(((q * mu_all[0]).sum(-1) / math.sqrt(dim)).mean())
            decisions = []
            n_r = 0
            # pass 1: page-level native / receiver logits (same content, same
            # query; only the RoPE phase differs)
            page_g = []
            for (start, end) in self.spans:
                cols = self.cols[start:end]
                block = keys.index_select(2, cols).float()          # [1,H,n,D]
                block_r = rerotate_keys(block, cos_d[:, start:end], sin_d[:, start:end])
                mu_n = block.mean(dim=2)[0]                          # [H, D]
                mu_r = block_r.mean(dim=2)[0]
                g_n = float(((q * mu_n).sum(-1) / math.sqrt(dim)).mean())
                g_r = float(((q * mu_r).sum(-1) / math.sqrt(dim)).mean())
                page_g.append((start, end, cols, block_r, g_n, g_r))
            # Position-Bias-Invariant routing (VLMAS_KV_ROUTE_MODE=share):
            # compare each page's RELATIVE access share within the visual bank,
            # Lambda_p = log pi_R - log pi_N = (G_R - G_N) - [LSE(G_R) - LSE(G_N)].
            # A bonus c^l common to every receiver logit cancels exactly
            # (softmax shift invariance); shares sum to 1 under both hypotheses
            # so all pages cannot gain at once. R iff Lambda > 0 (not a knob).
            if share_mode:
                gn = torch.tensor([g[4] for g in page_g])
                gr = torch.tensor([g[5] for g in page_g])
                lam = (gr - gn) - (torch.logsumexp(gr, 0) - torch.logsumexp(gn, 0))
                lambdas.extend(lam.tolist())
            for pi_, (start, end, cols, block_r, g_n, g_r) in enumerate(page_g):
                if share_mode:
                    choose_r = float(lam[pi_]) > 0.0
                elif tau is not None:
                    mu_all_r = rerotate_keys(
                        mu_all.unsqueeze(2), cos_d[:, start:start + 1],
                        sin_d[:, start:start + 1])[0, :, 0]         # [H, D]
                    g_pos = float(((q * mu_all_r).sum(-1) / math.sqrt(dim)).mean()) - g_all
                    margin = (g_r - g_n) - g_pos
                    margins.append(margin)
                    choose_r = margin > tau
                else:
                    choose_r = g_r > g_n
                if choose_r:
                    keys.index_copy_(2, cols, block_r.to(keys.dtype))
                    decisions.append("R")
                    n_r += 1
                else:
                    decisions.append("N")
            self.decisions.append(decisions)
            per_layer_R.append(n_r)
        total = len(cache.layers) * len(self.spans)
        routed = sum(per_layer_R)
        print(f"[KVRoute] resolved: R={routed}/{total} (layer×page), "
              f"per-layer R={per_layer_R}, band=[{band_start},{band_start + span})"
              f" b={self.b_pos}", flush=True)
        if lambdas:
            ls = sorted(lambdas)
            print(f"[KVRoute] mode=share Lambda: mean={sum(ls)/len(ls):.4f} "
                  f"min={ls[0]:.4f} med={ls[len(ls)//2]:.4f} max={ls[-1]:.4f} "
                  f"R-frac={sum(1 for x in ls if x > 0)/len(ls):.3f}", flush=True)
        if margins:
            ms = sorted(margins)
            print(f"[KVRoute] tau={tau} corrected-margin: mean={sum(ms)/len(ms):.4f} "
                  f"min={ms[0]:.4f} p25={ms[len(ms)//4]:.4f} med={ms[len(ms)//2]:.4f} "
                  f"p75={ms[3*len(ms)//4]:.4f} max={ms[-1]:.4f}", flush=True)
        if log:
            for li, d in enumerate(self.decisions):
                print(f"[KVRoute] L{li}: {''.join(d)}", flush=True)


__all__ = ["SinglePassRouter", "band_coordinates", "band_span"]


@torch.no_grad()
def apply_canonical(backbone, cache, vis_cols, vis_pos, kept_counts, b_pos):
    """Observation-Canonical Visual Addressing (VLMAS_KV_ROUTE_MODE=canonical).

    Every visual key's rotary phase is rewritten from its global coordinate
    r_{p,j} = o_p*1 + u_{p,j} to the crop-LOCAL coordinate u_{p,j} = (0,h,w)
    (the global cache offset o_p is dropped, within-crop 2-D geometry kept),
    and the Answerer reads it from the canonical reference c = 1 + max(u).
    Realized exactly at the answer boundary b through the equivalent key
    placement r_new = (b - c)*1 + u: native attention with the native query
    at b then yields q0^T R(u - c) k0, which contains neither o_p nor b.
    Text/latent keys and ALL values untouched. Prompt tokens before b see the
    canonical grid at a slightly larger offset (<= prompt length), the same
    approximation restage makes.
    """
    cols = vis_cols.to(device=backbone.device, dtype=torch.long)
    pos = vis_pos.to(device=backbone.device, dtype=torch.long)
    total = int(cache.layers[0].keys.shape[2])
    if int(cols.numel()) == 0 or int(cols.max()) >= total or \
            int(cols.numel()) != sum(kept_counts) or int(pos.shape[-1]) != int(cols.numel()):
        print(f"[KVCanon] SKIP: cols max={int(cols.max()) if cols.numel() else -1} "
              f"n={int(cols.numel())} vs cache={total} pages={sum(kept_counts)} "
              f"pos={int(pos.shape[-1])}", flush=True)
        return 0
    spans = _page_spans(kept_counts)
    blocks = []
    for start, end in spans:
        page = pos[:, start:end]
        u = page - page.min(dim=1, keepdim=True).values      # (t,h,w) local
        u[0] = 0                                             # t axis dropped
        blocks.append(u)
    mode = os.environ.get("VLMAS_KV_ROUTE_MODE", "").strip()
    if mode in ("canonical_t", "canonical_tperm"):
        # crop slot on the temporal axis: offset-invariant like canonical,
        # but crops no longer superimpose (inter-crop distance 1, not ~2000).
        # canonical_tperm = control: slots randomly permuted (seeded) so a
        # gain cannot come from ordinal preference (crop 1 first), only from
        # crops being distinguishable.
        slots = list(range(len(blocks)))
        if mode == "canonical_tperm":
            import random as _random
            _random.Random(42 + int(b_pos)).shuffle(slots)
        for u, slot in zip(blocks, slots):
            u[0] = slot
    u_all = torch.cat(blocks, dim=1)                          # [3, n]
    if mode == "derope":
        u_all = torch.zeros_like(u_all)                      # Full-deRoPE control: no spatial phase at all
    c = int(u_all.max().item()) + 1
    anchor = int(b_pos) - c
    if anchor < 0:
        print(f"[KVCanon] SKIP: b={b_pos} < c={c}", flush=True)
        return 0
    new_pos = u_all + anchor
    sample = cache.layers[0].keys
    cos_old, sin_old = backbone.lm.rotary_emb(sample, pos.unsqueeze(1))
    cos_new, sin_new = backbone.lm.rotary_emb(sample, new_pos.unsqueeze(1))
    cos_d, sin_d = delta_cos_sin(cos_old.float(), sin_old.float(),
                                 cos_new.float(), sin_new.float())
    for layer in cache.layers:
        block = layer.keys.index_select(2, cols).float()
        layer.keys.index_copy_(2, cols, rerotate_keys(block, cos_d, sin_d).to(layer.keys.dtype))
    print(f"[KVCanon] {mode} addressing: pages={list(kept_counts)} c={c} "
          f"b={int(b_pos)} anchor={anchor} n={int(cols.numel())} "
          f"(o_p dropped, (h,w) kept; values untouched)", flush=True)
    return int(cols.numel())


@torch.no_grad()
def offset_variance_diag(backbone, cache, vis_cols, vis_pos, kept_counts, b_pos,
                         q0_by_layer: dict, out_path: str):
    """Kill-test diagnostic (VLMAS_KV_ROUTE_MODE=diag, VLMAS_KV_ROUTE_DIAG=<dir>).

    From ONE base run: pre-RoPE boundary queries q0 (per layer) and the native
    visual keys, evaluate the mean visual logit G_V under four addressing
    semantics while the Answerer position is artificially shifted by delta:
      base      : keys at native r_{p,j},           query at b+delta
      restage   : keys at the recency band before b, query at b+delta
      canonical : keys at u_{p,j},                  query at c   (delta-free)
      derope    : keys at 0,                        query at 0   (delta-free)
    offset_variance[arm] = variance over delta of the layer-mean G_V.
    """
    import json
    cols = vis_cols.to(device=backbone.device, dtype=torch.long)
    pos = vis_pos.to(device=backbone.device, dtype=torch.long)
    n = int(cols.numel())
    total = int(cache.layers[0].keys.shape[2])
    if n == 0 or int(cols.max()) >= total or int(pos.shape[-1]) != n:
        print("[KVDiag] SKIP: bookkeeping mismatch", flush=True)
        return
    spans = _page_spans(kept_counts)
    u_blocks = []
    for start, end in spans:
        page = pos[:, start:end]
        u = page - page.min(dim=1, keepdim=True).values
        u[0] = 0
        u_blocks.append(u)
    u_all = torch.cat(u_blocks, dim=1)
    c = int(u_all.max().item()) + 1
    span = band_span(pos, kept_counts)
    band, _ = band_coordinates(pos, kept_counts, max(0, int(b_pos) - span))
    zero = torch.zeros_like(pos)
    sample = cache.layers[0].keys
    cos_nat, sin_nat = backbone.lm.rotary_emb(sample, pos.unsqueeze(1))
    def tables(coords):
        return backbone.lm.rotary_emb(sample, coords.unsqueeze(1))
    key_targets = {"base": pos, "restage": band, "canonical": u_all, "derope": zero}
    deltas = [0, 256, 1024, 4096]
    result = {arm: {str(d): [] for d in deltas} for arm in key_targets}
    for layer_index, layer in enumerate(cache.layers):
        q0 = q0_by_layer.get(layer_index)
        if q0 is None:
            continue
        k_nat = layer.keys.index_select(2, cols).float()            # [1,H,n,D] at native
        dim = k_nat.shape[-1]
        for arm, target in key_targets.items():
            cos_t, sin_t = tables(target)
            cos_d, sin_d = delta_cos_sin(cos_nat.float(), sin_nat.float(),
                                         cos_t.float(), sin_t.float())
            k_arm = rerotate_keys(k_nat, cos_d, sin_d)[0]            # [H,n,D]
            for d in deltas:
                if arm == "canonical":
                    qpos = c
                elif arm == "derope":
                    qpos = 0
                else:
                    qpos = int(b_pos) + d
                qp = torch.full((3, 1), qpos, device=backbone.device, dtype=torch.long)
                cos_q, sin_q = tables(qp)
                q_rot = rerotate_keys(q0.to(backbone.device).float()[None, :, None, :],
                                      cos_q.float(), sin_q.float())[0, :, 0, :]  # [H,D]
                g = float((torch.einsum("hd,hnd->hn", q_rot, k_arm) / math.sqrt(dim)).mean())
                result[arm][str(d)].append(g)
    summary = {}
    for arm, per_delta in result.items():
        means = {d: (sum(v) / len(v) if v else float("nan")) for d, v in per_delta.items()}
        vals = list(means.values())
        mu = sum(vals) / len(vals)
        var = sum((x - mu) ** 2 for x in vals) / len(vals)
        summary[arm] = {"G_by_delta": means, "offset_variance": var}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"b": int(b_pos), "c": c, "n_vis": n, "pages": list(kept_counts),
                   "summary": summary}, fh, indent=1)
    print("[KVDiag] offset variance: " + ", ".join(
        f"{a}={summary[a]['offset_variance']:.4f}" for a in summary), flush=True)


class DiagCapture:
    """Hooks capturing the PRE-RoPE boundary query per layer during the
    Answerer-prompt prefill; dumps offset-variance diag after generation."""

    def __init__(self, backbone, cache, b_pos, out_path):
        self.backbone, self.cache, self.b_pos, self.out_path = backbone, cache, int(b_pos), out_path
        self.captured, self.handles, self.done = {}, [], False

    def attach(self):
        for index, layer in enumerate(self.backbone.lm.layers):
            self.handles.append(layer.self_attn.register_forward_hook(
                self._hook(index), with_kwargs=True))

    def _hook(self, index):
        def hook(module, args, kwargs, _output):
            if self.done:
                return
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or int(hidden.shape[1]) < 2:
                return   # generation steps (1 token) are skipped; prefill only
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            q_pre = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
            u_q = q_pre[0, :, -1, :].float()
            h_kv = int(module.k_proj.out_features) // int(module.head_dim)
            self.captured[index] = reduce_query_heads(u_q, h_kv).cpu()
            if index == len(self.backbone.lm.layers) - 1:
                self.done = True
        return hook

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        cols = getattr(self.backbone, "_route_vis_cols", None)
        pos = getattr(self.backbone, "_route_vis_pos", None)
        pages = getattr(self.backbone, "_route_pages", None)
        if cols is None or pos is None or not pages or not self.captured:
            print("[KVDiag] SKIP: no bookkeeping/capture", flush=True)
            return
        offset_variance_diag(self.backbone, self.cache, cols, pos, pages,
                             self.b_pos, self.captured, self.out_path)
