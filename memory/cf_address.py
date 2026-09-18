"""Counterfactual Dual-Address Visual Memory (method draft 0910, Component 2).

Each retained WSI observation (crop page) is routed to ONE of two positional
addresses at the terminal boundary, chosen by its counterfactual effect on the
Answerer's full predictive distribution -- NOT by attention score, key
similarity, or any answer candidate:

  - native address  N : the page keeps its original acquisition RoPE positions
                        (far from the answer boundary).
  - receiver address R : the same content re-anchored to the terminal region
                        (recency-proximal); geometry preserved, values fixed.

At the fixed answer boundary b the vocabulary distribution is read three ways
per page p, all sharing the same non-visual context M_NV and the same probe:
    p_N   = all pages native
    p_-p  = page p removed, rest native
    p_pR  = page p at receiver, rest native
and the page is re-anchored to R iff
    JS(p_pR || p_-p)  >  JS(p_N || p_-p).
Otherwise it stays native. So recency is a *candidate* address whose validity
each observation earns against its own native utilization.

Position handling reuses the restage angle-subtraction identity: stored keys
are POST-RoPE at their native coordinates; a page is re-anchored by the delta
rotation from native to receiver coordinates. Column order is invisible to the
full-visibility terminal attention, so every page (native- or receiver-routed)
is physically appended at the cache tail and distinguished only by its RoPE
position -- exactly the property the restage/paging components already rely on.

Env: VLMAS_KV_RESTAGE_ORDER=counterfactual (with VLMAS_KV_RESTAGE=1
VLMAS_KV_PARK=1). Off / other order modes = byte-identical. VLMAS_KV_CFADDR_LOG=1
prints per-page (U_N, U_R, route).
"""
from __future__ import annotations

import os

import torch

from memory.restage import delta_cos_sin, rerotate_keys


def _page_spans(kept_counts: list[int]) -> list[tuple[int, int]]:
    spans, offset = [], 0
    for count in kept_counts:
        spans.append((offset, offset + count))
        offset += count
    return spans


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two vocab distributions (nats)."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (p.log() - m.log()))
    kl_qm = torch.sum(q * (q.log() - m.log()))
    return float(0.5 * (kl_pm + kl_qm))


def _receiver_coords(page_pos: torch.Tensor, anchor: int) -> torch.Tensor:
    """Rigid 3-D translation of a page's native coords so its min lands at
    ``anchor`` -- within-page (t,h,w) geometry preserved exactly."""
    base = page_pos.min(dim=1, keepdim=True).values
    return page_pos - base + anchor


class _AddressBank:
    """Precomputes each page's key block at native and receiver addresses.

    Native keys are the stored (post-RoPE) blocks as-is. Receiver keys are the
    stored blocks re-rotated by delta(native -> receiver). Values are shared.
    """

    def __init__(self, backbone, park_bank, kept_counts, position_cursor):
        self.spans = _page_spans(kept_counts)
        self.n_layers = len(park_bank["k"])
        self.device = backbone.device
        pos = park_bank["pos"].to(device=self.device, dtype=torch.long)
        sample = park_bank["k"][0]

        self.native_k = [[] for _ in range(len(self.spans))]
        self.recv_k = [[] for _ in range(len(self.spans))]
        self.values = [[] for _ in range(len(self.spans))]
        self.recv_coords = []

        # Each candidate page is tested at the receiver anchor = cursor, alone.
        span_max = 0
        for pi, (start, end) in enumerate(self.spans):
            page_pos = pos[:, start:end]
            recv = _receiver_coords(page_pos, position_cursor)
            self.recv_coords.append(recv)
            span_max = max(span_max, int(recv.max()) - position_cursor)
            cos_old, sin_old = backbone.lm.rotary_emb(sample, page_pos.unsqueeze(1))
            cos_new, sin_new = backbone.lm.rotary_emb(sample, recv.unsqueeze(1))
            cos_d, sin_d = delta_cos_sin(
                cos_old.float(), sin_old.float(),
                cos_new.float(), sin_new.float())
            for li in range(self.n_layers):
                block = park_bank["k"][li][:, :, start:end, :]
                self.native_k[pi].append(block)
                self.recv_k[pi].append(
                    rerotate_keys(block.float(), cos_d, sin_d).to(block.dtype))
                self.values[pi].append(park_bank["v"][li][:, :, start:end, :])
        self.native_max = int(pos.max())
        self.probe_pos = max(self.native_max, position_cursor + span_max,
                             position_cursor) + 1


@torch.no_grad()
def _boundary_distribution(backbone, cache, base_len, key_blocks, val_blocks,
                           probe_pos, probe_id):
    """Append the given per-layer key/value blocks, read the next-token
    distribution at a fixed boundary probe, then truncate back to base_len."""
    for li, layer in enumerate(cache.layers):
        if key_blocks:
            add_k = torch.cat([kb[li] for kb in key_blocks], dim=2).to(layer.keys.dtype)
            add_v = torch.cat([vb[li] for vb in val_blocks], dim=2).to(layer.values.dtype)
            layer.keys = torch.cat([layer.keys, add_k], dim=2)
            layer.values = torch.cat([layer.values, add_v], dim=2)
    ids = torch.tensor([[probe_id]], device=backbone.device, dtype=torch.long)
    pos = torch.full((3, 1, 1), int(probe_pos), device=backbone.device, dtype=torch.long)
    out = backbone.lm(
        inputs_embeds=backbone.lm.embed_tokens(ids),
        position_ids=pos, past_key_values=cache,
        use_cache=True, output_hidden_states=True)
    h = out.hidden_states[-1][:, -1, :]
    head = backbone.model.lm_head
    logits = head(h.to(head.weight.dtype))[0].float()
    dist = torch.softmax(logits, dim=-1)
    for layer in cache.layers:
        layer.keys = layer.keys[:, :, :base_len, :]
        layer.values = layer.values[:, :, :base_len, :]
    return dist


@torch.no_grad()
def route_and_restore(backbone, cache, park_bank, kept_counts, position_cursor):
    """Full Component-2 decision + commit. Returns (new_cursor, n_total,
    routes) or (None, None, None) to fall back to the standard park restore."""
    if len(kept_counts) < 2:
        return None, None, None
    if sum(kept_counts) != int(park_bank["pos"].shape[-1]):
        return None, None, None
    if not hasattr(cache, "layers"):
        return None, None, None

    bank = _AddressBank(backbone, park_bank, kept_counts, position_cursor)
    base_len = int(cache.layers[0].keys.shape[2])
    tok = backbone.processor.tokenizer
    probe_id = tok.convert_tokens_to_ids("<|im_start|>")
    if probe_id is None or probe_id < 0:
        probe_id = int(tok.eos_token_id)
    n_pages = len(kept_counts)
    all_pages = list(range(n_pages))

    def dist_for(config_k, config_v):
        return _boundary_distribution(
            backbone, cache, base_len, config_k, config_v,
            bank.probe_pos, probe_id)

    # p_N : all native
    p_N = dist_for([bank.native_k[p] for p in all_pages],
                   [bank.values[p] for p in all_pages])

    routes = []  # per page: "R" or "N"
    for p in all_pages:
        rest = [q for q in all_pages if q != p]
        p_mp = dist_for([bank.native_k[q] for q in rest],
                        [bank.values[q] for q in rest])
        p_pR = dist_for([bank.native_k[q] for q in rest] + [bank.recv_k[p]],
                        [bank.values[q] for q in rest] + [bank.values[p]])
        u_N = _js_divergence(p_N, p_mp)
        u_R = _js_divergence(p_pR, p_mp)
        routes.append("R" if u_R > u_N else "N")
        if os.environ.get("VLMAS_KV_CFADDR_LOG", "") == "1":
            print(f"[KVCFAddr] page {p}: U_N={u_N:.4e} U_R={u_R:.4e} "
                  f"-> {routes[-1]}", flush=True)

    # Commit: native-routed pages keep stored keys/positions; receiver-routed
    # pages are placed sequentially at recency. Physical order is invisible;
    # only RoPE position realizes native-vs-receiver. Values untouched.
    keys_out = [[] for _ in range(bank.n_layers)]
    vals_out = [[] for _ in range(bank.n_layers)]
    cursor = int(position_cursor)
    n_recv = 0
    pos_max = bank.native_max
    for p in all_pages:
        if routes[p] == "N":
            for li in range(bank.n_layers):
                keys_out[li].append(bank.native_k[p][li])
                vals_out[li].append(bank.values[p][li])
        else:
            page_pos = park_bank["pos"].to(device=backbone.device,
                                           dtype=torch.long)[:, bank.spans[p][0]:bank.spans[p][1]]
            recv = _receiver_coords(page_pos, cursor)
            sample = park_bank["k"][0]
            cos_old, sin_old = backbone.lm.rotary_emb(sample, page_pos.unsqueeze(1))
            cos_new, sin_new = backbone.lm.rotary_emb(sample, recv.unsqueeze(1))
            cos_d, sin_d = delta_cos_sin(cos_old.float(), sin_old.float(),
                                         cos_new.float(), sin_new.float())
            for li in range(bank.n_layers):
                block = park_bank["k"][li][:, :, bank.spans[p][0]:bank.spans[p][1], :]
                keys_out[li].append(
                    rerotate_keys(block.float(), cos_d, sin_d).to(block.dtype))
                vals_out[li].append(bank.values[p][li])
            cursor = int(recv.max()) + 1
            pos_max = max(pos_max, int(recv.max()))
            n_recv += 1

    for li, layer in enumerate(cache.layers):
        layer.keys = torch.cat([layer.keys] + [k.to(layer.keys.dtype)
                                               for k in keys_out[li]], dim=2)
        layer.values = torch.cat([layer.values] + [v.to(layer.values.dtype)
                                                   for v in vals_out[li]], dim=2)
    n_total = sum(kept_counts)
    new_cursor = pos_max + 1
    print(f"[KVCFAddr] routed {n_recv}/{n_pages} pages to receiver "
          f"(routes={''.join(routes)}), restored {n_total} cols", flush=True)
    return new_cursor, n_total, routes


__all__ = ["route_and_restore"]
