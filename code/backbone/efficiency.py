"""Non-invasive efficiency instrumentation for the LatentMAS backbones.

Purpose
-------
Quantify the KV-cache-carry vs re-prefill trade-off with two extra numbers per
item, WITHOUT touching any existing timing / token accounting:

  * FLOPs  — analytical, leading-order estimate accumulated over EVERY language-
             model forward pass (prefill AND each decode step) via a single
             forward pre-hook on the decoder module. Because it is driven by the
             real per-call (new_tokens, cached_len), it automatically reflects
             how much a run re-prefills: a --kv_carry run reuses the cache and
             pays a small prefill; a re-prefill run recomputes the growing
             context each pass and its total FLOPs is larger. Comparing the two
             runs' total_flops IS the KV-cache saving (no counterfactual needed).

  * TTFT   — time-to-first-token of the final answer decode, measured with a
             LogitsProcessor whose first invocation marks "first token ready".
             The processor returns the scores unchanged, so the generated text
             is identical (measurement is side-effect free).

Everything here is additive: if a backbone does not wire it up, nothing changes.

FLOPs model (leading order, per LM forward of q new tokens over c cached tokens):

    flops(q, c) = 2 * P_layer * q            # dense weight matmuls, new tokens only
                + 4 * L * d_model * q * (c+q) # attention scores + context mix

  P_layer  = parameter count of the transformer blocks (embeddings excluded,
             since embeddings are a lookup, not a matmul per hidden dim).
  L        = number of decoder layers.
  d_model  = hidden size.

This ignores softmax / norm / rotary / MLP-activation elementwise costs (all
lower order) and vision-encoder FLOPs (separate concern from the LM prefill that
the carry-vs-reprefill comparison is about). It is an estimate meant for
apples-to-apples run comparison, not a hardware bill.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class ModelFlops:
    """Config-derived constants for the analytical FLOPs estimate."""

    n_layers: int
    d_model: int
    p_layer: float  # transformer-block parameter count (embeddings excluded)

    @classmethod
    def from_language_model(cls, lm) -> "ModelFlops":
        cfg = getattr(lm, "config", None)
        n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        d_model = int(getattr(cfg, "hidden_size", 0) or 0)
        # Count block params only (exclude the token-embedding / lm-head lookup
        # tables, which do not do a per-hidden matmul the way weight matrices do).
        layers = getattr(lm, "layers", None)
        if layers is not None:
            p_layer = float(sum(p.numel() for p in layers.parameters()))
            if not n_layers:
                n_layers = len(layers)
        else:
            p_layer = float(sum(p.numel() for p in lm.parameters()))
        return cls(n_layers=n_layers, d_model=d_model, p_layer=p_layer)

    def forward_flops(self, new_tokens: int, cached_len: int) -> float:
        q = max(int(new_tokens), 0)
        c = max(int(cached_len), 0)
        if q == 0:
            return 0.0
        dense = 2.0 * self.p_layer * q
        attn = 4.0 * self.n_layers * self.d_model * q * (c + q)
        return dense + attn


def _seq_len_of(value: Any) -> int:
    """Best-effort cached length from a past_key_values object."""
    if value is None:
        return 0
    getter = getattr(value, "get_seq_length", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            pass
    # legacy tuple cache: ((k, v), ...) with k shape [B, H, S, D]
    try:
        return int(value[0][0].shape[-2])
    except Exception:
        return 0


@dataclass
class ForwardFlopsProfiler:
    """Accumulates analytical FLOPs over every decoder forward pass in an item.

    Registered once as a forward pre-hook on the language-model module; it reads
    the real (new_tokens, cached_len) per call, so prefill (large new_tokens) and
    decode (new_tokens == 1) are handled uniformly with no per-call-site edits.
    """

    mflops: ModelFlops
    prefill_flops: float = 0.0
    decode_flops: float = 0.0
    prefill_tokens: int = 0
    decode_steps: int = 0
    n_forwards: int = 0
    # counterfactual "no KV cache" cost: what each forward WOULD have cost if it
    # re-prefilled its whole context (c+q) from scratch instead of reusing the
    # cache. Trajectory-independent within a run → actual-minus-this isolates the
    # pure caching saving. Split by scope so the AGENTIC KV-carry saving (prefill/
    # append forwards, the method's novelty) is separable from the STANDARD
    # autoregressive decode-cache saving (q==1 forwards, which every LLM gets).
    prefill_flops_nocache: float = 0.0
    decode_flops_nocache: float = 0.0
    # scratch for TTFT arming (used by the backbone wrappers, not the hook)
    extra: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.prefill_flops = 0.0
        self.decode_flops = 0.0
        self.prefill_tokens = 0
        self.decode_steps = 0
        self.n_forwards = 0
        self.prefill_flops_nocache = 0.0
        self.decode_flops_nocache = 0.0
        self.extra = {}

    # torch forward pre-hook (registered with with_kwargs=True)
    def __call__(self, module, args, kwargs):
        # new-token count: inputs_embeds [B, S, D] or input_ids [B, S]
        emb = kwargs.get("inputs_embeds")
        if emb is None and args:
            first = args[0]
            if isinstance(first, torch.Tensor):
                emb = first
        ids = kwargs.get("input_ids")
        if emb is not None and hasattr(emb, "shape"):
            q = int(emb.shape[1]) if emb.dim() >= 2 else int(emb.shape[0])
        elif ids is not None and hasattr(ids, "shape"):
            q = int(ids.shape[1]) if ids.dim() >= 2 else int(ids.shape[0])
        else:
            return None
        c = _seq_len_of(kwargs.get("past_key_values"))
        flops = self.mflops.forward_flops(q, c)
        # counterfactual: recompute the full context from scratch (no cache reuse)
        flops_nocache = self.mflops.forward_flops(q + c, 0)
        self.n_forwards += 1
        if q == 1:
            self.decode_flops += flops
            self.decode_flops_nocache += flops_nocache
            self.decode_steps += 1
        else:
            self.prefill_flops += flops
            self.prefill_flops_nocache += flops_nocache
            self.prefill_tokens += q
        return None

    def summary(self, ttft_sec: Optional[float] = None) -> dict:
        total = self.prefill_flops + self.decode_flops
        nocache_total = self.prefill_flops_nocache + self.decode_flops_nocache
        # agentic KV-carry saving = re-prefilling every append vs reusing the KV
        # (the method's contribution). Decode saving = standard AR KV cache.
        carry_saving = self.prefill_flops_nocache - self.prefill_flops
        decode_saving = self.decode_flops_nocache - self.decode_flops
        total_saving = nocache_total - total
        out = {
            "flops_total": total,
            "flops_prefill": self.prefill_flops,
            "flops_decode": self.decode_flops,
            "prefill_tokens_total": self.prefill_tokens,
            "decode_steps_total": self.decode_steps,
            "n_lm_forwards": self.n_forwards,
            # counterfactual (no-cache re-prefill) and the isolated savings
            "flops_reprefill_nocache": nocache_total,
            "flops_prefill_nocache": self.prefill_flops_nocache,
            "flops_decode_nocache": self.decode_flops_nocache,
            "flops_carry_saving": carry_saving,
            "flops_decode_cache_saving": decode_saving,
            "flops_total_saving": total_saving,
            "carry_saving_ratio": (carry_saving / self.prefill_flops_nocache)
            if self.prefill_flops_nocache else None,
            "total_saving_ratio": (total_saving / nocache_total) if nocache_total else None,
        }
        if ttft_sec is not None:
            out["ttft_sec"] = round(float(ttft_sec), 4)
        return out


class FirstTokenTimer:
    """LogitsProcessor that records the wall-clock time of the first decode step.

    Attach to a `model.generate(..., logits_processor=[timer])` call. `t_first`
    minus the generate-call start time is the time-to-first-token (prefill
    latency). Returns scores unchanged, so the sampled tokens are unaffected.
    """

    def __init__(self) -> None:
        self.t_first: Optional[float] = None

    def __call__(self, input_ids, scores):
        if self.t_first is None:
            self.t_first = time.time()
        return scores


class EfficiencyMixin:
    """Optional mixin giving a backbone reset_eff / eff_summary and TTFT helpers.

    A backbone opts in by calling `self._init_efficiency()` after `self.lm` is
    set. Then:
      * reset_eff()       — zero the per-item counters (call before each item)
      * eff_summary()     — dict of accumulated FLOPs (+ last TTFT) for the item
      * _ttft_processors()/_record_ttft(...) — wrap a model.generate to time TTFT
    All are no-ops / return {} if efficiency was never initialised, so existing
    code paths are unaffected.
    """

    def _init_efficiency(self) -> None:
        try:
            mflops = ModelFlops.from_language_model(self.lm)
            self._eff = ForwardFlopsProfiler(mflops)
            self._eff_handle = self.lm.register_forward_pre_hook(
                self._eff, with_kwargs=True)
            self._last_ttft: Optional[float] = None
            print(f"[Backbone/eff] FLOPs profiler on: L={mflops.n_layers}, "
                  f"d={mflops.d_model}, block_params={mflops.p_layer/1e9:.2f}B")
        except Exception as exc:  # never break inference over instrumentation
            print(f"[Backbone/eff] disabled ({exc})")
            self._eff = None

    def reset_eff(self) -> None:
        if getattr(self, "_eff", None) is not None:
            self._eff.reset()
            self._last_ttft = None

    def eff_summary(self) -> dict:
        if getattr(self, "_eff", None) is None:
            return {}
        return self._eff.summary(ttft_sec=getattr(self, "_last_ttft", None))

    def _ttft_timer(self):
        """Return (timer, t_start) to pass into a timed generate, or (None, None)."""
        if getattr(self, "_eff", None) is None:
            return None, None
        return FirstTokenTimer(), time.time()

    def _record_ttft(self, timer, t_start) -> None:
        if timer is not None and getattr(timer, "t_first", None) is not None:
            self._last_ttft = timer.t_first - t_start
