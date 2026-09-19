"""0919 efficiency sidecar for the single_v7 baseline (measurement only; generation unchanged).

With SINGLE_EFF_JSONL=<path>, every transformers GenerationMixin.generate call appends one line:
  prefill_tokens (input_ids length, image tokens included), decode_steps (new tokens), wall_sec (generate call),
  ttft_sec (generate start -> first logits processor call, i.e. prefill forward + first step),
  peak_gpu_mem_bytes (torch.cuda.max_memory_allocated after a reset at call start),
  flops_total / flops_prefill / flops_decode: the same analytic LM formula as sender_relay_exp/eff_report.py
  flops(q, c) = 2 P q + 4 L d q (c + q), P = 3.63e9, L = 36, d = 2560 (vision encoder excluded),
  kv_mb: KV cache size after the prompt (2 * L * 8 KV heads * 128 * 2 bytes per token).
Lines are in call order = sample order of the (fresh, non-resumed) run.
"""
import json
import os
import time

_OUT = os.environ.get("SINGLE_EFF_JSONL", "").strip()
if _OUT:
    import transformers
    from transformers import LogitsProcessor, LogitsProcessorList

    P, L, D = 3.63e9, 36, 2560
    KV_BYTES = 2 * L * 8 * 128 * 2

    def _flops(q, c):
        return 2 * P * q + 4 * L * D * q * (c + q)

    class _FirstCall(LogitsProcessor):
        def __init__(self):
            self.t = None

        def __call__(self, input_ids, scores):
            if self.t is None:
                self.t = time.perf_counter()
            return scores

    _orig = transformers.GenerationMixin.generate

    def generate(self, *args, **kwargs):
        import torch
        ids = kwargs.get("input_ids", args[0] if args else None)
        q = int(ids.shape[-1]) if ids is not None else 0
        fc = _FirstCall()
        lp = kwargs.get("logits_processor") or LogitsProcessorList()
        lp.append(fc)
        kwargs["logits_processor"] = lp
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = _orig(self, *args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        seq = out.sequences if hasattr(out, "sequences") else out
        n = max(0, int(seq.shape[-1]) - q)
        f_pre = _flops(q, 0)
        f_dec = sum(_flops(1, q + i) for i in range(n))
        rec = {"prefill_tokens": q, "decode_steps": n, "wall_sec": round(t1 - t0, 4),
               "ttft_sec": round(fc.t - t0, 4) if fc.t else None,
               "peak_gpu_mem_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
               "flops_prefill": f_pre, "flops_decode": f_dec, "flops_total": f_pre + f_dec, "kv_mb": q * KV_BYTES / 2 ** 20}
        with open(_OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return out

    transformers.GenerationMixin.generate = generate
