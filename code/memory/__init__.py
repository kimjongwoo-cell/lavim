"""Diagnostic memory layer (S1–S6).

Currently implemented:
  - saliency.py   : S2 text-query-driven visual-token saliency (+ S1 per-ROI spans)
  - prune.py      : S3 intra-pass visual-KV compression (safe / method-literal Π_i)
  - unit.py       : S4 ROI memory unit u_i = (Π_i, z_i, meta)   [default off]
  - reallocate.py : S8 training-free attention reallocation (ViF-style)

Not yet implemented (core novelty — see method.md §Cross-Iter Scoring / Bank Update):
  - bank.py       : S5 spatial diagnostic bank (Sal/Rel/Cov, IoU, greedy B_roi/B_kv)
"""
