# Spatial latent: fixed 16-ID pilot

## Result

No observed improvement at gain 2, layers 8–20, with eight neighborhood steps
and two unbiased synthesis steps. Both real-parent and shuffled-parent variants
produced exactly the same final answer choice as Base on **all 16 IDs**.

| Method | Correct | Accuracy | Mean case wall time |
|---|---:|---:|---:|
| Existing Base, SDPA | 5/16 | 31.25% | 24.22 s |
| Existing Pruning-B, SDPA | 6/16 | 37.50% | 22.00 s |
| Real-neighborhood latent | 5/16 | 31.25% | 22.66 s |
| Shuffled-neighborhood latent | 5/16 | 31.25% | 23.13 s |

These are the same fixed IDs `0, 8, 16, ..., 120`, not the whole 128-question
evaluation. Real and shuffled pilots each completed 16/16 with zero failures.
Baseline timing is historical and can differ due to load and generated answer
length; the table does not establish a computational speedup. Average peak
allocated GPU memory for both new arms was about 13.60 GiB.

## What was verified

- Qwen3-VL-4B-Thinking, 12 patches, 10 latent steps, 12,288 maximum context.
- Exact matching patch IDs, magnifications, boxes, parent IDs, role prompts,
  slide IDs and labels across the compared runs.
- Every child was physically contained in its actual parent before grouping.
- Each new arm used all 12 patches once across four groups, each containing
  one 5x and two 20x patches (768 visual tokens per group).
- Real grouping preserved parent identity; shuffled grouping broke every
  fine-parent association while preserving group sizes and scales.
- Every case recorded exactly 13 biased layer calls on each of steps 1–8,
  and zero on steps 9–10. Active-step cache lengths remained within 12,288.
- No-op GPU smoke (gain 1, case 0) matched the old Base's entire answer object,
  patches and role prompts. Guided smoke changed answer text/rationale but not
  its final choice, so matching pilot choices are not evidence of an inert hook.
- Ten CPU behavioral tests passed: parent grouping, seeded control coverage,
  cache-offset mapping, invalid layout, actual SDPA output changes, scope,
  existing-mask preservation and adapter restoration.
- Ruff's E/F/I/UP/B checks, compile checks and shell syntax checks passed.
  Project-environment basedpyright: zero errors/warnings for implementation and
  tests. Standalone report check: zero errors, 16 inference/stub warnings around
  Arrow and unannotated analysis intermediates. The automatic editor LSP uses
  an unrelated environment and still reports missing packages/old Python.

## Interpretation

The pilot does not support an accuracy benefit from this spatial attention prior.
It does not establish that spatial context is unimportant in pathology, nor that
latent states contain no visual information. Potential explanations include weak
downstream use of the changed latent states, convergence during the last two
unbiased steps, or an ineffective grouping/bias schedule. This experiment alone
does not distinguish these explanations. Attention bias and hook activity do not
prove diagnostic spatial information was represented or used.

No 128-case expansion or gain tuning was performed after viewing these answers.
No latent-mediation claim is made. A subsequent causal test would need to isolate
latent-state influence while preserving original visual evidence and cache layout.

## Artifacts

- `spatial_pilot_comparison.json`: per-ID outcomes, source paths, source hashes,
  shared-ID validation and timing summary.
- `runs/spatial_spatial_pilot_gpu6/`: real-neighborhood outputs and per-case traces.
- `runs/spatial_shuffled_pilot_gpu8/`: shuffled-neighborhood outputs and traces.
- `runs/spatial_noop_smoke_case0_v2/`: unchanged-Base GPU control.
- `runs/spatial_guided_smoke_case0_v2/`: live intervention GPU smoke.
- `spatial_pilot_gpu6.log`, `shuffled_pilot_gpu8.log`: completion records.
- `spatial_latent/README.md`: method and reproduction commands.

All new source and artifacts are under `0912`; no shared pipeline source or
existing experiment results were modified. This is a training-free experimental
variant, not a new trained model.
