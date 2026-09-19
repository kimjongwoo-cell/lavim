# Spatial latent experiment

## Question

Does explicitly pairing high-magnification evidence with its real low-magnification neighborhood during Reasoner latent computation improve the existing Base?

## Fixed protocol

- Qwen3-VL-4B-Thinking; 12 patches; 10 latent steps; maximum context 12,288.
- Existing Base prompts, navigation, realignment, cache transport and Answerer.
- Four groups, each one 5x anchor plus its two 20x children. Two steps per group, followed by two unbiased synthesis steps.
- Add log(2) to the group's attention logits in Reasoner layers 8 through 20 (zero-based), only for single-query latent calls. Other keys remain accessible. SDPA remains active.
- Control: identical group sizes and scales, with the 20x patches assigned to different 5x parents using seed 42. No label-dependent choices.
- First pilot: dataset IDs 0, 8, 16, ..., 120, fixed before seeing new predictions. Compare with the same existing Base IDs. Include audit of identical patches/prompts and actual hook execution.
- This is a pilot, not a held-out novelty or efficacy claim. Do not tune on its answers.

## Task tracking

1. [completed] Implement isolated grouping, SDPA bias, runtime adapter and behavioral tests under 0912. Ten behavioral tests pass.
2. [completed] Verify CPU contracts and GPU smoke at the exact fixed protocol. Case 0: no-op answer object, patches and prompts match old Base exactly; guided mode fires 13 layers for each of 8 steps and zero on both synthesis steps.
3. [completed] Both 16-case pilots completed without failures. Paired results and audits are in `spatial_pilot_comparison.json`.
4. [completed] Findings and limits recorded in `SPATIAL_LATENT_RESULTS.md`. Neither intervention changed any final choice on the 16 IDs. No gain warranted a 128-case expansion or latent-mediation experiment in this turn.

## Isolation

No shared source file is edited. The experiment runner installs and restores adapters inside its own process. Output directories are unique and contain an experiment manifest and per-case traces. Existing jobs and results are preserved.

## Validation notes

- The automatic LSP hook uses an unrelated Python environment and reports missing Torch/Transformers and old-Python syntax. The explicit project-environment basedpyright check reports zero errors/warnings. No shared editor configuration was changed.
- Initial launcher smoke exposed the upstream CLI's requirement that output-root not already exist. The adapter now lets the upstream CLI create that directory and persists its manifest afterwards. Rejected output directories were preserved; successful smoke outputs have `_v2` suffixes.
- Smoke outputs: `runs/spatial_noop_smoke_case0_v2` and `runs/spatial_guided_smoke_case0_v2`.
