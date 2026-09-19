# Actual latent-KV to visual-KV relay

This implements the user's corrected request. It is separate from the rejected PVCR synthetic-V replacement experiment. New code is under `latent_visual_relay/`; old experimental and shared native files are unchanged.

## Data path

At each native sender latent step, observe attention from that latent query to the original visual keys, keeping the full softmax denominator. Visual values are never pooled or used as payload. The score only chooses which **original latent K/V head entries** to copy.

For each layer8–20 and KV head, compute visual/neighbor grounding per step. Spatial grounding is visual attention mass times positive core/context density enrichment over nonvisual background. Core/context density uses a harmonic mean, so one side alone is insufficient. Navigator uses thumbnail8-connected neighborhoods; Reasoner uses actual x20 children and their x5 parent. Reuse actual packed image-grid metadata, not arbitrary temporal step pairs.

Within each layer/head's10-step score vector, select values strictly above median+MAD and above zero. Compute these tiny selection statistics on CPU for deterministic median behavior. There is no forced top-k or last-step fallback. The union of selected steps determines dense cache rows; only selected layer/head entries carry copied data, and other entries are zero-filled and masked out of attention. Therefore10 appended rows can still contain only a subset of the1040 candidate mid-layer/head/step entries.

At the receiver handoff, physically insert the selected source K/V after the original visual block:

`[original prefix + original visual KV + selected latent K/V copies + remaining original cache]`

The same insertion runs at Navigator→Reasoner and Reasoner→Answerer. Original visual and original latent values are preserved bit-for-bit at insertion. Copied heads keep the source K and V exactly: no V replacement, no synthetic visual-V sum, no norm scaling, no projection into a different representation, and no additional log2 attention bias. Original rotary key phases and logical MRoPE cursor are retained; physical cache length grows by the number of inserted rows. This is valid for the installed backbone because it tracks its logical MRoPE cursor separately from physical cache length. The extra rows do not introduce new latent-generation steps.

The Answerer clone's inserted-head masks are scoped to that decode and restored afterward, so a format-repair decode does not inherit columns from a discarded clone. Empty selection inserts nothing. Both insertion and attention enforce the fixed12288 context limit; existing patches/prompts are not pruned to create space.

## Important interpretation limits

These are visual-grounded **latent head entries**, not proven pure visual components. A head can still contain text/context information. The native Base already retains the original latent block: copying selected entries adds another read opportunity and changes their effective attention weight; it does not create information absent from the source. Spatial-shuffled and ungrouped-visual modes are available as controls. Accuracy improvement and pathology-specific novelty remain empirical questions, not implementation claims.

## Execution

`bash 0912/run_latent_visual_relay.sh spatial 8 fresh_output_name 0 1`

Modes: relational (parent x5 + paired x20 morphology), spatial, shuffled, visual, off(observe without inserting), native(no adapter). Fixed Qwen3-VL-4B-Thinking, cumulative transport,10 latent steps,12 patches,12288 context, SDPA, seed42, deterministic/greedy, tcga_expert_vqa. Only GPU7/8 are accepted. Fresh output directories and source hashes preserve experiment identity.

## Verification

- New9 tests cover adaptive/empty selection, exact latent K and V provenance, preservation of original cache, per-head GQA masks, actual PyTorch SDPA output, core/context grounding and context limit rejection. Combined with the existing9 PVCR regression tests:18 passed.
- First spatial smoke IDs0/1:2 completed,0 failures. ID0 selected381 Navigator head entries across10 dense rows,42 Reasoner entries across6 dense rows; exact source-copy and original-cache checks both true. ID1 Reasoner used1 dense row. This smoke preceded the CPU-median determinism change; final-source results are recorded separately.
- Off smoke IDs0/1 completed; compare against native same-ID answers and patch coordinates before interpreting intervention results.
- Runtime audits: `runs/<name>/latent_visual_relay/<id>.json`; include scores, selected head masks, source columns, measured before/after cache lengths, insertion offsets, exact-copy checks and receiver attention-call counts.
- No GPU job belonging to another experiment is stopped or modified.

## Final-source evidence and full evaluation

- Final source: `runs/lvr_spatial_final_smoke_0912`, IDs0/1, `completed=2 failed=0 requested=2`. Every source hash in `latent_visual_relay_settings.json` matches delivered code. CPU-median warning is absent. Mean case wall time27.93s and peak reserved memory21,451,767,808bytes in this smoke; timing is not a fair full-dataset speed comparison because other GPU jobs were sharing the devices.
- Off/native full answer objects and patch ID/magnification/coordinates matched exactly on IDs0/1 (`cmp` exit0 for each).
- Final configured basedpyright:0 errors; ruff:all checks passed; combined tests18 passed in4.88s. The injected editor checker uses a different search root/interpreter; configured project checks use the actual experiment Python3.12 environment.
- Code responsibilities are separated into grounding selection, cache insertion, observation, attention masking, and native role adapters. No untyped escapes or native-source edits were added. Runtime adapter244 nonblank/noncomment lines is in the size warning band; split role wrappers before expanding that module further.
- Full128 launched in tmux: `lvr128_gpu7` → `runs/lvr128_gpu7_0912` (IDs0–63), `lvr128_gpu8` → `runs/lvr128_gpu8_0912` (IDs64–127). Logs: `lvr128_gpu7.log`, `lvr128_gpu8.log`. Completion/accuracy is not yet claimed.
- Both full-run workers have produced completed case artifacts, not just loading messages. The run manifests confirm64 disjoint IDs each; ID0 and ID64 audit records confirm exact latent-source copies and preservation of original cache. Cases with no qualifying Reasoner heads (e.g.ID2) correctly insert zero Reasoner rows instead of a fallback. CLI help exits0.
