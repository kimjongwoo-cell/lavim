# PVCR retained experiment audit, 2026-09-12

Scope: isolated `0912/pvcr`, `run_pvcr.py`, `run_pvcr.sh`, `test_pvcr.py`.
Runtime: Python 3.12 via uv and `/home/users/whddn12316/venvs/wsi-latentmas-py312`.
GPU 7 and 8 were both idle (34 MiB); no debugger ports or watchers started.
Git SHA: unavailable: this project is not a Git checkout. Record source hashes instead; never invent a commit.
Read references: debugging Python, setup, investigate; programming Python (earlier).

## Hypotheses and evidence checks

1. Wrong GQA grouping/mask causes invalid synthetic memory. Check explicit attention reference and causal mask tests.
2. Runtime callbacks miss latent steps or the receiver reads a different cache. Check 10 steps x 13 layers at each sender and receiver callback counters in real Qwen runs.
3. Spatial metadata does not align with packed visual tokens. Check actual grid_thw, 4 parent/8 child containment, and thumbnail-only Navigator layout during real inference.
4. In-place replacement leaks into later cases or persists after cache growth. Check restoration after growth/error and no-intervention reference equality.

## Artifacts planned before runtime work

- Retain tests, `PVCR_DESIGN.md`, audit JSON and experiment logs as requested experiment evidence.
- `0912/runs/pvcr_smoke_*`: two-ID smoke outputs; fresh names only, no overwrite.
- tmux sessions `pvcr_smoke_off` and `pvcr_smoke_spatial`: run to exit, no debug listeners.
- Additional controls and full 128-case runs only after smoke checks and review.
- No breakpoints, global env edits, or legacy source modifications.

## Observed so far

- Contract tests: `4 passed in 3.31s`.
- Correct-environment static check: `0 errors, 0 warnings, 0 notes` using `uvx basedpyright --project 0912/pyrightconfig.json` with explicit PVCR paths.
- Editor hook points at a different interpreter/search root and reports unresolved installed dependencies; explicit project check above resolves them without changing shared editor configuration.

## Runtime round 1

- Both smoke runs failed before latent work with `LayoutError: PVCR requires upstream transport and 10 sender steps`.
- Observed run manifest: `transport_mode: cumulative`. Existing Base manifest also says `cumulative`; preserve it, do not switch experimental protocol.
- Frozen legacy LayoutError then caused `TypeError: super(type, obj): obj must be an instance or subtype of type` when contextlib attached traceback. Isolated mutable PVCR error now preserves original exceptions; legacy sources unchanged.
- The latent block is before turn-closing tokens, not the closed cache's last ten columns. Use observed positions and compare Reasoner positions with native `_rpath_latent_cols`.
- Red test: shuffled neighborhood counts `[3,5,5,2,...]` vs real `[3,5,5,3,...]`. Fixed by deterministic distant column roll without dropping neighbors. `9 passed in 4.17s` after fixes.
- Planned: fresh smoke directories `pvcr_smoke_off_v2_0912`, `pvcr_smoke_spatial_v2_0912`; matched native and shuffled follow. Retain all failed logs as audit evidence.

## Runtime round 2

- Native, off, spatial, shuffled each completed 2/2 with zero terminal failures, IDs0/1. Native/off full answer objects and selected patch coordinates are exactly equal for both IDs.
- Spatial ID0: Navigator median V-norm ratio `0.0008298349566757679`; Reasoner median `0.0032207826152443886`. ID1 medians `0.0008377779158763587` and `0.00271703430917114`.
- Both senders observed `[1, 8, 10, 128]` per-layer synthetic values, 13 layers. ID0 receiver calls `navigator:156, reasoner:559`.
- Scientific finding: replacing latent V with such small raw visual contributions substantially suppresses the source memory. This is not established as the desired visual-context augmentation. Do not launch128 under this formulation.
- Review fixes: measured cache lengths replace the hard-coded token-growth claim; require structured_json so Answerer hook cannot silently be skipped; restrict shell GPU to7/8; record observed query heads and GQA ratio.
- Planned final instrumentation QA: `pvcr_smoke_audit_0912` ID0 on GPU8; diagnostic only, not128. Retain log `0912/pvcr_smoke_audit.log`. Then no background experiments remain.

- Instrumentation QA completed1/1. Observed Q32/KV8/group4 in both stages. Cache substitution/restoration lengths: Navigator3480→3480 and8391→8391; Reasoner8391→8391 and8781→8781. Native receiver generation accounts for the allowed intermediate growth.
- Final source delta: explicit isinstance checks for actual Qwen vision module/native latent columns, lint-only formatting and test context spelling. Plan one exact-final-source ID0 run `pvcr_smoke_final_0912`, log`0912/pvcr_smoke_final.log`, GPU8; no128.

## Final runtime record

- Final source smoke: `completed=1 failed=0 requested=1`; manifest source-hash check: `source_hash_mismatches []`.
- Exact source fingerprints: `runs/pvcr_smoke_final_0912/pvcr_settings.json`. Git commit SHA is unavailable (not a Git checkout). This record binds only the explicit source hashes, not an invented commit.
- Runtime audit PASS for execution, GQA shape observation, both receiver hooks, actual cache-length preservation, finite contributions, and restoration; method approval remains FAIL for the visual-component extraction/scale issue.
- Test suite9 passed; ruff `All checks passed!`; configured basedpyright `0 errors, 0 warnings, 0 notes`.
- No PVCR tmux sessions remain. GPU7 and8 each`34 MiB, 0%` after completion. No temporary debug statements/listeners or global shell environment changes were left. No existing legacy files or other users' jobs were modified.
- This journal has been promoted to retained experiment evidence; failed and successful run logs are intentionally preserved. No material user data was deleted.
