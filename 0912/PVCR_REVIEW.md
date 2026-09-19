# PVCR review: full evaluation withheld

## Outcome

The prototype executes both role boundaries, but it is not yet an acceptable implementation of latent-V visual-component extraction. It computes a visual-source attention sum using the latent query, then substitutes that synthetic sum for the original latent V. This is explicitly not a decomposition of information stored within latent V.

In two real cases, the synthetic vectors were much smaller than the original latent values. Replacing the latter can suppress existing latent information. An accuracy change under this intervention cannot by itself establish a benefit from spatial context. Do not claim improvement, information purity, or pathology-specific novelty; do not launch the128-case evaluation on this formulation.

## Measured evidence

Sources: `runs/pvcr_smoke_spatial_v2_0912/pvcr/{000,001}.json`; each stage has13 layers and10 steps. Norm ratios below are medians over10 recorded step-wise head/layer-averaged ratios, not every individual head.

| ID | Navigator synthetic/original V norm | Reasoner synthetic/original V norm |
|---|---:|---:|
| 0 | 0.000829835 | 0.003220783 |
| 1 | 0.000837778 | 0.002717034 |

Native, observation-only, spatial and shuffled arms each completed IDs0/1 with zero terminal failures. Native and observation-only full answer objects and patch coordinates matched exactly for both IDs. These are smoke checks, not accuracy estimates or a controlled timing benchmark.

Correct-environment tests:9 passed; explicit basedpyright reports zero errors; ruff clean. The editor's injected diagnostics use a different interpreter/search root and are not the experiment's configured type-check environment.

Later instrumentation run `runs/pvcr_smoke_audit_0912/pvcr/000.json` completed1/1 and recorded actual Q32/KV8/group4 at both stages. Source latent rows have shape `[1,8,10,128]` per layer, hence `[1,8,1,128]` for one step. Both K/V are organized per layer; no single flattened latent-KV vector represents all layers.

Final exact-source rerun `runs/pvcr_smoke_final_0912` also completed1/1. Every source hash in its `pvcr_settings.json` matches the delivered code (`source_hash_mismatches []`). It reproduced the measured shapes, receiver calls and cache-length evidence below. This establishes technical execution, not scientific effectiveness.

Measured cache lengths:

| Handoff | Before/after V substitution | Before/after restoration |
|---|---|---|
| Navigator→Reasoner | 3480 / 3480 | 8391 / 8391 |
| Reasoner→Answerer | 8391 / 8391 | 8781 / 8781 |

The intermediate growth is native receiver generation, not relay token insertion.

## Review lanes and corrections

Initial source snapshot identity is preserved in `runs/pvcr_smoke_spatial_v2_0912/pvcr_settings.json`. This is not a Git checkout; no commit SHA exists and none is fabricated. Lane verdicts are not reused as approval for later source changes.

| Lane | Initial verdict | Findings / disposition |
|---|---|---|
| Goal | FAIL | Hard-coded cache audit replaced by actual measurements; method-level suppression/extraction mismatch remains unresolved. |
| Code | FAIL | Alternative Answerer protocols could silently bypass the hook; runner now requires structured_json. |
| Security | FAIL | GPU launcher accepted arbitrary indices; now rejects anything except7/8 before model launch. |
| Context | FAIL | Added actual query-head/group recording and cache measurements;128 remains deliberately unrun because the method gate failed. |
| Hands-on QA | PASS, technical only |9 CPU tests, receiver SDPA driver, CLI rejection/help, and read-back of exact-source GPU smoke; see `pvcr_review_qa.md`. Scientific effectiveness remains unapproved. |

The QA PASS and root runtime-audit PASS bind to the exact SHA256 map in `runs/pvcr_smoke_final_0912/pvcr_settings.json`; commit SHA is N/A because no Git checkout exists. The overall review remains **FAILED**, not five-lane approval. Initial failing lane reviews are retained as findings; subsequent targeted corrections and current-source QA do not erase the unresolved method blocker.

## Next decision

Recommended redesign: keep the original latent K/V intact and use visual/neighbor grounding to select or weight an additional read from those latent rows/heads. Compare against equal-magnitude nonspatial and suppression controls. Grounding scores do not prove that the selected latent values are free of text information. This is a proposed next experiment, not an implemented or validated fix. User direction is required before replacing the experimental formulation.
