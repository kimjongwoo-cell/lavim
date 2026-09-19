# Cross-scale relation latent: fixed 16-ID pilot

## Research framing

The first pathology-specific challenge is efficient evidence retention under WSI
token redundancy; pruning addresses that challenge. The second is contextual
morphology: diagnoses depend on relations between cellular detail at high power
and tissue architecture, boundary, distribution, and stroma at lower power.
Plain LatentMAS receives both scales but does not explicitly represent that
parent-child relation during its recurrent latent computation.

This pilot tests one implementation of the second axis. It does not equate the
implementation with the broader challenge.

## Intervention

- Steps 1, 3, 5, 7 emphasize one 5x parent context.
- Steps 2, 4, 6, 8 emphasize that parent's two 20x children and the preceding
  context latent. Their K/V rows are designated relation rows.
- Step 9 emphasizes all four relation rows; step 10 emphasizes step 9.
- Layers 8–20 receive log(4) on the relevant source logits.
- The Answerer keeps original visual KV. The readout arm reserves 10% of layers
  8–20 attention output for the four existing relation rows; it does not append
  K/V or create position collisions.
- The shuffled arm preserves patch count and scale but assigns every 20x child
  to a different 5x parent. The no-readout arm forms real relation rows but gives
  them no reserved Answerer mass.

All settings were fixed before reading pilot predictions. Qwen3-VL-4B-Thinking,
12 patches, 10 latent steps, context 12,288 and fixed IDs 0, 8, ..., 120 were
retained.

## Paired result

| Method | Correct | Accuracy | Fixed Base errors | Regressed Base correct | Changed choices |
|---|---:|---:|---:|---:|---:|
| Base | 5/16 | 31.25% | 0 | 0 | 0 |
| Pruning-B | 6/16 | 37.50% | 1 | 0 | 2 |
| Real relation + 10% readout | 5/16 | 31.25% | 1 | 1 | 4 |
| Shuffled relation + 10% readout | 5/16 | 31.25% | 1 | 1 | 4 |
| Real relation + 0% readout | 5/16 | 31.25% | 0 | 0 | 0 |

Real and shuffled relation arms produced the same choice on all 16 cases. The
no-readout arm matched Base on all 16 cases. Therefore this pilot provides no
evidence that the parent-child relation caused the changed decisions. The four
changes are attributable to reserving Answerer attention for latent rows under
this design, not to correct spatial pairing.

## Mechanism and protocol audit

- Every relation run completed 16/16 with zero failure artifacts.
- Exact slide, gold label, role prompt, patch ID, magnification, box and source
  parent matched Base for every paired case.
- Each real child was geometrically contained by its declared parent. The
  shuffled condition broke every parent-child association.
- All ten steps fired on exactly 13 intended layers. Context steps addressed 256
  tokens, child-relation steps 512 tokens, and synthesis steps the latent tail.
- Readout traces identify four alternating Reasoner latent addresses. Every 10%
  arm recorded active Answerer calls; every 0% arm recorded zero.
- Maximum observed cache address stayed within 12,288.
- Fifteen CPU behavior tests passed. Project-environment basedpyright reported
  zero errors/warnings; Ruff E/F/I/UP/B checks, compile and shell syntax passed.

Mean wall times were 24.22 s Base, 22.00 s historical Pruning-B, 20.67 s real
relation, 21.78 s shuffled relation and 22.13 s no-readout. Different GPU load
and output length make this pilot unsuitable for latency claims.

## Decision

Do not expand this intervention to 128 cases and do not tune its gain or 10%
budget on these labels. It failed the spatial-specificity control.

The second research axis remains cross-scale contextual morphology. A subsequent
candidate must construct a relation value that changes when parent-child pairing
changes, rather than only changing which existing keys receive more attention.
An appropriate causal form is a question-conditioned contrast between parent
architecture and child morphology, injected into the latent update, followed by
the same real-versus-shuffled and pathway-cut controls.

## Artifacts

- `relation_pilot_comparison.json`: paired cases, aggregation and audit result.
- `runs/relation_spatial_alpha0.1_gpu6/`: real relation plus readout.
- `runs/relation_shuffled_alpha0.1_gpu8/`: shuffled relation plus readout.
- `runs/relation_spatial_alpha0.0_gpu6/`: real relation without readout.
- `runs/relation_real_smoke_case0/`: live GPU mechanism smoke.
- `spatial_latent/relation_attention.py`: relation and readout SDPA scopes.
- `spatial_latent/relation_runtime.py`: Base runtime adapter and trace writer.
- `test_relation_latent.py`: relation-specific behavior tests.

All experiment code and artifacts remain isolated under `0912`.
