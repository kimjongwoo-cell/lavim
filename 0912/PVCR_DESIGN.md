# PVCR experimental contract

Status: **method review failed; full128 evaluation withheld**. The diagnostic prototype runs, but raw synthetic-V replacement does not establish the desired latent-KV context augmentation. See `PVCR_REVIEW.md`.

## Corrections to the proposal

- A visual-source attention output is not an exact decomposition of cached latent V. The cached V is computed before that layer's attention/MLP. We construct synthetic memory read by the next role.
- Qwen3-VL-4B uses 32 query heads, 8 KV heads, head dimension 128. Average attention probabilities within each group of four query heads, then pool values in the corresponding KV head's own space. Never reshape concatenated attention outputs into KV heads.
- Use the full causal attention denominator, including text and prior latent tokens. Do not normalize solely over visual tokens. This excludes direct text-source summands, but visual values and queries can already contain contextual text information.
- Architectural appearance at x5 and cellular appearance at x20 need not have matching vector directions. Do not use their cosine as biological agreement or discard discordant rare evidence.
- The Navigator receives thumbnail and annotated root grid, not x20 crops. Its neighborhoods use actual merged thumbnail token coordinates; exclude the annotated root grid from the synthetic values. Reasoner neighborhoods use explicit parent-child metadata and geometric containment.
- Existing cumulative transport already preserves both sender latent blocks. Use their observed original cache addresses and keys; temporarily substitute their V for receiver computation, then restore. Turn-closing tokens follow the latent block; never assume the last ten cache columns are the latent block. No token duplication, position reassignment, or RoPE rotation.
- No arbitrary pairing of consecutive latent steps with tissue regions. Associate each step using measured attention and real region metadata.

## Operation

At layers 8 through 20, observe each of the sender's existing 10 latent forwards without changing its output. For full attention A and each spatial group r, compute core/context attention mass per token. Their harmonic mean provides a soft region weight. Normalize region weights only, while retaining the original full-denominator A in every value contribution. Pool each group's visual contribution, then combine groups using those weights. Zero support gives zero contribution; no forced fallback row selection.

Both boundaries use this same formula. Navigator core is a thumbnail token with its 8-connected neighbors as context. Reasoner core is the two x20 children and context is their actual x5 parent. This is an experimental spatial prior, not proof of pathology-specific novelty.

During Reasoner append, substitute Navigator latent values and add a fixed log(2) attention bias at their original columns. During Answerer decode, do the same with Reasoner values. Outside the receiver scope, restore original values. Prefill masks remain causal. No changes to Base prompts, visual acquisition, latent generation schedule, model weights, or token budgets.

## Evaluation

Fixed Qwen3-VL-4B-Thinking, SDPA, 10 latent steps per existing sender, patch budget 12, context 12288, IDs 0-127, GPUs 7 and 8. Do not tune against these evaluation labels. Record step/layer counts, GQA shapes, source mass, synthetic/original V norm ratio, receiver calls, and unchanged cache length at entry/exit of substitution (receiver generation itself can grow cache).

Controls exposed by the same entrypoint: off (observation only), original V with identical addressing/bias, ungrouped visual contribution, shuffled spatial relationships, Navigator only, Reasoner only, both. Smoke checks include off vs native Base on identical IDs, and real vs shuffled. Accuracy of 50% is a target, not a predicted outcome.

## Sources informing the audit

- Local Qwen implementation: `transformers/models/qwen3_vl/modeling_qwen3_vl.py`, attention projections and cache update preceding attention.
- GQA head sharing: https://arxiv.org/abs/2305.13245
- Existing spatial correlation work in pathology (not evidence that this relay is novel): https://arxiv.org/abs/2106.00908
