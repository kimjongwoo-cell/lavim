# LatentMAS Terminal-Parity Design

## Goal

Make the WSI Latent VL-MAS terminal path follow the cloned LatentMAS behavior:
carry the complete non-terminal embedding record into one free-text terminal
Judger call, without JSON Answerer generation or hidden truncation.

## Scope

- Keep the existing four roles and patch-selection pipeline.
- Keep Navigator JSON because deterministic crop selection requires typed IDs.
- Replace only the terminal Answerer JSON generation with a free-text Judger.
- Preserve strict-v2 dataset, sampling parameters, model, patch budget, and evaluator.

## Transport Contract

For every non-terminal agent, record the embeddings used by its HF rollout in
their original order: the agent input embedding block followed by all `M`
recurrent latent input embeddings. Do not replace recurrent inputs with
post-forward hidden states. Do not retain only a tail.

At the terminal boundary, concatenate every recorded block in agent order and
insert the complete tensor into the Judger prompt embeddings. For three
non-terminal agents and `M=80`, all 240 recurrent latent embeddings are carried,
in addition to the recorded agent input embeddings. vLLM builds its own KV cache
from these prompt embeddings; an HF KV object is not transferred across engines.

If the resulting prompt exceeds `max_model_len=8192`, fail the smoke test with an
explicit context-length artifact. Never silently truncate, prune, or lower M.

## Terminal Judger Contract

The terminal model generates free text once with the existing temperature,
top-p, seed, and token budget, matching upstream LatentMAS generation. Its prompt
ends with an explicit `FINAL_ANSWER:` instruction. A boundary parser extracts the
last non-empty `FINAL_ANSWER:` value and converts it into the existing typed
`AnswerOutput` artifact for unchanged evaluation.

For MCQ and bounded categorical questions, the extracted value must resolve to
exactly one supplied option after the evaluator's existing normalization. For
unbounded open questions, preserve the extracted phrase. Missing or ambiguous
terminal markers are recorded as failures; they are not silently replaced with
the first option. There is no Answerer JSON repair loop.

## Error Handling

- Navigator retains its existing bounded JSON recovery because invalid IDs
  cannot drive the cropper.
- Terminal Judger has one physical generation call.
- No `{` prefix, JSON schema, JSON whitespace grammar, or Answerer JSON fallback
  is used.
- A missing final marker, ambiguous bounded answer, empty output, context
  overflow, or model exception creates a typed failure artifact.

## Verification

1. Unit tests prove that transport contains every agent input embedding and all
   M latent embeddings in order.
2. Unit tests prove that terminal generation receives no JSON schema or prefix
   and executes exactly once.
3. Parser tests cover MCQ, categorical, open-ended, empty, missing-marker, and
   ambiguous outputs.
4. Existing focused latent/vLLM tests remain green.
5. Real M80 smoke tests on dataset indices 0 and 8 must have zero failures,
   zero Answerer retries, one terminal call each, and non-empty parsed answers.
6. Only after the smoke gate passes is a new 735-case output root launched.

## Non-Goals

- No pruning, attention reallocation, tail truncation, M reduction, prompt
  redesign, or evaluator change.
- No reuse of earlier partial M80 results generated under JSON Answerer or
  tail-limited transport.
