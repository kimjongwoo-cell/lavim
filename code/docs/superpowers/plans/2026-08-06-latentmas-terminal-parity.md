# LatentMAS Terminal Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry every non-terminal input and recurrent latent embedding into one free-text vLLM Judger call, matching upstream LatentMAS while preserving the WSI evaluator artifact.

**Architecture:** The Qwen HF rollout exports one transport block per non-terminal agent: its prefill input embeddings followed by all M recurrent inputs. The hybrid engine concatenates every block without truncation and performs one schema-free terminal decode. A small typed parser converts the Judger's final marker into the existing `AnswerOutput`, leaving Navigator JSON and the evaluator unchanged.

**Tech Stack:** Python 3.12, PyTorch, Transformers Qwen3-VL, vLLM prompt embeddings, Pydantic v2, pytest, Typer.

---

## File Structure

- Create `vision_text_mas/latent_judger_output.py`: terminal marker parsing and typed parse error only.
- Modify `vision_text_mas/latent_answerer.py`: free-text Judger prompt, one physical decode, and conversion into `AnswerOutput`.
- Modify `backbone/qwen3vl.py`: export each role's prefill input embeddings together with every recurrent latent input.
- Modify `vision_text_mas/latent_vllm_hybrid.py`: retain full transport blocks, enforce context length, and remove tail truncation.
- Modify `vision_text_mas/latent_vllm_hybrid_cli.py`: record full-transport/free-text contract in the run manifest.
- Modify focused tests under `tests/vision_text_mas/`; do not alter the evaluator.

`backbone/qwen3vl.py` and two existing transport modules already exceed the 250 pure-LOC guideline. This plan makes only return-contract edits in those files; splitting the legacy model implementation is excluded because it would mix a high-risk refactor into the inference correction.

### Task 1: Typed Free-Text Judger Boundary

**Files:**
- Create: `vision_text_mas/latent_judger_output.py`
- Create: `tests/vision_text_mas/test_latent_judger_output.py`

- [ ] **Step 1: Write failing parser tests**

```python
def test_extracts_the_single_last_final_answer_marker() -> None:
    raw = "Morphology supports the second option.\nFINAL_ANSWER: second"
    parsed = parse_latent_judger_output(raw, answer_options=("first", "second"))
    assert parsed.answer == "second"
    assert parsed.rationale == "Morphology supports the second option."


@pytest.mark.parametrize(
    "raw",
    ("", "no terminal marker", "FINAL_ANSWER: first\nFINAL_ANSWER: second"),
)
def test_rejects_missing_empty_or_ambiguous_final_markers(raw: str) -> None:
    with pytest.raises(LatentJudgerParseError):
        parse_latent_judger_output(raw, answer_options=("first", "second"))


def test_preserves_an_unbounded_open_answer_phrase() -> None:
    parsed = parse_latent_judger_output(
        "The tumor shows gland formation.\nFINAL_ANSWER: adenocarcinoma",
        answer_options=(),
    )
    assert parsed.answer == "adenocarcinoma"
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `PYTHONPATH=. pytest -q tests/vision_text_mas/test_latent_judger_output.py`

Expected: collection fails because `latent_judger_output` does not exist.

- [ ] **Step 3: Implement the typed parser**

```python
@dataclass(frozen=True, slots=True)
class ParsedLatentJudgerText:
    answer: str
    rationale: str


@dataclass(frozen=True, slots=True)
class LatentJudgerParseError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


FINAL_ANSWER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?im)^\s*FINAL_ANSWER\s*:\s*(?P<answer>\S.*)\s*$"
)


def parse_latent_judger_output(
    raw: str,
    *,
    answer_options: tuple[str, ...],
) -> ParsedLatentJudgerText:
    matches = tuple(FINAL_ANSWER_PATTERN.finditer(raw))
    if len(matches) != 1:
        raise LatentJudgerParseError(
            detail=f"expected exactly one FINAL_ANSWER marker; found {len(matches)}"
        )
    candidate = matches[0].group("answer").strip()
    answer = _canonical_answer(candidate, answer_options=answer_options)
    rationale = raw[: matches[0].start()].strip()
    return ParsedLatentJudgerText(
        answer=answer,
        rationale=rationale or "The Judger returned the final answer without prose.",
    )
```

The private `_canonical_answer` must return the unique exact/case-insensitive supplied option, reject zero or multiple bounded matches, and preserve stripped text when `answer_options` is empty.

- [ ] **Step 4: Run parser tests and confirm GREEN**

Run: `PYTHONPATH=. pytest -q tests/vision_text_mas/test_latent_judger_output.py`

Expected: all parser tests pass.

- [ ] **Step 5: Commit the parser boundary**

Stage only the two Task 1 files. Commit intent: `Prevent terminal formatting from determining answer validity` with `Tested:` and `Scope-risk: narrow` Lore trailers.

### Task 2: Replace JSON Answerer With One Free-Text Judger Call

**Files:**
- Modify: `vision_text_mas/latent_answerer.py`
- Modify: `tests/vision_text_mas/test_latent_answerer.py`
- Modify: `tests/vision_text_mas/test_latent_terminal_answerer.py`

- [ ] **Step 1: Replace JSON fake tests with a one-call free-text contract**

```python
@dataclass(slots=True)
class FinalizerClient:
    prompt: str = ""
    calls: int = 0

    def generate_final_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        final_tokens: int,
    ) -> RoleCall:
        self.prompt = user_prompt
        self.calls += 1
        return RoleCall(
            role="answerer",
            prompt=user_prompt,
            image_labels=(),
            reasoning="",
            final_outputs=("Evidence supports option two.\nFINAL_ANSWER: second",),
            format_repairs=0,
            physical_calls=1,
            elapsed_seconds=0.1,
        )


def test_terminal_judger_uses_one_free_text_call() -> None:
    result = build_answerer(FinalizerClient()).answer(**bounded_case())
    assert result.value.answer == "second"
    assert result.record.physical_calls == 1
    assert result.record.format_repairs == 0
```

Also assert that the prompt contains `FINAL_ANSWER:`, contains the supplied choices, and does not contain a JSON shape, `decision_basis` JSON field, or `{` prefix instruction.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `PYTHONPATH=. pytest -q tests/vision_text_mas/test_latent_answerer.py tests/vision_text_mas/test_latent_terminal_answerer.py`

Expected: tests fail because `LatentAnswererAgent` still calls `generate_json`.

- [ ] **Step 3: Implement the free-text protocol and conversion**

Change `LatentAnswererClient` to expose the existing `generate_final_text` capability. Build a Judger prompt that includes the question, choices when present, a statement that evidence is carried in latent state, and the final marker requirement. Call it once, parse `record.final_outputs[0]`, and return:

```python
return ParsedRoleCall(
    value=AnswerOutput(
        decision_basis=decision_basis,
        answer=parsed.answer,
        evidence_patches=cited_ids,
        rationale=parsed.rationale,
        confidence=0.0,
    ),
    record=record,
)
```

Convert `LatentJudgerParseError` into `PipelineFailure` with `FailureCode.ANSWER_CONTRACT`, stage `answerer`, and the raw output prefix in the failure detail. Delete the Answerer JSON schema, JSON repair, semantic repair, and fallback path from this class only.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `PYTHONPATH=. pytest -q tests/vision_text_mas/test_latent_answerer.py tests/vision_text_mas/test_latent_terminal_answerer.py`

Expected: one physical terminal call, zero format repairs, typed answer artifact.

- [ ] **Step 5: Commit the Judger boundary**

Stage only Task 2 files. Commit intent: `Match the upstream free-text Judger while preserving WSI evaluation artifacts`, documenting that Navigator JSON remains unchanged.

### Task 3: Export Every Agent Input and All M Latent Embeddings

**Files:**
- Modify: `backbone/qwen3vl.py`
- Modify: `tests/vision_text_mas/test_latent_qwen_readout.py`

- [ ] **Step 1: Add failing transport-order tests**

Extend fake-model tests for `continue_with_latent_steps` so the returned transport block is asserted as:

```python
assert torch.equal(
    result["transport_embeddings"],
    torch.tensor(
        [
            [1.0, 2.0],   # prompt input embedding
            [12.0, 13.0], # recurrent input 1
            [23.0, 24.0], # recurrent input 2
        ]
    ),
)
```

Add equivalent assertions for fresh multimodal and multimodal-on-KV return contracts using small fakes. The assertion must distinguish recurrent inputs from post-forward hidden states.

- [ ] **Step 2: Run the tests and confirm RED**

Run: `PYTHONPATH=. pytest -q tests/vision_text_mas/test_latent_qwen_readout.py`

Expected: `transport_embeddings` key is absent.

- [ ] **Step 3: Export upstream-style transport blocks**

Immediately after each role prefill, preserve `out.hidden_states[0][0].detach().cpu()` as the input embedding block. After the recurrent loop, concatenate it with `torch.stack(latent_trajectory)` and return it as `transport_embeddings` from:

- `grounded_prefill_and_latent`
- `continue_with_latent_steps`
- `grounded_prefill_and_latent_on_kv`

Do not change attention, pruning, MRoPE, KV, or latent-step calculations.

- [ ] **Step 4: Run the Qwen transport tests and confirm GREEN**

Run: `PYTHONPATH=. pytest -q tests/vision_text_mas/test_latent_qwen_readout.py`

Expected: all input-plus-M ordering tests pass.

- [ ] **Step 5: Commit the transport export**

Stage the backbone and its focused tests. Commit intent: `Preserve every upstream LatentMAS communication embedding` with `Scope-risk: moderate`.

### Task 4: Remove Tail Truncation and Enforce Context Capacity

**Files:**
- Modify: `vision_text_mas/latent_vllm_hybrid.py`
- Modify: `vision_text_mas/latent_vllm_hybrid_cli.py`
- Modify: `tests/vision_text_mas/test_latent_vllm_hybrid.py`
- Modify: `tests/vision_text_mas/test_latent_vllm_hybrid_cli.py`

- [ ] **Step 1: Write failing full-transport and overflow tests**

```python
def test_terminal_preserves_every_embedding_from_every_agent() -> None:
    cache = HybridLatentCache(
        hf_cache="unused",
        transport_blocks=(
            torch.arange(12, dtype=torch.float32).unsqueeze(1),
            torch.arange(20, 32, dtype=torch.float32).unsqueeze(1),
        ),
    )
    engine.decode(...)
    assert terminal.combined.squeeze(1).tolist() == [*range(12), *range(20, 32), -1.0]


def test_terminal_rejects_context_overflow_without_truncation() -> None:
    engine = make_engine(max_model_len=4)
    with pytest.raises(LatentVllmHybridError, match="exceeds terminal context"):
        engine.decode(cache=five_embedding_cache(), ...)
```

Assert the manifest contains `terminal_transport="all_agent_inputs_and_latents"`, `terminal_free_text=true`, and `terminal_tail_limit=null`.

- [ ] **Step 2: Run hybrid tests and confirm RED**

Run: `PYTHONPATH=. pytest -q tests/vision_text_mas/test_latent_vllm_hybrid.py tests/vision_text_mas/test_latent_vllm_hybrid_cli.py`

Expected: cache field/manifest assertions fail and terminal still keeps only the latest ten states.

- [ ] **Step 3: Implement full concatenation and capacity check**

Rename `latent_blocks` to `transport_blocks`, populate each block from `result["transport_embeddings"]`, and concatenate with:

```python
carried = torch.cat(cache.transport_blocks, dim=0)
combined = insert_latent_embeddings(prompt, carried, insertion_index=insertion_index)
if int(combined.shape[0]) > self._max_model_len:
    raise LatentVllmHybridError(
        f"terminal prompt exceeds terminal context: {combined.shape[0]} > "
        f"{self._max_model_len}"
    )
```

Pass `max_model_len` into the engine constructor from `from_pretrained`. Terminal free text passes `json_prefix=None` and `json_schema=None`. Record the three manifest fields from Step 1.

- [ ] **Step 4: Run hybrid tests and confirm GREEN**

Run: `PYTHONPATH=. pytest -q tests/vision_text_mas/test_latent_vllm_hybrid.py tests/vision_text_mas/test_latent_vllm_hybrid_cli.py`

Expected: every embedding is retained in order and overflow is explicit.

- [ ] **Step 5: Commit terminal transport parity**

Stage only Task 4 files. Commit intent: `Keep M semantics intact at the terminal boundary`, with the explicit context-overflow directive.

### Task 5: Regression and Real M80 Smoke Gate

**Files:**
- Modify only if a test reveals a defect in the files owned by Tasks 1-4.
- Runtime outputs: new non-overwriting directories under `runs/strict_fair_v2_hf_20260804/terminal_contract_step_sweep_20260806/`.

- [ ] **Step 1: Run focused unit and integration regression**

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/vision_text_mas/test_latent_judger_output.py \
  tests/vision_text_mas/test_latent_answerer.py \
  tests/vision_text_mas/test_latent_terminal_answerer.py \
  tests/vision_text_mas/test_latent_qwen_readout.py \
  tests/vision_text_mas/test_latent_vllm_hybrid.py \
  tests/vision_text_mas/test_latent_vllm_hybrid_cli.py \
  tests/vision_text_mas/test_latent_onepass.py \
  tests/vision_text_mas/test_native_vllm_backend.py
```

Expected: all selected tests pass with no failures.

- [ ] **Step 2: Run static and diff gates on touched files**

Run `git diff --check` on all Task 1-4 files and the repository's configured Ruff/basedpyright targets when available. Also run the programming skill's no-excuse checker over new Python modules.

Expected: zero new diagnostics attributable to this change.

- [ ] **Step 3: Execute real M80 indices 0 and 8**

Launch `vision_text_mas.latent_vllm_hybrid_cli` with the existing strict-v2 model, dataset, slide root, seed 42, temperature 0.6, top-p 0.95, max model length 8192, latent steps 80, and no pruning. Use a new output root containing `fulltransport_freetext_q000_q008`.

Expected for both cases:

- result artifact exists;
- terminal `RoleCall.physical_calls == 1`;
- terminal `RoleCall.format_repairs == 0`;
- raw output contains exactly one `FINAL_ANSWER:`;
- parsed answer is non-empty;
- no fallback rationale and no failure artifact;
- transport audit shows all 80 recurrent states per non-terminal agent.

- [ ] **Step 4: Stop on any smoke violation and preserve artifacts**

If either case violates the gate, do not launch 735 cases. Preserve the output root, diagnose the exact stage, add a failing regression, and repeat Tasks 1-3 as appropriate.

- [ ] **Step 5: Launch the clean full M80 run**

Only after Step 3 passes, start a new 735-case output root. Exclude every earlier JSON-Answerer or tail-limited partial result from evaluation. Monitor failures, missing final markers, and physical-call counts; stop the run immediately if any terminal case needs more than one call or produces a failure artifact.

- [ ] **Step 6: Commit final verification metadata**

Commit only code/test/manifest documentation changes, never generated WSI patches or run outputs. Record exact test commands and smoke artifacts in Lore trailers.
