# PVCR Manual QA

Scope: CPU/CLI hands-on verification of 0912/pvcr, 0912/run_pvcr.py, 0912/run_pvcr.sh, and 0912/test_pvcr.py. No code or run outputs were changed by this QA; the only file created by me is this report.

Technical verdict: PASS for the P0/P1 scenarios below. The final one-case instrumented run completed both relay boundaries with cache-length and GQA audit fields, and its recorded source SHA256 values match the current PVCR source files.

Scientific gate: FAIL separately, as recorded by the parent. The 128-case run was not launched after the scientific review raised concerns about contribution magnitude and replacement semantics. This report makes no method-effectiveness claim.

## manualQa

### surfaceEvidence

- scenarioId: PVCR-P0-CPU-CONTRACTS
  criterionReference: P0 — full-denominator attention, contribution pooling, GQA mapping, mask handling, causal receiver prefill, cache restoration/growth, unsupported spatial support, shuffled neighborhood shape.
  surface: CPU pytest suite.
  exactInvocation:
  ~~~sh
  PYTHONPATH=0912:. uv run --python /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python --no-project pytest -q 0912/test_pvcr.py
  ~~~
  verdict: PASS — 9 passed in 3.42s on the final current-source run.
  artifactRefs: [E-REPORT]

- scenarioId: PVCR-P0-CURRENT-SMOKE
  criterionReference: P0 — one shared PVCR module is exercised at Navigator→Reasoner and Reasoner→Answerer, retaining existing cache addresses and recording GQA/cache-length audits.
  surface: Parent-run Qwen3-VL GPU smoke; verified read-only from the launcher log, run settings, efficiency summary, and PVCR audit JSON.
  exactInvocation:
  ~~~sh
  bash 0912/run_pvcr.sh spatial 8 pvcr_smoke_final_0912 0
  ~~~
  verdict: PASS — completed=1, failed=0, requested=1. Audit JSON records 156 Navigator and 559 Reasoner receiver layer calls; both sender audits report query_heads=32, kv_heads=8, queries_per_kv_head=4, 10 latent columns, and layers 8–20. Cache lengths were 3480→3480 during Navigator-value substitution, 8391→8391 during its restoration, 8391→8391 during Reasoner-value substitution, and 8781→8781 during its restoration.
  artifactRefs: [E-FINAL-LOG, E-FINAL-AUDIT, E-FINAL-SETTINGS, E-FINAL-SUMMARY]

- scenarioId: PVCR-P0-CPU-RECEIVER-GQA
  criterionReference: P0 — receiver SDPA bias uses the synthetic value at the existing address, fires once at an active layer, preserves cache length, and restores original values.
  surface: CPU library driver invoking the registered Transformers SDPA adapter on layer 8 with a 4-query-head/2-KV-head fixture.
  exactInvocation:
  ~~~sh
  PYTHONPATH=0912:. uv run --python /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python --no-project python -c 'import torch
  from pathlib import Path
  from transformers.cache_utils import DynamicCache
  from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
  from pvcr.attention import attention_adapter
  from pvcr.runtime import PVCR
  from pvcr.receiver import RelayBank, layer_values
  class Layer(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.layer_idx = 8
          self.is_causal = True
          self.num_key_value_groups = 2
  cache = DynamicCache()
  keys = torch.zeros(1, 2, 3, 2)
  original = torch.ones_like(keys)
  cache.update(keys.clone(), original.clone(), 0)
  experiment = PVCR(Path("/tmp/pvcr-cpu-qa-unused"), "spatial")
  experiment.banks["navigator"] = RelayBank(torch.tensor([2]), {0: torch.full((1, 2, 1, 2), 7.0)})
  query = torch.zeros(1, 4, 1, 2)
  with attention_adapter(experiment.attention):
      with experiment.receiving(cache, "navigator"):
          output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](Layer(), query, keys, layer_values(cache, 0), None, scaling=1.0, is_causal=True)
  assert experiment.receiver_calls == {"navigator": 1}
  assert experiment.attention.receiver is None
  assert cache.get_seq_length() == 3
  assert torch.equal(layer_values(cache, 0), original)
  assert torch.allclose(output, torch.full_like(output, 4.0), atol=1e-6)
  print("PASS receiver SDPA GQA invocation; output=4.0; calls=navigator:1; cache_tokens=3; values_restored=true")'
  ~~~
  verdict: PASS — emitted the expected output and state assertions.
  artifactRefs: [E-REPORT]

- scenarioId: PVCR-P1-CLI-HELP
  criterionReference: P1 — the wrapper exposes the underlying Base CLI help without entering a model run.
  surface: CLI.
  exactInvocation:
  ~~~sh
  PYTHONPATH=0912:. uv run --python /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python --no-project python 0912/run_pvcr.py --help
  ~~~
  verdict: PASS — exit 0; usage and Base CLI options were displayed.
  artifactRefs: [E-REPORT]

- scenarioId: PVCR-P1-MISSING-BANK
  criterionReference: P1 — a receiver boundary without its sender bank fails before installing receiver state.
  surface: CPU runtime boundary.
  exactInvocation:
  ~~~sh
  PYTHONPATH=0912:. uv run --python /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python --no-project python -c 'from pathlib import Path
  from transformers.cache_utils import DynamicCache
  from pvcr.errors import LayoutError
  from pvcr.runtime import PVCR
  experiment = PVCR(Path("/tmp/pvcr-cpu-qa-unused"), "spatial")
  try:
      with experiment.receiving(DynamicCache(), "navigator"):
          raise AssertionError("receiver unexpectedly entered")
  except LayoutError as error:
      assert str(error) == "Missing navigator PVCR bank/cache at receiver boundary"
  assert experiment.attention.receiver is None
  assert experiment.receiver_calls == {}
  print("PASS missing-bank receiver boundary rejected before mutation")'
  ~~~
  verdict: PASS — emitted the expected boundary rejection and no state mutation.
  artifactRefs: [E-REPORT]

### adversarialCases

- scenarioId: PVCR-ADV-FIXED-STEPS
  criterionReference: P1 — reject unsupported sender-step configuration.
  adversarialClass: Invalid fixed adapter configuration.
  expectedBehavior: Reject --latent-steps 9 before model startup or output creation.
  exactInvocation:
  ~~~sh
  PVCR_VARIANT=spatial PYTHONPATH=0912:. uv run --python /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python --no-project python 0912/run_pvcr.py --variant base --latent-steps 9 --patch-budget 12 --max-model-len 12288 --backbone qwen3-vl --transport-mode cumulative --output-root /tmp/pvcr-qa-invalid-should-not-create --navigator-kv
  ~~~
  verdict: PASS — exit 1 with LayoutError: PVCR requires --latent-steps 10.
  artifactRefs: [E-REPORT]

- scenarioId: PVCR-ADV-OUTPUT-COLLISION
  criterionReference: P1 — refuse to reuse an existing output root.
  adversarialClass: Output-path collision.
  expectedBehavior: Reject the existing directory before app/model startup and preserve it.
  exactInvocation:
  ~~~sh
  PVCR_VARIANT=spatial PYTHONPATH=0912:. uv run --python /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python --no-project python 0912/run_pvcr.py --variant base --latent-steps 10 --patch-budget 12 --max-model-len 12288 --backbone qwen3-vl --transport-mode cumulative --answerer-protocol structured_json --output-root 0912/runs/pvcr_smoke_spatial_v2_0912 --navigator-kv
  ~~~
  verdict: PASS — exit 1 with LayoutError: Output already exists: 0912/runs/pvcr_smoke_spatial_v2_0912.
  artifactRefs: [E-REPORT]

- scenarioId: PVCR-ADV-LAUNCHER-GPU
  criterionReference: P1 — launcher rejects GPUs outside its explicit allowlist before launching Python.
  adversarialClass: Invalid GPU selection.
  expectedBehavior: Reject GPU index 0 before CUDA setup, uv, or model loading.
  exactInvocation:
  ~~~sh
  bash 0912/run_pvcr.sh spatial 0 pvcr_qa_should_not_create 0
  ~~~
  verdict: PASS — exit 2 with “This experiment is restricted to GPU 7 or 8”.
  artifactRefs: [E-REPORT]

- scenarioId: PVCR-ADV-RESTORE-ON-ERROR
  criterionReference: P0 — temporary cache substitution restores values when receiver work raises.
  adversarialClass: Receiver exception during substitution.
  expectedBehavior: Propagate the receiver error and restore original values without changing cache length.
  exactInvocation:
  ~~~sh
  PYTHONPATH=0912:. uv run --python /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python --no-project pytest -q 0912/test_pvcr.py
  ~~~
  verdict: PASS — test_value_substitution_keeps_keys_length_and_restores_on_error passed within the 9-test suite.
  artifactRefs: [E-REPORT]

- scenarioId: PVCR-ADV-MASKS-AND-EMPTY-SUPPORT
  criterionReference: P0 — preserve forbidden mask columns in both accepted mask formats and return zero for no shared spatial support.
  adversarialClass: Boolean/additive mask restrictions and empty spatial intersection.
  expectedBehavior: Forbidden columns retain zero probability; empty joint support yields finite zero contribution/support.
  exactInvocation:
  ~~~sh
  PYTHONPATH=0912:. uv run --python /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python --no-project pytest -q 0912/test_pvcr.py
  ~~~
  verdict: PASS — test_both_mask_formats_preserve_forbidden_columns and test_no_joint_spatial_support_returns_zero passed within the 9-test suite.
  artifactRefs: [E-REPORT]

## Execution notes

The first two attempts at the ad-hoc positive SDPA driver were fixture/oracle errors: the fake layer initially omitted Transformers’ num_key_value_groups metadata, then my expected weighted result was miscomputed. I corrected the fixture and expected value, and the final exact invocation above passed. These were harness errors, not product failures.

I inspected the earlier pvcr_smoke_off_v2 and pvcr_smoke_spatial_v2 logs and run artifacts. Both logs report 2/2 successful cases; the spatial JSON shows nonzero receiver calls at both boundaries. Those runs are stale relative to the final code: their embedded source SHA256 values differ from the current run_pvcr.py, run_pvcr.sh, attention.py, capture.py, receiver.py, and runtime.py, and their JSON lacks the new query_heads and receiver_cache_lengths fields. They are retained as historical evidence only. The final smoke artifacts below are the source-matched integration evidence.

No 128-case run was launched by this executor. The parent kept that experiment outside technical QA after the separate scientific review; its gate remains FAIL and is not converted into a technical pass by this report.

## Reviewed source SHA256

The final current-source smoke settings map matches every production source hash below. test_pvcr.py is recorded separately because the runtime manifest hashes launcher/runtime sources, not the test file.

| File | SHA256 |
|---|---|
| 0912/run_pvcr.py | 2744aeab601525db2b8e2281fb5d72efdb51201e740bc11fa54d2cb6420030a3 |
| 0912/run_pvcr.sh | 17ba0f351a89c975cbac37147b6befe5853804f43504217d85c9133c3fff5c49 |
| 0912/test_pvcr.py | 8e249523481558134c0020c5ae5b31e9c28ad852ec48ce25fd0f2c8752955c82 |
| 0912/pvcr/__init__.py | 0f1e02545130fb3eb31569f0a4717c78d5ac1e041836f9aceb4ee238d65467c8 |
| 0912/pvcr/attention.py | 147b052c2bb30f92de0f3a87f70034f4b4899cc110f3013d167c3ef647b3c902 |
| 0912/pvcr/capture.py | 89d6cbdfe3da9c47b2e505c6bf32b259161eeb8fcebf473622102d1dc454ba32 |
| 0912/pvcr/core.py | 5d71d73e06d9643675462feeedfb26ee60e43a506b9113510fe8f1152d1bf450 |
| 0912/pvcr/errors.py | 647fa4dc67d90e5a790560c2029fafeb320aa7693363498f5e47c311151b4ede |
| 0912/pvcr/layout.py | 5ee9aa8aff98894a24a3b7b53c1a642c56e8c219c0f1a81b69baf9ffe38e8761 |
| 0912/pvcr/receiver.py | 93361f2eaec73c0e98d8a8430fe6164f6ca756e7005eb7265e2e433b656c0173 |
| 0912/pvcr/runtime.py | 1fb14aaf96116c6dae652ccf2a144212d314fa1796a3de5665a043fbcf264aed |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| E-REPORT | QA transcript | Non-empty command invocations, outcomes, matrix, and source fingerprints in this report. | 0912/pvcr_review_qa.md |
| E-FINAL-LOG | Terminal transcript | Current-source one-case GPU smoke; reports completed=1, failed=0 and receiver counts. | 0912/pvcr_smoke_final.log |
| E-FINAL-AUDIT | JSON audit | Sender layers, cache-address columns, GQA counts, receiver calls, and before/after cache lengths for both handoffs. | 0912/runs/pvcr_smoke_final_0912/pvcr/000.json |
| E-FINAL-SETTINGS | JSON source manifest | Effective CLI argv and source SHA256 map; matches current production source files. | 0912/runs/pvcr_smoke_final_0912/pvcr_settings.json |
| E-FINAL-SUMMARY | JSON run summary | Successful/failed/requested case counts for the final smoke. | 0912/runs/pvcr_smoke_final_0912/efficiency_summary.json |
| E-V2-OFF-HISTORY | Historical JSON/log | Earlier off run inspected; source manifest does not match the final current source. | 0912/runs/pvcr_smoke_off_v2_0912/pvcr_settings.json |
| E-V2-SPATIAL-HISTORY | Historical JSON/log | Earlier spatial run inspected; source manifest does not match the final current source. | 0912/runs/pvcr_smoke_spatial_v2_0912/pvcr_settings.json |
