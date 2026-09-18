"""Comparable direct-HF Latent WSI-MAS ablation runner."""

from __future__ import annotations

import json
import dataclasses
import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, cast

import torch
import typer
from click import ClickException
from pydantic import ValidationError
from typing_extensions import assert_never

from vision_text_mas.artifacts import ArtifactStore
from vision_text_mas.direct_hf_ablation_presets import (
    AblationName,
    DirectHfAblationPreset,
    build_ablation_preset,
)
from vision_text_mas.direct_hf_attempts import (
    load_attempts,
    run_case_with_retries,
    summarize_successes,
)
from vision_text_mas.determinism import enable_deterministic_cuda
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.latent_answerer import AnswererProtocol
from vision_text_mas.latent_onepass_cli import SlideReuseCache, _requested_indices, _run_case
from vision_text_mas.live_prefix_metrics import select_latest_cases
from vision_text_mas.latent_qwen_engine import LatentQwenEngine
from vision_text_mas.io_pipeline import AsyncJsonWriter
from vision_text_mas.dataset import load_cases
from vision_text_mas.ablation_comparison_models import ComparisonError, ProtocolManifest


app = typer.Typer(no_args_is_help=True, add_completion=False)


def _terminal_failure_indices(output_root: Path) -> frozenset[int]:
    """Return dataset indices with a persisted terminal failure artifact."""
    indices: set[int] = set()
    for failure_path in output_root.rglob("failure*.json"):
        payload = json.loads(failure_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected failure object: {failure_path}")
        dataset_index = payload.get("dataset_index")
        if isinstance(dataset_index, int):
            indices.add(dataset_index)
    return frozenset(indices)


@app.command()
def run(
    variant: Annotated[AblationName, typer.Option(
        "--variant",
        help="base, pruning, pruning_v2, pruning_v3, pruning_v3_reallocation, pruning_v4, pruning_v5, pruning_v6, reallocation, reallocation_v2, pathology_reallocation, or both",
    )],
    dataset: Annotated[Path, typer.Option("--dataset", exists=True)],
    slide_root: Annotated[Path, typer.Option("--slide-root", exists=True)],
    model: Annotated[Path, typer.Option("--model", exists=True)],
    output_root: Annotated[Path, typer.Option("--output-root")],
    expected_base_protocol: Annotated[
        Path | None, typer.Option("--expected-base-protocol", exists=True)
    ] = None,
    expected_base_manifest: Annotated[
        Path | None, typer.Option("--expected-base-manifest", exists=True)
    ] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = False,
    dataset_index: Annotated[list[int], typer.Option("--dataset-index")] = [],
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    latent_steps: Annotated[int, typer.Option("--latent-steps", min=1, max=160)] = 5,
    patch_budget: Annotated[
        Literal[8, 12, 25], typer.Option("--patch-budget")
    ] = 8,
    max_model_len: Annotated[int, typer.Option("--max-model-len", min=1)] = 8_192,
    navigator_control_tokens: Annotated[
        int, typer.Option("--navigator-control-tokens", min=1, max=512)
    ] = 512,
    temperature: Annotated[float, typer.Option("--temperature", min=0.0, max=2.0)] = 0.6,
    top_p: Annotated[float, typer.Option("--top-p", min=0.0, max=1.0)] = 0.95,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    deterministic: Annotated[
        bool, typer.Option("--deterministic/--non-deterministic")
    ] = False,
    answerer_do_sample: Annotated[
        bool, typer.Option("--answerer-do-sample/--answerer-greedy")
    ] = False,
    answerer_thinking: Annotated[
        bool, typer.Option("--answerer-thinking/--no-answerer-thinking")
    ] = True,
    answerer_max_new_tokens: Annotated[
        int, typer.Option("--answerer-max-new-tokens", min=1, max=4_096)
    ] = 512,
    terminal_no_repeat_ngram_size: Annotated[
        int, typer.Option("--terminal-no-repeat-ngram-size", min=0, max=32)
    ] = 0,
    terminal_repetition_penalty: Annotated[
        float, typer.Option("--terminal-repetition-penalty", min=1.0, max=2.0)
    ] = 1.0,
    terminal_antiloop_method: Annotated[
        Literal["none", "boxed_ngram3", "boxed_ngram3_penalty", "tag_ngram3", "json_ngram3"],
        typer.Option("--terminal-antiloop-method"),
    ] = "none",
    relay_source: Annotated[
        Literal[
            "preset",
            "key_norm",
            "uniform",
            "sender_priority",
            "shuffled_sender_priority",
        ],
        typer.Option(
            "--relay-source",
            help=(
                "Sender-relay experiment: override the reallocation destination "
                "distribution at the Answerer handoff. 'preset' (default) keeps "
                "the variant's own relay_source untouched."
            ),
        ),
    ] = "preset",
    answerer_rationale: Annotated[
        bool, typer.Option("--answerer-rationale/--no-answerer-rationale")
    ] = True,
    answerer_protocol: Annotated[
        AnswererProtocol, typer.Option("--answerer-protocol")
    ] = AnswererProtocol.STRUCTURED_JSON,
    canonical_open_options: Annotated[
        bool, typer.Option("--canonical-open-options/--no-canonical-open-options")
    ] = True,
    save_navigation_pngs: Annotated[
        bool, typer.Option("--save-navigation-pngs/--no-save-navigation-pngs")
    ] = False,
    retain_navigator_kv: Annotated[
        bool,
        typer.Option(
            "--navigator-kv/--no-navigator-kv",
            help="Retain Navigator's latent KV for Reasoner and Answerer.",
        ),
    ] = True,
    transport_mode: Annotated[
        Literal["cumulative", "cumulative_nothink"],
        typer.Option("--transport-mode"),
    ] = "cumulative",
    realign_method: Annotated[
        Literal["identity_norm", "wa", "softmax"], typer.Option("--realign-method")
    ] = "wa",
    case_retries: Annotated[
        int, typer.Option("--case-retries", min=0, max=3)
    ] = 3,
    backbone: Annotated[
        Literal["qwen3-vl", "patho-r1", "octomed", "internvl3", "huatuo"],
        typer.Option("--backbone"),
    ] = "qwen3-vl",
    io_pipeline: Annotated[
        bool,
        typer.Option(
            "--io-pipeline/--no-io-pipeline",
            help="Prefetch the next slide, reuse bounded crop data, and queue JSON writes.",
        ),
    ] = True,
) -> None:
    """Run one fixed direct-HF ablation with success-only efficiency accounting."""
    if output_root.exists() and not resume:
        raise typer.BadParameter("output root must be new", param_hint="--output-root")
    try:
        preset = build_ablation_preset(variant, latent_steps=latent_steps)
        if relay_source != "preset":
            if preset.reallocation is None:
                raise typer.BadParameter(
                    f"--relay-source requires a reallocation variant, got {variant}",
                    param_hint="--relay-source",
                )
            preset = DirectHfAblationPreset(
                preset.name,
                preset.prune_config,
                dataclasses.replace(preset.reallocation, relay_source=relay_source),
            )
            print(f"[SenderRelay] relay_source override: {relay_source}", flush=True)
        # VLMAS_PATCH_BUDGET (opt-in): the expq harness hardcodes
        # --patch-budget 8, so the crop-scaling axis (fixed token budget B,
        # more crops) is enabled per-job through the ENVS field instead.
        env_budget = os.environ.get("VLMAS_PATCH_BUDGET", "")
        if env_budget in ("8", "12", "25"):
            patch_budget = cast(Literal[8, 12, 25], int(env_budget))
            print(f"[PatchBudget] env override: {patch_budget}", flush=True)
        match terminal_antiloop_method:
            case "none":
                pass
            case "boxed_ngram3":
                answerer_protocol = AnswererProtocol.BOXED
                terminal_no_repeat_ngram_size = 3
                terminal_repetition_penalty = 1.0
            case "boxed_ngram3_penalty":
                answerer_protocol = AnswererProtocol.BOXED
                terminal_no_repeat_ngram_size = 3
                terminal_repetition_penalty = 1.10
            case "tag_ngram3":
                answerer_protocol = AnswererProtocol.TAG
                terminal_no_repeat_ngram_size = 3
                terminal_repetition_penalty = 1.0
            case "json_ngram3":
                answerer_protocol = AnswererProtocol.JSON
                terminal_no_repeat_ngram_size = 3
                terminal_repetition_penalty = 1.0
            case unreachable:
                assert_never(unreachable)
        candidate_protocol = ProtocolManifest(
            backend="transformers-direct-kv",
            pipeline_mode=f"latent-onepass-{patch_budget}",
            agent_count=4,
            max_rounds=1,
            patch_budget=patch_budget,
            navigator_tool_decodes=1,
            terminal_answer_decodes=1,
            terminal_answer_marker=answerer_protocol is AnswererProtocol.BOXED,
            intermediate_transport=(
                f"sequential_hf_{transport_mode}"
                if retain_navigator_kv
                else f"sequential_hf_{transport_mode}_without_navigator_kv"
            ),
            thinking_roles=(),
            latent_steps=latent_steps,
            transport_mode=transport_mode,
            realign_method=realign_method,
            deterministic_cuda=deterministic,
            answerer_do_sample=answerer_do_sample,
            answerer_thinking=answerer_thinking,
            answerer_max_new_tokens=answerer_max_new_tokens,
            terminal_no_repeat_ngram_size=terminal_no_repeat_ngram_size,
            terminal_repetition_penalty=terminal_repetition_penalty,
            terminal_antiloop_method=terminal_antiloop_method,
            answerer_rationale=answerer_rationale,
            answerer_protocol=answerer_protocol.value,
            canonical_open_options=canonical_open_options,
            save_navigation_pngs=save_navigation_pngs,
            max_model_len=max_model_len,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            variant=variant,
            pruning=preset.prune_config is not None,
            reallocation=preset.reallocation is not None,
        )
        if expected_base_protocol is not None:
            base_protocol = ProtocolManifest.model_validate_json(
                expected_base_protocol.read_text(encoding="utf-8")
            )
            if candidate_protocol.fixed_signature() != base_protocol.fixed_signature():
                raise ComparisonError("fixed inference protocol drift from expected base")
        if expected_base_manifest is not None:
            base_manifest = json.loads(expected_base_manifest.read_text(encoding="utf-8"))
            if not isinstance(base_manifest, dict):
                raise ComparisonError("expected base manifest must be a JSON object")
            candidate_execution = {
                "agent_backends": {
                    "evidence_planner": "transformers",
                    "navigator": "transformers",
                    "reasoner": "transformers",
                    "answerer": "transformers",
                },
                "backbone": backbone,
                "terminal_backend": "transformers",
                "vllm": False,
                "case_retries": case_retries,
                "max_total_attempts": case_retries + 1,
                "retry_policy": "initial_attempt_plus_immediate_same_case_retries",
                "efficiency_denominator": "successful_final_attempts_only",
                "navigator_kv": retain_navigator_kv,
                "navigator_control_tokens": navigator_control_tokens,
                "patch_budget": patch_budget,
                "io_pipeline": io_pipeline,
            }
            drifted = tuple(
                key
                for key, value in candidate_execution.items()
                if base_manifest.get(key) != value
            )
            if drifted:
                raise ComparisonError(
                    "non-pruning execution drift from expected base: "
                    + ", ".join(drifted)
                )
        existing_manifest_path = output_root / "run_manifest.json"
        if existing_manifest_path.exists():
            existing_protocol = ProtocolManifest.model_validate_json(
                existing_manifest_path.read_text(encoding="utf-8")
            )
            if existing_protocol != candidate_protocol:
                raise ComparisonError("resume protocol differs from existing run manifest")
        if deterministic:
            # Navigator and intermediate role calls retain top-p sampling even
            # when the terminal answerer is greedy. CUDA cumsum therefore must
            # remain warn-only for every direct-HF protocol.
            enable_deterministic_cuda(warn_only=True)
        requested_indices = _requested_indices(dataset, tuple(dataset_index))
        completed_indices = (
            frozenset(case.dataset_index for case in select_latest_cases(output_root))
            | _terminal_failure_indices(output_root)
        ) if resume else frozenset()
        indices = tuple(
            index for index in requested_indices if index not in completed_indices
        )
        cases = load_cases(dataset, slide_root=slide_root, indices=indices)
        engine = LatentQwenEngine.from_pretrained(
            model,
            device,
            preserved_root=Path(__file__).parents[1],
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            terminal_do_sample=answerer_do_sample,
            answerer_thinking=answerer_thinking,
            terminal_no_repeat_ngram_size=terminal_no_repeat_ngram_size,
            terminal_repetition_penalty=terminal_repetition_penalty,
            transport_mode=transport_mode,
            realign_method=realign_method,
            prune_config=preset.prune_config,
            reallocation=preset.reallocation,
            backbone_name=backbone,
        )
    except (
        ComparisonError,
        OSError,
        PipelineFailure,
        RuntimeError,
        ValidationError,
        ValueError,
    ) as failure:
        raise ClickException(str(failure)) from failure

    output_root.mkdir(parents=True, exist_ok=resume)
    metrics_path = output_root / "attempt_metrics.jsonl"
    manifest = candidate_protocol.model_dump(mode="json")
    manifest.update({
        "agent_backends": {
            "evidence_planner": "transformers",
            "navigator": "transformers",
            "reasoner": "transformers",
            "answerer": "transformers",
        },
        "backbone": backbone,
        "terminal_backend": "transformers",
        "vllm": False,
        "case_retries": case_retries,
        "max_total_attempts": case_retries + 1,
        "retry_policy": "initial_attempt_plus_immediate_same_case_retries",
        "efficiency_denominator": "successful_final_attempts_only",
        "pruning_config": None if preset.prune_config is None else asdict(preset.prune_config),
        "reallocation_config": None if preset.reallocation is None else asdict(preset.reallocation),
        "navigator_kv": retain_navigator_kv,
        "navigator_control_tokens": navigator_control_tokens,
        "patch_budget": patch_budget,
        "io_pipeline": io_pipeline,
        "requested_indices": requested_indices,
        "resume": resume,
        "resumed_completed_indices": sorted(completed_indices),
    })
    manifest_path = output_root / "run_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    terminal_failures = 0
    slide_cache = SlideReuseCache(enable_io_pipeline=io_pipeline)
    artifact_writer = AsyncJsonWriter() if io_pipeline else None
    try:
        for position, case in enumerate(cases):
            if io_pipeline and position + 1 < len(cases):
                slide_cache.prefetch(cases[position + 1])
            try:
                run_case_with_retries(
                    case=case,
                    max_attempts=case_retries + 1,
                    metrics_path=metrics_path,
                    operation=lambda: _run_case(
                        case=case,
                        engine=engine,
                        output_root=output_root,
                        latent_steps=latent_steps,
                        patch_budget=patch_budget,
                        max_model_len=max_model_len,
                        navigator_control_tokens=navigator_control_tokens,
                        answerer_max_new_tokens=answerer_max_new_tokens,
                        answerer_include_rationale=answerer_rationale,
                        answerer_protocol=answerer_protocol,
                        canonical_open_options=canonical_open_options,
                        save_navigation_pngs=save_navigation_pngs,
                        retain_navigator_kv=retain_navigator_kv,
                        slide_cache=slide_cache,
                        artifact_writer=artifact_writer,
                    ),
                )
            except PipelineFailure as failure:
                terminal_failures += 1
                typer.echo(f"FAILED {case.dataset_index} {case.item.slide_id}: {failure}")
            except torch.OutOfMemoryError as error:
                terminal_failures += 1
                ArtifactStore(
                    output_root / f"{case.dataset_index:03d}_{case.item.slide_id}",
                    json_writer=artifact_writer,
                ).write_failure(
                    PipelineFailure(
                        code=FailureCode.MODEL_EXECUTION,
                        stage="cuda",
                        detail=f"CUDA out of memory: {error}",
                    ),
                    case,
                )
                typer.echo(f"FAILED {case.dataset_index} {case.item.slide_id}: CUDA out of memory")
    finally:
        slide_cache.shutdown()
        if artifact_writer is not None:
            artifact_writer.close()

    summary = summarize_successes(load_attempts(metrics_path))
    summary.update(
        {
            "requested_cases": len(cases),
            "terminal_failed_cases": terminal_failures,
            "efficiency_note": "Failure attempts and terminal failures are excluded from efficiency distributions.",
        }
    )
    (output_root / "efficiency_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    typer.echo(f"completed={summary['successful_cases']} failed={terminal_failures} requested={len(cases)}")
    if terminal_failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
