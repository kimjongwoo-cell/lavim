"""Prompt-matched four-agent VL-MAS with explicit text transport."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from click import ClickException

from vision_text_mas.artifacts import ArtifactStore
from vision_text_mas.cli import FAST_EVAL_DIRECT_JSON_ROLES, _load_client, _open_slide
from vision_text_mas.contracts import RunResult
from vision_text_mas.controller_onepass import OnePassController
from vision_text_mas.dataset import load_cases
from vision_text_mas.errors import PipelineFailure
from vision_text_mas.evidence_planner import EvidencePlannerAgent
from vision_text_mas.latent_answerer import AnswererProtocol, FINAL_ANSWER_TOKEN_BUDGET
from vision_text_mas.latent_onepass_cli import SlideReuseCache, _requested_indices
from vision_text_mas.io_pipeline import AsyncJsonWriter
from vision_text_mas.matched_text_answerer import MatchedTextAnswererAgent
from vision_text_mas.onepass_navigator import OnePassNavigator
from vision_text_mas.onepass_navigator_agent import OneShotNavigatorAgent
from vision_text_mas.reasoner import ReasonerAgent


app = typer.Typer(no_args_is_help=True, add_completion=False)


def _direct_json_roles(*, fast_json: bool) -> frozenset[str]:
    """Select the optional shortcut without changing the role contract."""
    return (
        FAST_EVAL_DIRECT_JSON_ROLES - {"answerer"}
        if fast_json
        else frozenset()
    )


@app.command()
def run(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True)],
    slide_root: Annotated[Path, typer.Option("--slide-root", exists=True)],
    model: Annotated[Path, typer.Option("--model", exists=True)],
    output_root: Annotated[Path, typer.Option("--output-root")],
    dataset_index: Annotated[list[int], typer.Option("--dataset-index")] = [],
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    max_model_len: Annotated[int, typer.Option("--max-model-len", min=1)] = 8_192,
    temperature: Annotated[float, typer.Option("--temperature", min=0.0)] = 0.6,
    top_p: Annotated[float, typer.Option("--top-p", min=0.0, max=1.0)] = 0.95,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    answerer_max_new_tokens: Annotated[
        int, typer.Option("--answerer-max-new-tokens", min=1, max=4_096)
    ] = FINAL_ANSWER_TOKEN_BUDGET,
    answerer_protocol: Annotated[
        AnswererProtocol, typer.Option("--answerer-protocol")
    ] = AnswererProtocol.STRUCTURED_JSON,
    canonical_open_options: Annotated[
        bool,
        typer.Option("--canonical-open-options/--no-canonical-open-options"),
    ] = True,
    fast_json: Annotated[
        bool,
        typer.Option("--fast-json/--standard-json"),
    ] = True,
    constrained_json: Annotated[
        bool,
        typer.Option("--constrained-json/--no-constrained-json"),
    ] = False,
    io_pipeline: Annotated[
        bool,
        typer.Option(
            "--io-pipeline/--no-io-pipeline",
            help="Prefetch the next slide, reuse bounded crop data, and queue JSON writes.",
        ),
    ] = True,
) -> None:
    """Run Planner, Navigator, Reasoner, and matched Answerer without KV transfer."""
    if output_root.exists():
        raise typer.BadParameter("output root must be new", param_hint="--output-root")
    try:
        indices = _requested_indices(dataset, tuple(dataset_index))
        cases = load_cases(dataset, slide_root=slide_root, indices=indices)
        client = _load_client(
            model,
            device,
            direct_json_roles=_direct_json_roles(fast_json=fast_json),
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_model_len=max_model_len,
            constrained_json=constrained_json,
        )
    except PipelineFailure as failure:
        raise ClickException(str(failure)) from failure
    answerer = MatchedTextAnswererAgent(
        client,
        final_tokens=answerer_max_new_tokens,
        protocol=answerer_protocol,
        canonical_open_options=canonical_open_options,
    )
    output_root.mkdir(parents=True)
    completed: list[RunResult] = []
    failed = 0
    slide_cache = SlideReuseCache(enable_io_pipeline=True) if io_pipeline else None
    artifact_writer = AsyncJsonWriter() if io_pipeline else None
    try:
        for position, case in enumerate(cases):
            if io_pipeline and position + 1 < len(cases):
                slide_cache.prefetch(cases[position + 1])
            case_root = output_root / f"{case.dataset_index:03d}_{case.item.slide_id}"
            artifacts = ArtifactStore(case_root, json_writer=artifact_writer)
            controller = OnePassController(
                planner=EvidencePlannerAgent(client),
                navigator=OnePassNavigator(
                    agent=OneShotNavigatorAgent(client),
                    artifact_root=case_root / "navigation",
                ),
                reasoner=ReasonerAgent(client),
                answerer=answerer,
                artifacts=artifacts,
            )
            try:
                slide = _open_slide(case) if slide_cache is None else slide_cache.acquire(case)
                try:
                    completed.append(controller.run_case(case, slide=slide))
                finally:
                    if slide_cache is None:
                        slide.close()
            except PipelineFailure as failure:
                failed += 1
                typer.echo(f"FAILED {case.dataset_index} {case.item.slide_id}: {failure}")
    finally:
        if slide_cache is not None:
            slide_cache.shutdown()
        if artifact_writer is not None:
            artifact_writer.close()
    correct = sum(result.correct for result in completed)
    typer.echo(f"completed={len(completed)} failed={failed} correct={correct}/{len(cases)}")
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
