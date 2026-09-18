"""One-load sequential batch CLI for Vision-TextMAS."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

import typer
from click import ClickException
from typing_extensions import assert_never

from vision_text_mas.answerer import AnswererAgent
from vision_text_mas.artifacts import ArtifactStore
from vision_text_mas.contracts import CaseInput, RunResult
from vision_text_mas.controller import Controller
from vision_text_mas.controller_onepass import OnePassController
from vision_text_mas.dataset import load_cases
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.evidence_planner import EvidencePlannerAgent
from vision_text_mas.flat_image_slide import FlatImageSlide
from vision_text_mas.dicom_slide import DicomSlide
from vision_text_mas.navigator import Navigator
from vision_text_mas.onepass_navigator import OnePassNavigator
from vision_text_mas.onepass_navigator_agent import OneShotNavigatorAgent
from vision_text_mas.patch_navigator import PatchNavigatorAgent
from vision_text_mas.qwen_client import QwenJsonClient
from vision_text_mas.reasoner import ReasonerAgent
from vision_text_mas.verifier import VerifierAgent

if TYPE_CHECKING:
    import openslide


app = typer.Typer(no_args_is_help=True, add_completion=False)
FAST_EVAL_DIRECT_JSON_ROLES: Final = frozenset(
    {"evidence_planner", "navigator", "patch_navigator", "verifier", "answerer"}
)
PipelineMode = Literal["strict-v2", "onepass-8"]


@app.callback()
def main() -> None:
    """Run a fair, isolated Vision-TextMAS pipeline."""


def _load_client(
    model_path: Path,
    device: str,
    *,
    backend: Literal["transformers", "vllm", "native-vllm", "hf-service"] = "transformers",
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    hf_service_url: str = "http://127.0.0.1:8123/v1",
    served_model_name: str = "qwen3-vl-thinking",
    direct_json_roles: frozenset[str] = frozenset(),
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 42,
    max_model_len: int = 8_192,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    request_max_tokens: int | None = None,
    constrained_json: bool = False,
) -> QwenJsonClient:
    try:
        match backend:
            case "transformers":
                return QwenJsonClient.from_pretrained(
                    model_path,
                    device,
                    direct_json_roles=direct_json_roles,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    max_model_len=max_model_len,
                    request_max_tokens=request_max_tokens,
                    constrained_json=constrained_json,
                )
            case "vllm":
                return QwenJsonClient.from_vllm(
                    base_url=vllm_base_url,
                    model_name=served_model_name,
                    direct_json_roles=direct_json_roles,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    request_max_tokens=request_max_tokens,
                )
            case "native-vllm":
                return QwenJsonClient.from_native_vllm(
                    model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    direct_json_roles=direct_json_roles,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    max_model_len=max_model_len,
                    request_max_tokens=request_max_tokens,
                )
            case "hf-service":
                return QwenJsonClient.from_hf_service(
                    base_url=hf_service_url,
                    model_name=served_model_name,
                    direct_json_roles=direct_json_roles,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    request_max_tokens=request_max_tokens,
                )
            case unreachable:
                assert_never(unreachable)
    except (OSError, RuntimeError) as error:
        raise PipelineFailure(
            code=FailureCode.MODEL_EXECUTION,
            stage="model-load",
            detail=str(error),
        ) from error


def _open_slide(case: CaseInput) -> openslide.OpenSlide | FlatImageSlide | DicomSlide:
    if case.slide_path.is_dir():
        try:
            return DicomSlide(case.slide_path)
        except (OSError, ValueError) as error:
            raise PipelineFailure(
                code=FailureCode.IMAGE_IO,
                stage="slide-open",
                detail=f"failed to open DICOM series {case.slide_path}: {error}",
            ) from error
    if case.slide_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        try:
            return FlatImageSlide(case.slide_path)
        except (OSError, ValueError) as error:
            raise PipelineFailure(
                code=FailureCode.IMAGE_IO,
                stage="slide-open",
                detail=f"failed to open {case.slide_path}: {error}",
            ) from error
    import openslide

    try:
        return openslide.OpenSlide(str(case.slide_path))
    except (OSError, openslide.OpenSlideError) as error:
        raise PipelineFailure(
            code=FailureCode.IMAGE_IO,
            stage="slide-open",
            detail=f"failed to open {case.slide_path}: {error}",
        ) from error


def _validate_pipeline_config(*, pipeline_mode: PipelineMode, max_rounds: int) -> None:
    """Reject settings that would reintroduce a OnePass retrieval loop."""
    if pipeline_mode == "onepass-8" and max_rounds != 1:
        raise typer.BadParameter(
            "onepass-8 runs exactly one round; set --max-rounds 1",
            param_hint="--max-rounds",
        )


def _build_controller(
    *,
    client: QwenJsonClient,
    artifacts: ArtifactStore,
    case_root: Path,
    max_rounds: int,
    pipeline_mode: PipelineMode,
) -> Controller | OnePassController:
    """Compose the selected agent topology over one shared model client."""
    planner = EvidencePlannerAgent(client)
    reasoner = ReasonerAgent(client)
    answerer = AnswererAgent(client)
    match pipeline_mode:
        case "strict-v2":
            return Controller(
                planner=planner,
                navigator=Navigator(
                    selector=PatchNavigatorAgent(client),
                    artifact_root=case_root / "navigation",
                ),
                reasoner=reasoner,
                verifier=VerifierAgent(client),
                answerer=answerer,
                artifacts=artifacts,
                max_rounds=max_rounds,
            )
        case "onepass-8":
            return OnePassController(
                planner=planner,
                navigator=OnePassNavigator(
                    agent=OneShotNavigatorAgent(client),
                    artifact_root=case_root / "navigation",
                ),
                reasoner=reasoner,
                answerer=answerer,
                artifacts=artifacts,
            )
        case unreachable:
            assert_never(unreachable)


def _run_one(
    *,
    case: CaseInput,
    client: QwenJsonClient,
    output_root: Path,
    max_rounds: int,
    pipeline_mode: PipelineMode,
) -> RunResult:
    case_root = output_root / f"{case.dataset_index:03d}_{case.item.slide_id}"
    artifacts = ArtifactStore(case_root)
    controller = _build_controller(
        client=client,
        artifacts=artifacts,
        case_root=case_root,
        max_rounds=max_rounds,
        pipeline_mode=pipeline_mode,
    )
    try:
        slide = _open_slide(case)
    except PipelineFailure as failure:
        artifacts.write_failure(failure, case)
        raise
    try:
        return controller.run_case(case, slide=slide)
    finally:
        slide.close()


@app.command()
def run(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", exists=True, file_okay=True, dir_okay=False),
    ],
    slide_root: Annotated[
        Path,
        typer.Option("--slide-root", exists=True, file_okay=False, dir_okay=True),
    ],
    model: Annotated[
        Path,
        typer.Option("--model", exists=True, file_okay=False, dir_okay=True),
    ],
    output_root: Annotated[Path, typer.Option("--output-root")],
    dataset_index: Annotated[
        list[int],
        typer.Option("--dataset-index", help="Repeat once per requested row."),
    ],
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    max_model_len: Annotated[int, typer.Option("--max-model-len", min=1)] = 8_192,
    max_rounds: Annotated[int, typer.Option("--max-rounds", min=1, max=3)] = 3,
    pipeline_mode: Annotated[
        PipelineMode,
        typer.Option(
            "--pipeline-mode",
            help="strict-v2 uses Verifier; onepass-8 runs Planner/Navigator/Reasoner/Answerer once.",
        ),
    ] = "strict-v2",
    fast_eval: Annotated[
        bool,
        typer.Option(
            "--fast-eval",
            help="Use direct structured output for routing roles.",
        ),
    ] = False,
    backend: Annotated[
        Literal["transformers", "vllm", "native-vllm", "hf-service"],
        typer.Option("--backend", help="Physical generation engine."),
    ] = "transformers",
    tensor_parallel_size: Annotated[
        int,
        typer.Option("--tensor-parallel-size", min=1),
    ] = 1,
    gpu_memory_utilization: Annotated[
        float,
        typer.Option("--gpu-memory-utilization", min=0.1, max=0.99),
    ] = 0.9,
    vllm_base_url: Annotated[
        str,
        typer.Option("--vllm-base-url", help="OpenAI-compatible vLLM URL."),
    ] = "http://127.0.0.1:8000/v1",
    hf_service_url: Annotated[
        str,
        typer.Option("--hf-service-url", help="Direct-HF batch-service URL."),
    ] = "http://127.0.0.1:8123/v1",
    served_model_name: Annotated[
        str,
        typer.Option("--served-model-name", help="Model name exposed by vLLM."),
    ] = "qwen3-vl-thinking",
    temperature: Annotated[
        float,
        typer.Option("--temperature", min=0.0, max=2.0),
    ] = 0.6,
    top_p: Annotated[
        float,
        typer.Option("--top-p", min=0.0, max=1.0),
    ] = 0.95,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    request_max_tokens: Annotated[
        int | None,
        typer.Option("--request-max-tokens", min=1),
    ] = None,
) -> None:
    """Run selected WSI-VQA rows sequentially with one shared Qwen load."""
    _validate_pipeline_config(pipeline_mode=pipeline_mode, max_rounds=max_rounds)
    if output_root.exists():
        raise typer.BadParameter(
            "output root must be new to prevent artifact overwrite",
            param_hint="--output-root",
        )
    try:
        cases = load_cases(
            dataset,
            slide_root=slide_root,
            indices=tuple(dataset_index),
        )
        client = _load_client(
            model,
            device,
            backend=backend,
            vllm_base_url=vllm_base_url,
            hf_service_url=hf_service_url,
            served_model_name=served_model_name,
            direct_json_roles=(
                FAST_EVAL_DIRECT_JSON_ROLES if fast_eval else frozenset()
            ),
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            request_max_tokens=request_max_tokens,
        )
    except PipelineFailure as failure:
        raise ClickException(str(failure)) from failure
    output_root.mkdir(parents=True)
    completed: list[RunResult] = []
    failed = 0
    for case in cases:
        try:
            completed.append(
                _run_one(
                    case=case,
                    client=client,
                    output_root=output_root,
                    max_rounds=max_rounds,
                    pipeline_mode=pipeline_mode,
                )
            )
        except PipelineFailure as failure:
            failed += 1
            typer.echo(f"FAILED {case.dataset_index} {case.item.slide_id}: {failure}")
    correct = sum(result.correct for result in completed)
    typer.echo(
        f"completed={len(completed)} failed={failed} correct={correct}/{len(cases)}"
    )
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
