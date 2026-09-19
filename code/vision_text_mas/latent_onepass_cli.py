"""All-HF, cumulative-KV runner for the four-agent Latent VL-MAS experiment."""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Annotated, Callable, Literal, Protocol, TypeVar, cast

import torch
import typer
from click import ClickException
from PIL import Image
from pydantic import TypeAdapter

from vision_text_mas.artifacts import ArtifactStore
from vision_text_mas.cli import _open_slide
from vision_text_mas.contracts import CaseInput, DatasetItem, RunResult
from vision_text_mas.dataset import load_cases
from vision_text_mas.determinism import enable_deterministic_cuda
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.latent_backend import LatentBackendAdapter
from vision_text_mas.latent_onepass import LatentRoleClient
from vision_text_mas.latent_answerer import LatentAnswererAgent
from vision_text_mas.latent_onepass_agents import (
    LatentEvidencePlannerAgent,
    LatentReasonerAgent,
)
from vision_text_mas.latent_answerer import AnswererProtocol, FINAL_ANSWER_TOKEN_BUDGET
from vision_text_mas.latent_onepass_controller import LatentOnePassController
from vision_text_mas.latent_qwen_engine import LatentQwenEngine
from vision_text_mas.latent_state import LatentCaseState
from vision_text_mas.io_pipeline import AsyncJsonWriter, BoundedImageCache
from vision_text_mas.navigation.navigation_render import SlideLike
from vision_text_mas.navigation.onepass_navigator import OnePassNavigator
from vision_text_mas.navigation.onepass_navigator_agent import OneShotNavigatorAgent
from vision_text_mas.qwen_client import QwenJsonClient

if TYPE_CHECKING:
    from vision_text_mas.navigation.onepass_navigation_roots import RootGrid


app = typer.Typer(no_args_is_help=True, add_completion=False)
CacheT = TypeVar("CacheT")


class ManagedSlide(SlideLike, Protocol):
    def close(self) -> None: ...


class SlideReuseCache:
    """Reuse an immutable slide handle and thumbnails across its questions."""

    def __init__(self, *, enable_io_pipeline: bool = False) -> None:
        self._path: Path | None = None
        self._slide: ManagedSlide | None = None
        self._thumbnails: dict[tuple[int, int], Image.Image] = {}
        self._root_grid: RootGrid | None = None
        self._regions = BoundedImageCache(max_bytes=256 * 1024 * 1024)
        self._enable_io_pipeline = enable_io_pipeline
        self._prefetch_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="slide-prefetch")
            if enable_io_pipeline
            else None
        )
        self._prefetch_path: Path | None = None
        self._prefetch_future: Future[tuple[ManagedSlide, Image.Image, RootGrid]] | None = None

    def acquire(self, case: CaseInput) -> "SlideReuseCache":
        if self._path != case.slide_path:
            self.close()
            if self._prefetch_path == case.slide_path and self._prefetch_future is not None:
                self._slide, thumbnail, self._root_grid = self._prefetch_future.result()
                self._thumbnails[(1_024, 1_024)] = thumbnail
            else:
                self._slide = cast(ManagedSlide, _open_slide(case))
            self._path = case.slide_path
            self._prefetch_path = None
            self._prefetch_future = None
        return self

    @staticmethod
    def _prepare(case: CaseInput) -> tuple[ManagedSlide, Image.Image, RootGrid]:
        from vision_text_mas.navigation.navigation_render import slide_thumbnail
        from vision_text_mas.navigation.onepass_navigation_roots import prepare_root_grid

        slide = cast(ManagedSlide, _open_slide(case))
        try:
            thumbnail = slide_thumbnail(slide)
            return slide, thumbnail, prepare_root_grid(slide, thumbnail=thumbnail)
        except BaseException:
            slide.close()
            raise

    def prefetch(self, case: CaseInput) -> None:
        if not self._enable_io_pipeline or case.slide_path == self._path:
            return
        if self._prefetch_path == case.slide_path:
            return
        if self._prefetch_future is not None:
            prefetched_slide, _, _ = self._prefetch_future.result()
            prefetched_slide.close()
        if self._prefetch_executor is None:
            raise RuntimeError("prefetch executor is unavailable")
        self._prefetch_path = case.slide_path
        self._prefetch_future = self._prefetch_executor.submit(self._prepare, case)

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._require_slide().dimensions

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self._require_slide().level_downsamples

    def get_thumbnail(self, size: tuple[int, int]):
        thumbnail = self._thumbnails.get(size)
        if thumbnail is None:
            thumbnail = self._require_slide().get_thumbnail(size)
            self._thumbnails[size] = thumbnail
        return thumbnail.copy()

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return self._require_slide().get_best_level_for_downsample(downsample)

    def read_region(self, *args, **kwargs):
        if not self._enable_io_pipeline:
            return self._require_slide().read_region(*args, **kwargs)
        key = (args, tuple(sorted(kwargs.items())))
        return self._regions.get_or_load(
            key,
            lambda: self._require_slide().read_region(*args, **kwargs),
        )

    def cached_root_grid(self, thumbnail: Image.Image) -> RootGrid:
        from vision_text_mas.navigation.onepass_navigation_roots import prepare_root_grid

        if not self._enable_io_pipeline:
            return prepare_root_grid(self, thumbnail=thumbnail)
        if self._root_grid is None:
            self._root_grid = prepare_root_grid(self, thumbnail=thumbnail)
        return self._root_grid

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()
        self._path = None
        self._slide = None
        self._thumbnails.clear()
        self._root_grid = None
        self._regions.clear()

    def shutdown(self) -> None:
        self.close()
        if self._prefetch_future is not None:
            prefetched_slide, _, _ = self._prefetch_future.result()
            prefetched_slide.close()
            self._prefetch_future = None
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True, cancel_futures=False)

    def _require_slide(self) -> ManagedSlide:
        if self._slide is None:
            raise RuntimeError("slide cache is not initialized")
        return self._slide


def _run_with_immediate_retry(
    operation: Callable[[], RunResult], *, max_attempts: int = 3
) -> RunResult:
    """Retry one isolated case immediately up to the requested total attempts."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(max_attempts):
        try:
            return operation()
        except (PipelineFailure, torch.OutOfMemoryError):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt + 1 == max_attempts:
                raise
    raise AssertionError("retry loop must either return or raise")


def _requested_indices(dataset: Path, requested: tuple[int, ...]) -> tuple[int, ...]:
    """Expand an omitted index list to every validated WSI-VQA test row."""
    if requested:
        return requested
    items = TypeAdapter(list[DatasetItem]).validate_json(dataset.read_bytes())
    return tuple(range(len(items)))


def _write_latent_audits(client: LatentRoleClient[CacheT], case_root: Path) -> None:
    """Persist tensor-free proof of the Planner->Navigator->Reasoner transfer."""
    path = case_root / "latent_audits.json"
    payload = [asdict(audit) for audit in client.state.audits]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_case(
    *,
    case: CaseInput,
    engine: LatentQwenEngine[CacheT],
    output_root: Path,
    latent_steps: int,
    max_model_len: int,
    answerer_max_new_tokens: int,
    patch_budget: Literal[8, 25] = 8,
    navigator_control_tokens: int = 512,
    answerer_include_rationale: bool = False,
    answerer_protocol: AnswererProtocol = AnswererProtocol.TAG,
    canonical_open_options: bool = False,
    save_navigation_pngs: bool = True,
    retain_navigator_kv: bool = True,
    slide_cache: SlideReuseCache | None = None,
    artifact_writer: AsyncJsonWriter | None = None,
) -> RunResult:
    """Run one full prompt-KV plus latent-KV chain for a single slide."""
    case_root = output_root / f"{case.dataset_index:03d}_{case.item.slide_id}"
    artifacts = ArtifactStore(case_root, json_writer=artifact_writer)
    temporary_navigation = (
        None
        if save_navigation_pngs
        else TemporaryDirectory(prefix=f"latent_navigation_{case.dataset_index:03d}_")
    )
    navigation_root = (
        case_root / "navigation"
        if temporary_navigation is None
        else Path(temporary_navigation.name)
    )
    client = LatentRoleClient(
        backend=LatentBackendAdapter(engine),
        initial_state=LatentCaseState.empty(case.item.slide_id, case.dataset_index),
        latent_steps=latent_steps,
        max_model_len=max_model_len,
        retain_navigator_kv=retain_navigator_kv,
    )
    controller = LatentOnePassController(
        planner=LatentEvidencePlannerAgent(client),
        navigator=OnePassNavigator(
            agent=OneShotNavigatorAgent(
                cast(QwenJsonClient, cast(object, client)),
                control_tokens=navigator_control_tokens,
            ),
            artifact_root=navigation_root,
        ),
        reasoner=LatentReasonerAgent(client),
        answerer=LatentAnswererAgent(
            client,
            final_tokens=answerer_max_new_tokens,
            include_rationale=answerer_include_rationale,
            protocol=answerer_protocol,
            canonical_open_options=canonical_open_options,
        ),
        artifacts=artifacts,
        patch_budget=patch_budget,
    )
    owns_slide = slide_cache is None
    try:
        slide = _open_slide(case) if slide_cache is None else slide_cache.acquire(case)
    except PipelineFailure as failure:
        artifacts.write_failure(failure, case)
        if temporary_navigation is not None:
            temporary_navigation.cleanup()
        raise
    try:
        return controller.run_case(case, slide=slide)
    finally:
        if owns_slide:
            slide.close()
        _write_latent_audits(client, case_root)
        client.release()
        if temporary_navigation is not None:
            temporary_navigation.cleanup()


@app.command()
def run(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True)],
    slide_root: Annotated[Path, typer.Option("--slide-root", exists=True)],
    model: Annotated[Path, typer.Option("--model", exists=True)],
    output_root: Annotated[Path, typer.Option("--output-root")],
    dataset_index: Annotated[
        list[int], typer.Option("--dataset-index", help="Repeat per requested row.")
    ] = [],
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    latent_steps: Annotated[int, typer.Option("--latent-steps", min=1, max=160)] = 5,
    max_model_len: Annotated[int, typer.Option("--max-model-len", min=1)] = 8_192,
    temperature: Annotated[float, typer.Option("--temperature", min=0.0, max=2.0)] = 0.6,
    top_p: Annotated[float, typer.Option("--top-p", min=0.0, max=1.0)] = 0.95,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    deterministic: Annotated[
        bool, typer.Option("--deterministic/--non-deterministic")
    ] = False,
    answerer_do_sample: Annotated[
        bool,
        typer.Option("--answerer-do-sample/--answerer-greedy"),
    ] = True,
    answerer_thinking: Annotated[
        bool,
        typer.Option(
            "--answerer-thinking/--no-answerer-thinking",
            help="Keep the final boxed/free-text Answerer in Qwen Thinking mode.",
        ),
    ] = True,
    answerer_max_new_tokens: Annotated[
        int,
        typer.Option("--answerer-max-new-tokens", min=1, max=4_096),
    ] = FINAL_ANSWER_TOKEN_BUDGET,
    answerer_rationale: Annotated[
        bool,
        typer.Option("--answerer-rationale/--no-answerer-rationale"),
    ] = True,
    answerer_protocol: Annotated[
        AnswererProtocol, typer.Option("--answerer-protocol")
    ] = AnswererProtocol.STRUCTURED_JSON,
    canonical_open_options: Annotated[
        bool,
        typer.Option(
            "--canonical-open-options/--no-canonical-open-options",
            help="Constrain OPEN answers to the dataset's declared canonical labels "
            "(histological_type/vital_status/receptor/her2). On by default.",
        ),
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
        Literal[
            "latent_only",
            "sequential_info_only",
            "cumulative",
            "cumulative_nothink",
            "upstream",
        ],
        typer.Option("--transport-mode"),
    ] = "cumulative",
    realign_method: Annotated[
        Literal["identity_norm", "wa", "softmax"],
        typer.Option(
            "--realign-method",
            help="Latent-step realign map: wa (default), identity_norm, or softmax.",
        ),
    ] = "wa",
    thinking_role: Annotated[
        list[str],
        typer.Option(
            "--thinking-role",
            help="Repeat per intermediate role: evidence_planner, navigator, reasoner.",
        ),
    ] = [],
) -> None:
    """Run all four agents with direct-HF latent-KV transport."""
    if output_root.exists():
        raise typer.BadParameter("output root must be new", param_hint="--output-root")
    thinking_roles = frozenset(thinking_role)
    allowed_thinking_roles = frozenset({"evidence_planner", "navigator", "reasoner"})
    unknown_thinking_roles = thinking_roles - allowed_thinking_roles
    if unknown_thinking_roles:
        raise typer.BadParameter(
            "unsupported thinking role(s): " + ", ".join(sorted(unknown_thinking_roles)),
            param_hint="--thinking-role",
        )
    if thinking_roles and transport_mode != "cumulative_nothink":
        raise typer.BadParameter(
            "role-specific thinking requires --transport-mode cumulative_nothink "
            "to close the selected thinking turn before the next role",
            param_hint="--transport-mode",
        )
    try:
        if deterministic:
            enable_deterministic_cuda()
        indices = _requested_indices(dataset, tuple(dataset_index))
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
            transport_mode=transport_mode,
            thinking_roles=thinking_roles,
            realign_method=realign_method,
        )
    except (OSError, PipelineFailure, ValueError) as failure:
        raise ClickException(str(failure)) from failure
    output_root.mkdir(parents=True)
    manifest = {
        "backend": "transformers-direct-kv",
        "agent_backends": {
            "evidence_planner": "transformers",
            "navigator": "transformers",
            "reasoner": "transformers",
            "answerer": "transformers",
        },
        "terminal_backend": "transformers",
        "vllm": False,
        "terminal_protocol": answerer_protocol.value,
        "terminal_answer_marker": answerer_protocol is AnswererProtocol.BOXED,
        "pipeline_mode": "latent-onepass-8",
        "agent_count": 4,
        "intermediate_transport": f"sequential_hf_{transport_mode}",
        "transport_mode": transport_mode,
        "realign_method": realign_method,
        "canonical_open_options": canonical_open_options,
        "save_navigation_pngs": save_navigation_pngs,
        "navigator_kv": retain_navigator_kv,
        "thinking_roles": sorted(thinking_roles),
        "navigator_tool_decodes": 1,
        "terminal_answer_decodes": 1,
        "patch_budget": 8,
        "max_rounds": 1,
        "latent_steps": latent_steps,
        "pruning": False,
        "reallocation": False,
        "max_model_len": max_model_len,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "deterministic_cuda": deterministic,
        "answerer_do_sample": answerer_do_sample,
        "answerer_thinking": answerer_thinking,
        "answerer_max_new_tokens": answerer_max_new_tokens,
        "requested_indices": indices,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    completed = 0
    failed = 0
    for case in cases:
        try:
            _run_with_immediate_retry(
                lambda: _run_case(
                    case=case,
                    engine=engine,
                    output_root=output_root,
                    latent_steps=latent_steps,
                    max_model_len=max_model_len,
                    answerer_max_new_tokens=answerer_max_new_tokens,
                    answerer_include_rationale=answerer_rationale,
                    answerer_protocol=answerer_protocol,
                        canonical_open_options=canonical_open_options,
                        save_navigation_pngs=save_navigation_pngs,
                        retain_navigator_kv=retain_navigator_kv,
                )
            )
            completed += 1
        except PipelineFailure as failure:
            failed += 1
            typer.echo(f"FAILED {case.dataset_index} {case.item.slide_id}: {failure}")
        except torch.OutOfMemoryError as error:
            failed += 1
            ArtifactStore(
                output_root / f"{case.dataset_index:03d}_{case.item.slide_id}"
            ).write_failure(
                PipelineFailure(
                    code=FailureCode.MODEL_EXECUTION,
                    stage="cuda",
                    detail=f"CUDA out of memory: {error}",
                ),
                case,
            )
            torch.cuda.empty_cache()
    typer.echo(f"completed={completed} failed={failed} requested={len(cases)}")
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
