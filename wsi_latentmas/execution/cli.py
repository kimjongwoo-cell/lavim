"""Single public CLI for all final experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer
from click import ClickException

from ..config.experiment import (
    DatasetName,
    ExperimentSpec,
    HyperParameters,
    MethodName,
    ModelName,
    canonical_output_root,
)
from .runner import ExperimentLaunchError, run_experiment


app = typer.Typer(no_args_is_help=True, add_completion=False)


def _gpus(value: str) -> tuple[int, ...]:
    """Parse a non-empty physical GPU list."""
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise typer.BadParameter("GPUs must be comma-separated integers") from error
    if not parsed or any(gpu < 0 for gpu in parsed):
        raise typer.BadParameter("GPUs must be non-negative integers")
    return parsed


@app.command()
def run(
    dataset: Annotated[DatasetName, typer.Option("--dataset")],
    model: Annotated[ModelName, typer.Option("--model")],
    method: Annotated[MethodName, typer.Option("--method")],
    gpus: Annotated[str, typer.Option("--gpus")],
    steps: Annotated[int, typer.Option("--steps", min=1)] = 5,
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
    max_model_len: Annotated[int, typer.Option("--max-model-len", min=1)] = 8192,
    patch_budget: Annotated[int, typer.Option("--patch-budget", min=1)] = 8,
    navigator_control_tokens: Annotated[int, typer.Option("--navigator-control-tokens", min=1)] = 512,
    temperature: Annotated[float, typer.Option("--temperature", min=0.0)] = 0.6,
    top_p: Annotated[float, typer.Option("--top-p", min=0.0, max=1.0)] = 0.95,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1)] = 512,
    cpu_threads: Annotated[int, typer.Option("--cpu-threads", min=1)] = 4,
    single_case_workers: Annotated[int, typer.Option("--single-case-workers", min=1)] = 1,
    io_pipeline: Annotated[bool, typer.Option("--io-pipeline/--no-io-pipeline")] = True,
    navigator_kv: Annotated[bool, typer.Option("--navigator-kv/--no-navigator-kv")] = True,
    canonical_open_options: Annotated[bool, typer.Option("--canonical-open-options/--no-canonical-open-options")] = True,
    deterministic: Annotated[bool, typer.Option("--deterministic/--no-deterministic")] = True,
    greedy_decoding: Annotated[bool, typer.Option("--greedy-decoding/--sampling")] = True,
    single_do_sample: Annotated[bool, typer.Option("--single-do-sample/--single-greedy")] = True,
    single_reasoner_style: Annotated[bool, typer.Option("--single-reasoner-style/--single-direct-style")] = True,
    single_direct_final_only: Annotated[bool, typer.Option("--single-direct-final-only/--single-reasoner-final")] = False,
    answerer_thinking: Annotated[bool, typer.Option("--answerer-thinking/--no-answerer-thinking")] = False,
    answerer_rationale: Annotated[bool, typer.Option("--answerer-rationale/--no-answerer-rationale")] = True,
    answerer_protocol: Annotated[Literal["structured_json", "tag"], typer.Option("--answerer-protocol")] = "structured_json",
    transport_mode: Annotated[Literal["cumulative", "cumulative_nothink"], typer.Option("--transport-mode")] = "cumulative",
    realign_method: Annotated[Literal["identity_norm", "wa", "softmax"], typer.Option("--realign-method")] = "wa",
    fast_json: Annotated[bool, typer.Option("--fast-json/--no-fast-json")] = True,
    constrained_json: Annotated[bool, typer.Option("--constrained-json/--no-constrained-json")] = False,
    save_navigation_pngs: Annotated[bool, typer.Option("--save-navigation-pngs/--no-save-navigation-pngs")] = False,
    terminal_no_repeat_ngram_size: Annotated[int, typer.Option("--terminal-no-repeat-ngram-size", min=0, max=32)] = 0,
    terminal_repetition_penalty: Annotated[float, typer.Option("--terminal-repetition-penalty", min=1.0, max=2.0)] = 1.0,
    terminal_anti_repeat: Annotated[Literal["none", "boxed_ngram3", "boxed_ngram3_penalty", "tag_ngram3", "json_ngram3"], typer.Option("--terminal-anti-repeat")] = "none",
    case_retries: Annotated[int, typer.Option("--case-retries", min=0)] = 0,
    relay_source: Annotated[Literal["preset", "key_norm", "uniform", "sender_priority", "shuffled_sender_priority"], typer.Option("--relay-source")] = "preset",
) -> None:
    """Run Single, VL-MAS, Latent Base, Pruning V3, or a method ablation."""
    gpu_ids = _gpus(gpus)
    spec = ExperimentSpec(
        dataset=dataset,
        model=model,
        method=method,
        steps=steps,
        gpus=gpu_ids,
        output_root=output_root
        or (
            canonical_output_root(dataset, model, method, steps)
            if relay_source == "preset"
            else canonical_output_root(dataset, model, method, steps).with_name(
                f"{canonical_output_root(dataset, model, method, steps).name}"
                f"-relay-{relay_source}"
            )
        ),
        resume=resume,
        dry_run=dry_run,
        hyperparameters=HyperParameters(
            max_model_len=max_model_len,
            patch_budget=patch_budget,
            navigator_control_tokens=navigator_control_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_new_tokens=max_new_tokens,
            cpu_threads=cpu_threads,
            single_case_workers=single_case_workers,
            io_pipeline=io_pipeline,
            navigator_kv=navigator_kv,
            canonical_open_options=canonical_open_options,
            deterministic=deterministic,
            greedy_decoding=greedy_decoding,
            single_do_sample=single_do_sample,
            single_reasoner_style=single_reasoner_style,
            single_direct_final_only=single_direct_final_only,
            answerer_thinking=answerer_thinking,
            answerer_rationale=answerer_rationale,
            answerer_protocol=answerer_protocol,
            transport_mode=transport_mode,
            realign_method=realign_method,
            fast_json=fast_json,
            constrained_json=constrained_json,
            save_navigation_pngs=save_navigation_pngs,
            terminal_no_repeat_ngram_size=terminal_no_repeat_ngram_size,
            terminal_repetition_penalty=terminal_repetition_penalty,
            terminal_anti_repeat=terminal_anti_repeat,
            case_retries=case_retries,
            relay_source=relay_source,
        ),
    )
    try:
        run_experiment(spec)
    except ExperimentLaunchError as error:
        raise ClickException(str(error)) from error


def main() -> None:
    """Expose a conventional Python entry point."""
    app()


if __name__ == "__main__":
    main()
