"""Build and run the only supported final experiment commands."""

from __future__ import annotations

import os
import shlex
import subprocess
import json
from dataclasses import asdict, dataclass
from typing import BinaryIO
from pathlib import Path
from typing import Final

from ..config.experiment import ExperimentSpec
from ..data.catalog import count_cases, partition_case_indices


ROOT: Final = Path(__file__).resolve().parents[2]
PYTHON: Final = ROOT.parent / "venvs" / "wsi-latentmas-py312" / "bin" / "python"
class ExperimentLaunchError(RuntimeError):
    """Raised when a worker process exits unsuccessfully."""


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    """One GPU-specific command and its immutable environment."""

    gpu: int
    argv: tuple[str, ...]
    environment: dict[str, str]
    log_path: Path


def _shared_environment(spec: ExperimentSpec, gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(spec.hyperparameters.cpu_threads),
            "MKL_NUM_THREADS": str(spec.hyperparameters.cpu_threads),
            "OPENBLAS_NUM_THREADS": str(spec.hyperparameters.cpu_threads),
            "NUMEXPR_NUM_THREADS": str(spec.hyperparameters.cpu_threads),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_VISIBLE_DEVICES": str(gpu),
        }
    )
    return environment


def _latent_method(method: str) -> str:
    methods: Final = {
        "latent-base": "base",
        "pruning-v3": "pruning_v3",
        "pruning-a": "pruning_a",
        "reallocation-v2": "reallocation_v2",
        "both": "both",
        "pruning-v3-reallocation-v2": "pruning_v3_reallocation",
        "reallocation-v2-answerer-only": "reallocation_v2_answerer_only",
        "pruning-v3-reallocation-v2-answerer-only": "pruning_v3_reallocation_answerer_only",
    }
    return methods[method]


def _build_single_command(spec: ExperimentSpec, root: Path, indices: tuple[int, ...], gpu: int) -> WorkerCommand:
    parameters = spec.hyperparameters
    thinking = "false" if spec.method == "single-no-thinking" else "true"
    environment = _shared_environment(spec, gpu)
    environment.update(
        {
            "FAIR_CONTRACT_FILE": str(spec.contract_path),
            "WSIVQA_MODEL": str(spec.model_path),
            "WSIVQA_DATASET": str(spec.dataset_path),
            "WSIVQA_SLIDE_ROOT": str(spec.slide_root),
            "SINGLE_HF_BACKEND": "transformers",
            "SINGLE_CANONICAL_OPEN_OPTIONS": "1" if parameters.canonical_open_options else "0",
            "SINGLE_ANSWERER_PROTOCOL": parameters.answerer_protocol,
            "SINGLE_OUTPUT_PROTOCOL": parameters.answerer_protocol,
            "SINGLE_DIRECT_FINAL_ONLY": "1" if parameters.single_direct_final_only else "0",
            "SINGLE_NATIVE_THINKING": thinking,
            "SINGLE_REASONER_STYLE": "1" if parameters.single_reasoner_style else "0",
            "DO_SAMPLE": "false" if parameters.greedy_decoding else ("true" if parameters.single_do_sample else "false"),
            "SINGLE_TEMPERATURE": "0.0" if parameters.greedy_decoding else str(parameters.temperature),
            "SINGLE_TOP_P": "1.0" if parameters.greedy_decoding else str(parameters.top_p),
            "SINGLE_SEED": str(parameters.seed),
            "SINGLE_MAX_NEW_TOKENS": str(parameters.max_new_tokens),
            "SINGLE_MAX_MODEL_LEN": str(parameters.max_model_len),
            "HF_CASE_WORKERS": str(parameters.single_case_workers),
            "PYTHON_BIN": str(PYTHON),
        }
    )
    if spec.thumbnail_seed_dir is not None:
        environment["SINGLE_THUMBNAIL_SEED_DIR"] = str(spec.thumbnail_seed_dir)
    return WorkerCommand(
        gpu=gpu,
        argv=("bash", str(ROOT / "scripts" / "run_single_v7_fair_hf_batched.sh"), str(root), *(str(index) for index in indices)),
        environment=environment,
        log_path=root.parent / f"gpu{gpu}.log",
    )


def _build_text_mas_command(spec: ExperimentSpec, root: Path, indices: tuple[int, ...], gpu: int) -> WorkerCommand:
    parameters = spec.hyperparameters
    arguments = [
        str(PYTHON), "-m", "wsi_latentmas.pipeline.text_mas",
        "--dataset", str(spec.dataset_path), "--slide-root", str(spec.slide_root),
        "--model", str(spec.model_path), "--output-root", str(root), "--device", "cuda:0",
        "--max-model-len", str(parameters.max_model_len),
        "--temperature", "0.0" if parameters.greedy_decoding else str(parameters.temperature),
        "--top-p", "1.0" if parameters.greedy_decoding else str(parameters.top_p),
        "--seed", str(parameters.seed),
        "--answerer-max-new-tokens", str(parameters.max_new_tokens),
        "--answerer-protocol", parameters.answerer_protocol,
    ]
    if parameters.canonical_open_options:
        arguments.append("--canonical-open-options")
    if parameters.fast_json:
        arguments.append("--fast-json")
    if parameters.constrained_json:
        arguments.append("--constrained-json")
    if parameters.io_pipeline:
        arguments.append("--io-pipeline")
    for index in indices:
        arguments.extend(("--dataset-index", str(index)))
    return WorkerCommand(
        gpu=gpu,
        argv=tuple(arguments),
        environment=_shared_environment(spec, gpu),
        log_path=root.parent / f"gpu{gpu}.log",
    )


def _build_latent_mas_command(spec: ExperimentSpec, root: Path, indices: tuple[int, ...], gpu: int) -> WorkerCommand:
    parameters = spec.hyperparameters
    arguments = [
        str(PYTHON), "-m", "wsi_latentmas.pipeline.latent_mas",
        "--variant", _latent_method(spec.method), "--backbone", "qwen3-vl",
        "--dataset", str(spec.dataset_path), "--slide-root", str(spec.slide_root),
        "--model", str(spec.model_path), "--output-root", str(root), "--device", "cuda:0",
        "--latent-steps", str(spec.steps), "--patch-budget", str(parameters.patch_budget),
        "--max-model-len", str(parameters.max_model_len),
        "--navigator-control-tokens", str(parameters.navigator_control_tokens),
        "--temperature", "0.0" if parameters.greedy_decoding else str(parameters.temperature),
        "--top-p", "1.0" if parameters.greedy_decoding else str(parameters.top_p),
        "--seed", str(parameters.seed),
        "--answerer-max-new-tokens", str(parameters.max_new_tokens),
        "--terminal-no-repeat-ngram-size", str(parameters.terminal_no_repeat_ngram_size),
        "--terminal-repetition-penalty", str(parameters.terminal_repetition_penalty),
        "--terminal-antiloop-method", parameters.terminal_anti_repeat,
        "--answerer-protocol", parameters.answerer_protocol,
        "--transport-mode", parameters.transport_mode, "--realign-method", parameters.realign_method,
        "--case-retries", str(parameters.case_retries),
    ]
    if parameters.relay_source != "preset":
        arguments.extend(("--relay-source", parameters.relay_source))
    if parameters.deterministic:
        arguments.append("--deterministic")
    arguments.append("--answerer-greedy" if parameters.greedy_decoding else "--answerer-do-sample")
    if not parameters.answerer_thinking:
        arguments.append("--no-answerer-thinking")
    if parameters.canonical_open_options:
        arguments.append("--canonical-open-options")
    if parameters.navigator_kv:
        arguments.append("--navigator-kv")
    if parameters.io_pipeline:
        arguments.append("--io-pipeline")
    if parameters.answerer_rationale:
        arguments.append("--answerer-rationale")
    arguments.append("--save-navigation-pngs" if parameters.save_navigation_pngs else "--no-save-navigation-pngs")
    for index in indices:
        arguments.extend(("--dataset-index", str(index)))
    if spec.resume:
        arguments.append("--resume")
    return WorkerCommand(gpu, tuple(arguments), _shared_environment(spec, gpu), root.parent / f"gpu{gpu}.log")


def build_worker_commands(spec: ExperimentSpec) -> tuple[WorkerCommand, ...]:
    """Build one equivalent command per requested physical GPU."""
    total = count_cases(spec)
    commands: list[WorkerCommand] = []
    for shard, gpu in enumerate(spec.gpus):
        worker_root = spec.output_root / f"gpu{gpu}"
        indices = partition_case_indices(total, shard, len(spec.gpus))
        if spec.method in {"single", "single-no-thinking"}:
            command = _build_single_command(spec, worker_root, indices, gpu)
        elif spec.method == "vlmas":
            command = _build_text_mas_command(spec, worker_root, indices, gpu)
        else:
            command = _build_latent_mas_command(spec, worker_root, indices, gpu)
        commands.append(command)
    return tuple(commands)


def run_experiment(spec: ExperimentSpec) -> None:
    """Run all GPU shards, or print exactly what would run in dry-run mode."""
    commands = build_worker_commands(spec)
    print(f"hyperparameters={json.dumps(asdict(spec.hyperparameters), sort_keys=True)}")
    for command in commands:
        print(f"GPU {command.gpu}: {shlex.join(command.argv)}")
    if spec.dry_run:
        return
    if spec.output_root.exists() and not spec.resume:
        raise ExperimentLaunchError(f"output already exists: {spec.output_root}")
    spec.output_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[WorkerCommand, subprocess.Popen[bytes], BinaryIO]] = []
    for command in commands:
        command.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = command.log_path.open("wb")
        process = subprocess.Popen(command.argv, cwd=ROOT, env=command.environment, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((command, process, log_file))
    failures: list[int] = []
    for command, process, log_file in processes:
        status = process.wait()
        log_file.close()
        if status != 0:
            failures.append(command.gpu)
    if failures:
        raise ExperimentLaunchError(f"GPU worker failures: {failures}")
