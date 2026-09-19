# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# ─── How to run ───
# PYTHONPATH=. uv run -m dashboard.server --port 8767

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from time import monotonic
from typing import Final
from urllib.parse import urlsplit

from openpyxl import load_workbook

from eval.metrics import EvalRow, compute_coco_scores, evaluate, expand_letter
from vision_text_mas.live_prefix_metrics import (
    Indicator,
    cumulative_points,
    latent_indicator,
    load_eval_rows,
    load_strict_functions,
    select_latest_cases,
)

CODE_ROOT: Final = Path(__file__).resolve().parents[1]
ROOT: Final = CODE_ROOT.parent / "results"
CANONICAL_MATRIX_ROOT: Final = ROOT / "runs/qwen_full_matrix_local_canonical_finalmatched_20260829"
MATCHED_SINGLE_ROOT: Final = ROOT / "runs/qwen_single_matched_to_vlmas_sampling_20260830"
LATEST_SINGLE_NOTHINKING_ROOT: Final = ROOT / "runs/qwen_single_nothinking_r2_20260902"
FINAL_SINGLE_NOTHINKING_ROOT: Final = ROOT / "runs/qwen_single_nothinking_final_v3_20260902"
RECOVERY_SINGLE_NOTHINKING_ROOT: Final = ROOT / "runs/qwen_single_nothinking_final_v4_20260902"
CONTROLLED_SINGLE_NOTHINKING_ROOT: Final = ROOT / "runs/qwen_single_nothinking_controlled_20260902"
PAGE: Final = CODE_ROOT / "dashboard" / "index.html"
STATUS_CACHE_SECONDS: Final = 0.75
LIVE_QUALITY_REFRESH_CASES: Final = 16
LATENCY_GPUS: Final = frozenset({7, 8})
OFFICIAL_METRICS_CACHE: Final = ROOT / "runs/dashboard/official_metrics.json"
PERFORMANCE_WORKBOOK: Final = ROOT / "main_performance.xlsx"
MULTIPATH_DATASET: Final = Path(
    "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/multipathqa_no_panda.json"
)
MULTIPATH_ROOT: Path = Path("/home/users/whddn12316/datasets/MultiPathQA")
GTEX_DICOM_TARGET: Final = 957
MULTIPATH_BENCHMARKS: Final = {
    "tcga_expert_vqa": "ExpertVQA",
    "tcga_slidebench": "SlideBench",
    "tcga": "TCGA",
    "gtex": "GTEx",
    "panda": "PANDA",
}
MULTIPATH_BALANCED_TASKS: Final = frozenset({"tcga", "gtex", "panda"})
MULTIPATH_READY_ROOT: Final = MULTIPATH_ROOT / "ready_wsivqa/full_no_panda"
MULTIPATH_RUN_BENCHMARKS: Final = {
    "tcga_expert_vqa": ("ExpertVQA", 128),
    "tcga_slidebench": ("SlideBench", 197),
    "tcga": ("TCGA", 221),
    "gtex": ("GTEx", 190),
}
_status_cache = b""
_status_cache_at = 0.0
_status_cache_lock = Lock()
_live_quality_cache: dict[str, tuple[int, QualityMetric]] = {}
_live_quality_lock = Lock()
_live_quality_refreshing: set[str] = set()
_live_quality_refresh_lock = Lock()
_multipath_quality_cache: dict[
    tuple[str, str, int], tuple[QualityMetric, ...]
] = {}
_gpu_status_cache: tuple[GpuStatus, ...] = ()
_gpu_status_cache_at = 0.0
_gpu_status_refreshing = False
_gpu_status_lock = Lock()
_pathagent_score_lock = Lock()
_pathagent_score_executor: Final = ProcessPoolExecutor(max_workers=1)


@dataclass(frozen=True, slots=True)
class JobSpec:
    name: str
    gpu: int
    target: int
    root: Path
    process_marker: str
    single_predictions: bool = False
    recovery_root: Path | None = None
    extra_roots: tuple[Path, ...] = ()
    speed_root: Path | None = None
    speed_peer_root: Path | None = None
    speed_extra_roots: tuple[Path, ...] = ()
    speed_warmup_pairs: int = 0
    speed_warmup_attempts_per_root: int = 0
    active_gpus: tuple[int, ...] = ()
    evaluation_summary: Path | None = None
    latent_steps: int | None = None
    model_size: str | None = None
    queue_group: str | None = None
    queue_order: int | None = None
    thinking_mode: str = "No Thinking"


@dataclass(frozen=True, slots=True)
class JobStatus:
    name: str
    state: str
    state_label: str
    target: int
    success: int
    failure: int
    processed: int
    percent: int
    seconds_per_case: float | None
    recent_seconds_per_case: float | None
    eta_seconds: float | None
    retry_count: int | None = None
    queue_position: int | None = None
    queue_wait_seconds: float | None = None
    allocated_gpus: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GpuStatus:
    index: int
    utilization: int
    memory_used_gib: float
    memory_total_gib: float
    pstate: str
    active_workload: str | None


def gtex_sync_status() -> JobStatus:
    """Expose local GTEx DICOM synchronization as a live dashboard job."""
    downloaded = sum(1 for _ in (MULTIPATH_ROOT / "slides/gtex_idc").rglob("*.dcm"))
    completed = downloaded >= GTEX_DICOM_TARGET
    return JobStatus(
        name="MultiPathQA/GTEx · 데이터 동기화",
        state="completed" if completed else "running",
        state_label="완료" if completed else "동기화 중",
        target=GTEX_DICOM_TARGET,
        success=downloaded,
        failure=0,
        processed=downloaded,
        percent=min(100, round(downloaded / GTEX_DICOM_TARGET * 100)),
        seconds_per_case=None,
        recent_seconds_per_case=None,
        eta_seconds=None,
    )


def multipath_dataset_for_spec(spec: JobSpec) -> Path:
    """Select the immutable benchmark dataset associated with one run."""
    if "multipathqa_gtex_" in str(spec.root):
        return MULTIPATH_ROOT / "ready_wsivqa/gtex_complete/gtex.json"
    for benchmark in MULTIPATH_RUN_BENCHMARKS:
        if f"/{benchmark}/" in str(spec.root):
            return MULTIPATH_READY_ROOT / f"{benchmark}.json"
    return MULTIPATH_DATASET


def dataset_label_for_spec(spec: JobSpec) -> str:
    """Return the independently evaluated dataset represented by a run."""
    if "multipathqa_gtex_" in str(spec.root):
        return "MultiPathQA/GTEx"
    for benchmark, (label, _) in MULTIPATH_RUN_BENCHMARKS.items():
        if f"/{benchmark}/" in str(spec.root):
            return f"MultiPathQA/{label}"
    if spec.queue_group == "multipathqa-qwen":
        return "MultiPathQA"
    if spec.queue_group in {"slidebench-bcnb-qwen2b", "slidebench-bcnb-qwen4b"}:
        return "SlideBench-BCNB"
    return "WSI-VQA"


@dataclass(frozen=True, slots=True)
class LiveMetric:
    name: str
    completed: int
    total_accuracy: float | None
    mcq_accuracy: float | None
    open_accuracy: float | None
    exact: float | None = None
    token_f1: float | None = None
    bleu_1: float | None = None
    bleu_2: float | None = None
    bleu_3: float | None = None
    bleu_4: float | None = None


@dataclass(frozen=True, slots=True)
class QualityMetric:
    backbone: str
    mode: str
    samples: int
    total: float | None
    mcq: float | None
    open: float | None
    exact: float | None
    token_f1: float | None
    bleu_1: float | None
    bleu_2: float | None
    bleu_3: float | None
    bleu_4: float | None
    seconds_per_case: float | None = None
    state: str = "완료"
    latent_steps: int | None = None
    model_size: str | None = None
    meteor: float | None = None
    rouge_l: float | None = None
    dataset: str = "WSI-VQA"
    thinking_mode: str = "No Thinking"


def merge_open_only_quality(
    metric: QualityMetric,
    closed_samples: int,
    closed_accuracy: float,
) -> QualityMetric:
    """Keep a completed multiple-choice score while an Open-only run grows."""
    if metric.open is None:
        return metric
    samples = closed_samples + metric.samples
    total = round(
        (closed_accuracy * closed_samples + metric.open * metric.samples) / samples,
        2,
    )
    return replace(metric, samples=samples, total=total, mcq=closed_accuracy)


JOBS: Final = (
    JobSpec(
        "Qwen 2B Single · aligned Latent Base protocol · step 5",
        5,
        735,
        ROOT / "runs/canonical/wsi-vqa/qwen-2b/single-aligned-final-v3/gpu5",
        "qwen-2b/single-aligned-final-v3",
        single_predictions=True,
        extra_roots=tuple(
            ROOT / f"runs/canonical/wsi-vqa/qwen-2b/single-aligned-final-v3/gpu{gpu}"
            for gpu in (6, 7, 8)
        ),
        active_gpus=(5, 6, 7, 8),
        model_size="2B",
        queue_group="qwen-2b-local-matrix",
        queue_order=0,
        speed_warmup_attempts_per_root=1,
        thinking_mode="No Thinking",
    ),
    JobSpec(
        "Qwen 2B Single No Thinking",
        5,
        735,
        ROOT / "runs/qwen_single_nothinking_phased_20260902/wsi/2b/restart_gpu5",
        "qwen_single_nothinking_final_v4_20260902/wsi/2b/gpu",
        single_predictions=True,
        extra_roots=tuple(
            ROOT / f"runs/qwen_single_nothinking_phased_20260902/wsi/2b/restart_gpu{gpu}"
            for gpu in (6, 7, 8)
        ) + tuple(
            RECOVERY_SINGLE_NOTHINKING_ROOT / f"wsi/2b/gpu{gpu}"
            for gpu in (5, 6, 7, 8)
        ),
        active_gpus=(5, 6, 7, 8),
        model_size="2B",
        queue_group="qwen-2b-local-matrix",
        queue_order=1,
    ),
    JobSpec(
        "Qwen 2B VLMAS",
        6,
        735,
        ROOT / "runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/vlmas_gpu6",
        "qwen3vl2b_firstcall_four_modes_local_v6_20260823/vlmas_gpu",
        extra_roots=tuple(
            ROOT / f"runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/vlmas_gpu{gpu}"
            for gpu in (6, 7, 8)
        ),
        active_gpus=(7, 8),
        model_size="2B",
        queue_group="qwen2b-six-modes",
        queue_order=6,
    ),
    JobSpec(
        "Qwen 4B Single No Thinking",
        5,
        735,
        CONTROLLED_SINGLE_NOTHINKING_ROOT / "wsi/4b/gpu5",
        "qwen_single_nothinking_controlled_20260902/wsi/4b/gpu",
        single_predictions=True,
        extra_roots=tuple(
            CONTROLLED_SINGLE_NOTHINKING_ROOT / f"wsi/4b/gpu{gpu}"
            for gpu in (5, 6, 7, 8)
        ),
        active_gpus=(5, 6, 7, 8),
        model_size="4B",
        queue_group="qwen-4b-local-matrix",
        queue_order=0,
    ),
    JobSpec(
        "Qwen 4B Single · matched base · step 10",
        5,
        735,
        ROOT / "runs/canonical/wsi-vqa/qwen-4b/single-matched-base-step10/gpu5",
        "qwen-4b/single-matched-base-step10",
        single_predictions=True,
        extra_roots=tuple(
            ROOT / f"runs/canonical/wsi-vqa/qwen-4b/single-matched-base-step10/gpu{gpu}"
            for gpu in (6, 7, 8)
        ) + tuple(
            ROOT / f"runs/canonical/wsi-vqa/qwen-4b/single-matched-base-step10-resume-v4/gpu{gpu}"
            for gpu in (5, 6, 7, 8)
        ),
        active_gpus=(5, 6, 7, 8),
        model_size="4B",
        queue_group="qwen-4b-local-matrix",
        queue_order=1,
        speed_warmup_attempts_per_root=1,
        thinking_mode="Thinking",
    ),
    JobSpec(
        "Qwen 2B Single · matched base · step 10",
        5,
        735,
        ROOT / "runs/canonical/wsi-vqa/qwen-2b/single-matched-base-step10-after4/gpu5",
        "qwen-2b/single-matched-base-step10-after4",
        single_predictions=True,
        extra_roots=(
            ROOT / "runs/canonical/wsi-vqa/qwen-2b/single-matched-base-step10-after4/gpu6",
            ROOT / "runs/canonical/wsi-vqa/qwen-2b/single-matched-base-step10-after4/gpu7",
            ROOT / "runs/canonical/wsi-vqa/qwen-2b/single-matched-base-step10-after4/gpu8",
        ),
        active_gpus=(5, 6, 7, 8),
        model_size="2B",
        queue_group="qwen-2b-local-matrix",
        queue_order=0,
        speed_warmup_attempts_per_root=1,
        thinking_mode="Thinking",
    ),
    JobSpec(
        "Qwen 8B Single · matched base · step 10",
        5,
        735,
        ROOT / "runs/canonical/wsi-vqa/qwen-8b/single-matched-base-step10-after4/gpu5",
        "qwen-8b/single-matched-base-step10-after4",
        single_predictions=True,
        extra_roots=(
            ROOT / "runs/canonical/wsi-vqa/qwen-8b/single-matched-base-step10-after4/gpu6",
            ROOT / "runs/canonical/wsi-vqa/qwen-8b/single-matched-base-step10-after4/gpu7",
            ROOT / "runs/canonical/wsi-vqa/qwen-8b/single-matched-base-step10-after4/gpu8",
        ),
        active_gpus=(5, 6, 7, 8),
        model_size="8B",
        queue_group="qwen-8b-local-matrix",
        queue_order=0,
        speed_warmup_attempts_per_root=1,
        thinking_mode="Thinking",
    ),
    *tuple(
        JobSpec(
            f"MultiPathQA/{label} · Qwen {model_size} {'Single No Thinking' if mode == 'Single' else mode}",
            5,
            target,
            (
                (
                    CONTROLLED_SINGLE_NOTHINKING_ROOT
                    / f"multipath/{benchmark}/{model_size.lower()}/gpu5"
                )
                if mode == "Single" and model_size in {"4B", "8B"}
                else FINAL_SINGLE_NOTHINKING_ROOT / f"multipath/{benchmark}/{model_size.lower()}/gpu5"
                if mode == "Single"
                else ROOT / f"runs/multipathqa_remaining_qwen_base_allsteps_gpu5678_20260902/{benchmark}/{model_size.lower()}/{root_name}/gpu5"
            ),
            (
                f"qwen_single_nothinking_final_v5_20260902/multipath/{benchmark}/{model_size.lower()}/gpu"
                if mode == "Single"
                else f"multipathqa_remaining_qwen_base_allsteps_gpu5678_20260902/{benchmark}/{model_size.lower()}/{root_name}"
            ),
            single_predictions=mode == "Single",
            extra_roots=tuple(
                (
                    (
                        CONTROLLED_SINGLE_NOTHINKING_ROOT
                        / f"multipath/{benchmark}/{model_size.lower()}/gpu{gpu}"
                    )
                    if model_size in {"4B", "8B"}
                    else FINAL_SINGLE_NOTHINKING_ROOT / f"multipath/{benchmark}/{model_size.lower()}/gpu{gpu}"
                    if mode == "Single"
                    else ROOT / f"runs/multipathqa_remaining_qwen_base_allsteps_gpu5678_20260902/{benchmark}/{model_size.lower()}/{root_name}/gpu{gpu}"
                )
                for gpu in (6, 7, 8)
            ) + (
                ()
                if mode == "Single"
                else ()
            ),
            active_gpus=(5, 6, 7, 8),
            model_size=model_size,
            queue_group=(
                "multipathqa-single-nothinking"
                if mode == "Single"
                else "multipathqa-remaining-base-allsteps"
            ),
            queue_order=benchmark_index * 18 + model_index * 6 + mode_offset,
        )
        for benchmark_index, (benchmark, (label, target)) in enumerate(
            MULTIPATH_RUN_BENCHMARKS.items()
        )
        for model_index, model_size in enumerate(("2B", "4B", "8B"))
        for mode_offset, (mode, root_name) in enumerate(
            (("Single", "single"), ("VLMAS", "vlmas")), start=1
        )
    ),
    *tuple(
        JobSpec(
            f"Qwen 2B {label}",
            6,
            735,
            ROOT / f"runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/{root_name}_gpu6",
            f"qwen3vl2b_firstcall_four_modes_local_v6_20260823/{root_name}_gpu",
            extra_roots=tuple(
                ROOT / f"runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/{root_name}_gpu{gpu}"
                for gpu in (6, 7, 8)
            ) + tuple(
                ROOT / f"runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/{root_name}_resume_gpu{gpu}"
                for gpu in (6, 7, 8)
            ) + tuple(
                ROOT / f"runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/{root_name}_v2_gpu{gpu}"
                for gpu in (6, 7, 8)
            ) + tuple(
                ROOT / f"runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/{root_name}_v3_gpu{gpu}"
                for gpu in (6, 7, 8)
            ) + tuple(
                ROOT / f"runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/{root_name}_v4_gpu{gpu}"
                for gpu in (7, 8)
            ) + tuple(
                ROOT / f"runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/{root_name}_missing_gpu{gpu}"
                for gpu in (7, 8)
            ),
            active_gpus=(7, 8),
            latent_steps=5,
            model_size="2B",
            queue_group="qwen2b-six-modes",
            queue_order=order,
        )
        for label, root_name, order in (
            ("Latent base", "latent_base", 2),
            ("Both", "both", 5),
        )
    ),
    JobSpec(
        "Qwen 2B Latent base · step 20 · I/O pipeline", 7, 735,
        ROOT / "runs/qwen2b_step20_base_vs_pruning_v1_50_20260825/base_gpu7",
        "qwen2b_step20_base_vs_pruning_v1_",
        extra_roots=(
            ROOT / "runs/qwen2b_step20_base_vs_pruning_v1_50_20260825/base_gpu8",
            ROOT / "runs/qwen2b_step20_base_vs_pruning_v1_paired_live_685_20260825/base_50_391_gpu7",
            ROOT / "runs/qwen2b_step20_base_vs_pruning_v1_paired_live_685_20260825/base_392_734_gpu8",
            ROOT / "runs/qwen2b_step20_base_vs_pruning_v1_io_pipeline_r2_20260825/base_gpu7",
        ),
        speed_root=ROOT / "runs/qwen2b_step20_base_vs_pruning_v1_io_pipeline_r2_20260825/base_gpu7",
        speed_peer_root=ROOT / "runs/qwen2b_step20_base_vs_pruning_v1_io_pipeline_r2_20260825/pruning_gpu8",
        active_gpus=(7,), latent_steps=20, model_size="2B",
    ),
    JobSpec(
        "Qwen 2B Pruning V3 · step 20 · I/O pipeline", 7, 735,
        ROOT / "runs/qwen2b_step20_pruning_v3_dynamic_pilot_20260825/pruning_v3_clean_gpu7",
        "qwen2b_step20_pruning_v3_dynamic_pilot_20260825/pruning_v3_",
        extra_roots=(
            ROOT / "runs/qwen2b_step20_pruning_v3_dynamic_pilot_20260825/pruning_v3_clean_gpu8",
            ROOT / "runs/qwen2b_step20_pruning_v3_dynamic_pilot_20260825/pruning_v3_full_gpu7",
            ROOT / "runs/qwen2b_step20_pruning_v3_dynamic_pilot_20260825/pruning_v3_full_gpu8",
        ),
        active_gpus=(7, 8), latent_steps=20, model_size="2B",
    ),
    JobSpec(
        "Qwen 2B Pruning V3 · step 5", 7, 735,
        ROOT / "runs/qwen2b_step5_pruning_v3_full_20260826/gpu7",
        "qwen2b_step5_pruning_v3_full_20260826/gpu7",
        active_gpus=(7,), latent_steps=5, model_size="2B",
    ),
    JobSpec(
        "Qwen 2B Pruning V4 · hierarchy · step 5 · GPU 7/8 split · 735", 7, 735,
        ROOT / "runs/qwen2b_step5_pruning_v4_hierarchy_full_gpu7_20260827",
        "qwen2b_step5_pruning_v4_hierarchy_v4single_",
        extra_roots=(
            ROOT / "runs/qwen2b_step5_pruning_v4_hierarchy_full_gpu8_20260827",
            ROOT / "runs/qwen2b_step5_pruning_v4_hierarchy_v4single_gate50_gpu7_20260827",
            ROOT / "runs/qwen2b_step5_pruning_v4_hierarchy_v4single_gate50_gpu8_20260827",
            ROOT / "runs/qwen2b_step5_pruning_v4_hierarchy_v4single_remaining685_gpu7_20260827",
            ROOT / "runs/qwen2b_step5_pruning_v4_hierarchy_v4single_remaining685_gpu8_20260827",
        ),
        active_gpus=(7, 8), latent_steps=5, model_size="2B",
    ),
    JobSpec(
        "Qwen 4B Pruning V4 · significance adaptive · 8 patches · step 5 · GPU 7/8 split · 735", 7, 735,
        ROOT / "runs/qwen4b_step5_patch8_pruning_v4_significance_full735v2_lowmem_gpu7_20260828",
        "qwen4b_step5_patch8_pruning_v4_significance_full735v2_lowmem_gpu",
        extra_roots=(
            ROOT / "runs/qwen4b_step5_patch8_pruning_v4_significance_full735v2_lowmem_gpu8_20260828",
        ),
        active_gpus=(7, 8), latent_steps=5, model_size="4B",
        queue_group="qwen-v4-significance", queue_order=2,
    ),
    JobSpec(
        "Qwen 8B Pruning V4 · significance adaptive · 8 patches · step 5 · GPU 7/8 split · 735", 7, 735,
        ROOT / "runs/qwen8b_step5_patch8_pruning_v4_significance_full735v2_lowmem_gpu7_20260828",
        "qwen8b_step5_patch8_pruning_v4_significance_full735v2_lowmem_gpu",
        extra_roots=(
            ROOT / "runs/qwen8b_step5_patch8_pruning_v4_significance_full735v2_lowmem_gpu8_20260828",
        ),
        active_gpus=(7, 8), latent_steps=5, model_size="8B",
        queue_group="qwen-v4-significance", queue_order=3,
    ),
    JobSpec(
        "Qwen 2B Latent base · 25 patches · step 5 · I/O pipeline · GPU 8 · 735", 8, 735,
        ROOT / "runs/qwen2b_step5_patch25_latent_base_earlyvision_full735_v1_gpu8_20260828",
        "qwen2b_step5_patch25_latent_base_earlyvision_full735_v1_gpu8_20260828",
        speed_warmup_attempts_per_root=1,
        active_gpus=(8,), latent_steps=5, model_size="2B",
    ),
    JobSpec(
        "Qwen 2B Pruning V4 · early-vision hierarchy · 25 patches · step 5 · I/O pipeline · GPU 7 · 735", 7, 735,
        ROOT / "runs/qwen2b_step5_patch25_pruning_v4_hierarchy_earlyvision_full735_v1_gpu7_20260828",
        "qwen2b_step5_patch25_pruning_v4_hierarchy_earlyvision_full735_v1_gpu7_20260828",
        speed_warmup_attempts_per_root=1,
        active_gpus=(7,), latent_steps=5, model_size="2B",
    ),
    JobSpec(
        "Qwen 2B Pruning V5 · Two-stage · step 5 · GPU 7 · 735", 7, 735,
        ROOT / "runs/qwen2b_step5_pruning_v5_twostage_full735_gpu7_20260827",
        "qwen2b_step5_pruning_v5_twostage_full735_gpu7_20260827",
        active_gpus=(7,), latent_steps=5, model_size="2B",
    ),
    JobSpec(
        "Qwen 2B Pruning V6 · Stable streaming · step 5 · GPU 8 · 735", 8, 735,
        ROOT / "runs/qwen2b_step5_pruning_v6_streaming_full735_gpu8_20260827",
        "qwen2b_step5_pruning_v6_streaming_full735_gpu8_20260827",
        active_gpus=(8,), latent_steps=5, model_size="2B",
    ),
    JobSpec(
        "Qwen 2B Latent base · 8 patches · step 5 · GPU 7/8 split · 735", 7, 735,
        ROOT / "runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/latent_base_v3_gpu6",
        "qwen3vl2b_firstcall_four_modes_local_v6_20260823/latent_base_",
        extra_roots=(
            ROOT / "runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/latent_base_v3_gpu7",
            ROOT / "runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/latent_base_v3_gpu8",
            ROOT / "runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/latent_base_v4_gpu7",
            ROOT / "runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/latent_base_v4_gpu8",
            ROOT / "runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/latent_base_missing_gpu7",
            ROOT / "runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/latent_base_missing_gpu8",
        ),
        speed_root=ROOT / "runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/latent_base_v4_gpu7",
        speed_peer_root=ROOT / "runs/qwen3vl2b_firstcall_four_modes_local_v6_20260823/latent_base_v4_gpu8",
        active_gpus=(7, 8), latent_steps=5, model_size="2B",
    ),
    JobSpec(
        "Qwen 2B Pruning V4 · significance adaptive · 8 patches · step 5 · GPU 7/8 split · 735", 7, 735,
        ROOT / "runs/qwen2b_step5_patch8_pruning_v4_stepwise_significance_mean_full735v1_gpu7_20260828",
        "qwen2b_step5_patch8_pruning_v4_stepwise_significance_mean_full735v1_gpu",
        extra_roots=(
            ROOT / "runs/qwen2b_step5_patch8_pruning_v4_stepwise_significance_mean_full735v1_gpu8_20260828",
        ),
        active_gpus=(7, 8), latent_steps=5, model_size="2B",
        queue_group="qwen-v4-significance", queue_order=1,
    ),
    JobSpec(
        "Qwen 2B Pruning V4.1 · QK probe · 8 patches · step 5 · GPU 7/8 split · 50", 7, 50,
        ROOT / "runs/qwen2b_step5_patch8_pruning_v4_significance_v41_qkprobe_gate50_gpu7_20260828",
        "qwen2b_step5_patch8_pruning_v4_significance_v41_qkprobe_gate50_gpu",
        extra_roots=(
            ROOT / "runs/qwen2b_step5_patch8_pruning_v4_significance_v41_qkprobe_gate50_gpu8_20260828",
        ),
        active_gpus=(7, 8), latent_steps=5, model_size="2B",
        queue_group="qwen-v4-significance", queue_order=2,
    ),
    JobSpec(
        "Qwen 2B Pruning V4.1 · QK probe · 8 patches · step 5 · GPU 7/8 split · 735", 7, 735,
        ROOT / "runs/qwen2b_step5_patch8_pruning_v4_stepwise_v41_qkprobe_full735_gpu7_20260828",
        "qwen2b_step5_patch8_pruning_v4_stepwise_v41_qkprobe_full735_gpu",
        extra_roots=(
            ROOT / "runs/qwen2b_step5_patch8_pruning_v4_stepwise_v41_qkprobe_full735_gpu8_20260828",
        ),
        active_gpus=(7, 8), latent_steps=5, model_size="2B",
        queue_group="qwen-v4-significance", queue_order=3,
    ),
    JobSpec(
        "Qwen 4B Pruning V4.1 · QK probe · 8 patches · step 5 · GPU 7/8 split · 735", 7, 735,
        ROOT / "runs/qwen4b_step5_patch8_pruning_v4_significance_v41_qkprobe_full735_gpu7_20260828",
        "qwen4b_step5_patch8_pruning_v4_significance_v41_qkprobe_full735_gpu",
        extra_roots=(
            ROOT / "runs/qwen4b_step5_patch8_pruning_v4_significance_v41_qkprobe_full735_gpu8_20260828",
        ),
        active_gpus=(7, 8), latent_steps=5, model_size="4B",
        queue_group="qwen-v4-significance", queue_order=4,
    ),
    JobSpec(
        "Qwen 4B Pruning V4.1 · QK probe · TAG answer · 8 patches · step 5 · GPU 7/8 split · 50", 7, 50,
        ROOT / "runs/qwen4b_step5_patch8_pruning_v4_significance_v41_qkprobe_tag_gate50_gpu7_20260828",
        "qwen4b_step5_patch8_pruning_v4_significance_v41_qkprobe_tag_gate50_gpu",
        extra_roots=(
            ROOT / "runs/qwen4b_step5_patch8_pruning_v4_significance_v41_qkprobe_tag_gate50_gpu8_20260828",
        ),
        active_gpus=(7, 8), latent_steps=5, model_size="4B",
        queue_group="qwen-v4-significance", queue_order=5,
    ),
    JobSpec(
        "Qwen 2B Reallocation V2 · step 5 · I/O pipeline · GPU 7/8 split · 735", 5, 735,
        ROOT / "runs/qwen2b_step5_reallocation_v2_full_gpu5678_r2_20260826/gpu5",
        "qwen2b_step5_reallocation_v2",
        extra_roots=(
            ROOT / "runs/qwen2b_step5_reallocation_v2_full_gpu5678_r2_20260826/gpu6",
            ROOT / "runs/qwen2b_step5_reallocation_v2_full_gpu5678_r2_20260826/gpu7",
            ROOT / "runs/qwen2b_step5_reallocation_v2_full_gpu5678_r2_20260826/gpu8",
            ROOT / "runs/qwen2b_step5_reallocation_v2_remaining_gpu78_20260826/gpu7",
            ROOT / "runs/qwen2b_step5_reallocation_v2_remaining_gpu78_20260826/gpu8",
        ),
        speed_root=ROOT / "runs/qwen2b_step5_reallocation_v2_full_gpu5678_r2_20260826/gpu5",
        active_gpus=(7, 8), latent_steps=5, model_size="2B",
    ),
    JobSpec(
        "Qwen 2B Reallocation · pathology-aware · step 5",
        0,
        735,
        ROOT / "runs/qwen2b_step5_pathology_reallocation_only_735_20260831/gpu0",
        "qwen2b_step5_pathology_reallocation_only_735_20260831",
        extra_roots=tuple(
            ROOT / f"runs/qwen2b_step5_pathology_reallocation_only_735_20260831/gpu{gpu}"
            for gpu in (1, 2, 3, 4, 7, 8)
        ),
        active_gpus=(0, 1, 2, 3, 4, 7, 8),
        latent_steps=5,
        model_size="2B",
        speed_warmup_attempts_per_root=1,
        queue_group="qwen-reallocation-pathology",
    ),
    JobSpec(
        "Qwen 2B Reallocation · pathology-aware V2 · step 5 · gate 50",
        0,
        50,
        ROOT / "runs/qwen2b_step5_pathology_reallocation_v2_reasoner_gate50_20260831/gpu0",
        "qwen2b_step5_pathology_reallocation_v2_reasoner_gate50_20260831",
        extra_roots=tuple(
            ROOT / f"runs/qwen2b_step5_pathology_reallocation_v2_reasoner_gate50_20260831/gpu{gpu}"
            for gpu in (1, 2, 3, 4, 7, 8)
        ),
        active_gpus=(0, 1, 2, 3, 4, 7, 8),
        latent_steps=5,
        model_size="2B",
        speed_warmup_attempts_per_root=1,
        queue_group="qwen-reallocation-pathology",
    ),
    JobSpec(
        "Qwen 2B Reallocation · pathology-aware V2 · step 5 · 735",
        0,
        735,
        ROOT / "runs/qwen2b_step5_pathology_reallocation_v2_reasoner_full735_20260831/gpu0",
        "qwen2b_step5_pathology_reallocation_v2_reasoner_full735_20260831",
        extra_roots=tuple(
            ROOT / f"runs/qwen2b_step5_pathology_reallocation_v2_reasoner_full735_20260831/gpu{gpu}"
            for gpu in (1, 2, 3, 4, 7, 8)
        ),
        active_gpus=(0, 1, 2, 3, 4, 7, 8),
        latent_steps=5,
        model_size="2B",
        speed_warmup_attempts_per_root=1,
        queue_group="qwen-reallocation-pathology",
    ),
    JobSpec(
        "Qwen 2B Reallocation · pathology-aware V3 · relevance × hierarchy · gate 50",
        0,
        50,
        ROOT / "runs/qwen2b_step5_pathology_reallocation_v3_relevance_hierarchy_gate50_20260831/gpu0",
        "qwen2b_step5_pathology_reallocation_v3_relevance_hierarchy_gate50_20260831",
        extra_roots=tuple(
            ROOT / f"runs/qwen2b_step5_pathology_reallocation_v3_relevance_hierarchy_gate50_20260831/gpu{gpu}"
            for gpu in (1, 2, 3, 4, 7, 8)
        ),
        active_gpus=(0, 1, 2, 3, 4, 7, 8),
        latent_steps=5,
        model_size="2B",
        speed_warmup_attempts_per_root=1,
        queue_group="qwen-reallocation-pathology",
    ),
    JobSpec(
        "Qwen 2B Reallocation · pathology context · step 5 · gate 50",
        0,
        50,
        ROOT / "runs/qwen2b_step5_pathology_context_reallocation_gate50_20260831/gpu0",
        "qwen2b_step5_pathology_context_reallocation_gate50_20260831",
        extra_roots=tuple(
            ROOT / f"runs/qwen2b_step5_pathology_context_reallocation_gate50_20260831/gpu{gpu}"
            for gpu in (1, 2, 3, 4, 7, 8)
        ),
        active_gpus=(0, 1, 2, 3, 4, 7, 8),
        latent_steps=5,
        model_size="2B",
        speed_warmup_attempts_per_root=1,
        queue_group="qwen-reallocation-pathology",
    ),
    JobSpec(
        "Qwen 2B Reallocation · pathology context · step 5 · 735",
        0,
        735,
        ROOT / "runs/qwen2b_step5_pathology_context_reallocation_full735_20260831/gpu0",
        "qwen2b_step5_pathology_context_reallocation_full735_20260831",
        extra_roots=tuple(
            ROOT / f"runs/qwen2b_step5_pathology_context_reallocation_full735_20260831/gpu{gpu}"
            for gpu in (1, 2, 3, 4, 7, 8)
        ),
        active_gpus=(0, 1, 2, 3, 4, 7, 8),
        latent_steps=5,
        model_size="2B",
        speed_warmup_attempts_per_root=1,
        queue_group="qwen-reallocation-pathology-context-matrix",
        queue_order=1,
    ),
    *tuple(
        JobSpec(
            f"Qwen {model_size} Reallocation · pathology context · step {latent_steps} · 735",
            0,
            735,
            ROOT / (
                f"runs/qwen{model_size.lower()}_pathology_context_reallocation_"
                f"step{latent_steps}_full735_20260831/gpu0"
            ),
            (
                f"qwen{model_size.lower()}_pathology_context_reallocation_"
                f"step{latent_steps}_full735_20260831"
            ),
            extra_roots=tuple(
                ROOT / (
                    f"runs/qwen{model_size.lower()}_pathology_context_reallocation_"
                    f"step{latent_steps}_full735_20260831/gpu{gpu}"
                )
                for gpu in (1, 2, 3, 4, 7, 8)
            ),
            active_gpus=(0, 1, 2, 3, 4, 7, 8),
            latent_steps=latent_steps,
            model_size=model_size,
            speed_warmup_attempts_per_root=1,
            queue_group="qwen-reallocation-pathology-context-matrix",
            queue_order=queue_order,
        )
        for queue_order, (model_size, latent_steps) in enumerate((
                ("2B", 10), ("2B", 20), ("2B", 30),
                ("4B", 5), ("4B", 10), ("4B", 20), ("4B", 30),
                ("8B", 5), ("8B", 10), ("8B", 20), ("8B", 30),
            ), start=2)
    ),
    JobSpec(
        "Qwen 4B Pruning V3 · step 5", 7, 735,
        ROOT / "runs/qwen4b_step5_pruning_v3_full_2gpu_io_20260826/gpu7",
        "qwen4b_step5_pruning_v3_full_2gpu_io_20260826/gpu",
        extra_roots=(ROOT / "runs/qwen4b_step5_pruning_v3_full_2gpu_io_20260826/gpu8",),
        active_gpus=(7, 8), latent_steps=5, model_size="4B",
    ),
    JobSpec(
        "Qwen 4B Reallocation V2 · step 5 · I/O pipeline · GPU 7/8 split · 735", 7, 735,
        ROOT / "runs/qwen4b_step5_reallocation_v2_rerun_gpu78_20260827/gpu7",
        "qwen4b_step5_reallocation_v2",
        extra_roots=(
            ROOT / "runs/qwen4b_step5_reallocation_v2_rerun_gpu78_20260827/gpu8",
            ROOT / "runs/qwen4b_step5_reallocation_v2_remaining1024_gpu78_20260827/gpu7",
            ROOT / "runs/qwen4b_step5_reallocation_v2_remaining1024_gpu78_20260827/gpu8",
            ROOT / "runs/qwen4b_step5_reallocation_v2_actual1024_gpu78_20260827/gpu7",
            ROOT / "runs/qwen4b_step5_reallocation_v2_actual1024_gpu78_20260827/gpu8",
            ROOT / "runs/qwen4b_step5_reallocation_v2_remaining2048_gpu78_20260827/gpu7",
            ROOT / "runs/qwen4b_step5_reallocation_v2_remaining2048_gpu78_20260827/gpu8",
            ROOT / "runs/qwen4b_step5_reallocation_v2_continue2048_gpu78_20260827/gpu7",
            ROOT / "runs/qwen4b_step5_reallocation_v2_continue2048_gpu78_20260827/gpu8",
            ROOT / "runs/qwen4b_step5_reallocation_v2_recovery_gpu78_20260827/gpu7",
            ROOT / "runs/qwen4b_step5_reallocation_v2_recovery_gpu78_20260827/gpu8",
        ),
        speed_root=ROOT / "runs/qwen4b_step5_reallocation_v2_continue2048_gpu78_20260827/gpu8",
        active_gpus=(7, 8), latent_steps=5, model_size="4B",
    ),
    JobSpec(
        "Qwen 8B Reallocation V2 · step 5 · I/O pipeline · GPU 7/8 split · 735", 7, 735,
        ROOT / "runs/qwen8b_step5_reallocation_v2_full_gpu78_r2_20260827/gpu7",
        "qwen8b_step5_reallocation_v2_full_gpu78_r2_20260827/gpu",
        extra_roots=(ROOT / "runs/qwen8b_step5_reallocation_v2_full_gpu78_r2_20260827/gpu8",),
        speed_root=ROOT / "runs/qwen8b_step5_reallocation_v2_full_gpu78_r2_20260827/gpu8",
        active_gpus=(7, 8), latent_steps=5, model_size="8B",
    ),
    JobSpec(
        "Qwen 8B Single No Thinking", 7, 735,
        CONTROLLED_SINGLE_NOTHINKING_ROOT / "wsi/8b/gpu5",
        "qwen_single_nothinking_controlled_20260902/wsi/8b/gpu",
        single_predictions=True,
        extra_roots=tuple(
            CONTROLLED_SINGLE_NOTHINKING_ROOT / f"wsi/8b/gpu{gpu}"
            for gpu in (5, 6, 7, 8)
        ),
        active_gpus=(5, 6, 7, 8), model_size="8B",
        queue_group="qwen-8b-local-matrix", queue_order=1,
    ),
    JobSpec(
        "Qwen 8B Latent base", 7, 735,
        ROOT / "runs/qwen8b_firstcall_structured_20260824/latent_base_gpu7",
        "qwen8b_firstcall_structured_20260824/latent_base_gpu",
        extra_roots=(
            ROOT / "runs/qwen8b_firstcall_structured_20260824/latent_base_gpu8",
            ROOT / "runs/qwen8b_firstcall_structured_20260824/latent_base_gpu5_borrowed_20260825",
        ),
        active_gpus=(5, 7, 8), latent_steps=5, model_size="8B",
        queue_group="qwen8b-three-modes", queue_order=2,
    ),
    JobSpec(
        "OctoMed 7B Single", 7, 735,
        ROOT / "runs/octomed7b_thinking_single_three_modes_20260825/single_gpu7",
        "octomed7b_thinking_single_three_modes_20260825/single_gpu",
        single_predictions=True,
        extra_roots=(
            ROOT / "runs/octomed7b_thinking_single_three_modes_20260825/single_gpu8",
        ),
        active_gpus=(7, 8), model_size="7B",
        queue_group="octomed-three-modes", queue_order=1,
    ),
    JobSpec(
        "OctoMed 7B Latent base", 7, 735,
        ROOT / "runs/octomed7b_thinking_single_three_modes_20260825/base_gpu7",
        "octomed7b_thinking_single_three_modes_20260825/base_gpu",
        extra_roots=(
            ROOT / "runs/octomed7b_thinking_single_three_modes_20260825/base_gpu8",
        ),
        active_gpus=(7, 8), latent_steps=5, model_size="7B",
        queue_group="octomed-three-modes", queue_order=2,
    ),
    JobSpec(
        "OctoMed 7B Pruning · Navigator KV off · 96 tok", 7, 735,
        ROOT / "runs/octomed7b_thinking_single_three_modes_20260825/pruning_gpu7",
        "octomed7b_thinking_single_three_modes_20260825/pruning_gpu",
        extra_roots=(
            ROOT / "runs/octomed7b_thinking_single_three_modes_20260825/pruning_gpu8",
        ),
        active_gpus=(7, 8), latent_steps=5, model_size="7B",
        queue_group="octomed-three-modes", queue_order=3,
    ),
    JobSpec(
        "Qwen 2B Reallocation · pathology relay",
        6,
        735,
        ROOT / "runs/qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v6_gpu6",
        "qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v6_gpu",
        extra_roots=tuple(
            ROOT / f"runs/qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v6_gpu{gpu}"
            for gpu in (7, 8)
        ),
        active_gpus=(6, 7, 8),
        latent_steps=5,
        model_size="2B",
        queue_group="qwen2b-six-modes",
        queue_order=4,
    ),
    JobSpec(
        "Qwen 2B Reallocation · relay v7",
        6,
        735,
        ROOT / "runs/qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v7_gpu6",
        "qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v7_gpu",
        extra_roots=tuple(
            ROOT / f"runs/qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v7_gpu{gpu}"
            for gpu in (7, 8)
        ),
        active_gpus=(6, 7, 8),
        latent_steps=5,
        model_size="2B",
        queue_group="qwen2b-six-modes",
        queue_order=3,
    ),
    JobSpec(
        "Qwen 2B Reallocation · relay v8",
        6,
        735,
        ROOT / "runs/qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v8_gpu6",
        "qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v8_gpu",
        extra_roots=tuple(
            ROOT / f"runs/qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v8_gpu{gpu}"
            for gpu in (7, 8)
        ),
        active_gpus=(6, 7, 8),
        latent_steps=5,
        model_size="2B",
        queue_group="qwen2b-six-modes",
        queue_order=2,
    ),
    JobSpec(
        "Qwen 2B Reallocation · relay v9",
        6,
        735,
        ROOT / "runs/qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v9_gpu6",
        "qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v9_gpu",
        extra_roots=tuple(
            ROOT / f"runs/qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v9_gpu{gpu}"
            for gpu in (7, 8)
        ),
        active_gpus=(6, 7, 8),
        latent_steps=5,
        model_size="2B",
        queue_group="qwen2b-six-modes",
        queue_order=1,
    ),
    JobSpec(
        "Qwen 2B Reallocation · relay v10 sweep",
        6,
        192,
        ROOT / "runs/qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v10_gpu6",
        "qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v10_gpu",
        extra_roots=tuple(
            ROOT / f"runs/qwen3vl2b_thinking_6modes_20260822_r4/reallocation_pathology_relay_v10_gpu{gpu}"
            for gpu in (7, 8)
        ),
        active_gpus=(6, 7, 8),
        latent_steps=5,
        model_size="2B",
        queue_group="qwen2b-six-modes",
    ),
    JobSpec(
        "Qwen 8B VLMAS",
        8,
        735,
        ROOT / "runs/qwen3vl_8b_matrix_20260823/vlmas_gpu8",
        "qwen3vl_8b_matrix_20260823/vlmas_gpu",
        active_gpus=(8,),
        model_size="8B",
    ),
    JobSpec(
        "InternVL Reallocation · current",
        6,
        735,
        ROOT / "runs/internvl3_reallocation_3gpu_20260822_gpu6",
        "internvl3_reallocation_3gpu_20260822",
        extra_roots=(
            ROOT / "runs/internvl3_reallocation_3gpu_20260822_gpu7",
            ROOT / "runs/internvl3_reallocation_3gpu_20260822_gpu8",
        ),
        active_gpus=(6, 7, 8),
        latent_steps=5,
    ),
    JobSpec(
        "Lingshu Pruning · Navigator KV off",
        7,
        735,
        ROOT / "runs/lingshu_pruning_navigator_kv_off_nav96_gpu7_20260821",
        "lingshu_pruning_nav96_gpu",
        extra_roots=(
            ROOT / "runs/lingshu_pruning_nav96_gpu6_shard_20260821",
            ROOT / "runs/lingshu_pruning_nav96_gpu7_shard_20260821",
            ROOT / "runs/lingshu_pruning_nav96_gpu8_shard_20260821",
        ),
        active_gpus=(6, 7, 8),
    ),
    JobSpec(
        "Lingshu Reallocation · 96 tok",
        6,
        735,
        ROOT / "runs/lingshu_reallocation_nav96_gpu8_20260821",
        "lingshu_reallocation_nav96",
        extra_roots=tuple(
            ROOT / f"runs/lingshu_reallocation_nav96_attempt{attempt}_gpu{gpu}_20260821"
            for attempt in range(1, 4)
            for gpu in (6, 7, 8)
        ),
        active_gpus=(6, 7, 8),
        evaluation_summary=ROOT / "runs/evaluation_lingshu_reallocation_nav96_20260821/general/lingshu_reallocation_nav96_summary.json",
    ),
    JobSpec(
        "Lingshu Both · 96 tok",
        6,
        735,
        ROOT / "runs/lingshu_both_nav96_attempt1_gpu6_20260821",
        "lingshu_both_nav96",
        extra_roots=tuple(
            ROOT / f"runs/lingshu_both_nav96_attempt{attempt}_gpu{gpu}_20260821"
            for attempt in range(1, 4)
            for gpu in (6, 7, 8)
            if not (attempt == 1 and gpu == 6)
        ),
        active_gpus=(6, 7, 8),
        evaluation_summary=ROOT / "runs/evaluation_lingshu_both_nav96_20260821/general/lingshu_both_nav96_summary.json",
    ),
    JobSpec(
        "Lingshu Latent base · step 5 · tissue-safe Navigator",
        7,
        735,
        ROOT / "runs/lingshu_step5_nav512_slots_r3_20260827/base_gpu7",
        "lingshu_step5_nav512_slots_r3_20260827/base_gpu",
        extra_roots=(ROOT / "runs/lingshu_step5_nav512_slots_r3_20260827/base_gpu8",),
        active_gpus=(7, 8),
        latent_steps=5,
        speed_warmup_attempts_per_root=1,
    ),
    JobSpec(
        "InternVL Latent base · paired control · GPU 7/8 · 30",
        7,
        30,
        ROOT / "runs/internvl3_step5_base_noio_repro_30_gpu7_20260828",
        "internvl3_step5_base_noio_repro_30_gpu7_20260828",
        extra_roots=(
            ROOT / "runs/internvl3_step5_base_noio_repro_30_gpu8_20260828",
        ),
        active_gpus=(7, 8),
        latent_steps=5,
        speed_warmup_attempts_per_root=1,
    ),
    JobSpec(
        "InternVL Pruning V3 · paired Base · GPU 7/8 · 735",
        7,
        735,
        ROOT / "runs/internvl3_step5_pruning_v3_paired_30_gpu7_20260828",
        "internvl3_step5_pruning_v3_paired_30_gpu7_20260828",
        extra_roots=(
            ROOT / "runs/internvl3_step5_pruning_v3_paired_30_gpu8_20260828",
        ),
        active_gpus=(7, 8),
        latent_steps=5,
        speed_warmup_attempts_per_root=1,
    ),
    JobSpec(
        "InternVL3.5 Single · InternVL3 prompt · slide-wise · 735",
        0,
        735,
        ROOT / "runs/internvl35_single_internvl3prompt_greedy_slidewise_full735_gpu0_20260829",
        "internvl35_single_internvl3prompt_greedy_slidewise_full735_gpu",
        single_predictions=True,
        extra_roots=tuple(
            ROOT / f"runs/internvl35_single_internvl3prompt_greedy_slidewise_full735_gpu{gpu}_20260829"
            for gpu in (1, 2, 3, 4, 7, 8)
        ),
        active_gpus=(0, 1, 2, 3, 4, 7, 8),
        model_size="8B",
    ),
    JobSpec(
        "InternVL3.5 VL-MAS · InternVL3 prompt · slide-wise · 735",
        0,
        735,
        ROOT / "runs/internvl35_vlmas_internvl3prompt_slidewise_full735_gpu0_20260829",
        "internvl35_vlmas_internvl3prompt_slidewise_full735_gpu",
        extra_roots=tuple(
            ROOT / f"runs/internvl35_vlmas_internvl3prompt_slidewise_full735_gpu{gpu}_20260829"
            for gpu in (1, 2, 3, 4, 7, 8)
        ),
        active_gpus=(0, 1, 2, 3, 4, 7, 8),
        model_size="8B",
    ),
    JobSpec(
        "Qwen 2B Latent base · Open only · step 5 · GPU 0/1 · 345",
        0,
        345,
        ROOT / "runs/qwen2b_latent_base_step5_open_only_no_canonical_345_20260829_gpu0",
        "qwen2b_latent_base_step5_open_only_no_canonical_345_20260829_gpu",
        extra_roots=(
            ROOT / "runs/qwen2b_latent_base_step5_open_only_no_canonical_345_20260829_gpu1",
        ),
        speed_extra_roots=(
            ROOT / "runs/qwen2b_latent_base_step5_no_canonical_open_full735_20260829_gpu0",
            ROOT / "runs/qwen2b_latent_base_step5_no_canonical_open_full735_20260829_gpu1",
        ),
        active_gpus=(0, 1),
        speed_warmup_attempts_per_root=1,
        latent_steps=5,
        model_size="2B",
    ),
    JobSpec(
        "Qwen 8B VL-MAS · no candidate · no Reasoner timeout + I/O · GPU 7/8 · 735",
        7,
        735,
        ROOT / "runs/qwen8b_vlmas_no_candidate_reasoner_no_timeout_full735_20260829_gpu7",
        "qwen8b_vlmas_no_candidate_reasoner_no_timeout_full735_20260829_gpu",
        extra_roots=(
            ROOT / "runs/qwen8b_vlmas_no_candidate_reasoner_no_timeout_full735_20260829_gpu8",
        ),
        active_gpus=(7, 8),
        speed_warmup_attempts_per_root=1,
        model_size="8B",
    ),
    JobSpec(
        "Lingshu Reallocation V2 · step 5 · tissue-safe Navigator",
        8,
        735,
        ROOT / "runs/lingshu_step5_nav512_slots_r3_20260827/reallocation_v2_gpu8",
        "lingshu_step5_nav512_slots_r3_20260827/reallocation_v2_gpu",
        active_gpus=(8,),
        latent_steps=5,
        speed_warmup_attempts_per_root=1,
    ),
    JobSpec(
        "Qwen VLMAS",
        6,
        735,
        ROOT / "runs/qwen_vlmas_gpu6_735_20260821",
        "qwen_vlmas_gpu6_735_20260821",
        extra_roots=tuple(
            ROOT / f"runs/qwen_vlmas_gpu{gpu}_shard_20260821" for gpu in (6, 7, 8)
        ),
        active_gpus=(6, 7, 8),
        evaluation_summary=ROOT / "runs/evaluation_qwen_vlmas_20260822/general/qwen_vlmas_20260822_summary.json",
    ),
    *tuple(
        JobSpec(
            f"Qwen {model_size} Latent base · no candidate · step {latent_steps} · GPU {gpu} · 735",
            gpu,
            735,
            ROOT / f"runs/qwen{model_size.lower()}_latent_base_step{latent_steps}_no_candidate_full735_20260829_gpu{gpu}",
            f"qwen{model_size.lower()}_latent_base_step{latent_steps}_no_candidate_full735_20260829_gpu{gpu}",
            active_gpus=(gpu,),
            latent_steps=latent_steps,
            model_size=model_size,
            speed_warmup_attempts_per_root=1,
        )
        for model_size, latent_steps, gpu in (
            ("2B", 5, 0), ("2B", 10, 1), ("2B", 20, 2), ("2B", 30, 3),
            ("4B", 5, 4), ("4B", 10, 4), ("4B", 20, 4), ("4B", 30, 4),
            ("8B", 5, 0), ("8B", 10, 1), ("8B", 20, 2), ("8B", 30, 3),
        )
    ),
    *tuple(
        JobSpec(
            f"Qwen {model_size} Single · Reasoner-thinking + Answerer JSON · 735",
            gpus[0],
            735,
            MATCHED_SINGLE_ROOT / f"{model_size.lower()}/single_gpu{gpus[0]}",
            f"qwen_single_matched_to_vlmas_sampling_20260830/{model_size.lower()}/single_gpu",
            single_predictions=True,
            extra_roots=tuple(
                MATCHED_SINGLE_ROOT / f"{model_size.lower()}/single_gpu{gpu}"
                for gpu in gpus[1:]
            ),
            active_gpus=gpus,
            model_size=model_size,
            queue_group=f"qwen-{model_size.lower()}-local-matrix",
            queue_order=0,
            thinking_mode="Thinking",
        )
        for model_size, gpus in (
            ("2B", (0, 1)),
            ("8B", (4, 7, 8)),
        )
    ),
    *tuple(
        JobSpec(
            f"Qwen {model_size} VLMAS",
            gpus[0],
            735,
            CANONICAL_MATRIX_ROOT / f"{model_size.lower()}/vlmas_gpu{gpus[0]}",
            f"qwen_full_matrix_local_canonical_finalmatched_20260829/{model_size.lower()}/vlmas_gpu",
            extra_roots=tuple(
                CANONICAL_MATRIX_ROOT / f"{model_size.lower()}/vlmas_gpu{gpu}"
                for gpu in gpus[1:]
            ),
            active_gpus=gpus,
            model_size=model_size,
            speed_warmup_attempts_per_root=1,
            queue_group=f"qwen-{model_size.lower()}-local-matrix",
            queue_order=1,
        )
        for model_size, gpus in (
            ("2B", (0, 1)),
            ("4B", (2, 3)),
            ("8B", (4, 7, 8)),
        )
    ),
    *tuple(
        JobSpec(
            f"Qwen {model_size} Latent base · step {latent_steps}",
            gpus[0],
            735,
            (
                CANONICAL_MATRIX_ROOT / f"{model_size.lower()}/latent_base_step{latent_steps}_gpu{gpus[0]}"
                if model_size != "8B"
                else ROOT / f"runs/qwen8b_navkv_canonical_allsteps_20260830/latent_base_navkv_step{latent_steps}_gpu{ {5: 3, 10: 4, 20: 7, 30: 8}[latent_steps] }"
            ),
            (
                f"qwen_full_matrix_local_canonical_finalmatched_20260829/{model_size.lower()}/latent_base_step{latent_steps}_gpu"
                if model_size != "8B"
                else f"qwen8b_navkv_canonical_allsteps_20260830/latent_base_navkv_step{latent_steps}_gpu"
            ),
            extra_roots=(
                *tuple(
                    (
                        CANONICAL_MATRIX_ROOT / f"{model_size.lower()}/latent_base_step{latent_steps}_gpu{gpu}"
                        if model_size != "8B"
                        else ROOT / f"runs/qwen8b_navkv_canonical_allsteps_20260830/latent_base_navkv_step{latent_steps}_gpu{ {5: 3, 10: 4, 20: 7, 30: 8}[latent_steps] }_resume_gpu{gpu}"
                    )
                    for gpu in (gpus[1:] if model_size != "8B" else ((0,) if latent_steps == 5 else (1,) if latent_steps == 10 else (2,) if latent_steps == 20 else ()))
                ),
                *((CANONICAL_MATRIX_ROOT / "4b/latent_base_step30_gpu1",)
                  if model_size == "4B" and latent_steps == 30 else ()),
                *(
                    (
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step20_resume_gpu0",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step20_resume_gpu1",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step20_resume_gpu4",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step20_resume2_gpu0",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step20_resume2_gpu1",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step20_resume2_gpu4",
                    )
                    if model_size == "4B" and latent_steps == 20
                    else ()
                ),
                *(
                    (
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step30_resume_gpu2",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step30_resume_gpu3",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step30_resume_gpu7",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step30_resume_gpu8",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step30_resume2_gpu2",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step30_resume2_gpu3",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step30_resume2_gpu7",
                        CANONICAL_MATRIX_ROOT / "4b/latent_base_step30_resume2_gpu8",
                    )
                    if model_size == "4B" and latent_steps == 30
                    else ()
                ),
            ),
            active_gpus=(
                (0, 3) if latent_steps == 5 else (1, 4) if latent_steps == 10 else (2, 7) if latent_steps == 20 else (0, 1, 2, 3, 4, 7, 8)
            ) if model_size == "8B" else (
                (0, 1, 4) if latent_steps == 20 else (2, 3, 7, 8)
            ) if model_size == "4B" and latent_steps in (20, 30) else gpus,
            latent_steps=latent_steps,
            model_size=model_size,
            speed_warmup_attempts_per_root=1,
            queue_group=f"qwen-{model_size.lower()}-local-matrix",
            queue_order=queue_order,
        )
        for model_size, gpus in (
            ("2B", (0, 1)),
            ("4B", (2, 3)),
            ("8B", (4, 7, 8)),
        )
        for queue_order, latent_steps in enumerate((5, 10, 20, 30), start=2)
    ),
    JobSpec(
        "Qwen 2B Pruning V3 · step 30 · pilot 200",
        0,
        200,
        ROOT / "runs/qwen2b_pruning_v3_oneshot_allagents_step30_200_gpu01_20260831/gpu0",
        "qwen2b_pruning_v3_oneshot_allagents_step30_200_gpu01_20260831/gpu",
        extra_roots=tuple(
            ROOT / f"runs/qwen2b_pruning_v3_oneshot_allagents_step30_200_gpu01_20260831/gpu{gpu}"
            for gpu in (1,)
        ),
        active_gpus=(0, 1),
        latent_steps=30,
        model_size="2B",
        speed_warmup_attempts_per_root=1,
        queue_group="qwen-pruning-v3-priority",
        queue_order=0,
    ),
    *tuple(
        JobSpec(
            f"Qwen {model_size} Pruning V3 · step {latent_steps}",
            0,
            735,
            ROOT / f"runs/qwen_pruning_v3_rolepreserve_prefill_2block_matrix_20260831/{model_size.lower()}/step{latent_steps}/gpu0",
            f"qwen_pruning_v3_rolepreserve_prefill_2block_matrix_20260831/{model_size.lower()}/step{latent_steps}/gpu",
            extra_roots=tuple(
                ROOT / f"runs/qwen_pruning_v3_rolepreserve_prefill_2block_matrix_20260831/{model_size.lower()}/step{latent_steps}/gpu{gpu}"
                for gpu in (1, 2, 3, 4, 7, 8)
            ),
            active_gpus=(0, 1, 2, 3, 4, 7, 8),
            latent_steps=latent_steps,
            model_size=model_size,
            speed_warmup_attempts_per_root=1,
            queue_group="qwen-pruning-v3-navkv-matrix",
            queue_order=("2B", "4B", "8B").index(model_size) * 4
            + (5, 10, 20, 30).index(latent_steps)
            + 1,
        )
        for model_size in ("2B", "4B", "8B")
        for latent_steps in (5, 10, 20, 30)
    ),
    JobSpec(
        "SlideBench-BCNB · Qwen 2B Single · 1,058 slides / 7,274 QA",
        0,
        7274,
        ROOT / "runs/slidebench_bcnb_qwen2b_single_grouped_full_20260831",
        "slidebench_bcnb_qwen2b_single_grouped_full_20260831",
        single_predictions=True,
        active_gpus=(0,),
        model_size="2B",
        queue_group="slidebench-bcnb-qwen2b",
        queue_order=1,
    ),
    JobSpec(
        "SlideBench-BCNB · Qwen 4B Single · 1,058 slides / 7,274 QA",
        0,
        7274,
        ROOT / "runs/slidebench_bcnb_qwen4b_single_slidewise_20260831/gpu0",
        "slidebench_bcnb_qwen4b_single_slidewise_20260831/gpu",
        single_predictions=True,
        extra_roots=tuple(
            ROOT / f"runs/slidebench_bcnb_qwen4b_single_slidewise_20260831/gpu{gpu}"
            for gpu in (1, 2, 3, 4, 7, 8)
        ),
        active_gpus=(0, 1, 2, 3, 4, 7, 8),
        model_size="4B",
        queue_group="slidebench-bcnb-qwen4b",
        queue_order=1,
        speed_warmup_attempts_per_root=1,
    ),
    *tuple(
        JobSpec(
            f"MultiPathQA · Qwen {model_size} {mode_label}",
            gpu,
            736,
            ROOT / f"runs/multipathqa_no_panda_qwen_gpu678_v4_20260901/{model_size.lower()}/{root_name}",
            f"multipathqa_no_panda_qwen_gpu678_v4_20260901/{model_size.lower()}/{root_name}",
            single_predictions=mode_label == "Single",
            active_gpus=(gpu,),
            latent_steps=5 if mode_label == "Latent base" else None,
            model_size=model_size,
            queue_group="multipathqa-qwen",
            queue_order=queue_order,
        )
        for model_size, gpu in (("2B", 6), ("4B", 7), ("8B", 8))
        for queue_order, (mode_label, root_name) in enumerate(
            (("Single", "single"), ("VLMAS", "vlmas"), ("Latent base", "latent_base_step5")),
            start=1,
        )
    ),
    *tuple(
        JobSpec(
            f"MultiPathQA/GTEx · Qwen {model_size} Latent base · step {step}",
            5,
            190,
            ROOT / f"runs/multipathqa_gtex_qwen_base_allsteps_gpu5678_20260901/{model_size.lower()}/step{step}/gpu5",
            f"multipathqa_gtex_qwen_base_allsteps_gpu5678_20260901/{model_size.lower()}/step{step}",
            extra_roots=(
                ROOT / f"runs/multipathqa_gtex_qwen_base_allsteps_gpu5678_20260901/{model_size.lower()}/step{step}/gpu6",
                ROOT / f"runs/multipathqa_gtex_qwen_base_allsteps_gpu5678_20260901/{model_size.lower()}/step{step}/gpu7",
                ROOT / f"runs/multipathqa_gtex_qwen_base_allsteps_gpu5678_20260901/{model_size.lower()}/step{step}/gpu8",
                *(
                    (ROOT / f"runs/multipathqa_gtex_qwen_gpu678_v2_20260901/{model_size.lower()}/latent_base_step5",)
                    if step == 5
                    else ()
                ),
            ),
            active_gpus=(5, 6, 7, 8),
            latent_steps=step,
            model_size=model_size,
            queue_group="multipathqa-gtex-base-allsteps",
            queue_order=queue_order,
        )
        for queue_order, (model_size, step) in enumerate(
            (
                (model_size, step)
                for model_size in ("2B", "4B", "8B")
                for step in (5, 10, 20, 30)
            ),
            start=1,
        )
    ),
    *tuple(
        JobSpec(
            f"MultiPathQA/{label} · Qwen {model_size} Latent base · step {step}",
            5,
            target,
            ROOT / f"runs/multipathqa_remaining_qwen_base_allsteps_gpu5678_20260902/{benchmark}/{model_size.lower()}/step{step}/gpu5",
            f"multipathqa_remaining_qwen_base_allsteps_gpu5678_20260902/{benchmark}/{model_size.lower()}/step{step}",
            extra_roots=tuple(
                ROOT / f"runs/multipathqa_remaining_qwen_base_allsteps_gpu5678_20260902/{benchmark}/{model_size.lower()}/step{step}/gpu{gpu}"
                for gpu in (6, 7, 8)
            ),
            active_gpus=(5, 6, 7, 8),
            latent_steps=step,
            model_size=model_size,
            queue_group="multipathqa-remaining-base-allsteps",
            queue_order=benchmark_index * 18 + model_index * 6 + step_index + 3,
        )
        for benchmark_index, (benchmark, (label, target)) in enumerate(
            MULTIPATH_RUN_BENCHMARKS.items()
        )
        for model_index, model_size in enumerate(("2B", "4B", "8B"))
        for step_index, step in enumerate((5, 10, 20, 30))
    ),
)

GTEX_RESULT_SPECS: Final = tuple(
    JobSpec(
        f"MultiPathQA/GTEx · Qwen {model_size} {mode}",
        gpu,
        190,
        ROOT / f"runs/multipathqa_gtex_qwen_gpu678_v2_20260901/{model_size.lower()}/{root_name}",
        f"multipathqa_gtex_qwen_gpu678_v2_20260901/{model_size.lower()}/{root_name}",
        single_predictions=mode == "Single",
        active_gpus=(gpu,),
        model_size=model_size,
        queue_group="multipathqa-gtex-results",
    )
    for model_size, gpu in (("2B", 6), ("4B", 7), ("8B", 8))
    for mode, root_name in (("Single", "single"), ("VLMAS", "vlmas"))
)

GTEX_BASE_RESULT_SPECS: Final = tuple(
    spec
    for spec in JOBS
    if spec.queue_group == "multipathqa-gtex-base-allsteps"
)

MULTIPATH_REMAINING_RESULT_SPECS: Final = tuple(
    spec
    for spec in JOBS
    if spec.queue_group == "multipathqa-remaining-base-allsteps"
)

MULTIPATH_REMAINING_BASE_RESULT_SPECS: Final = tuple(
    spec for spec in MULTIPATH_REMAINING_RESULT_SPECS if spec.latent_steps is not None
)

QUALITY_METRICS: Final = (
    QualityMetric("Qwen", "Single", 735, 49.12, 44.36, 54.49, 48.98, 58.53, 65.47, 62.48, 60.84, 57.87, seconds_per_case=41.199574982065734, state="완료", model_size="4B", meteor=37.37, rouge_l=58.26),
    QualityMetric("Qwen", "VLMAS", 735, 43.54, 44.36, 42.61, 41.77, 51.19, 67.56, 74.59, 78.33, 81.07, seconds_per_case=101.00315706105903, model_size="4B"),
    QualityMetric("Qwen", "Latent base", 735, 42.99, 48.46, 36.81, 37.82, 46.86, 62.63, 70.97, 75.23, 78.38, latent_steps=5, model_size="4B"),
    QualityMetric("Qwen", "Latent base", 735, 48.03, 50.00, 45.80, 41.36, 52.10, None, None, None, None, seconds_per_case=33.04539523857784, latent_steps=10, model_size="4B"),
    QualityMetric("Qwen", "Reallocation", 735, 44.63, 50.26, 38.26, 40.27, 48.55, 62.60, 70.54, 74.58, 77.58, latent_steps=5, model_size="4B"),
    QualityMetric("Qwen", "Both", 735, 51.02, 52.05, 49.86, 48.03, 57.58, 71.26, 77.88, 81.30, 83.81, latent_steps=5, model_size="4B"),
    QualityMetric("Qwen", "Single", 735, 42.99, 35.64, 51.30, 42.99, 53.09, 56.70, 53.62, 51.88, 49.52, seconds_per_case=26.221427587911386, state="완료", model_size="2B", meteor=32.64, rouge_l=52.83),
    QualityMetric("Qwen", "VLMAS", 735, 44.63, 39.74, 50.14, None, None, None, None, None, None, seconds_per_case=97.4, model_size="2B"),
    QualityMetric("Qwen", "Latent base", 735, 48.98, 43.85, 54.78, None, None, None, None, None, None, seconds_per_case=30.7, latent_steps=5, model_size="2B"),
    QualityMetric("Qwen", "Latent base · 8 patches", 735, 48.98, 43.85, 54.78, None, None, None, None, None, None, seconds_per_case=30.7, latent_steps=5, model_size="2B"),
    QualityMetric("Qwen", "Latent base", 735, 47.89, 43.08, 53.33, 46.80, 58.13, 74.25, 79.28, 83.33, 86.11, seconds_per_case=35.944838760969496, state="완료", latent_steps=20, model_size="2B"),
    QualityMetric("Qwen", "Pruning V3", 735, 49.12, 43.59, 55.36, 48.44, 59.59, 74.13, 78.38, 81.89, 84.30, seconds_per_case=32.20193312139358, latent_steps=5, model_size="2B"),
    QualityMetric("Qwen", "Pruning V3", 735, 47.48, 43.85, 51.59, 46.80, 57.35, 72.58, 77.81, 81.93, 84.72, seconds_per_case=32.80247197798042, state="완료", latent_steps=20, model_size="2B"),
    QualityMetric("Qwen", "Pruning V4", 50, 50.0, 48.28, 52.38, 50.0, 59.56, 73.81, 80.36, 85.0, 88.09, seconds_per_case=25.643539288900794, state="파일럿 완료", latent_steps=5, model_size="2B"),
    QualityMetric("Qwen", "Pruning V4 · 8 patches", 735, 49.93, 45.13, 55.36, 49.39, 60.37, 75.88, 81.71, 85.50, 88.08, seconds_per_case=27.314381908721664, state="완료", latent_steps=5, model_size="2B"),
    QualityMetric("Qwen", "Pruning V4.1 · QK probe · gate 50", 50, 48.0, 48.28, 47.62, None, None, None, None, None, None, seconds_per_case=26.39073788956739, state="게이트 통과", latent_steps=5, model_size="2B"),
    QualityMetric("Qwen", "Pruning V4.1 · QK probe · 8 patches", 735, 49.66, 45.64, 54.20, 49.25, 60.19, 75.31, 81.39, 85.30, 87.93, seconds_per_case=26.523219930140193, state="완료", latent_steps=5, model_size="2B"),
    QualityMetric("Qwen", "Pruning V4.1 · QK probe · 8 patches", 0, None, None, None, None, None, None, None, None, None, state="실행 중", latent_steps=5, model_size="4B"),
    QualityMetric("Qwen", "Pruning V4.1 · QK probe · TAG answer", 0, None, None, None, None, None, None, None, None, None, state="실행 중", latent_steps=5, model_size="4B"),
    QualityMetric("Qwen", "Pruning V4 · 8 patches", 0, None, None, None, None, None, None, None, None, None, state="2B 완료 후 대기", latent_steps=5, model_size="4B"),
    QualityMetric("Qwen", "Pruning V4 · 8 patches", 0, None, None, None, None, None, None, None, None, None, state="4B 완료 후 대기", latent_steps=5, model_size="8B"),
    QualityMetric("Qwen", "Pruning V5", 99, 49.49, 46.30, 53.33, 49.49, 55.84, 72.34, 78.70, 82.99, 85.66, seconds_per_case=32.16743591001652, state="중단", latent_steps=5, model_size="2B"),
    QualityMetric("Qwen", "Pruning V6", 112, 49.11, 47.54, 50.98, 48.21, 55.43, 72.00, 78.66, 83.18, 85.94, seconds_per_case=28.39737376154101, state="중단", latent_steps=5, model_size="2B"),
    QualityMetric("Qwen", "Reallocation", 735, 48.57, 43.08, 54.78, None, None, None, None, None, None, seconds_per_case=35.0, latent_steps=5, model_size="2B"),
    QualityMetric("Qwen", "Reallocation V2", 735, 50.34, 43.85, 57.68, 49.80, 60.99, 76.84, 80.18, 83.63, 86.01, seconds_per_case=37.07376705601811, latent_steps=5, model_size="2B"),
    QualityMetric("Qwen", "Single", 735, 51.43, 48.46, 54.78, 51.16, 60.27, 67.39, 64.63, 62.70, 59.40, seconds_per_case=43.11209733372643, state="완료", model_size="8B", meteor=38.62, rouge_l=59.95),
    *tuple(
        QualityMetric(
            "Qwen", mode, 0, None, None, None, None, None, None, None, None, None,
            state="미실행", latent_steps=latent_steps, model_size="8B",
        )
        for mode, latent_steps in (
            ("Latent base", 5),
            ("Reallocation", 5), ("Both", 5),
        )
    ),
    QualityMetric("Qwen", "VLMAS", 735, 47.62, 46.41, 48.99, 46.80, 56.87, 66.04, 64.37, 63.85, 62.33, seconds_per_case=96.34543517117902, state="완료", model_size="8B", meteor=39.46, rouge_l=56.30),
    QualityMetric("Qwen", "Latent base", 735, 51.02, 51.03, 51.01, 50.20, 59.71, 72.12, 77.34, 80.83, 83.30, seconds_per_case=47.91258407264289, latent_steps=5, model_size="8B", meteor=52.00, rouge_l=53.45),
    QualityMetric("Qwen", "Pruning V3", 735, 51.29, 51.79, 50.72, 50.48, 59.83, 71.74, 76.92, 80.31, 82.75, seconds_per_case=40.46430409101478, latent_steps=5, model_size="8B"),
    QualityMetric("Qwen", "Pruning V3", 733, 48.98, 48.72, 49.27, 46.11, 56.25, 70.29, 76.32, 79.88, 82.40, seconds_per_case=35.563581684450966, state="degraded", latent_steps=5, model_size="4B"),
    QualityMetric("InternVL", "Single", 735, 38.91, 53.08, 22.90, 38.78, 47.99, 54.39, 64.76, 69.56, 72.94),
    QualityMetric("InternVL", "VLMAS", 735, 40.82, 39.74, 42.03, 38.64, 48.13, 64.18, 70.83, 74.18, 76.60),
    QualityMetric("InternVL", "Latent base", 735, 45.71, 45.64, 45.80, 45.17, 54.35, 68.01, 74.59, 77.88, 80.17, latent_steps=5),
    QualityMetric("InternVL", "Pruning V3", 0, None, None, None, None, None, None, None, None, None, state="중단됨", latent_steps=5),
    QualityMetric("InternVL", "Single 3.5", 0, None, None, None, None, None, None, None, None, None, state="대기", model_size="8B"),
    QualityMetric("InternVL", "Pruning", 735, 44.49, 47.44, 41.16, 42.31, 52.64, 64.06, 70.33, 73.76, 76.39, latent_steps=5),
    QualityMetric("InternVL", "Both", 735, 44.49, 47.44, 41.16, 42.31, 52.64, 64.06, 70.33, 73.76, 76.39, latent_steps=5),
    QualityMetric("InternVL", "Reallocation", 735, 45.71, 45.64, 45.80, 45.17, 54.28, 68.01, 74.59, 77.88, 80.17, seconds_per_case=26.55, latent_steps=5),
    QualityMetric("Lingshu", "Single", 735, 23.40, 40.26, 4.35, 23.13, 33.23, 42.40, 56.06, 63.36, 67.88),
    QualityMetric("Lingshu", "VLMAS", 735, 43.95, 42.82, 45.22, 41.09, 49.81, 63.60, 68.51, 71.69, 74.45, seconds_per_case=156.1919031227058),
    QualityMetric("Lingshu", "Latent base", 735, 47.89, 49.49, 46.09, 45.03, 54.82, 68.39, 74.45, 79.26, 82.57, latent_steps=5),
    QualityMetric("Lingshu", "Pruning V3", 0, None, None, None, None, None, None, None, None, None, state="실행 중", latent_steps=5),
)

RUNNING_QUALITY_LABELS: Final = {
    **{
        f"MultiPathQA/{label} · Qwen {model_size} Single No Thinking": (
            "Qwen", "Single No Thinking"
        )
        for label, _ in MULTIPATH_RUN_BENCHMARKS.values()
        for model_size in ("2B", "4B", "8B")
    },
    "Qwen 2B Pruning V3 · step 30 · pilot 200": ("Qwen", "Pruning V3 · pilot 200"),
    "Qwen 2B Single · aligned Latent Base protocol · step 5": ("Qwen", "Single"),
    "Qwen 2B Single No Thinking": ("Qwen", "Single No Thinking"),
    "Qwen 2B VLMAS": ("Qwen", "VLMAS"),
    "Qwen 2B Latent base": ("Qwen", "Latent base"),
    "Qwen 2B Latent base · Open only · step 5 · GPU 0/1 · 345": (
        "Qwen", "Latent base · Open-only no-canonical",
    ),
    "Qwen 2B Latent base · step 20 · I/O pipeline": ("Qwen", "Latent base"),
    "Qwen 2B Pruning V3 · step 20 · I/O pipeline": ("Qwen", "Pruning V3"),
    "Qwen 2B Pruning V3 · step 5": ("Qwen", "Pruning V3"),
    "Qwen 2B Pruning V4 · hierarchy · step 5 · GPU 7/8 split · 735": (
        "Qwen", "Pruning V4",
    ),
    "Qwen 2B Latent base · 25 patches · step 5 · I/O pipeline · GPU 8 · 735": (
        "Qwen", "Latent base · 25 patches",
    ),
    "Qwen 2B Pruning V4 · early-vision hierarchy · 25 patches · step 5 · I/O pipeline · GPU 7 · 735": (
        "Qwen", "Pruning V4 · 25 patches",
    ),
    "Qwen 2B Pruning V5 · Two-stage · step 5 · GPU 7 · 735": (
        "Qwen", "Pruning V5",
    ),
    "Qwen 2B Pruning V6 · Stable streaming · step 5 · GPU 8 · 735": (
        "Qwen", "Pruning V6",
    ),
    "Qwen 2B Latent base · 8 patches · step 5 · GPU 7/8 split · 735": (
        "Qwen", "Latent base · 8 patches",
    ),
    "Qwen 2B Pruning V4 · significance adaptive · 8 patches · step 5 · GPU 7/8 split · 735": (
        "Qwen", "Pruning V4 · 8 patches",
    ),
    "Qwen 2B Pruning V4.1 · QK probe · 8 patches · step 5 · GPU 7/8 split · 50": (
        "Qwen", "Pruning V4.1 · QK probe · gate 50",
    ),
    "Qwen 2B Pruning V4.1 · QK probe · 8 patches · step 5 · GPU 7/8 split · 735": (
        "Qwen", "Pruning V4.1 · QK probe · 8 patches",
    ),
    "Qwen 4B Pruning V4.1 · QK probe · 8 patches · step 5 · GPU 7/8 split · 735": (
        "Qwen", "Pruning V4.1 · QK probe · 8 patches",
    ),
    "Qwen 4B Pruning V4.1 · QK probe · TAG answer · 8 patches · step 5 · GPU 7/8 split · 50": (
        "Qwen", "Pruning V4.1 · QK probe · TAG answer",
    ),
    "Qwen 4B Pruning V4 · significance adaptive · 8 patches · step 5 · GPU 7/8 split · 735": (
        "Qwen", "Pruning V4 · 8 patches",
    ),
    "Qwen 8B Pruning V4 · significance adaptive · 8 patches · step 5 · GPU 7/8 split · 735": (
        "Qwen", "Pruning V4 · 8 patches",
    ),
    "Qwen 2B Reallocation V2 · step 5 · I/O pipeline · GPU 7/8 split · 735": (
        "Qwen", "Reallocation V2",
    ),
    "Qwen 2B Reallocation · pathology-aware · step 5": (
        "Qwen", "Reallocation Pathology",
    ),
    "Qwen 2B Reallocation · pathology-aware V2 · step 5 · gate 50": (
        "Qwen", "Reallocation Pathology V2 · gate 50",
    ),
    "Qwen 2B Reallocation · pathology-aware V2 · step 5 · 735": (
        "Qwen", "Reallocation Pathology V2",
    ),
    "Qwen 2B Reallocation · pathology-aware V3 · relevance × hierarchy · gate 50": (
        "Qwen", "Reallocation Pathology V3 · gate 50",
    ),
    "Qwen 2B Reallocation · pathology context · step 5 · gate 50": (
        "Qwen", "Reallocation Pathology Context · gate 50",
    ),
    "Qwen 2B Reallocation · pathology context · step 5 · 735": (
        "Qwen", "Reallocation Pathology Context",
    ),
    **{
        f"Qwen {model_size} Reallocation · pathology context · step {latent_steps} · 735": (
            "Qwen", "Reallocation Pathology Context",
        )
        for model_size, latent_steps in (
            ("2B", 10), ("2B", 20), ("2B", 30),
            ("4B", 5), ("4B", 10), ("4B", 20), ("4B", 30),
            ("8B", 5), ("8B", 10), ("8B", 20), ("8B", 30),
        )
    },
    "Qwen 4B Pruning V3 · step 5": ("Qwen", "Pruning V3"),
    "Qwen 4B Reallocation V2 · step 5 · I/O pipeline · GPU 7/8 split · 735": (
        "Qwen", "Reallocation V2",
    ),
    "Qwen 2B Reallocation": ("Qwen", "Reallocation"),
    "Qwen 2B Reallocation · pathology relay": (
        "Qwen",
        "Reallocation",
    ),
    "Qwen 2B Reallocation · relay v7": ("Qwen", "Reallocation"),
    "Qwen 2B Reallocation · relay v8": ("Qwen", "Reallocation"),
    "Qwen 2B Reallocation · relay v9": ("Qwen", "Reallocation"),
    "Qwen 2B Reallocation · relay v10 sweep": ("Qwen", "Reallocation"),
    "Qwen 8B Latent base": ("Qwen", "Latent base"),
    "Qwen 8B Single No Thinking": ("Qwen", "Single No Thinking"),
    "Qwen 4B Single No Thinking": ("Qwen", "Single No Thinking"),
    "Qwen 4B Single · matched base · step 10": ("Qwen", "Single · matched base"),
    "Qwen 2B Single · matched base · step 10": ("Qwen", "Single · matched base"),
    "Qwen 8B Single · matched base · step 10": ("Qwen", "Single · matched base"),
    "Qwen 8B Reallocation V2 · step 5 · I/O pipeline · GPU 7/8 split · 735": ("Qwen", "Reallocation V2"),
    "Qwen 8B VLMAS": ("Qwen", "VLMAS"),
    "Qwen 2B Both": ("Qwen", "Both"),
    "OctoMed 7B Single": ("OctoMed", "Single"),
    "OctoMed 7B Latent base": ("OctoMed", "Latent base"),
    "OctoMed 7B Pruning · Navigator KV off · 96 tok": (
        "OctoMed",
        "Pruning",
    ),
    "InternVL Reallocation · current": ("InternVL", "Reallocation · current"),
    "Qwen VLMAS": ("Qwen", "VLMAS"),
    **{
        f"Qwen {model_size} Latent base · no candidate · step {latent_steps} · GPU {gpu} · 735": ("Qwen", "Latent base")
        for model_size, latent_steps, gpu in (
            ("2B", 5, 0), ("2B", 10, 1), ("2B", 20, 2), ("2B", 30, 3),
            ("4B", 5, 4), ("4B", 10, 4), ("4B", 20, 4), ("4B", 30, 4),
            ("8B", 5, 0), ("8B", 10, 1), ("8B", 20, 2), ("8B", 30, 3),
        )
    },
    **{
        f"Qwen {model_size} Single · Reasoner-thinking + Answerer JSON · 735": ("Qwen", "Single")
        for model_size in ("2B", "4B", "8B")
    },
    **{
        f"Qwen {model_size} VLMAS": ("Qwen", "VLMAS")
        for model_size in ("2B", "4B", "8B")
    },
    **{
        f"Qwen {model_size} Latent base · step {latent_steps}": ("Qwen", "Latent base")
        for model_size in ("2B", "4B", "8B")
        for latent_steps in (5, 10, 20, 30)
    },
    **{
        f"Qwen 8B Latent base · Navigator KV · step {latent_steps}": (
            "Qwen", "Latent base · Navigator KV"
        )
        for latent_steps in (5, 10, 20, 30)
    },
    **{
        f"Qwen {model_size} Pruning V3 · step {latent_steps}": (
            "Qwen", "Pruning V3"
        )
        for model_size in ("2B", "4B", "8B")
        for latent_steps in (5, 10, 20, 30)
    },
    **{
        f"Qwen {model_size} Latent base · step {latent_steps}": ("Qwen", "Latent base")
        for model_size in ("2B", "4B", "8B")
        for latent_steps in (5, 10, 20, 30)
    },
    "Lingshu Pruning · Navigator KV off": (
        "Lingshu",
        "Pruning",
    ),
    "Lingshu Reallocation · 96 tok": ("Lingshu", "Reallocation"),
    "Lingshu Both · 96 tok": ("Lingshu", "Both"),
    "Lingshu Latent base · step 5 · tissue-safe Navigator": ("Lingshu", "Latent base"),
    "InternVL Latent base · paired control · GPU 7/8 · 30": ("InternVL", "Latent base"),
    "InternVL Pruning V3 · paired Base · GPU 7/8 · 735": ("InternVL", "Pruning V3"),
    "InternVL3.5 Single · InternVL3 prompt · slide-wise · 735": ("InternVL", "Single 3.5"),
    "InternVL3.5 VL-MAS · InternVL3 prompt · slide-wise · 735": ("InternVL", "VLMAS 3.5"),
    "Qwen 8B VL-MAS · no candidate · no Reasoner timeout + I/O · GPU 7/8 · 735": ("Qwen", "VLMAS"),
    "Lingshu Reallocation V2 · step 5 · tissue-safe Navigator": ("Lingshu", "Reallocation V2"),
}

QWEN2B_OPEN_ONLY_BASE_NAME: Final = "Qwen 2B Latent base · Open only · step 5 · GPU 0/1 · 345"
QWEN2B_LATENT_BASE_CLOSED_SAMPLES: Final = 390
QWEN2B_LATENT_BASE_CLOSED_ACCURACY: Final = 43.85

PREFIX_QUALITY_SPECS: Final = (
    (
        "Lingshu",
        "Both",
        "완료",
        JobSpec(
            "Lingshu Both · 96 tok",
            6,
            735,
            ROOT / "runs/lingshu_both_nav96_attempt1_gpu6_20260821",
            "lingshu_both_nav96",
            extra_roots=(
                ROOT / "runs/lingshu_both_nav96_attempt1_gpu7_20260821",
                ROOT / "runs/lingshu_both_nav96_attempt1_gpu8_20260821",
            ),
            active_gpus=(6, 7, 8),
            evaluation_summary=ROOT / "runs/evaluation_lingshu_both_nav96_20260821/general/lingshu_both_nav96_summary.json",
            latent_steps=5,
        ),
    ),
    (
        "Lingshu",
        "Pruning",
        "완료",
        JobSpec(
            "Lingshu Pruning · Navigator KV off",
            7,
            735,
            ROOT / "runs/lingshu_pruning_navigator_kv_off_nav96_gpu7_20260821",
            "lingshu_pruning_nav96_gpu",
            extra_roots=(
                ROOT / "runs/lingshu_pruning_nav96_gpu6_shard_20260821",
                ROOT / "runs/lingshu_pruning_nav96_gpu7_shard_20260821",
                ROOT / "runs/lingshu_pruning_nav96_gpu8_shard_20260821",
            ),
            active_gpus=(6, 7, 8),
            evaluation_summary=ROOT / "runs/evaluation_lingshu_pruning_nav96_20260821/general/lingshu_pruning_nav96_summary.json",
            latent_steps=5,
        ),
    ),
    (
        "Qwen",
        "VLMAS",
        "중단됨",
        JobSpec(
            "Qwen VLMAS",
            6,
            735,
            ROOT / "runs/qwen_vlmas_gpu6_735_20260821",
            "qwen_vlmas_gpu6_735_20260821",
            extra_roots=tuple(
                ROOT / f"runs/qwen_vlmas_gpu{gpu}_shard_20260821" for gpu in (6, 7, 8)
            ),
            active_gpus=(6, 7, 8),
            evaluation_summary=ROOT / "runs/evaluation_qwen_vlmas_20260822/general/qwen_vlmas_20260822_summary.json",
            model_size="4B",
        ),
    ),
    (
        "Lingshu",
        "Reallocation",
        "중단됨",
        JobSpec(
            "Lingshu Reallocation · 96 tok",
            8,
            735,
            ROOT / "runs/lingshu_reallocation_nav96_gpu8_20260821",
            "lingshu_reallocation_nav96_gpu8_20260821",
            evaluation_summary=ROOT / "runs/evaluation_lingshu_reallocation_nav96_20260821/general/lingshu_reallocation_nav96_summary.json",
            latent_steps=5,
        ),
    ),
)

QUALITY_BACKBONE_ORDER: Final = {
    "Qwen": 0,
    "Qwen 8B": 1,
    "InternVL": 2,
    "Lingshu": 3,
    "OctoMed": 4,
}


def quality_order(metric: QualityMetric) -> tuple[int, int]:
    """Keep active prefix rows beside their matching completed mode."""
    mode = metric.mode.lower()
    if mode.endswith("· current"):
        return 3, 0
    mode_order = (
        0 if mode.startswith("single")
        else 1 if mode.startswith("vlmas")
        else 2 if mode.startswith("latent base")
        else 3 if mode.startswith("pruning")
        else 4 if mode.startswith("reallocation")
        else 5 if mode.startswith("both")
        else 6
    )
    return QUALITY_BACKBONE_ORDER[metric.backbone], mode_order

REFERENCE_ROWS: Final = load_eval_rows(
    ROOT / "runs/internvl_single_gpu7_735_20260819/eval/internvl_general_eval.jsonl"
)
STRICT_METRICS: Final = load_strict_functions(CODE_ROOT / "eval/metrics.py")


def command_lines() -> str:
    """Return executable names and arguments for local processes."""
    result = subprocess.run(
        ["ps", "-eo", "comm=,args="], check=True, capture_output=True, text=True
    )
    return result.stdout


def is_python_process_line(line: str) -> bool:
    """Recognize absolute-path Python launchers in ``ps`` output."""
    executable = line.lstrip().split(maxsplit=1)[0]
    return executable.rsplit("/", 1)[-1].startswith("python")


def process_marker_matches(spec: JobSpec, line: str) -> bool:
    """Match a run only when its marker owns the launched output root."""
    marker = spec.process_marker
    stable_marker = re.sub(r"_gpu\d*$", "", marker)
    if marker not in line and stable_marker not in line:
        return False
    try:
        arguments = shlex.split(line)
    except ValueError:
        return False
    output_roots = [
        value
        for option, value in zip(arguments, arguments[1:], strict=False)
        if option in {"--output-root", "--output-dir", "--answers-file"}
    ]
    output_roots.extend(
        argument.split("=", 1)[1]
        for argument in arguments
        if argument.startswith(("--output-root=", "--output-dir=", "--answers-file="))
    )
    if any(marker in root or stable_marker in root for root in output_roots):
        return True
    return (
        "run_single_v7_fair_hf_batched.sh" in line
        and (marker in line or stable_marker in line)
    )


def count_cases(spec: JobSpec) -> tuple[int, int]:
    """Count completed and failed artifacts for one run."""
    if spec.single_predictions:
        return len(single_prediction_rows(spec)), 0
    if not any(root.exists() for root in (spec.root, *spec.extra_roots)):
        return 0, 0
    roots = (spec.root, *spec.extra_roots)
    successes = tuple(path for root in roots for path in root.rglob("result.json"))
    failures = tuple(path for root in roots for path in root.rglob("failure.json"))
    recovery_results = (
        ()
        if spec.recovery_root is None
        else tuple(spec.recovery_root.rglob("result.json"))
    )
    success_indices = {path.parent.name.split("_", 1)[0] for path in (*successes, *recovery_results)}
    failure_indices = {path.parent.name.split("_", 1)[0] for path in failures}
    return len(success_indices), len(failure_indices - success_indices)


def latest_cases(spec: JobSpec):
    """Merge the newest artifact per dataset index across active shards."""
    latest = {}
    for root in (spec.root, *spec.extra_roots):
        for case in select_latest_cases(root):
            previous = latest.get(case.dataset_index)
            if previous is None or case.modified_ns > previous.modified_ns:
                latest[case.dataset_index] = case
    return tuple(sorted(latest.values(), key=lambda case: (case.modified_ns, case.dataset_index)))


def latency_roots(spec: JobSpec) -> tuple[Path, ...]:
    """Return only run shards produced by the designated latency GPUs."""
    roots = (spec.root, *spec.extra_roots, *spec.speed_extra_roots)
    if len(roots) == 1 and len(spec.active_gpus or (spec.gpu,)) == 1:
        return roots
    selected = tuple(
        root
        for root in roots
        if (match := re.search(r"(?:^|_)gpu(\d+)(?:$|_)", root.name)) is not None
        and int(match.group(1)) in LATENCY_GPUS
    )
    if selected:
        return selected
    if len(roots) == len(spec.active_gpus):
        return tuple(
            root
            for root, gpu in zip(roots, spec.active_gpus, strict=True)
            if gpu in LATENCY_GPUS
        )
    return (spec.root,) if spec.gpu in LATENCY_GPUS else ()


def latest_latency_cases(spec: JobSpec):
    """Merge completed cases from GPU 7 and 8 for latency fallback metrics."""
    roots = latency_roots(spec)
    if not roots:
        return ()
    return latest_cases(replace(spec, root=roots[0], extra_roots=roots[1:]))


def mean_attempt_seconds(roots: tuple[Path, ...]) -> float | None:
    """Return mean recorded attempt wall time across all active run shards."""
    values: list[float] = []
    for root in roots:
        metrics_paths = (root / "attempt_metrics.jsonl",)
        if not metrics_paths[0].exists():
            metrics_paths = tuple(root.rglob("attempt_metrics.jsonl"))
        for metrics in metrics_paths:
            for line in metrics.read_text().splitlines():
                record = json.loads(line)
                wall_seconds = record.get("wall_seconds")
                if isinstance(wall_seconds, (int, float)):
                    values.append(float(wall_seconds))
    return sum(values) / len(values) if values else None


def cumulative_attempt_seconds(spec: JobSpec) -> float | None:
    """Average successful wall time once per dataset ID, using the newest result."""
    latest_roots: dict[int, tuple[int, Path]] = {}
    roots = latency_roots(spec)
    for root in roots:
        for case in select_latest_cases(root):
            previous = latest_roots.get(case.dataset_index)
            if previous is None or case.modified_ns > previous[0]:
                latest_roots[case.dataset_index] = (case.modified_ns, root)
    attempts_by_root = {root: _successful_attempt_seconds(root) for root in roots}
    warmup_indices = {
        root: {
            dataset_index
            for dataset_index, _ in sorted(
                attempts.items(), key=lambda item: item[1][0]
            )[: spec.speed_warmup_attempts_per_root]
        }
        for root, attempts in attempts_by_root.items()
    }
    values = [
        attempts_by_root[root][dataset_index][1]
        for dataset_index, (_, root) in latest_roots.items()
        if dataset_index in attempts_by_root[root]
        and dataset_index not in warmup_indices[root]
    ]
    return sum(values) / len(values) if values else None


def role_call_seconds(spec: JobSpec) -> float | None:
    """Average successful VL-MAS role-call time per completed case."""
    totals: list[float] = []
    for case in latest_latency_cases(spec):
        raw = json.loads(case.result_path.read_text(encoding="utf-8"))
        calls = raw.get("role_calls") if isinstance(raw, dict) else None
        if not isinstance(calls, list):
            continue
        values = [
            float(call["elapsed_seconds"])
            for call in calls
            if isinstance(call, dict)
            and isinstance(call.get("elapsed_seconds"), (int, float))
        ]
        if values:
            totals.append(sum(values))
    return sum(totals) / len(totals) if totals else None


def recent_role_call_seconds(spec: JobSpec) -> float | None:
    """Return agent-call time for the globally latest 20 completed cases."""
    totals: list[float] = []
    for case in latest_latency_cases(spec)[-20:]:
        raw = json.loads(case.result_path.read_text(encoding="utf-8"))
        calls = raw.get("role_calls") if isinstance(raw, dict) else None
        if not isinstance(calls, list):
            continue
        values = [
            float(call["elapsed_seconds"])
            for call in calls
            if isinstance(call, dict)
            and isinstance(call.get("elapsed_seconds"), (int, float))
        ]
        if values:
            totals.append(sum(values))
    return sum(totals) / len(totals) if totals else None


def _successful_attempt_seconds(root: Path) -> dict[int, tuple[int, float]]:
    """Return the latest successful attempt and its file position per dataset ID."""
    metrics = root / "attempt_metrics.jsonl"
    if not metrics.exists():
        return {}
    attempts: dict[int, tuple[int, float]] = {}
    for position, line in enumerate(metrics.read_text().splitlines()):
        record = json.loads(line)
        dataset_index = record.get("dataset_index")
        wall_seconds = record.get("wall_seconds")
        if (
            record.get("success") is True
            and isinstance(dataset_index, int)
            and isinstance(wall_seconds, (int, float))
        ):
            attempts[dataset_index] = (position, float(wall_seconds))
    return attempts


def paired_attempt_seconds(
    root: Path,
    peer_root: Path,
    limit: int | None = None,
    warmup_pairs: int = 0,
) -> tuple[float | None, float | None]:
    """Return matched speed means after excluding designated warm-up pairs."""
    own_attempts = _successful_attempt_seconds(root)
    peer_attempts = _successful_attempt_seconds(peer_root)
    shared_indices = own_attempts.keys() & peer_attempts.keys()
    ordered_indices = sorted(
        shared_indices,
        key=lambda index: max(own_attempts[index][0], peer_attempts[index][0]),
    )
    ordered_indices = ordered_indices[warmup_pairs:]
    if limit is not None:
        ordered_indices = ordered_indices[-limit:]
    if not ordered_indices:
        return None, None
    own_values = [own_attempts[index][1] for index in ordered_indices]
    peer_values = [peer_attempts[index][1] for index in ordered_indices]
    return sum(own_values) / len(own_values), sum(peer_values) / len(peer_values)


def recent_attempt_seconds(root: Path) -> float | None:
    """Return the trailing 20-case wall-time mean for an active run."""
    metrics = root / "attempt_metrics.jsonl"
    metrics_paths = (metrics,) if metrics.exists() else tuple(root.rglob("attempt_metrics.jsonl"))
    values = [
        float(record["wall_seconds"])
        for metrics_path in metrics_paths
        for line in metrics_path.read_text().splitlines()
        if isinstance(record := json.loads(line), dict)
        and isinstance(record.get("wall_seconds"), (int, float))
    ][-20:]
    return sum(values) / len(values) if values else None


def recent_attempt_seconds_for_spec(spec: JobSpec) -> float | None:
    """Return the latest 20 successful completions across all worker shards."""
    roots = latency_roots(spec)
    latest_roots: dict[int, tuple[int, Path]] = {}
    for root in roots:
        for case in select_latest_cases(root):
            previous = latest_roots.get(case.dataset_index)
            if previous is None or case.modified_ns > previous[0]:
                latest_roots[case.dataset_index] = (case.modified_ns, root)
    attempts_by_root = {root: _successful_attempt_seconds(root) for root in roots}
    warmup_indices = {
        root: {
            dataset_index
            for dataset_index, _ in sorted(
                attempts.items(), key=lambda item: item[1][0]
            )[: spec.speed_warmup_attempts_per_root]
        }
        for root, attempts in attempts_by_root.items()
    }
    values = [
        attempts_by_root[root][dataset_index][1]
        for dataset_index, (_, root) in sorted(
            latest_roots.items(), key=lambda item: item[1][0]
        )
        if dataset_index in attempts_by_root[root]
        and dataset_index not in warmup_indices[root]
    ][-20:]
    return sum(values) / len(values) if values else None


def retry_count(spec: JobSpec) -> int | None:
    """Count case attempts beyond the first plus format-repair decodes."""
    roots = (spec.speed_root,) if spec.speed_root is not None else (spec.root, *spec.extra_roots)
    highest_attempt: dict[int, int] = {}
    format_repairs = 0
    metric_records = 0
    for root in roots:
        metrics = root / "attempt_metrics.jsonl"
        if not metrics.exists():
            continue
        for line in metrics.read_text().splitlines():
            record = json.loads(line)
            metric_records += 1
            dataset_index = record.get("dataset_index")
            attempt = record.get("attempt")
            if isinstance(dataset_index, int) and isinstance(attempt, int):
                highest_attempt[dataset_index] = max(
                    highest_attempt.get(dataset_index, 1), attempt
                )
            repairs = record.get("format_repairs")
            if isinstance(repairs, int):
                format_repairs += repairs
    if metric_records == 0:
        return None
    return sum(attempt - 1 for attempt in highest_attempt.values()) + format_repairs


def artifact_seconds_per_case(spec: JobSpec, processed: int) -> float | None:
    """Estimate wall-clock throughput from artifact timestamps when metrics are absent."""
    if processed < 2:
        return None
    if spec.single_predictions:
        roots = (spec.root, *spec.extra_roots)
        prediction_paths = tuple(
            root / "worker_0" / "predictions" / "qwen3vl_predictions.jsonl"
            for root in roots
        )
        manifests = tuple(root / "run_manifest.json" for root in roots)
        existing_predictions = tuple(path for path in prediction_paths if path.exists())
        existing_manifests = tuple(path for path in manifests if path.exists())
        workers = len(spec.active_gpus or (spec.gpu,))
        if existing_predictions and existing_manifests:
            elapsed = max(path.stat().st_mtime for path in existing_predictions) - min(
                path.stat().st_mtime for path in existing_manifests
            )
            return max(0.0, elapsed) / max(1, processed / workers)
        return None
    roots = (spec.speed_root,) if spec.speed_root is not None else (spec.root, *spec.extra_roots)
    intervals = []
    for root in roots:
        artifacts = tuple(root.rglob("result.json")) + tuple(root.rglob("failure.json"))
        timestamps = tuple(path.stat().st_mtime for path in artifacts)
        if len(timestamps) > 1:
            intervals.append((max(timestamps) - min(timestamps)) / (len(timestamps) - 1))
    return sum(intervals) / len(intervals) if intervals else None


def single_prediction_paths(spec: JobSpec) -> tuple[Path, ...]:
    """Return worker outputs, or the direct JSONL emitted by a one-process runner."""
    worker_paths = tuple(
        path
        for root in (spec.root, *spec.extra_roots)
        for path in sorted(root.glob("worker_*/predictions/qwen3vl_predictions.jsonl"))
    )
    if worker_paths:
        return worker_paths
    return tuple(
        path
        for root in (spec.root, *spec.extra_roots)
        if (path := root / "predictions/qwen3vl_predictions.jsonl").exists()
    )


def active_single_prediction_paths(spec: JobSpec) -> tuple[Path, ...]:
    """Prefer the current one-worker rebalance shards for live throughput."""
    paths = single_prediction_paths(spec)
    rebalanced_paths = tuple(path for path in paths if "single_rebalanced_" in str(path))
    return rebalanced_paths or paths


def single_prediction_rows(spec: JobSpec) -> dict[int, dict[str, object]]:
    """Map each Single worker-local row back to its unique dataset index."""
    rows: dict[int, dict[str, object]] = {}
    for root in (spec.root, *spec.extra_roots):
        manifest_path = root / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        requested_indices = manifest.get("requested_indices")
        worker_count = manifest.get("worker_count")
        if not isinstance(requested_indices, list) or not isinstance(worker_count, int):
            continue
        for path in sorted(root.glob("worker_*/predictions/qwen3vl_predictions.jsonl")):
            worker_name = path.parent.parent.name
            worker_id = int(worker_name.removeprefix("worker_"))
            worker_indices = requested_indices[worker_id::worker_count]
            for line in path.read_text().splitlines():
                row = json.loads(line)
                local_index = row.get("question_id")
                if not isinstance(local_index, str) or not local_index.isdigit():
                    continue
                dataset_position = int(local_index)
                if dataset_position >= len(worker_indices):
                    continue
                dataset_index = worker_indices[dataset_position]
                if isinstance(dataset_index, int):
                    rows[dataset_index] = row
    if rows:
        return rows
    for root_position, root in enumerate((spec.root, *spec.extra_roots)):
        path = root / "predictions/qwen3vl_predictions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            dataset_index = row.get("question_id") if isinstance(row, dict) else None
            if isinstance(dataset_index, str) and dataset_index.isdigit():
                rows[root_position * spec.target + int(dataset_index)] = row
    return rows


def single_prediction_seconds(spec: JobSpec, recent: bool = False) -> float | None:
    """Average recorded inference time from a Single runner's worker outputs."""
    roots = latency_roots(spec)
    paths = tuple(
        path
        for path in active_single_prediction_paths(spec)
        if any(path.is_relative_to(root) for root in roots)
    )
    values: list[float] = []
    recent_per_shard = max(1, math.ceil(20 / len(paths))) if paths else 0
    for path in paths:
        path_values = [
            float(record["inference_time_sec"])
            for line in path.read_text().splitlines()
            if isinstance(record := json.loads(line), dict)
            and isinstance(record.get("inference_time_sec"), (int, float))
        ]
        path_values = path_values[spec.speed_warmup_attempts_per_root :]
        values.extend(path_values[-recent_per_shard:] if recent else path_values)
    return sum(values) / len(values) if values else None


def recent_artifact_seconds_per_case(spec: JobSpec) -> float | None:
    """Estimate current throughput from the newest twenty completed artifacts."""
    roots = (spec.speed_root,) if spec.speed_root is not None else (spec.root, *spec.extra_roots)
    intervals = []
    for root in roots:
        artifacts = sorted(
            (*root.rglob("result.json"), *root.rglob("failure.json")),
            key=lambda path: path.stat().st_mtime,
        )[-21:]
        if len(artifacts) > 1:
            timestamps = tuple(path.stat().st_mtime for path in artifacts)
            intervals.append((timestamps[-1] - timestamps[0]) / (len(timestamps) - 1))
    return sum(intervals) / len(intervals) if intervals else None


def active_worker_count(spec: JobSpec, commands: str) -> int:
    """Count currently running shard roots for an independently scheduled job."""
    roots = (spec.root, *spec.extra_roots)
    command_lines = commands.splitlines()
    count = sum(
        any(
            process_marker_matches(spec, line)
            and str(root.relative_to(ROOT)) in line
            and is_python_process_line(line)
            for line in command_lines
        )
        for root in roots
    )
    return count or len(spec.active_gpus or (spec.gpu,))


def seconds_per_case(spec: JobSpec, processed: int) -> float | None:
    """Prefer recorded per-attempt timings; otherwise use observable artifact cadence."""
    if spec.single_predictions:
        return single_prediction_seconds(spec)
    if spec.name == "Qwen 4B Reallocation V2 · step 5 · I/O pipeline · GPU 7/8 split · 735":
        return recent_attempt_seconds(spec.speed_root or spec.root) or cumulative_attempt_seconds(spec)
    if spec.speed_root is not None and spec.speed_peer_root is not None:
        own_seconds, _ = paired_attempt_seconds(
            spec.speed_root, spec.speed_peer_root,
            warmup_pairs=spec.speed_warmup_pairs,
        )
        return own_seconds
    return cumulative_attempt_seconds(spec) or role_call_seconds(spec)


def job_status(spec: JobSpec, commands: str) -> JobStatus:
    """Derive an observable job state from artifacts and current commands."""
    success, failure = count_cases(spec)
    processed = min(success + failure, spec.target)
    running = any(
        is_python_process_line(line)
        and process_marker_matches(spec, line)
        and ("vision_text_mas." in line or "run_thinking.py" in line)
        for line in commands.splitlines()
    )
    complete = processed >= spec.target
    if running:
        state, label = "running", "실행 중"
    elif complete and failure == 0:
        state, label = "completed", "완료"
    elif failure > 0:
        state, label = "degraded", "실패 복구 대기"
    else:
        state, label = "queued", "대기"
    cumulative_seconds = seconds_per_case(spec, processed)
    if spec.single_predictions:
        recent_seconds = single_prediction_seconds(spec, recent=True)
    elif spec.speed_root is not None and spec.speed_peer_root is not None:
        recent_seconds, _ = paired_attempt_seconds(
            spec.speed_root, spec.speed_peer_root, limit=20,
            warmup_pairs=spec.speed_warmup_pairs,
        )
    else:
        recent_seconds = recent_attempt_seconds_for_spec(spec)
    if recent_seconds is None:
        recent_seconds = recent_role_call_seconds(spec) or cumulative_seconds
    interval_seconds = recent_seconds if recent_seconds is not None else cumulative_seconds
    eta_seconds = None
    if running and processed < spec.target and interval_seconds is not None:
        eta_seconds = (spec.target - processed) * interval_seconds / active_worker_count(spec, commands)
    return JobStatus(
        spec.name,
        state,
        label,
        spec.target,
        success,
        failure,
        processed,
        round(processed * 100 / spec.target),
        cumulative_seconds,
        recent_seconds,
        eta_seconds,
        retry_count(spec),
        allocated_gpus=spec.active_gpus or (spec.gpu,),
    )


def attach_queue_estimates(
    specs: tuple[JobSpec, ...], statuses: tuple[JobStatus, ...]
) -> tuple[JobStatus, ...]:
    """Annotate explicitly sequenced jobs with their position and observable wait."""
    updated = list(statuses)
    grouped: dict[str, list[int]] = {}
    for index, spec in enumerate(specs):
        if spec.queue_group is not None and spec.queue_order is not None:
            grouped.setdefault(spec.queue_group, []).append(index)
    for indices in grouped.values():
        elapsed_wait: float | None = 0.0
        for position, index in enumerate(sorted(indices, key=lambda item: specs[item].queue_order or 0), start=1):
            status = updated[index]
            if status.state == "queued":
                updated[index] = replace(
                    status,
                    queue_position=position,
                    queue_wait_seconds=elapsed_wait,
                )
            elif status.state == "running":
                updated[index] = replace(status, queue_position=position)
                elapsed_wait = status.eta_seconds
                continue
            if status.state != "completed":
                elapsed_wait = None
    return tuple(updated)


def gpu_statuses(
    jobs: tuple[JobStatus, ...],
    specs: tuple[JobSpec, ...],
) -> tuple[GpuStatus, ...]:
    """Read utilization and memory for the assigned GPUs."""
    active_workloads = {
        gpu_index: ", ".join(
            status.name
            for status, status_spec in zip(jobs, specs, strict=True)
            if status.state == "running" and gpu_index in (status_spec.active_gpus or (status_spec.gpu,))
        )
        for gpu_index in range(9)
    }
    query = "index,utilization.gpu,memory.used,memory.total,pstate"
    result = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    statuses: list[GpuStatus] = []
    for line in result.stdout.splitlines():
        index, utilization, used, total, pstate = (part.strip() for part in line.split(","))
        if 0 <= int(index) <= 8:
            gpu_index = int(index)
            statuses.append(
                GpuStatus(
                    gpu_index,
                    int(utilization),
                    round(int(used) / 1024, 1),
                    round(int(total) / 1024, 1),
                    pstate,
                    active_workloads.get(gpu_index) or None,
                )
            )
    return tuple(statuses)


def cached_gpu_statuses(
    jobs: tuple[JobStatus, ...], specs: tuple[JobSpec, ...]
) -> tuple[GpuStatus, ...]:
    """Refresh slow nvidia-smi telemetry off the one-second API request path."""
    global _gpu_status_cache_at, _gpu_status_refreshing
    with _gpu_status_lock:
        if monotonic() - _gpu_status_cache_at < 1.0:
            return _gpu_status_cache
        if _gpu_status_refreshing:
            return _gpu_status_cache
        _gpu_status_refreshing = True

    def refresh() -> None:
        global _gpu_status_cache, _gpu_status_cache_at, _gpu_status_refreshing
        statuses = gpu_statuses(jobs, specs)
        with _gpu_status_lock:
            _gpu_status_cache = statuses
            _gpu_status_cache_at = monotonic()
            _gpu_status_refreshing = False

    Thread(target=refresh, name="dashboard-gpu-status", daemon=True).start()
    return _gpu_status_cache


def live_metric(spec: JobSpec) -> LiveMetric:
    """Compute strict accuracy over the completed prefix of one active run."""
    if spec.single_predictions:
        return single_prediction_live_metric(spec)
    if not any(root.exists() for root in (spec.root, *spec.extra_roots)):
        return LiveMetric(spec.name, 0, None, None, None)
    indicators = [
        latent_indicator(case, REFERENCE_ROWS[case.dataset_index], STRICT_METRICS)
        for case in latest_cases(spec)
        if case.dataset_index in REFERENCE_ROWS
    ]
    points = cumulative_points(indicators)
    if not points:
        return LiveMetric(spec.name, 0, None, None, None)
    point = points[-1]
    return LiveMetric(
        spec.name,
        point.completed_cases,
        round(point.total_accuracy * 100, 2),
        None if point.mcq_accuracy is None else round(point.mcq_accuracy * 100, 2),
        None if point.open_substring_accuracy is None else round(point.open_substring_accuracy * 100, 2),
    )


def single_prediction_live_metric(spec: JobSpec) -> LiveMetric:
    """Compute strict prefix accuracy directly from a Single runner's JSONL rows."""
    indicators: list[Indicator] = []
    for row in single_prediction_rows(spec).values():
        prediction = row.get("prediction")
        ground_truth = row.get("ground_truth")
        choices = row.get("choices")
        if not isinstance(prediction, str) or not isinstance(ground_truth, str):
            continue
        if not isinstance(choices, list) or not all(
            isinstance(choice, str) for choice in choices
        ):
            continue
        prediction_eval = STRICT_METRICS.expand_letter(prediction.strip(), choices)
        ground_truth_eval = STRICT_METRICS.expand_letter(ground_truth.strip(), choices)
        if choices:
            correct = int(
                bool(
                    STRICT_METRICS.acc_of_seq(
                        choices, ground_truth_eval, prediction_eval
                    )
                )
            )
            indicators.append(Indicator(correct, 1, correct, 1, 0, 0))
        else:
            correct = int(
                STRICT_METRICS.substring_correct(
                    ground_truth_eval, prediction_eval
                )
            )
            indicators.append(Indicator(correct, 1, 0, 0, correct, 1))
    points = cumulative_points(indicators)
    if not points:
        return LiveMetric(spec.name, 0, None, None, None)
    point = points[-1]
    return LiveMetric(
        spec.name,
        point.completed_cases,
        round(point.total_accuracy * 100, 2),
        None if point.mcq_accuracy is None else round(point.mcq_accuracy * 100, 2),
        None
        if point.open_substring_accuracy is None
        else round(point.open_substring_accuracy * 100, 2),
    )


def prefix_quality_metric(
    backbone: str,
    mode: str,
    state: str,
    spec: JobSpec,
    live: LiveMetric,
) -> QualityMetric:
    """Render a partial run's strict prefix alongside completed comparisons."""
    display_seconds = seconds_per_case(spec, live.completed)
    return QualityMetric(
        backbone,
        mode,
        live.completed,
        live.total_accuracy,
        live.mcq_accuracy,
        live.open_accuracy,
        None,
        None,
        None,
        None,
        None,
        None,
        display_seconds,
        state,
        spec.latent_steps,
        spec.model_size,
        thinking_mode=spec.thinking_mode,
    )


def evaluated_quality_metric(
    backbone: str,
    mode: str,
    state: str,
    spec: JobSpec,
    live: LiveMetric,
) -> QualityMetric:
    """Use the completed 735-case evaluator summary when it is available."""
    summary_path = spec.evaluation_summary
    if summary_path is None or not summary_path.exists():
        return prefix_quality_metric(backbone, mode, state, spec, live)
    summary = json.loads(summary_path.read_text())
    if summary.get("num_rows") != 735:
        return prefix_quality_metric(backbone, mode, state, spec, live)
    display_seconds = seconds_per_case(spec, 735)
    return QualityMetric(
        backbone,
        mode,
        735,
        round(float(summary["total_accuracy"]) * 100, 2),
        round(float(summary["mcq_accuracy"]) * 100, 2),
        round(float(summary["open_substring_accuracy"]) * 100, 2),
        round(float(summary["normalized_exact_match"]) * 100, 2),
        round(float(summary["token_f1"]) * 100, 2),
        round(float(summary["open_BLEU_1"]) * 100, 2),
        round(float(summary["open_BLEU_2"]) * 100, 2),
        round(float(summary["open_BLEU_3"]) * 100, 2),
        round(float(summary["open_BLEU_4"]) * 100, 2),
        display_seconds,
        "완료",
        spec.latent_steps,
        spec.model_size,
        meteor=round(float(summary["open_METEOR"]) * 100, 2),
        rouge_l=round(float(summary["open_ROUGE_L"]) * 100, 2),
        thinking_mode=spec.thinking_mode,
    )


def live_full_quality(backbone: str, mode: str, state: str, spec: JobSpec) -> QualityMetric:
    """Evaluate every available result once per completed-prefix size."""
    single_rows = single_prediction_rows(spec) if spec.single_predictions else {}
    cases = () if spec.single_predictions else latest_cases(spec)
    completed = len(single_rows) if spec.single_predictions else len(cases)
    cached = _live_quality_cache.get(spec.name)
    if cached is not None and cached[0] == completed:
        return cached[1]
    rows: list[EvalRow] = []
    for dataset_index, record in single_rows.items():
        choices = record.get("choices")
        prediction = record.get("prediction")
        ground_truth = record.get("ground_truth")
        if (
            not isinstance(choices, list)
            or not isinstance(prediction, str)
            or not isinstance(ground_truth, str)
        ):
            continue
        inference_time = record.get("inference_time_sec")
        rows.append(
            EvalRow(
                source_file=spec.name,
                question_id=str(dataset_index),
                long_id=str(record.get("slide_id", "")),
                question=str(record.get("question", "")),
                prediction=prediction,
                ground_truth=ground_truth,
                choices=[str(choice) for choice in choices],
                is_mcq=bool(choices),
                inference_time_sec=(
                    float(inference_time)
                    if isinstance(inference_time, (int, float))
                    else None
                ),
            )
        )
    for case in cases:
        reference = REFERENCE_ROWS.get(case.dataset_index)
        if reference is None:
            continue
        result = json.loads(case.result_path.read_text())
        answer = result.get("answer")
        prediction = answer.get("answer", "") if isinstance(answer, dict) else answer
        choices = reference.get("choices")
        if not isinstance(prediction, str) or not isinstance(choices, list):
            continue
        rows.append(
            EvalRow(
                source_file=str(case.result_path),
                question_id=str(case.dataset_index),
                long_id=str(reference.get("slide_id", "")),
                question=str(reference.get("question", "")),
                prediction=prediction,
                ground_truth=str(result.get("gold_answer", reference.get("ground_truth", ""))),
                choices=[str(choice) for choice in choices],
                is_mcq=bool(choices),
            )
        )
    if not rows:
        metric = QualityMetric(
            backbone, mode, 0, None, None, None, None, None, None, None, None, None,
            state=state, latent_steps=spec.latent_steps, model_size=spec.model_size,
            thinking_mode=spec.thinking_mode,
        )
    else:
        summary, _ = evaluate(rows, include_text_metrics=False)
        metric = QualityMetric(
            backbone,
            mode,
            len(rows),
            round(float(summary["total_accuracy"]) * 100, 2),
            round(float(summary["mcq_accuracy"]) * 100, 2),
            round(float(summary["open_substring_accuracy"]) * 100, 2),
            round(float(summary["normalized_exact_match"]) * 100, 2),
            round(float(summary["token_f1"]) * 100, 2),
            None,
            None,
            None,
            None,
            (
                seconds_per_case(spec, len(rows))
                or next(
                    (
                        existing.seconds_per_case
                        for existing in QUALITY_METRICS
                        if existing.backbone == backbone
                        and existing.mode == mode
                        and existing.model_size == spec.model_size
                        and existing.latent_steps == spec.latent_steps
                    ),
                    None,
                )
            ),
            state,
            spec.latent_steps,
            spec.model_size,
            thinking_mode=spec.thinking_mode,
        )
        if dataset_label_for_spec(spec) == "WSI-VQA":
            pathagent_gts = {
                row.question_id: [expand_letter(row.ground_truth.strip(), row.choices)]
                for row in rows
            }
            pathagent_res = {
                row.question_id: [expand_letter(row.prediction.strip(), row.choices)]
                for row in rows
            }
            with _pathagent_score_lock:
                score_future = _pathagent_score_executor.submit(
                    compute_coco_scores,
                    pathagent_gts,
                    pathagent_res,
                )
                sample_scores, _ = score_future.result()
            metric = replace(
                metric,
                bleu_1=round(float(sample_scores["BLEU_1"]) * 100, 2),
                bleu_2=round(float(sample_scores["BLEU_2"]) * 100, 2),
                bleu_3=round(float(sample_scores["BLEU_3"]) * 100, 2),
                bleu_4=round(float(sample_scores["BLEU_4"]) * 100, 2),
                meteor=round(float(sample_scores["METEOR"]) * 100, 2),
                rouge_l=round(float(sample_scores["ROUGE_L"]) * 100, 2),
            )
    with _live_quality_lock:
        _live_quality_cache[spec.name] = (completed, metric)
    return metric


def cached_or_schedule_quality(
    backbone: str,
    mode: str,
    state: str,
    spec: JobSpec,
    processed: int,
) -> QualityMetric | None:
    """Return the last full metric while refreshing a changed prefix asynchronously."""
    with _live_quality_lock:
        cached = _live_quality_cache.get(spec.name)
        if cached is not None and cached[0] == processed and cached[1].state == state:
            return cached[1]
        if (
            cached is not None
            and processed < spec.target
            and processed - cached[0] < LIVE_QUALITY_REFRESH_CASES
        ):
            return cached[1]
        if spec.name in _live_quality_refreshing:
            return None if cached is None else cached[1]
        _live_quality_refreshing.add(spec.name)

    def refresh() -> None:
        try:
            with _live_quality_refresh_lock:
                live_full_quality(backbone, mode, state, spec)
        finally:
            with _live_quality_lock:
                _live_quality_refreshing.discard(spec.name)

    Thread(target=refresh, name=f"dashboard-metrics-{spec.name}", daemon=True).start()
    return None if cached is None else cached[1]


def uses_live_corpus_metrics(spec: JobSpec) -> bool:
    """Reserve CPU for active inference; corpus metrics run after completion."""
    return spec.queue_group in {
        "qwen-2b-local-matrix",
        "qwen-4b-local-matrix",
        "qwen-8b-local-matrix",
        "qwen-reallocation-pathology",
        "qwen-reallocation-pathology-context-matrix",
        "multipathqa-single-nothinking",
    }


def multipath_quality_metrics(
    spec: JobSpec,
    state: str,
    dataset_path: Path | None = None,
    processed: int | None = None,
) -> tuple[QualityMetric, ...]:
    """Compute the completed prefix accuracy for each MultiPathQA benchmark."""
    selected_dataset = dataset_path or multipath_dataset_for_spec(spec)
    if not selected_dataset.is_file():
        return ()
    cache_key: tuple[str, str, int] | None = None
    if processed is not None:
        cache_key = (str(spec.root), state, processed)
    if cache_key is not None and cache_key in _multipath_quality_cache:
        return _multipath_quality_cache[cache_key]
    records = json.loads(selected_dataset.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        return ()
    predictions: dict[int, str] = {}
    elapsed_by_index: dict[int, float] = {}
    if spec.single_predictions:
        for dataset_index, row in single_prediction_rows(spec).items():
            prediction = row.get("prediction")
            if isinstance(prediction, str):
                predictions[dataset_index] = prediction
            elapsed = row.get("inference_time_sec")
            if isinstance(elapsed, (int, float)) and elapsed >= 0:
                elapsed_by_index[dataset_index] = float(elapsed)
    else:
        for case in latest_cases(spec):
            result = json.loads(case.result_path.read_text(encoding="utf-8"))
            answer = result.get("answer")
            prediction = answer.get("answer") if isinstance(answer, dict) else answer
            if isinstance(prediction, str):
                predictions[case.dataset_index] = prediction
            role_calls = result.get("role_calls")
            if isinstance(role_calls, list):
                elapsed_values: list[float] = []
                for call in role_calls:
                    if not isinstance(call, dict):
                        continue
                    elapsed = call.get("elapsed_seconds")
                    if isinstance(elapsed, (int, float)):
                        elapsed_values.append(float(elapsed))
                if elapsed_values:
                    total_elapsed = 0.0
                    for elapsed_value in elapsed_values:
                        total_elapsed += elapsed_value
                    elapsed_by_index[case.dataset_index] = total_elapsed

    correct_by_task: dict[str, dict[str, list[int]]] = {
        task: {} for task in MULTIPATH_BENCHMARKS
    }
    elapsed_by_task: dict[str, list[float]] = {
        task: [] for task in MULTIPATH_BENCHMARKS
    }
    for dataset_index, prediction in predictions.items():
        if not 0 <= dataset_index < len(records):
            continue
        record = records[dataset_index]
        if not isinstance(record, dict):
            continue
        task = record.get("Task")
        choices = record.get("Choice")
        ground_truth = record.get("Answer")
        if (
            not isinstance(task, str)
            or task not in correct_by_task
            or not isinstance(choices, list)
            or not all(isinstance(choice, str) for choice in choices)
            or not isinstance(ground_truth, str)
        ):
            continue
        prediction_eval = STRICT_METRICS.expand_letter(prediction.strip(), choices)
        ground_truth_eval = STRICT_METRICS.expand_letter(ground_truth.strip(), choices)
        correct_by_task[task].setdefault(ground_truth_eval, []).append(
            int(bool(STRICT_METRICS.acc_of_seq(choices, ground_truth_eval, prediction_eval)))
        )
        elapsed = elapsed_by_index.get(dataset_index)
        if elapsed is not None:
            elapsed_by_task[task].append(elapsed)

    mode = (
        "Single No Thinking"
        if spec.name.endswith("Single No Thinking")
        else "Latent base"
        if " Latent base · step " in spec.name
        else next(
            (
                label
                for label in ("Single", "VLMAS", "Latent base")
                if spec.name.endswith(label)
            ),
            spec.name,
        )
    )
    present_tasks = frozenset(
        record.get("Task") for record in records if isinstance(record, dict)
    )
    metrics = tuple(
        QualityMetric(
            backbone="Qwen",
            mode=mode,
            samples=sum(len(scores) for scores in scores_by_class.values()),
            total=(
                round(
                    sum(sum(scores) / len(scores) for scores in scores_by_class.values())
                    / len(scores_by_class)
                    * 100,
                    2,
                )
                if task in MULTIPATH_BALANCED_TASKS and scores_by_class
                else round(
                    sum(sum(scores) for scores in scores_by_class.values())
                    / sum(len(scores) for scores in scores_by_class.values())
                    * 100,
                    2,
                )
                if scores_by_class
                else None
            ),
            mcq=(
                round(
                    sum(sum(scores) / len(scores) for scores in scores_by_class.values())
                    / len(scores_by_class)
                    * 100,
                    2,
                )
                if task in MULTIPATH_BALANCED_TASKS and scores_by_class
                else round(
                    sum(sum(scores) for scores in scores_by_class.values())
                    / sum(len(scores) for scores in scores_by_class.values())
                    * 100,
                    2,
                )
                if scores_by_class
                else None
            ),
            open=None,
            exact=None,
            token_f1=None,
            bleu_1=None,
            bleu_2=None,
            bleu_3=None,
            bleu_4=None,
            seconds_per_case=(
                round(sum(elapsed_by_task[task]) / len(elapsed_by_task[task]), 2)
                if elapsed_by_task[task]
                else None
            ),
            state=state,
            latent_steps=spec.latent_steps,
            model_size=spec.model_size,
            dataset=f"MultiPathQA/{benchmark}",
        )
        for task, benchmark in MULTIPATH_BENCHMARKS.items()
        if task in present_tasks
        for scores_by_class in (correct_by_task[task],)
    )
    if cache_key is not None:
        for old_key in tuple(_multipath_quality_cache):
            if old_key[0] == str(spec.root):
                del _multipath_quality_cache[old_key]
        _multipath_quality_cache[cache_key] = metrics
    return metrics


def quality_dict(metric: QualityMetric) -> dict[str, str | int | float | None]:
    """Serialize a quality row with its latent-step value for UI filtering."""
    payload = asdict(metric)
    payload["score_label"] = (
        "BAcc."
        if metric.dataset in {"MultiPathQA/TCGA", "MultiPathQA/GTEx", "MultiPathQA/PANDA"}
        else "Acc."
        if metric.dataset.startswith("MultiPathQA/")
        else "MCQ Acc."
    )
    return payload


def cached_official_metrics() -> tuple[QualityMetric, ...]:
    """Load completed official-metric recomputations when available."""
    if not OFFICIAL_METRICS_CACHE.exists():
        return ()
    raw = json.loads(OFFICIAL_METRICS_CACHE.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return ()
    return tuple(QualityMetric(**item) for item in raw if isinstance(item, dict))


def workbook_quality_metrics() -> tuple[QualityMetric, ...]:
    """Load dashboard performance rows exclusively from the shared workbook."""
    if not PERFORMANCE_WORKBOOK.exists():
        return ()
    workbook = load_workbook(PERFORMANCE_WORKBOOK, read_only=True, data_only=True)
    sheet = workbook["Main Performance"]
    rows = sheet.iter_rows(values_only=True)
    headers = next(rows, ())
    columns = {str(value): index for index, value in enumerate(headers) if isinstance(value, str)}

    def value(row: tuple[object, ...], name: str) -> object | None:
        index = columns.get(name)
        return row[index] if index is not None and index < len(row) else None

    def number(row: tuple[object, ...], name: str) -> float | None:
        raw = value(row, name)
        return float(raw) if isinstance(raw, (int, float)) else None

    metrics: list[QualityMetric] = []
    for row in rows:
        samples = value(row, "Samples")
        backbone = value(row, "Backbone")
        mode = value(row, "Method")
        if not isinstance(samples, int) or samples <= 0:
            continue
        if not isinstance(backbone, str) or not isinstance(mode, str):
            continue
        dataset = value(row, "Dataset")
        parameter = value(row, "Parameter")
        latent_steps = value(row, "Latent step")
        metrics.append(QualityMetric(
            backbone, mode, samples,
            number(row, "WSI Total"), number(row, "WSI Closed"), number(row, "WSI Open"),
            None, None, number(row, "BLEU-1"), None, None, number(row, "BLEU-4"),
            seconds_per_case=number(row, "Seconds / question"),
            state=str(value(row, "Status") or "완료"),
            latent_steps=latent_steps if isinstance(latent_steps, int) else None,
            model_size=parameter if isinstance(parameter, str) else None,
            meteor=number(row, "METEOR"), rouge_l=number(row, "ROUGE-L"),
            dataset=dataset if isinstance(dataset, str) else "WSI-VQA",
            thinking_mode=(
                str(value(row, "Thinking mode"))
                if isinstance(value(row, "Thinking mode"), str)
                else "No Thinking" if mode == "Single No Thinking" else "Thinking"
            ),
        ))
    return tuple(metrics)


def is_historical_qwen_baseline(metric: QualityMetric) -> bool:
    """Exclude pre-canonical Qwen baselines from the current Aug 29–30 table."""
    return (
        metric.backbone == "Qwen"
        and metric.model_size in {"2B", "4B", "8B"}
        and metric.mode in {"VLMAS", "Latent base", "Pruning V3"}
    )


def quality_specs_for_dashboard(
    current_specs: tuple[JobSpec, ...], baseline_specs: tuple[JobSpec, ...]
) -> tuple[JobSpec, ...]:
    """Keep completed GTEx step-matrix runs in the performance table."""
    multipath_specs = (
        (*GTEX_BASE_RESULT_SPECS, *MULTIPATH_REMAINING_RESULT_SPECS)
        if any(
            spec.queue_group in {
                "multipathqa-qwen",
                "multipathqa-remaining-base-allsteps",
            }
            for spec in current_specs
        )
        else ()
    )
    return tuple(
        {
            spec.name: spec
            for spec in (
                *current_specs,
                *baseline_specs,
                *multipath_specs,
            )
        }.values()
    )


def status_payload() -> bytes:
    """Build the live dashboard response."""
    global _status_cache, _status_cache_at
    with _status_cache_lock:
        if _status_cache and monotonic() - _status_cache_at < STATUS_CACHE_SECONDS:
            return _status_cache
        commands = command_lines()
        active_queue_groups = frozenset(
            spec.queue_group
            for spec in JOBS
            if spec.queue_group is not None
            and any(
                is_python_process_line(line) and process_marker_matches(spec, line)
                for line in commands.splitlines()
            )
        )
        visible_queue_groups = active_queue_groups
        if not active_queue_groups:
            visible_queue_groups |= frozenset({
                "qwen-pruning-v3-priority",
                "qwen-reallocation-pathology-context-matrix",
            })
        selected_specs = tuple(
            spec
            for spec in JOBS
            if spec.queue_group in {
                "qwen-2b-local-matrix",
                "qwen-4b-local-matrix",
                "qwen-8b-local-matrix",
                "qwen-pruning-v3-priority",
                "qwen-pruning-v3-navkv-matrix",
                "qwen-reallocation-pathology",
                "qwen-reallocation-pathology-context-matrix",
                "slidebench-bcnb-qwen2b",
                "slidebench-bcnb-qwen4b",
                "multipathqa-qwen",
                "multipathqa-single-nothinking",
                "multipathqa-remaining-base-allsteps",
            }
            and spec.queue_group in visible_queue_groups
            and not (spec.queue_group == "multipathqa-qwen" and spec.target == 736)
            and (
                spec.latent_steps is not None
                or spec.queue_group in {
                    "slidebench-bcnb-qwen2b",
                    "slidebench-bcnb-qwen4b",
                    "multipathqa-qwen",
                    "multipathqa-remaining-base-allsteps",
                    "qwen-2b-local-matrix",
                    "qwen-4b-local-matrix",
                    "qwen-8b-local-matrix",
                }
            )
        )
        current_specs = selected_specs or JOBS
        current_statuses = tuple(job_status(spec, commands) for spec in current_specs)
        all_jobs = attach_queue_estimates(current_specs, current_statuses)
        current_status_by_name = {
            spec.name: status
            for spec, status in zip(current_specs, all_jobs, strict=True)
        }
        jobs = all_jobs
        baseline_specs = tuple(
            spec
            for spec in JOBS
            if spec.name in RUNNING_QUALITY_LABELS
            and spec.model_size in {"2B", "4B", "8B"}
            and (
                spec.name in {
                    "Qwen 2B Single No Thinking",
                    "Qwen 4B Single No Thinking",
                    "Qwen 8B Single No Thinking",
                }
                or
                spec.name.endswith("Single · Reasoner-thinking + Answerer JSON · 735")
                or (
                    spec.name in {"Qwen 2B VLMAS", "Qwen 4B VLMAS", "Qwen 8B VLMAS"}
                    and CANONICAL_MATRIX_ROOT in spec.root.parents
                )
                or spec.name.startswith("Qwen ")
                and " Latent base · step " in spec.name
                and "no candidate" not in spec.name
            )
        )
        quality_specs = quality_specs_for_dashboard(current_specs, baseline_specs)
        quality_statuses = tuple(
            current_status_by_name[spec.name]
            if spec.name in current_status_by_name
            else job_status(spec, commands)
            for spec in quality_specs
        )
        quality_status_by_key = {
            (
                RUNNING_QUALITY_LABELS[spec.name][0],
                RUNNING_QUALITY_LABELS[spec.name][1],
                spec.model_size,
                spec.latent_steps,
            ): status
            for status, spec in zip(quality_statuses, quality_specs, strict=True)
            if spec.name in RUNNING_QUALITY_LABELS
        }
        prefix_quality: tuple[QualityMetric, ...] = ()
        dynamic_quality: list[QualityMetric] = []
        dynamic_names = frozenset(spec.name for spec in quality_specs)
        # Refresh quality asynchronously so progress/GPU polling stays responsive.
        for status, spec in zip(quality_statuses, quality_specs, strict=True):
            if (
                status.processed > 0
                and status.state in {"running", "completed"}
                and dataset_label_for_spec(spec).startswith("MultiPathQA/")
            ):
                dynamic_quality.extend(
                    multipath_quality_metrics(spec, status.state, processed=status.processed)
                )
                continue
            if (
                status.processed > 0
                and status.state in {"running", "completed"}
                and status.name in dynamic_names
                and status.name in RUNNING_QUALITY_LABELS
                and uses_live_corpus_metrics(spec)
            ):
                backbone, mode = RUNNING_QUALITY_LABELS[status.name]
                state = status.state
                metric = cached_or_schedule_quality(
                    backbone, mode, state, spec, status.processed
                )
                if metric is not None:
                    dynamic_quality.append(metric)
                elif spec.single_predictions:
                    live = live_metric(spec)
                    if live.completed > 0:
                        dynamic_quality.append(
                            prefix_quality_metric(backbone, mode, state, spec, live)
                        )
        quality_rows: dict[tuple[object, ...], QualityMetric] = {}
        for metric in (*workbook_quality_metrics(), *dynamic_quality):
            key = (
                metric.dataset,
                metric.backbone,
                metric.mode,
                metric.model_size,
                metric.latent_steps,
                metric.thinking_mode,
            )
            existing = quality_rows.get(key)
            if existing is not None:
                metric = replace(
                    metric,
                    bleu_1=metric.bleu_1 if metric.bleu_1 is not None else existing.bleu_1,
                    bleu_2=metric.bleu_2 if metric.bleu_2 is not None else existing.bleu_2,
                    bleu_3=metric.bleu_3 if metric.bleu_3 is not None else existing.bleu_3,
                    bleu_4=metric.bleu_4 if metric.bleu_4 is not None else existing.bleu_4,
                    meteor=metric.meteor if metric.meteor is not None else existing.meteor,
                    rouge_l=metric.rouge_l if metric.rouge_l is not None else existing.rouge_l,
                )
            quality_rows[key] = metric
        legacy_mode_order = ("Single", "VLMAS", "Latent base", "Pruning", "Reallocation", "Both")
        groups = (
            ("Qwen", "2B"),
            ("Qwen", "4B"),
            ("Qwen", "8B"),
            ("OctoMed", "7B"),
            ("InternVL", None),
            ("Lingshu", None),
        )
        ordered_quality: list[QualityMetric] = []
        for backbone, model_size in groups:
            if backbone == "Qwen" and model_size in {"2B", "4B", "8B"}:
                modes = (
                    "Single", "Single · matched base", "Single No Thinking", "VLMAS",
                    "Latent base", "Pruning V3",
                )
            elif backbone == "Lingshu":
                modes = (
                    "Single", "VLMAS", "Latent base", "Pruning V3",
                    "Reallocation", "Reallocation V2", "Both",
                )
            elif backbone == "InternVL":
                modes = (
                    "Single", "Single 3.5", "VLMAS", "VLMAS 3.5", "Latent base", "Pruning V3", "Pruning",
                    "Reallocation", "Both",
                )
            else:
                modes = legacy_mode_order
            for mode in modes:
                latent_step_values = (
                    (None,)
                    if mode in {"Single", "Single No Thinking", "Single 3.5", "VLMAS", "VLMAS 3.5"}
                    else (5,)
                    if mode in {"Latent base · 8 patches", "Latent base · 25 patches"}
                    else (5,)
                    if mode == "Pruning V4"
                    else (5,)
                    if mode in {
                        "Pruning V4 · 8 patches",
                        "Pruning V4.1 · QK probe · gate 50",
                        "Pruning V4.1 · QK probe · 8 patches",
                        "Pruning V4.1 · QK probe · TAG answer",
                        "Pruning V4 · 25 patches",
                    }
                    else (5,)
                    if mode in {"Pruning V5", "Pruning V6"}
                    else (5,)
                    if mode in {
                        "Reallocation Pathology",
                        "Reallocation Pathology V2 · gate 50",
                        "Reallocation Pathology V2",
                        "Reallocation Pathology V3 · gate 50",
                        "Reallocation Pathology Context · gate 50",
                    }
                    else (5, 10, 20, 30)
                    if backbone == "Qwen" and mode in {
                        "Pruning V3", "Reallocation Pathology Context",
                    }
                    else (5, 10, 20, 30)
                    if backbone == "Qwen" and mode == "Latent base"
                    else ((5, 10, 20) if model_size == "2B" else (5, 10, 20))
                )
                for latent_steps in latent_step_values:
                    row_model_size = (
                        "8B"
                        if backbone == "InternVL" and mode in {"Single 3.5", "VLMAS 3.5"}
                        else model_size
                    )
                    key = (backbone, mode, row_model_size, latent_steps)
                    metric_key = (
                        "WSI-VQA",
                        backbone,
                        mode,
                        row_model_size,
                        latent_steps,
                        "No Thinking",
                    )
                    run_status = quality_status_by_key.get(key)
                    ordered_quality.append(
                        quality_rows.get(
                            metric_key,
                            QualityMetric(
                                backbone,
                                mode,
                                run_status.processed if run_status is not None else 0,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                state=(
                                    "평가 중"
                                    if run_status is not None and run_status.state == "completed"
                                    else run_status.state
                                    if run_status is not None
                                    else "미실행"
                                ),
                                latent_steps=latent_steps,
                                model_size=row_model_size,
                            ),
                        )
                    )
        ordered_quality.extend(
            metric
            for metric in dynamic_quality
            if metric.thinking_mode != "No Thinking"
        )
        for status, spec in zip(quality_statuses, quality_specs, strict=True):
            if spec.queue_group in {
                "multipathqa-qwen",
                "multipathqa-gtex-base-allsteps",
                "multipathqa-remaining-base-allsteps",
            }:
                ordered_quality.extend(
                    multipath_quality_metrics(
                        spec, status.state, processed=status.processed
                    )
                )
        for spec in GTEX_RESULT_SPECS:
            success, failure = count_cases(spec)
            processed = min(success + failure, spec.target)
            ordered_quality.extend(
                multipath_quality_metrics(
                    spec,
                    "completed" if processed >= spec.target else "stopped",
                    processed=processed,
                )
            )
        # The workbook is the durable performance source.  The compact legacy
        # ordering above is only a presentation preference; it must never drop
        # a completed row from another dataset or thinking configuration.
        def quality_key(metric: QualityMetric) -> tuple[object, ...]:
            return (
                metric.dataset,
                metric.backbone,
                metric.mode,
                metric.model_size,
                metric.latent_steps,
                metric.thinking_mode,
            )

        persisted_quality = {
            quality_key(metric): metric for metric in quality_rows.values()
        }
        for metric in ordered_quality:
            key = quality_key(metric)
            existing = persisted_quality.get(key)
            if existing is not None:
                metric = replace(
                    metric,
                    bleu_1=metric.bleu_1 if metric.bleu_1 is not None else existing.bleu_1,
                    bleu_2=metric.bleu_2 if metric.bleu_2 is not None else existing.bleu_2,
                    bleu_3=metric.bleu_3 if metric.bleu_3 is not None else existing.bleu_3,
                    bleu_4=metric.bleu_4 if metric.bleu_4 is not None else existing.bleu_4,
                    meteor=metric.meteor if metric.meteor is not None else existing.meteor,
                    rouge_l=metric.rouge_l if metric.rouge_l is not None else existing.rouge_l,
                )
            persisted_quality[key] = metric
        matrix_datasets = (
            "WSI-VQA",
            "MultiPathQA/ExpertVQA",
            "MultiPathQA/SlideBench",
            "MultiPathQA/TCGA",
            "MultiPathQA/GTEx",
        )
        matrix_modes = (
            "Single",
            "Single · matched base",
            "Single No Thinking",
            "VLMAS",
            "Latent base",
            "Pruning V3",
        )
        ordered_quality = []
        for dataset in matrix_datasets:
            for model_size in ("2B", "4B", "8B"):
                for mode in matrix_modes:
                    steps = (
                        (None,)
                        if mode in {
                            "Single",
                            "Single · matched base",
                            "Single No Thinking",
                            "VLMAS",
                        }
                        else (5, 10, 20, 30)
                    )
                    for latent_step in steps:
                        matches = [
                            metric
                            for metric in persisted_quality.values()
                            if metric.dataset == dataset
                            and metric.backbone == "Qwen"
                            and metric.model_size == model_size
                            and metric.mode == mode
                            and metric.latent_steps == latent_step
                        ]
                        if matches:
                            ordered_quality.append(max(matches, key=lambda metric: metric.samples))
                        else:
                            pending_nothinking_multipath = (
                                dataset.startswith("MultiPathQA/")
                                and mode == "Single No Thinking"
                            )
                            ordered_quality.append(QualityMetric(
                                "Qwen", mode, 0,
                                None, None, None, None, None, None, None, None, None,
                                state="대기" if pending_nothinking_multipath else "미실행",
                                latent_steps=latent_step,
                                model_size=model_size,
                                dataset=dataset,
                                thinking_mode=(
                                    "Thinking"
                                    if mode in {"Single", "Single · matched base"}
                                    else "No Thinking"
                                ),
                            ))
        sync_status = gtex_sync_status()
        sync_jobs = (
            [{**asdict(sync_status), "dataset": "MultiPathQA/GTEx"}]
            if sync_status.state != "completed"
            else []
        )
        body = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "gpus": [
                asdict(status)
                for status in cached_gpu_statuses(all_jobs, current_specs)
            ],
            "jobs": [
                {
                    **asdict(status),
                    "dataset": dataset_label_for_spec(spec),
                }
                for status, spec in zip(jobs, current_specs, strict=True)
            ] + sync_jobs,
            "live_metrics": [
                asdict(LiveMetric(
                    f"{metric.backbone}{(' ' + metric.model_size) if metric.model_size else ''} {metric.mode}",
                    metric.samples,
                    metric.total,
                    metric.mcq,
                    metric.open,
                    metric.exact,
                    metric.token_f1,
                    metric.bleu_1,
                    metric.bleu_2,
                    metric.bleu_3,
                    metric.bleu_4,
                ))
                for metric in ordered_quality
            ],
            "quality_metrics": [quality_dict(metric) for metric in ordered_quality],
        }
        _status_cache = json.dumps(body, ensure_ascii=False).encode()
        _status_cache_at = monotonic()
        return _status_cache


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve the dashboard page and its local status API."""

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.read_bytes())
        elif path == "/api/status":
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(status_payload())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: str) -> None:
        """Keep routine polling out of the terminal."""


def main() -> None:
    """Run the local read-only dashboard server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    with ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
