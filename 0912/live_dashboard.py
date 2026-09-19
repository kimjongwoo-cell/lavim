# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# noqa: SIZE_OK — the dashboard is deployed as one standalone local script; splitting its handlers is outside this benchmark-mapping change.
# ─── How to run ───
# uv run 0912/live_dashboard.py --port 8767

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean
from threading import Lock
from time import monotonic
from typing import Final

ROOT: Final = Path(__file__).resolve().parent
RUNS: Final = ROOT / "runs"
NAV3_RUNS: Final = (
    RUNS / "20260914_nav3_per_root_visual_smoke_gpu7",
    RUNS / "20260914_nav3_per_root_visual_smoke_id1_gpu7",
    RUNS / "20260914_nav3_per_root_visual_smoke_id2_gpu8",
)
NAV3_ASSETS: Final = ROOT / "nav3_dashboard_assets"
NAV1_RUN_NAMES: Final = frozenset(
    {
        "base_sdpa_gpu6",
        "pruning_b_relay2_full128_part0_63_gpu6_0912",
        "pruning_b_relay2_full128_part64_127_gpu8_0912",
    }
)
EXPERT_DATASET: Final = (
    ROOT.parent
    / "../datasets/MultiPathQA/ready_wsivqa/full_no_panda/tcga_expert_vqa.json"
).resolve()
ALL_AGENT_RUNS: Final = (
    ROOT.parent / "sender_relay_exp/runs/pruning_b_all_agents_prompt1",
    ROOT.parent / "sender_relay_exp/runs/pruning_b_all_agents_prompt1_resume1",
    ROOT.parent / "sender_relay_exp/runs/pruning_b_nav_reasoner_relay3_prompt1",
    ROOT.parent / "sender_relay_exp/runs/pruning_b_reasoner_relay3_prompt1",
    ROOT.parent / "sender_relay_exp/runs/pruning_b_reasoner_relay2_prompt1",
    ROOT.parent / "sender_relay_exp/runs/pruning_b_reasoner_triple_relay2_prompt1",
    ROOT.parent / "sender_relay_exp/runs/pruning_b_reasoner_dual_triple_paired_prompt1",
    ROOT.parent / "sender_relay_exp/runs/pruning_b_reasoner_context_relay3_prompt1",
)
PAGE: Final = ROOT / "live_dashboard.html"
PAIRED_RUNS: Final = ROOT.parent / "sender_relay_exp/runs/pruning_b_reasoner_dual_triple_paired_prompt1"
BENCHMARK_RUNS: Final = ROOT.parent / "sender_relay_exp/runs"
PATHOLOGY_SPARSEVLM_RUNS: Final = (
    BENCHMARK_RUNS / "sparsevlm_pathology_vs_both_gtex20_rank75_20260915",
    BENCHMARK_RUNS / "sparsevlm_pathology_vs_both_gtex190_rank75_20260915",
)
TARGETS: Final = {
    "tcga_expert_vqa": 128,
    "gtex": 190,
    "tcga_slidebench": 197,
    "tcga": 221,
    "panda": 196,
}
BALANCED: Final = frozenset({"gtex", "tcga", "panda"})
VISIBLE_VARIANTS: Final = frozenset(
    {
        "base",
        "pruning_b",
        "latent_kv_relay2",
        "pruning_b_latent_kv_relay2_dual",
        "pruning_b_reasoner_context_relay3",
        "nav3_base",
        "nav3_both",
        "nav3_both_keep50",
        "pruning_sparsevlm_pathology_nav3_latent_kv_relay2_dual",
        "nav3_x20_x40",
        "nav3_vlmas",
        "nav1_base",
        "nav1_both",
        "pruning_eadaprune_pathology_latent_kv_relay2_dual",
        "pruning_atp_sap_pathology_latent_kv_relay2_dual",
        "pruning_sparsevlm_pathology_latent_kv_relay2_dual",
    }
)
EXPERT_PROMPT1_PREFIXES: Final = {
    "base": ("nav2v2_base_expert_",),
    "pruning_b": ("nav2v2_pruning_b_expert_",),
    "latent_kv_relay2": ("nav2v2_relay2_expert_",),
    "pruning_a_latent_kv_relay2_dual": (
        "nav2_prompt1_pruning_a_dual_both_tcga_expert_vqa_",
    ),
    "pruning_b_latent_kv_relay2_dual": (
        "nav2v2_factorial_pruning_relay_dual_expert_",
    ),
    "pruning_sparsevlm_latent_kv_relay2_dual": (
        "nav2_prompt1_qgap_dual_tcga_expert_vqa_",
    ),
}
COLLECT_CACHE_SECONDS: Final = 30.0
_collect_cache: list[Row] | None = None
_collect_cache_at = 0.0
_collect_nav3_signature: tuple[int, int] | None = None
_collect_active_keys: frozenset[tuple[str, str]] | None = None
_collect_lock = Lock()


@dataclass(frozen=True, slots=True)
class Row:
    dataset: str
    variant: str
    completed: int
    target: int
    score: float | None
    score_name: str
    correct: int
    seconds_per_case: float | None
    state: str


@dataclass(frozen=True, slots=True)
class PatchOrigin:
    patch_id: str
    magnification: int
    root_id: int
    candidate_id: int
    tissue_fraction: float
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SelectionCase:
    dataset_index: int
    question: str
    slide_width: int
    slide_height: int
    thumbnail_url: str
    evidence_board_url: str
    x5_patches: tuple[PatchOrigin, ...]
    x20_patches: tuple[PatchOrigin, ...]


def _patch_origin(raw: dict[str, object]) -> PatchOrigin:
    box = raw["box"]
    if not isinstance(box, dict):
        raise TypeError("patch box must be an object")
    return PatchOrigin(
        patch_id=str(raw["patch_id"]),
        magnification=int(raw["magnification"]),
        root_id=int(raw["root_id"]),
        candidate_id=int(raw["candidate_id"]),
        tissue_fraction=float(raw["tissue_fraction"]),
        x=int(box["x"]),
        y=int(box["y"]),
        width=int(box["width"]),
        height=int(box["height"]),
    )


def collect_nav3_selection_cases() -> tuple[SelectionCase, ...]:
    """Expose audited Nav3 patch origins without scanning unrelated runs."""
    questions = {
        index: str(item["Question"])
        for index, item in enumerate(json.loads(EXPERT_DATASET.read_text()))
    }
    cases: list[SelectionCase] = []
    result_paths = sorted(
        result_path
        for run_root in NAV3_RUNS
        for result_path in run_root.glob("*/result.json")
    )
    for result_path in result_paths:
        payload = json.loads(result_path.read_text())
        index = int(payload["dataset_index"])
        patches = tuple(_patch_origin(raw) for raw in payload["patches"])
        thumbnail_name = f"case_{index:03d}_thumbnail.jpg"
        dimensions = json.loads(
            (NAV3_ASSETS / f"case_{index:03d}_thumbnail.json").read_text()
        )
        cases.append(
            SelectionCase(
                dataset_index=index,
                question=questions[index],
                slide_width=int(dimensions["width"]),
                slide_height=int(dimensions["height"]),
                thumbnail_url=f"/nav3-assets/{thumbnail_name}",
                evidence_board_url=f"/nav3-board/{index}",
                x5_patches=tuple(patch for patch in patches if patch.magnification == 5),
                x20_patches=tuple(patch for patch in patches if patch.magnification == 20),
            )
        )
    return tuple(cases)


def normalized(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def dataset_from_slide(slide_id: str) -> str | None:
    for name in TARGETS:
        if slide_id.startswith(f"{name}__"):
            return name
    return None


def variant_label(variant: str) -> str:
    labels = {
        "base": "Nav2 Base",
        "pruning_a": "Pruning-A",
        "pruning_b": "Pruning-B",
        "latent_kv_relay2": "Relay2",
        "latent_kv_relay2_dual": "Dual Relay",
        "pruning_a_latent_kv_relay2_dual": "Pruning-A + Dual Relay",
        "pruning_b_latent_kv_relay2_dual": "Nav2 Both (Pruning-B + Relay2)",
        "pruning_b_reasoner_context_relay3": "Both (Pruning-B + Relay3 Context)",
        "nav3_base": "Nav3 Base",
        "nav3_both": "Nav3 Both (Pruning-B + Dual Relay2)",
        "nav3_both_keep50": "Nav3 Both (Pruning-B · Visual KV 50% 보존 + Dual Relay2)",
        "nav3_x20_x40": "Nav3 Base (x20 + x40, 20-case)",
        "nav3_vlmas": "Nav3 Text VLM-MAS (20-case)",
        "nav1_base": "Nav1 Base",
        "nav1_both": "Nav1 Both (Pruning-B + Relay2)",
        "pruning_sparsevlm_latent_kv_relay2_dual": "QGAP + Dual Both",
        "pruning_sparsevlm_pathology_latent_kv_relay2_dual": (
            "Pathology-context SparseVLM + Dual Relay2"
        ),
        "pruning_sparsevlm_pathology_nav3_latent_kv_relay2_dual": (
            "Nav3 Pathology-context SparseVLM + Dual Relay2"
        ),
        "pruning_eadaprune_pathology_latent_kv_relay2_dual": (
            "Nav3 E-AdaPrune Pathology + Dual Relay2"
        ),
        "pruning_atp_sap_pathology_latent_kv_relay2_dual": (
            "Nav3 ATP-SAP Pathology + Dual Relay2"
        ),
        "pruning_b_latent_kv_relay2_all_agents": "Pruning-B + Relay2 All Agents",
        "pruning_b_nav_reasoner_relay3_dual": "Pruning-B Nav/Reasoner + Dual Relay3",
        "pruning_b_reasoner_relay3_dual": "Pruning-B Reasoner-only + Dual Relay3",
        "pruning_b_reasoner_relay2_dual": "Pruning-B Reasoner-only + Dual Relay2",
        "pruning_b_reasoner_relay2_triple": "Pruning-B Reasoner-only + Triple Relay2",
        "pruning_b_reasoner_relay2_dual_paired": "Pruning-B Reasoner-only + Dual Relay2 (paired rerun)",
        "pruning_b_reasoner_relay2_triple_paired": "Pruning-B Reasoner-only + Triple Relay2 (paired rerun)",
    }
    return labels.get(variant, variant)


def nav3_variant_for_run_name(run_name: str, variant: str) -> str:
    """Normalize Nav3 campaign prefixes to the dashboard's Nav3 rows."""
    # noqa: SIZE_OK — this standalone dashboard keeps run-name routing colocated.
    if run_name.startswith("nav1_prompt1_base_"):
        return "nav1_base"
    if run_name.startswith(("gtex_base_12patch_", "tcga_slidebench_base_12patch_")):
        return "nav1_base"
    if run_name.startswith("nav3_prompt1_eadaprune_pathology_dual_"):
        return "pruning_eadaprune_pathology_latent_kv_relay2_dual"
    if run_name.startswith("nav3_prompt1_atp_sap_pathology_dual_"):
        return "pruning_atp_sap_pathology_latent_kv_relay2_dual"
    if run_name.startswith("nav3_prompt1_pathology_dual_"):
        return "pruning_sparsevlm_pathology_nav3_latent_kv_relay2_dual"
    if run_name.startswith("nav3_prompt1_pruning_b_keep50_dual_"):
        return "nav3_both_keep50"
    if run_name.startswith("nav3_prompt1_pruning_b_both_"):
        return "nav3_both"
    if run_name.startswith("nav3_prompt1_vlmas_panda20_"):
        return "nav3_vlmas"
    if run_name.startswith("nav3_x20x40_prompt1_base_panda_"):
        return "nav3_x20_x40"
    if run_name.startswith("nav3_prompt1_base_"):
        return "nav3_base"
    if run_name.startswith("nav3_prompt1_both_"):
        return "nav3_both"
    return variant


def active_run_keys() -> frozenset[tuple[str, str]]:
    """Return dashboard keys backed by a currently live pipeline process."""
    active: set[tuple[str, str]] = set()
    for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            arguments = [
                raw.decode(errors="replace")
                for raw in cmdline_path.read_bytes().split(b"\0")
                if raw
            ]
        except OSError:
            continue
        joined = " ".join(arguments)
        if not any(
            module in joined
            for module in (
                "wsi_latentmas.pipeline.latent_mas",
                "wsi_latentmas.pipeline.text_mas",
                "vision_text_mas.latent_hf_ablation_cli",
                "pruning_b_keep50_override.py",
            )
        ):
            continue

        def option(name: str) -> str | None:
            try:
                return arguments[arguments.index(name) + 1]
            except (ValueError, IndexError):
                return None

        dataset_path = option("--dataset")
        output_root = option("--output-root")
        if dataset_path is None or output_root is None:
            continue
        dataset = Path(dataset_path).stem
        run_name = Path(output_root).name
        variant = nav3_variant_for_run_name(
            run_name, option("--variant") or "base"
        )
        active.add((dataset, variant_label(variant)))
    return frozenset(active)


def paired_run_started(dataset: str, variant: str) -> bool:
    """A shard manifest is written at startup, before its first case finishes."""
    return any(
        manifest.exists()
        for manifest in (PAIRED_RUNS / dataset).rglob("run_manifest.json")
        if json.loads(manifest.read_text()).get("variant") == variant
    )


def pathology_context_run_started(dataset: str) -> bool:
    """Report active Relay3 only when its output contains a real run manifest."""
    root = ROOT.parent / "sender_relay_exp/runs/pruning_b_reasoner_context_relay3_prompt1"
    for manifest in (root / dataset).rglob("run_manifest.json"):
        try:
            if json.loads(manifest.read_text()).get("variant") == (
                "pruning_b_reasoner_context_relay3"
            ):
                return True
        except (OSError, json.JSONDecodeError):
            continue
    return False


def nav3_run_started(dataset: str, label: str) -> bool:
    """Match an exact Nav3 dataset name rather than a shared tcga prefix."""
    return any(
        BENCHMARK_RUNS.glob(
            f"nav3_prompt1_{label}_{dataset}_gpu[78]_*/run_manifest.json"
        )
    )


def nav3_adaptive_pruner_started(dataset: str, method: str) -> bool:
    """Detect an adaptive-pruner run manifest across the four requested GPUs."""
    return any(
        BENCHMARK_RUNS.glob(
            f"nav3_prompt1_{method}_pathology_dual_{dataset}_gpu[0-3]_*/run_manifest.json"
        )
    )


def is_prompt1_source(run_name: str, dataset: str, variant: str) -> bool:
    """Exclude prompt experiments that reused the same manifest variant name."""
    if variant == "pruning_eadaprune_pathology_latent_kv_relay2_dual":
        return (
            run_name.startswith("nav3_prompt1_eadaprune_pathology_dual_")
            and "_smoke_" not in run_name
        )
    if variant == "pruning_atp_sap_pathology_latent_kv_relay2_dual":
        return (
            run_name.startswith("nav3_prompt1_atp_sap_pathology_dual_")
            and "_smoke_" not in run_name
        )
    if variant == "nav3_both_keep50":
        return run_name.startswith("nav3_prompt1_pruning_b_keep50_dual_")
    if variant in {"nav3_base", "nav3_both"}:
        return run_name.startswith("nav3_prompt1_")
    if variant == "nav3_x20_x40":
        return run_name.startswith("nav3_x20x40_prompt1_base_panda_")
    if variant == "nav3_vlmas":
        return run_name.startswith("nav3_prompt1_vlmas_panda20_")
    if variant in {"nav1_base", "nav1_both"}:
        return (
            run_name in NAV1_RUN_NAMES
            or run_name.startswith("nav1_prompt1_base_")
            or run_name.startswith(("gtex_base_12patch_", "tcga_slidebench_base_12patch_"))
        )
    if variant in {
        "pruning_b_latent_kv_relay2_all_agents",
        "pruning_sparsevlm_pathology_latent_kv_relay2_dual",
        "pruning_sparsevlm_pathology_nav3_latent_kv_relay2_dual",
        "pruning_eadaprune_pathology_latent_kv_relay2_dual",
        "pruning_atp_sap_pathology_latent_kv_relay2_dual",
        "pruning_b_nav_reasoner_relay3_dual",
        "pruning_b_reasoner_relay3_dual",
        "pruning_b_reasoner_relay2_dual",
        "pruning_b_reasoner_relay2_triple",
        "pruning_b_reasoner_relay2_dual_paired",
        "pruning_b_reasoner_relay2_triple_paired",
        "pruning_b_reasoner_context_relay3",
    }:
        return True
    if dataset == "tcga_expert_vqa":
        return run_name.startswith(EXPERT_PROMPT1_PREFIXES.get(variant, ()))
    if dataset in {"gtex", "tcga_slidebench"} and variant in {
        "pruning_b",
        "latent_kv_relay2",
    }:
        return run_name.startswith(f"{dataset}_{variant}_12patch_")
    if dataset == "tcga_slidebench" and variant == "base":
        return run_name.startswith("tcga_slidebench_base_12patch_")
    return "nav2" in run_name and "prompt2" not in run_name


def collect_uncached() -> list[Row]:
    grouped: dict[tuple[str, str], dict[int, tuple[float, str, str]]] = {}
    timings: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
    prompt1_roots = (
        path
        for path in RUNS.iterdir()
        if path.is_dir()
        and (
            ("nav2" in path.name and "prompt2" not in path.name)
            or path.name.startswith("sparsevlm_pathology_vs_both_gtex")
            or path.name.startswith("tcga_slidebench_base_12patch_")
            or path.name.startswith("gtex_pruning_b_12patch_")
            or path.name.startswith("gtex_latent_kv_relay2_12patch_")
            or path.name.startswith("tcga_slidebench_pruning_b_12patch_")
            or path.name.startswith("tcga_slidebench_latent_kv_relay2_12patch_")
        )
    )
    local_manifests = (
        manifest
        for prompt1_root in prompt1_roots
        for manifest in prompt1_root.rglob("run_manifest.json")
    )
    all_agent_manifests = (
        manifest
        for root in ALL_AGENT_RUNS
        if root.exists()
        for manifest in root.rglob("run_manifest.json")
    )
    nav3_manifests = (
        *BENCHMARK_RUNS.glob("nav3_prompt1_*/run_manifest.json"),
        *BENCHMARK_RUNS.glob("nav3_x20x40_prompt1_*/run_manifest.json"),
        *BENCHMARK_RUNS.glob("nav3_prompt1_vlmas_panda20_*/run_manifest.json"),
    )
    pathology_sparsevlm_manifests = (
        manifest
        for root in PATHOLOGY_SPARSEVLM_RUNS
        if root.exists()
        for manifest in root.rglob("run_manifest.json")
    )
    nav1_manifests = (
        *RUNS.glob("nav1_prompt1_base_*/run_manifest.json"),
        *RUNS.glob("gtex_base_12patch_*/run_manifest.json"),
        *RUNS.glob("tcga_slidebench_base_12patch_*/run_manifest.json"),
        *(RUNS / run_name / "run_manifest.json" for run_name in NAV1_RUN_NAMES),
    )
    for manifest_path in (
        *local_manifests,
        *all_agent_manifests,
        *nav3_manifests,
        *nav1_manifests,
        *pathology_sparsevlm_manifests,
    ):
        run_dir = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        variant = nav3_variant_for_run_name(
            run_dir.name, str(manifest.get("variant", "unknown"))
        )
        if run_dir.name == "base_sdpa_gpu6":
            variant = "nav1_base"
        elif run_dir.name.startswith("pruning_b_relay2_full128_part"):
            variant = "nav1_both"
        if variant not in VISIBLE_VARIANTS:
            continue
        source_name = run_dir.name
        for parent in run_dir.parents:
            if parent == RUNS.parent:
                break
            if "nav2" in parent.name:
                source_name = parent.name
                break
        for result_path in run_dir.glob("*/result.json"):
            try:
                payload = json.loads(result_path.read_text())
                index = int(payload["dataset_index"])
                dataset = dataset_from_slide(str(payload["slide_id"]))
                answer = str(payload.get("answer", {}).get("answer", ""))
                gold = str(payload.get("gold_answer", ""))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if dataset is None:
                continue
            if not is_prompt1_source(source_name, dataset, variant):
                continue
            key = (dataset, variant)
            stamp = result_path.stat().st_mtime
            current = grouped.setdefault(key, {}).get(index)
            if current is None or stamp > current[0]:
                grouped[key][index] = (stamp, normalized(gold), normalized(answer))
        metrics_path = run_dir / "attempt_metrics.jsonl"
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines():
                try:
                    metric = json.loads(line)
                    if not metric.get("success"):
                        continue
                    index = int(metric["dataset_index"])
                    dataset = dataset_from_slide(str(metric["slide_id"]))
                    seconds = float(
                        metric.get("wall_seconds", metric["role_call_seconds"])
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if dataset is None:
                    continue
                if not is_prompt1_source(source_name, dataset, variant):
                    continue
                key = (dataset, variant)
                stamp = metrics_path.stat().st_mtime
                current = timings.setdefault(key, {}).get(index)
                if current is None or stamp > current[0]:
                    timings[key][index] = (stamp, seconds)

    rows: list[Row] = []
    for (dataset, variant), cases in grouped.items():
        pairs = [(gold, answer) for _, gold, answer in cases.values()]
        correct = sum(gold == answer for gold, answer in pairs)
        if dataset in BALANCED:
            labels = sorted({gold for gold, _ in pairs})
            recalls = [
                sum(gold == answer for gold, answer in pairs if gold == label)
                / sum(gold == label for gold, _ in pairs)
                for label in labels
            ]
            score = 100.0 * mean(recalls) if recalls else None
            score_name = "BAcc"
        else:
            score = 100.0 * correct / len(pairs) if pairs else None
            score_name = "Acc"
        seconds = [value for _, value in timings.get((dataset, variant), {}).values()]
        target = 20 if variant in {"nav3_x20_x40", "nav3_vlmas"} else TARGETS[dataset]
        rows.append(
            Row(
                dataset=dataset,
                variant=variant_label(variant),
                completed=len(pairs),
                target=target,
                score=round(score, 2) if score is not None else None,
                score_name=score_name,
                correct=correct,
                seconds_per_case=round(mean(seconds), 2) if seconds else None,
                state="completed" if len(pairs) >= target else "running",
            )
        )
    relay3_label = variant_label("pruning_b_nav_reasoner_relay3_dual")
    present_relay3 = {row.dataset for row in rows if row.variant == relay3_label}
    for dataset, target in TARGETS.items():
        if dataset in present_relay3:
            continue
        rows.append(
            Row(
                dataset=dataset,
                variant=relay3_label,
                completed=0,
                target=target,
                score=None,
                score_name="BAcc" if dataset in BALANCED else "Acc",
                correct=0,
                seconds_per_case=None,
                state="queued",
            )
        )
    reasoner_relay3_label = variant_label("pruning_b_reasoner_relay3_dual")
    present_reasoner_relay3 = {
        row.dataset for row in rows if row.variant == reasoner_relay3_label
    }
    for dataset, target in TARGETS.items():
        if dataset in present_reasoner_relay3:
            continue
        rows.append(
            Row(
                dataset=dataset,
                variant=reasoner_relay3_label,
                completed=0,
                target=target,
                score=None,
                score_name="BAcc" if dataset in BALANCED else "Acc",
                correct=0,
                seconds_per_case=None,
                state="queued",
            )
        )
    reasoner_relay2_label = variant_label("pruning_b_reasoner_relay2_dual")
    present_reasoner_relay2 = {
        row.dataset for row in rows if row.variant == reasoner_relay2_label
    }
    for dataset, target in TARGETS.items():
        if dataset in present_reasoner_relay2:
            continue
        rows.append(
            Row(
                dataset=dataset,
                variant=reasoner_relay2_label,
                completed=0,
                target=target,
                score=None,
                score_name="BAcc" if dataset in BALANCED else "Acc",
                correct=0,
                seconds_per_case=None,
                state="queued",
            )
        )
    reasoner_triple_relay2_label = variant_label(
        "pruning_b_reasoner_relay2_triple"
    )
    present_reasoner_triple_relay2 = {
        row.dataset for row in rows if row.variant == reasoner_triple_relay2_label
    }
    for dataset, target in TARGETS.items():
        if dataset in present_reasoner_triple_relay2:
            continue
        rows.append(
            Row(
                dataset=dataset,
                variant=reasoner_triple_relay2_label,
                completed=0,
                target=target,
                score=None,
                score_name="BAcc" if dataset in BALANCED else "Acc",
                correct=0,
                seconds_per_case=None,
                state="queued",
            )
        )
    for variant in (
        "pruning_b_reasoner_relay2_dual_paired",
        "pruning_b_reasoner_relay2_triple_paired",
    ):
        label = variant_label(variant)
        present = {row.dataset for row in rows if row.variant == label}
        for dataset, target in TARGETS.items():
            if dataset in present:
                continue
            rows.append(
                Row(
                    dataset=dataset,
                    variant=label,
                    completed=0,
                    target=target,
                    score=None,
                    score_name="BAcc" if dataset in BALANCED else "Acc",
                    correct=0,
                    seconds_per_case=None,
                    state="running" if paired_run_started(dataset, variant) else "queued",
                )
            )
    retained_labels = frozenset(
        {
            "Nav2 Base",
            "Pruning-B",
            "Relay2",
            "Nav2 Both (Pruning-B + Relay2)",
            "Both (Pruning-B + Relay3 Context)",
            "Nav3 Base",
            "Nav3 Both (Pruning-B + Dual Relay2)",
            "Nav3 Both (Pruning-B · Visual KV 50% 보존 + Dual Relay2)",
            "Nav3 Pathology-context SparseVLM + Dual Relay2",
            "Nav3 Base (x20 + x40, 20-case)",
            "Nav3 Text VLM-MAS (20-case)",
            "Nav1 Base",
            "Nav1 Both (Pruning-B + Relay2)",
            "Nav3 E-AdaPrune Pathology + Dual Relay2",
            "Nav3 ATP-SAP Pathology + Dual Relay2",
            "Pathology-context SparseVLM + Dual Relay2",
        }
    )
    labels_by_variant = {
        variant: variant_label(variant)
        for variant in (
            "base",
            "pruning_b",
            "latent_kv_relay2",
            "pruning_b_latent_kv_relay2_dual",
            "pruning_sparsevlm_pathology_latent_kv_relay2_dual",
            "pruning_sparsevlm_pathology_nav3_latent_kv_relay2_dual",
            "pruning_eadaprune_pathology_latent_kv_relay2_dual",
            "pruning_atp_sap_pathology_latent_kv_relay2_dual",
            "pruning_b_reasoner_context_relay3",
            "nav3_base",
            "nav3_both",
            "nav3_both_keep50",
            "nav3_x20_x40",
            "nav3_vlmas",
        )
    }
    present = {(row.dataset, row.variant) for row in rows}
    for dataset, target in TARGETS.items():
        for variant, label in labels_by_variant.items():
            if variant in {"nav3_x20_x40", "nav3_vlmas"} and dataset != "panda":
                continue
            if (dataset, label) in present:
                continue
            rows.append(
                Row(
                    dataset=dataset,
                    variant=label,
                    completed=0,
                    target=target,
                    score=None,
                    score_name="BAcc" if dataset in BALANCED else "Acc",
                    correct=0,
                    seconds_per_case=None,
                    state=(
                        "queued"
                        if variant == "pruning_atp_sap_pathology_latent_kv_relay2_dual"
                        and nav3_adaptive_pruner_started(dataset, "eadaprune")
                        else
                        "running"
                        if variant == "pruning_b_reasoner_context_relay3"
                        and pathology_context_run_started(dataset)
                        else "running"
                        if variant == "nav3_base"
                        and nav3_run_started(dataset, "base")
                        else "running"
                        if variant == "nav3_both"
                        and nav3_run_started(dataset, "both")
                        else "queued"
                        if variant in {"nav3_base", "nav3_both"}
                        else "not_run"
                    ),
                )
            )
    active = active_run_keys()
    normalized_rows = (
        replace(
            row,
            state=(
                "completed"
                if row.completed >= row.target
                else "running"
                if (row.dataset, row.variant) in active
                else "stopped"
                if row.completed > 0
                else "queued"
                if row.state == "queued"
                else "not_run"
            ),
        )
        for row in rows
    )
    return sorted(
        (row for row in normalized_rows if row.variant in retained_labels),
        key=lambda row: (row.dataset, row.variant),
    )


def nav3_result_signature() -> tuple[int, int]:
    """Return a cheap change token for the actively growing Nav3 campaign."""
    result_paths = BENCHMARK_RUNS.glob(
        "nav3_prompt1_*_gpu[0-8]_*/[0-9][0-9][0-9]_*/result.json"
    )
    count = 0
    newest_mtime_ns = 0
    for result_path in result_paths:
        try:
            mtime_ns = result_path.stat().st_mtime_ns
        except OSError:
            continue
        count += 1
        newest_mtime_ns = max(newest_mtime_ns, mtime_ns)
    return count, newest_mtime_ns


def collect() -> list[Row]:
    """Cache stable history, but refresh immediately when a Nav3 result lands."""
    global _collect_active_keys, _collect_cache, _collect_cache_at
    global _collect_nav3_signature
    with _collect_lock:
        current_nav3_signature = nav3_result_signature()
        current_active_keys = active_run_keys()
        if (
            _collect_cache is not None
            and monotonic() - _collect_cache_at < COLLECT_CACHE_SECONDS
            and current_nav3_signature == _collect_nav3_signature
            and current_active_keys == _collect_active_keys
        ):
            return _collect_cache
        _collect_cache = collect_uncached()
        _collect_cache_at = monotonic()
        _collect_nav3_signature = current_nav3_signature
        _collect_active_keys = current_active_keys
        return _collect_cache


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        request_path = self.path.split("?", 1)[0]
        if request_path == "/api/status":
            body = json.dumps(
                {
                    "updated_at": datetime.now().astimezone().isoformat(),
                    "rows": [asdict(row) for row in collect()],
                    "nav3_cases": [
                        asdict(case) for case in collect_nav3_selection_cases()
                    ],
                }
            ).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        elif request_path.startswith("/nav3-assets/"):
            name = Path(request_path).name
            asset = NAV3_ASSETS / name
            body = asset.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
        elif request_path.startswith("/nav3-board/"):
            index = int(request_path.rsplit("/", 1)[-1])
            boards = tuple(
                run_root
                / f"{index:03d}_tcga_expert_vqa__{index}"
                / "evidence_board_round_1.png"
                for run_root in NAV3_RUNS
            )
            board = next(candidate for candidate in boards if candidate.is_file())
            body = board.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
        else:
            body = PAGE.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: str) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
