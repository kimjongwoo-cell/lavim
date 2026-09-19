from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_dashboard():
    path = Path(__file__).with_name("live_dashboard.py")
    spec = importlib.util.spec_from_file_location("live_dashboard_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_invalidates_cache_when_nav3_results_change(monkeypatch) -> None:
    dashboard = load_dashboard()
    snapshots = [["first"], ["second"]]
    signatures = iter(((1, 1.0), (2, 2.0)))

    monkeypatch.setattr(dashboard, "collect_uncached", lambda: snapshots.pop(0))
    monkeypatch.setattr(dashboard, "nav3_result_signature", lambda: next(signatures))
    dashboard._collect_cache = None
    dashboard._collect_cache_at = 0.0

    assert dashboard.collect() == ["first"]
    assert dashboard.collect() == ["second"]


def test_nav3_pruning_b_dual_run_maps_to_nav3_both_row() -> None:
    dashboard = load_dashboard()

    assert dashboard.nav3_variant_for_run_name(
        "nav3_prompt1_pruning_b_both_tcga_gpu7_20260914",
        "pruning_b_latent_kv_relay2_dual",
    ) == "nav3_both"


def test_nav3_keep50_run_maps_to_separate_prompt1_row() -> None:
    dashboard = load_dashboard()
    run_name = "nav3_prompt1_pruning_b_keep50_dual_gtex_gpu7_20260914"

    variant = dashboard.nav3_variant_for_run_name(
        run_name,
        "pruning_b_latent_kv_relay2_dual",
    )

    assert variant == "nav3_both_keep50"
    assert dashboard.variant_label(variant) == (
        "Nav3 Both (Pruning-B · Visual KV 50% 보존 + Dual Relay2)"
    )
    assert dashboard.is_prompt1_source(run_name, "gtex", variant)


def test_pathology_sparsevlm_variant_has_distinct_prompt1_dashboard_row() -> None:
    dashboard = load_dashboard()
    variant = "pruning_sparsevlm_pathology_latent_kv_relay2_dual"

    assert variant in dashboard.VISIBLE_VARIANTS
    assert dashboard.variant_label(variant) == (
        "Pathology-context SparseVLM + Dual Relay2"
    )
    assert dashboard.is_prompt1_source("sparsevlm_pathology_vs_both_gtex190_rank75_20260915", "gtex", variant)


def test_nav3_pathology_dual_run_maps_to_a_separate_dashboard_row() -> None:
    dashboard = load_dashboard()
    run_name = "nav3_prompt1_pathology_dual_gtex_gpu0_20260915"
    manifest_variant = "pruning_sparsevlm_pathology_latent_kv_relay2_dual"

    variant = dashboard.nav3_variant_for_run_name(run_name, manifest_variant)

    assert variant == "pruning_sparsevlm_pathology_nav3_latent_kv_relay2_dual"
    assert variant in dashboard.VISIBLE_VARIANTS
    assert dashboard.variant_label(variant) == (
        "Nav3 Pathology-context SparseVLM + Dual Relay2"
    )
    assert dashboard.is_prompt1_source(run_name, "gtex", variant)


def test_nav3_adaptive_pruner_runs_get_distinct_dashboard_rows() -> None:
    dashboard = load_dashboard()
    cases = (
        (
            "nav3_prompt1_eadaprune_pathology_dual_gtex_gpu0_20260915",
            "pruning_eadaprune_pathology_latent_kv_relay2_dual",
            "Nav3 E-AdaPrune Pathology + Dual Relay2",
        ),
        (
            "nav3_prompt1_atp_sap_pathology_dual_gtex_gpu0_20260915",
            "pruning_atp_sap_pathology_latent_kv_relay2_dual",
            "Nav3 ATP-SAP Pathology + Dual Relay2",
        ),
    )

    for run_name, manifest_variant, label in cases:
        variant = dashboard.nav3_variant_for_run_name(run_name, manifest_variant)

        assert variant == manifest_variant
        assert variant in dashboard.VISIBLE_VARIANTS
        assert dashboard.variant_label(variant) == label
        assert dashboard.is_prompt1_source(run_name, "gtex", variant)

    assert not dashboard.is_prompt1_source(
        "nav3_prompt1_eadaprune_pathology_dual_gtex_smoke_20260915",
        "gtex",
        "pruning_eadaprune_pathology_latent_kv_relay2_dual",
    )


def test_nav1_base_run_prefix_maps_to_nav1_dashboard_row() -> None:
    dashboard = load_dashboard()
    run_name = "nav1_prompt1_base_panda_gpu7_20260915"

    variant = dashboard.nav3_variant_for_run_name(run_name, "base")

    assert variant == "nav1_base"
    assert dashboard.variant_label(variant) == "Nav1 Base"
    assert dashboard.is_prompt1_source(run_name, "panda", variant)
    assert dashboard.nav3_variant_for_run_name(
        "gtex_base_12patch_part0_63_gpu6_0913", "base"
    ) == "nav1_base"
    assert dashboard.nav3_variant_for_run_name(
        "tcga_slidebench_base_12patch_part0_98_gpu6_0913", "base"
    ) == "nav1_base"


def test_nav3_signature_includes_results_from_gpus_zero_through_three(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard()
    result = (
        tmp_path
        / "nav3_prompt1_pathology_dual_gtex_gpu0_20260915"
        / "001_gtex__case"
        / "result.json"
    )
    result.parent.mkdir(parents=True)
    result.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(dashboard, "BENCHMARK_RUNS", tmp_path)

    count, newest_mtime_ns = dashboard.nav3_result_signature()

    assert count == 1
    assert newest_mtime_ns == result.stat().st_mtime_ns
