"""Tests for the model-free parts of glvr_diag (grouping, stats, env handling)."""
from __future__ import annotations

import os

import torch

import glvr_diag as D


def test_groups_partition_the_prefix():
    base = 60
    role_visual = {"planner": torch.arange(2, 12), "reasoner": torch.arange(20, 40)}
    role_latent = {"planner": torch.arange(12, 15), "reasoner": torch.arange(40, 50)}
    g = D.provenance_groups(role_visual=role_visual, role_latent=role_latent, base_len=base)
    union = torch.cat([g["sink"], g["H_V"], g["H_C"]]).sort().values
    assert torch.equal(union, torch.arange(base)), union
    assert g["H_V"].numel() == 30
    assert set(g["latent"].tolist()) <= set(g["H_C"].tolist())
    assert 0 not in set(g["H_C"].tolist())
    assert not (set(g["H_V"].tolist()) & set(g["H_C"].tolist()))


def test_groups_drop_columns_past_base_len():
    g = D.provenance_groups(role_visual={"r": torch.arange(50, 80)},
                            role_latent={"r": torch.arange(80, 90)}, base_len=60)
    assert int(g["H_V"].max()) < 60
    assert g["latent"].numel() == 0
    assert g["H_C_nolatent"].numel() == g["H_C"].numel()


def test_groups_with_empty_registry():
    g = D.provenance_groups(role_visual={}, role_latent={}, base_len=10)
    assert g["H_V"].numel() == 0
    assert g["H_C"].numel() == 9      # everything except the sink


def test_donor_selection_by_env():
    g = D.provenance_groups(role_visual={"r": torch.arange(1, 5)},
                            role_latent={"r": torch.arange(5, 8)}, base_len=12)
    for kind, key in (("hc", "H_C"), ("hc_nolatent", "H_C_nolatent"), ("latent", "latent")):
        os.environ["VLMAS_GLVR_DONOR"] = kind
        assert torch.equal(D.donor_cols(g), g[key]), kind
    os.environ["VLMAS_GLVR_DONOR"] = "nonsense"
    assert torch.equal(D.donor_cols(g), g["H_C"])
    os.environ.pop("VLMAS_GLVR_DONOR")


def test_lambdas_parsing():
    os.environ["VLMAS_GLVR_LAMBDAS"] = "0, 0.25 ,0.5,0.75,1"
    assert D.lambdas() == [0.0, 0.25, 0.5, 0.75, 1.0]
    os.environ["VLMAS_GLVR_LAMBDAS"] = ""
    assert D.lambdas() == [0.0, 0.25, 0.5, 0.75, 1.0]
    os.environ.pop("VLMAS_GLVR_LAMBDAS")


def test_trajectory_stats_identical_rows_collapse():
    P = torch.softmax(torch.randn(1, 30), dim=-1).repeat(10, 1)
    U = torch.randn(1, 8).repeat(10, 1)
    s = D.trajectory_stats(P, U)
    assert abs(s["cos_mean"] - 1.0) < 1e-4
    assert abs(s["bc_mean"] - 1.0) < 1e-4
    assert s["jsd_mean"] < 1e-6
    assert s["eff_rank"] < 1.05


def test_trajectory_stats_diverse_rows():
    torch.manual_seed(0)
    P = torch.softmax(torch.randn(10, 30) * 5, dim=-1)
    U = torch.randn(10, 8)
    s = D.trajectory_stats(P, U)
    assert s["cos_mean"] < 0.6
    assert s["bc_mean"] < 0.9
    assert s["eff_rank"] > 3.0


def test_summarize_layers_picks_requested_layers_and_mean():
    stats = [{"layer": li, "rho_H_C": 0.1 * li, "rho_H_V": 0.01 * li} for li in range(36)]
    out = D.summarize_layers(stats, layers=(18, 27, 35))
    assert set(out) == {"18", "27", "35", "mean"}
    assert abs(out["18"]["rho_H_C"] - 1.8) < 1e-6
    assert abs(out["mean"]["rho_H_V"] - sum(0.01 * li for li in range(36)) / 36) < 1e-6


def test_summarize_layers_handles_missing_layer():
    out = D.summarize_layers([{"layer": 0, "rho_H_C": 0.5}], layers=(18,))
    assert set(out) == {"mean"}


def test_enabled_reads_env():
    os.environ.pop("VLMAS_GLVR", None)
    assert not D.enabled()
    os.environ["VLMAS_GLVR"] = "/tmp/x.jsonl"
    assert D.enabled()
    os.environ.pop("VLMAS_GLVR")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print("all good" if not fails else f"{fails} failed")
    raise SystemExit(1 if fails else 0)
