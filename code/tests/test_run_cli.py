from __future__ import annotations

from pathlib import Path
from subprocess import run
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_run_cli_prints_canonical_latent_command_when_dry_run() -> None:
    """Given a canonical latent request, run.py prints without launching a GPU job."""
    completed = run(
        (
            sys.executable,
            "run.py",
            "--dataset",
            "wsi-vqa",
            "--model",
            "qwen-4b",
            "--method",
            "latent-base",
            "--steps",
            "10",
            "--gpus",
            "2,3",
            "--patch-budget",
            "12",
            "--temperature",
            "0.25",
            "--no-io-pipeline",
            "--no-navigator-kv",
            "--dry-run",
        ),
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "wsi_latentmas.pipeline.latent_mas" in completed.stdout
    assert "--patch-budget 12" in completed.stdout
    assert '"temperature": 0.25' in completed.stdout
    assert "--temperature 0.0 --top-p 1.0" in completed.stdout
    assert "--io-pipeline" not in completed.stdout
    assert "--navigator-kv" not in completed.stdout


def test_greedy_decoding_is_shared_by_single_vlmas_and_latent() -> None:
    """All final methods must use the same greedy structured-answer policy."""
    commands = {
        method: run(
            (
                sys.executable,
                "run.py",
                "--dataset",
                "expertvqa",
                "--model",
                "qwen-2b",
                "--method",
                method,
                "--gpus",
                "0",
                "--dry-run",
            ),
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        for method in ("single", "vlmas", "latent-base")
    }

    assert all(completed.returncode == 0 for completed in commands.values())
    assert '"greedy_decoding": true' in commands["single"].stdout
    assert "--temperature 0.0 --top-p 1.0" in commands["vlmas"].stdout
    assert "--answerer-greedy" in commands["latent-base"].stdout
    assert "--temperature 0.0 --top-p 1.0" in commands["latent-base"].stdout
    assert "--answerer-protocol structured_json" in commands["vlmas"].stdout
    assert "--answerer-protocol structured_json" in commands["latent-base"].stdout
    assert commands["latent-base"].stdout.count("--answerer-protocol structured_json") == 1


def test_run_cli_reports_the_engine_role_selected_by_each_method() -> None:
    """Each public method tag must resolve to one explicit execution engine."""
    commands = {
        method: run(
            (
                sys.executable,
                "run.py",
                "--dataset",
                "wsi-vqa",
                "--model",
                "qwen-2b",
                "--method",
                method,
                "--gpus",
                "0",
                "--dry-run",
            ),
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        for method in ("single", "vlmas", "latent-base", "pruning-v3")
    }

    assert all(command.returncode == 0 for command in commands.values())
    assert "engine_role=single" in commands["single"].stdout
    assert "engine_role=text-mas" in commands["vlmas"].stdout
    assert "engine_role=latent-mas" in commands["latent-base"].stdout
    assert "engine_role=latent-mas" in commands["pruning-v3"].stdout
