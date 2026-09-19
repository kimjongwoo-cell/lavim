#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: bash 0912/run_latent_visual_relay.sh spatial 8 new_run_name 0 1
# Uses the existing project uv interpreter; no package installations or upgrades.
"""Run actual latent-KV extraction/insertion under the fixed native protocol."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

from latent_visual_relay.capture import Mode
from latent_visual_relay.runtime import LatentVisualRelay
from pvcr.errors import LayoutError
from pydantic import BaseModel, ConfigDict

from vision_text_mas.latent_hf_ablation_cli import app


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    mode: Mode | Literal["native"]
    relay_log_gain: float = 0.0
    receiver_relation_gain: float | None = None


def main() -> None:
    args = sys.argv[1:]
    if args == ["--help"]:
        app()
        return
    fixed = {
        "--variant": "base",
        "--backbone": "qwen3-vl",
        "--latent-steps": "10",
        "--patch-budget": "12",
        "--max-model-len": "12288",
        "--transport-mode": "cumulative",
        "--answerer-protocol": "structured_json",
    }
    for flag, expected in fixed.items():
        if (
            args.count(flag) != 1
            or args.index(flag) + 1 >= len(args)
            or args[args.index(flag) + 1] != expected
        ):
            raise LayoutError(f"Latent visual relay requires {flag} {expected}")
    if args.count("--output-root") != 1 or "--navigator-kv" not in args:
        raise LayoutError("Require fresh output and Navigator KV retention")
    output = Path(args[args.index("--output-root") + 1])
    if output.exists():
        raise LayoutError(f"Output exists: {output}")
    settings = Settings.model_validate(
        {
            "mode": os.environ["LVR_MODE"],
            "relay_log_gain": os.environ.get("LVR_RELAY_LOG_GAIN", "0"),
            "receiver_relation_gain": os.environ.get("LVR_RECEIVER_RELATION_GAIN"),
        }
    )
    root = Path(__file__).resolve().parent
    sources = [
        Path(__file__),
        Path(__file__).with_suffix(".sh"),
        *sorted((root / "latent_visual_relay").glob("*.py")),
        *sorted((root / "pvcr").glob("*.py")),
    ]
    experiment = (
        None
        if settings.mode == "native"
        else LatentVisualRelay(output, settings.mode, settings.relay_log_gain)
    )
    if experiment is not None:
        default_relation_gain = (
            1.0 if settings.mode == "attention_context_read" else 0.0
        )
        experiment.receiver_relation_gain = (
            settings.receiver_relation_gain
            if settings.receiver_relation_gain is not None
            else default_relation_gain
        )
    manifest = {
        **settings.model_dump(),
        "effective_receiver_relation_gain": (
            0.0 if experiment is None else experiment.receiver_relation_gain
        ),
        "argv": args,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    os.environ["VLMAS_ATTN_IMPLEMENTATION"] = "sdpa"
    scope = nullcontext() if experiment is None else experiment.installed()
    with scope:
        try:
            app()
        finally:
            if output.is_dir():
                (output / "latent_visual_relay_settings.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )


if __name__ == "__main__":
    main()
