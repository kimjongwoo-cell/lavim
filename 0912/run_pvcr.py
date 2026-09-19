#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: bash 0912/run_pvcr.sh spatial 7 output_name 0 1
"""PVCR runs the native Base CLI under isolated process-local adapters."""

import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

from pvcr.errors import LayoutError
from pvcr.layout import Variant
from pvcr.runtime import PVCR
from pydantic import BaseModel, ConfigDict

from vision_text_mas.latent_hf_ablation_cli import app


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    variant: Variant | Literal["native"]
    boundary: Literal["both", "navigator", "reasoner"] = "both"


def main() -> None:
    args = sys.argv[1:]
    if args == ["--help"]:
        app()
        return
    for flag, value in {
        "--variant": "base",
        "--latent-steps": "10",
        "--patch-budget": "12",
        "--max-model-len": "12288",
        "--backbone": "qwen3-vl",
        "--transport-mode": "cumulative",
        "--answerer-protocol": "structured_json",
    }.items():
        if (
            args.count(flag) != 1
            or args.index(flag) + 1 >= len(args)
            or args[args.index(flag) + 1] != value
        ):
            raise LayoutError(f"PVCR requires {flag} {value}")
    if args.count("--output-root") != 1 or "--navigator-kv" not in args:
        raise LayoutError("PVCR requires output root and retained Navigator KV")
    output_root = Path(args[args.index("--output-root") + 1])
    if output_root.exists():
        raise LayoutError(f"Output already exists: {output_root}")
    settings = Settings.model_validate(
        {
            "variant": os.environ["PVCR_VARIANT"],
            "boundary": os.environ.get("PVCR_BOUNDARY", "both"),
        }
    )
    os.environ["VLMAS_ATTN_IMPLEMENTATION"] = "sdpa"
    sources = [
        Path(__file__),
        Path(__file__).with_suffix(".sh"),
        *sorted(Path(__file__).with_name("pvcr").glob("*.py")),
    ]
    manifest = {
        **settings.model_dump(),
        "argv": args,
        "python": sys.version,
        "source_sha256": {
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
        },
    }
    scope = (
        nullcontext()
        if settings.variant == "native"
        else PVCR(output_root, settings.variant, settings.boundary).installed()
    )
    with scope:
        try:
            app()
        finally:
            if output_root.is_dir():
                (output_root / "pvcr_settings.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )


if __name__ == "__main__":
    main()
