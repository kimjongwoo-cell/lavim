#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: bash 0912/run_spatial_pilot.sh spatial 6
# Uses the project's pinned model environment via uv; no dependency upgrades.
"""Run spatial latent reasoning through the unchanged Base command line."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from spatial_latent.adaptive_consistency_runtime import AdaptiveConsistencyExperiment
from spatial_latent.contextual_fusion_runtime import ContextualFusionExperiment
from spatial_latent.grouping import GroupMode, LayoutError
from spatial_latent.neighborhood_refinement_runtime import (
    NeighborhoodRefinementExperiment,
)
from spatial_latent.relation_consistency_runtime import ConsistencyRelationExperiment
from spatial_latent.relation_contrast_runtime import ContrastRelationExperiment
from spatial_latent.relation_runtime import RelationExperiment
from spatial_latent.runtime import SpatialExperiment

from vision_text_mas.latent_hf_ablation_cli import app


class ExperimentSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    mode: GroupMode
    gain: float = Field(default=2.0, ge=1.0, le=4.0)
    relation: bool = False
    contrast: bool = False
    consistency: bool = False
    adaptive_consistency: bool = False
    contextual_fusion: bool = False
    neighborhood_refinement: bool = False
    answerer_alpha: float = Field(default=0.0, ge=0.0, le=0.25)
    contrast_strength: float = Field(default=0.1, ge=0.0, le=0.5)
    consistency_strength: float = Field(default=0.1, ge=0.0, le=0.5)
    agreement_temperature: float = Field(default=0.05, gt=0.0, le=1.0)
    fusion_strength: float = Field(default=0.4, ge=0.0, le=0.75)


def main() -> None:
    """Validate the fixed protocol and persist the isolated adapter's settings."""
    args = sys.argv[1:]
    expected = {
        "--variant": "base",
        "--latent-steps": "10",
        "--patch-budget": "12",
        "--max-model-len": "12288",
    }
    for flag, value in expected.items():
        if args.count(flag) != 1 or args[
            args.index(flag) + 1 : args.index(flag) + 2
        ] != [value]:
            raise LayoutError(f"This experiment requires {flag} {value}")
    if args.count("--output-root") != 1:
        raise LayoutError("Exactly one --output-root is required")
    output_root = Path(args[args.index("--output-root") + 1])
    settings = ExperimentSettings.model_validate(
        {
            "mode": os.environ["SPATIAL_MODE"],
            "gain": os.environ.get("SPATIAL_GAIN", "2"),
            "relation": os.environ.get("RELATION_LATENT", "0") == "1",
            "contrast": os.environ.get("RELATION_CONTRAST", "0") == "1",
            "consistency": os.environ.get("RELATION_CONSISTENCY", "0") == "1",
            "adaptive_consistency": os.environ.get("ADAPTIVE_CONSISTENCY", "0") == "1",
            "contextual_fusion": os.environ.get("CONTEXTUAL_FUSION", "0") == "1",
            "neighborhood_refinement": os.environ.get(
                "NEIGHBORHOOD_REFINEMENT", "0"
            )
            == "1",
            "answerer_alpha": os.environ.get("RELATION_ALPHA", "0"),
            "contrast_strength": os.environ.get("CONTRAST_STRENGTH", "0.1"),
            "consistency_strength": os.environ.get("CONSISTENCY_STRENGTH", "0.1"),
            "agreement_temperature": os.environ.get("AGREEMENT_TEMPERATURE", "0.05"),
            "fusion_strength": os.environ.get("FUSION_STRENGTH", "0.4"),
        }
    )
    manifest = {
        **settings.model_dump(mode="json"),
        "base_variant": "base",
        "layers_zero_based": [8, 20],
        "group_steps": 8,
        "synthesis_steps": 2,
        "shuffle_seed": 42,
        "attention_backend": "sdpa",
        "arguments": args,
        "source_root": str(Path(__file__).resolve().parents[1]),
    }
    path = output_root / "spatial_experiment.json"
    if output_root.exists():
        raise LayoutError(f"Experiment output already exists: {output_root}")
    os.environ["VLMAS_ATTN_IMPLEMENTATION"] = "sdpa"
    experiment = NeighborhoodRefinementExperiment(
        settings.mode, settings.gain
    ) if settings.neighborhood_refinement else ContextualFusionExperiment(
        settings.mode,
        output_root,
        settings.gain,
        settings.fusion_strength,
    ) if settings.contextual_fusion else AdaptiveConsistencyExperiment(
        settings.mode,
        output_root,
        settings.gain,
        settings.answerer_alpha,
        settings.consistency_strength,
        settings.agreement_temperature,
    ) if settings.adaptive_consistency else ConsistencyRelationExperiment(
        settings.mode,
        output_root,
        settings.gain,
        settings.answerer_alpha,
        settings.consistency_strength,
    ) if settings.consistency else ContrastRelationExperiment(
        settings.mode,
        output_root,
        settings.gain,
        settings.answerer_alpha,
        settings.contrast_strength,
    ) if settings.contrast else (
        RelationExperiment(
            settings.mode,
            output_root,
            settings.gain,
            settings.answerer_alpha,
        )
        if settings.relation
        else SpatialExperiment(settings.mode, settings.gain)
    )
    with experiment.installed():
        try:
            app()
        finally:
            if output_root.is_dir():
                path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
