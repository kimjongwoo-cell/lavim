"""Run one-off Pruning-B + Dual Relay experiments at 50% visual retention.

How to run:
    PYTHONPATH=/path/to/repo python sender_relay_exp/pruning_b_keep50_override.py \
        run --variant pruning_b_latent_kv_relay2_dual ...

Only this process overrides the preset builder; the shared experiment presets
and all other runs remain unchanged.
"""

# /// script
# requires-python = ">=3.12"
# ///

from __future__ import annotations

from dataclasses import replace
from typing import cast

from vision_text_mas.direct_hf_ablation_presets import (
    AblationName,
    DirectHfAblationPreset,
    build_ablation_preset,
)


TARGET_VARIANT = "pruning_b_latent_kv_relay2_dual"


def pruning_b_keep50_preset(*, latent_steps: int = 5) -> DirectHfAblationPreset:
    """Keep Pruning-B's hierarchy/rescue policy and relay, changing ratio only."""
    preset = build_ablation_preset(cast(AblationName, TARGET_VARIANT), latent_steps=latent_steps)
    if preset.prune_config is None:
        raise RuntimeError("Pruning-B preset did not provide a pruning config")
    return replace(
        preset,
        prune_config=replace(preset.prune_config, keep_ratio=0.50),
    )


def main() -> None:
    """Run the normal CLI with a process-local 50% Pruning-B override."""
    import vision_text_mas.latent_hf_ablation_cli as cli

    original_builder = cli.build_ablation_preset

    def build_for_this_run(
        name: AblationName, *, latent_steps: int = 5
    ) -> DirectHfAblationPreset:
        if name != TARGET_VARIANT:
            raise ValueError(f"This runner only supports {TARGET_VARIANT}, got {name}")
        preset = original_builder(name, latent_steps=latent_steps)
        if preset.prune_config is None:
            raise RuntimeError("Pruning-B preset did not provide a pruning config")
        overridden = replace(
            preset,
            prune_config=replace(preset.prune_config, keep_ratio=0.50),
        )
        print("[Pruning-B keep-ratio override] retention=0.50", flush=True)
        return overridden

    cli.build_ablation_preset = build_for_this_run
    cli.app()


if __name__ == "__main__":
    main()
