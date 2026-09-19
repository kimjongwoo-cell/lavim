from __future__ import annotations

from dataclasses import dataclass

import torch

from vision_text_mas.direct_hf_ablation_presets import build_ablation_preset
from vision_text_mas.latent_context_relay import (
    append_latent_kv_relay,
    extract_terminal_latent_kv,
    select_visual_grounded_latent_steps,
)


@dataclass
class _FakeLayer:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class _FakeCache:
    layers: list[_FakeLayer]


def test_append_latent_kv_relay_copies_reasoner_kv_to_answerer_cache() -> None:
    """Given a Reasoner cache, its final latent K/V entries are appended at handoff."""
    # Given: a minimal dynamic-cache shaped object with two latent entries.
    layer = _FakeLayer(
        keys=torch.tensor([[[[1.0], [2.0], [3.0]]]]),
        values=torch.tensor([[[[4.0], [5.0], [6.0]]]]),
    )
    cache = _FakeCache(layers=[layer])
    relay = extract_terminal_latent_kv(cache, latent_steps=2)

    # When: the Reasoner latent relay is applied for Answerer decoding.
    append_latent_kv_relay(cache, relay)

    # Then: only the copied final latent K/V entries are appended.
    assert torch.equal(layer.keys.flatten(), torch.tensor([1.0, 2.0, 3.0, 2.0, 3.0]))
    assert torch.equal(layer.values.flatten(), torch.tensor([4.0, 5.0, 6.0, 5.0, 6.0]))


def test_select_visual_grounded_latent_steps_keeps_steps_that_read_visual_kv() -> None:
    """Given latent-to-visual attention, only visual-reading latent steps are relayed."""
    # Given: three latent steps, with the first two reading visual KV substantially.
    latent_to_vision = {20: torch.tensor([[[0.8, 0.1]], [[0.5, 0.1]], [[0.1, 0.0]]])}

    # When: visual-grounded latent steps are selected.
    selected = select_visual_grounded_latent_steps(latent_to_vision)

    # Then: the low-visual-attention latent step is omitted from the relay.
    assert torch.equal(selected, torch.tensor([0, 1]))


def test_latent_kv_relay_preset_registers_reasoner_memory_as_answerer_visual_kv() -> None:
    """Given the KV relay variant, Answerer reads relayed K/V through visual attention."""
    # Given: the public latent-KV relay variant.
    preset = build_ablation_preset("latent_kv_relay")

    # When: its reallocation contract is inspected.
    reallocation = preset.reallocation

    # Then: only latent-derived contextual evidence activates the new pathway.
    assert reallocation is not None
    assert reallocation.latent_kv_relay is True
    assert reallocation.latent_roles == ()
    assert reallocation.answerer_alpha == 0.05
