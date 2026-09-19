"""Contract checks for the latent pathology-grounding reallocation arm."""

from pathlib import Path
import ast
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_text_mas.direct_hf_ablation_presets import build_ablation_preset
from vision_text_mas.pathology_reallocation import build_spatial_relay_weights
import torch


def test_reallocation_a_uses_reasoner_to_answerer_spatial_relay() -> None:
    """Reallocation A passes Reasoner focus plus spatial context to Answerer."""
    preset = build_ablation_preset("reallocation_a", latent_steps=10)

    assert preset.prune_config is None
    assert preset.reallocation is not None
    assert preset.reallocation.latent_roles == ()
    assert preset.reallocation.pathology_aware
    assert preset.reallocation.spatial_context_relay
    assert preset.reallocation.answerer_alpha == 0.05
    assert preset.reallocation.answerer_prefill_only
    assert preset.reallocation.relay_source == "sender_priority"


def test_pruning_b_reallocation_a_combines_the_current_arms() -> None:
    """Both keeps Pruning B's Reasoner KV reduction and A's receiver relay."""
    preset = build_ablation_preset("pruning_b_reallocation_a", latent_steps=10)

    assert preset.prune_config is not None
    assert preset.prune_config.reasoner_only
    assert preset.reallocation is not None
    assert preset.reallocation.spatial_context_relay
    assert preset.reallocation.relay_source == "sender_priority"


def test_reallocation_a_uses_continuation_span_map() -> None:
    """Given a KV continuation, when Reallocation A maps patches, then it uses its local spans."""
    source_path = Path(__file__).resolve().parents[1] / "backbone" / "qwen3vl.py"
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "grounded_prefill_and_latent_on_kv"
    )

    assert not any(
        isinstance(node, ast.Name) and node.id == "prefill_spans"
        for node in ast.walk(function)
    )


def test_spatial_relay_preserves_reasoner_focus_and_local_context() -> None:
    """A focused Reasoner token gives its local visual neighbourhood relay weight."""
    focus = torch.zeros((1, 1, 25), dtype=torch.float32)
    focus[..., 12] = 1.0

    weights = build_spatial_relay_weights(
        latent_to_vision_attn={8: focus},
        grids=torch.tensor([[1, 10, 10]], dtype=torch.long),
        spans=[("V", 25)],
        spatial_merge_size=2,
        context_strength=0.35,
    )

    assert weights.numel() == 25
    assert weights[12] > weights[0]
    assert weights[11] > weights[0]


def test_spatial_relay_keeps_reasoner_focus_after_pruning() -> None:
    """Given pruned visual KV, Both still passes aligned focus to the Answerer."""
    focus = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32)

    weights = build_spatial_relay_weights(
        latent_to_vision_attn={24: focus},
        grids=torch.tensor([[1, 10, 10]], dtype=torch.long),
        spans=[("V", 4)],
        spatial_merge_size=2,
        context_strength=0.35,
    )

    assert weights.numel() == 4
    assert weights[3] > weights[0]
