"""Agreement-weighted Answerer readout for cross-scale consistency latents."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import torch

from spatial_latent.adaptive_readout_attention import adaptive_pair_sdpa
from spatial_latent.attention import BiasProbe
from spatial_latent.grouping import GroupMode, LayoutError
from spatial_latent.relation_consistency_runtime import ConsistencyRelationExperiment
from vision_text_mas import latent_hf_ablation_cli as cli
from vision_text_mas.contracts import CaseInput, RunResult


class AdaptiveConsistencyExperiment(ConsistencyRelationExperiment):
    """Allocate Answerer relation budget across paired latent rows by agreement."""

    def __init__(
        self,
        mode: GroupMode,
        output_root: Path,
        gain: float = 4.0,
        answerer_alpha: float = 0.1,
        consistency_strength: float = 0.1,
        temperature: float = 0.05,
    ) -> None:
        super().__init__(
            mode, output_root, gain, answerer_alpha, consistency_strength
        )
        self.temperature = temperature
        self.readout_weights: list[float] = []

    def answerer_readout(
        self, latent_columns: torch.Tensor
    ) -> AbstractContextManager[BiasProbe]:
        """Read parent-context/relation pairs proportional to their agreement."""
        if len(self.consistency_scores) != 4:
            raise LayoutError("Adaptive readout requires four consistency scores")
        scores = torch.tensor(self.consistency_scores, dtype=torch.float32)
        weights = torch.softmax(scores / self.temperature, dim=0)
        self.readout_weights = weights.tolist()
        pairs = latent_columns[:8].reshape(4, 2)
        return adaptive_pair_sdpa(pairs, weights, alpha=self.answerer_alpha)

    @contextmanager
    def installed(self) -> Iterator[None]:
        """Persist Answerer allocation with the inherited agreement trace."""
        with super().installed():
            original_case = cli.run_case_with_retries

            def run_case(
                *,
                case: CaseInput,
                max_attempts: int,
                operation: Callable[[], RunResult],
                metrics_path: Path,
            ) -> RunResult:
                result = original_case(
                    case=case,
                    max_attempts=max_attempts,
                    operation=operation,
                    metrics_path=metrics_path,
                )
                path = (
                    self.output_root
                    / "adaptive_readout"
                    / f"{case.dataset_index:03d}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "dataset_index": case.dataset_index,
                            "agreement": self.consistency_scores,
                            "weights": self.readout_weights,
                            "temperature": self.temperature,
                            "alpha": self.answerer_alpha,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return result

            cli.run_case_with_retries = run_case
            try:
                yield None
            finally:
                cli.run_case_with_retries = original_case
