"""Runtime controls for reproducible CUDA latent-rollout experiments."""

from __future__ import annotations

import os

import torch


def enable_deterministic_cuda(*, warn_only: bool = False) -> None:
    """Enable deterministic kernels before the model first touches CUDA.

    The latent transport contains only five BF16 vectors, so small reduction
    differences can otherwise select a different free-text terminal trajectory.
    ``CUBLAS_WORKSPACE_CONFIG`` must be present before the first CUDA operation.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
