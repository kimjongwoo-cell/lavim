# `vision_text_mas` layout

This package implements the WSI Visual-Language Multi-Agent System.

## Navigation

`navigation/` owns WSI patch selection and multi-scale coordinate handling.

- `navigator*.py`: iterative multi-scale navigation
- `onepass_navigation_*.py`: one-pass hierarchy construction and materialization
- `onepass_navigator*.py`: one-pass navigator agent
- `patch_navigator*.py`: patch ranking and structured output schema
- `navigation_*.py`: navigation contracts, prompts, rendering, and session state

## Remaining top-level modules

The remaining modules are still top-level because they are used by the currently
running baseline experiment. They will be moved only after the active run is
complete and each engine has been validated independently.

- `latent_*.py`: Latent VL-MAS cache, engine, terminal generation, pruning, and reallocation
- `answerer*.py`, `reasoner.py`, `verifier*.py`, `evidence*.py`: agent roles and evidence handling
- `controller*.py`, `matched_*.py`, `cli.py`: Text VL-MAS orchestration and entrypoints
- `qwen_*.py`, `json_*.py`, `direct_hf_*.py`: Qwen backend and constrained generation
- `dataset.py`, `*_slide.py`, `io_pipeline.py`, `artifacts.py`: data and artifact I/O
- `contracts.py`, `errors.py`, `geometry.py`, `determinism.py`: shared primitives
