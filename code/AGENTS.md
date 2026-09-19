# Project Agent Instructions

## Dashboard updates are mandatory

- Every experiment launch, queue change, restart, stop, failure recovery, partial result, and completion must be reflected in the dashboard immediately.
- While an experiment is running, keep its status, completed/requested count, failed count, assigned GPUs, cumulative seconds per case, warmup-excluded seconds per case, and ETA current.
- Recompute and publish all available evaluation metrics as successful cases arrive. Never leave a completed run displayed with zero cases, blank metrics, or stale speed.
- Preserve existing completed rows and metrics unless the user explicitly requests deletion. A new run must not erase an older result.
- Dashboard values and the main performance table must come from the authoritative result files for that exact run. Do not mix attempts, protocols, model sizes, latent steps, or variants.
- When a run is replaced, clearly distinguish the new run and update the dashboard before reporting its status to the user.
- Dashboard updating is part of experiment completion, not an optional follow-up. Do not wait for the user to request it.
