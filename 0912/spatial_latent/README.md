# Spatial latent pilot

Training-free intervention on **Reasoner computation**, not latent-KV relay.
The underlying model remains Qwen3-VL-4B-Thinking.

## Operation

The existing pipeline supplies four 5x patches and eight 20x child patches.
The adapter validates every child's physical containment in its declared parent,
then groups the parent and its two children. Absolute visual cache indices are
resolved from contiguous image-token runs, including the carried cache offset.

Steps 1–2 read group 1, 3–4 group 2, 5–6 group 3, and 7–8 group 4 with a soft
attention prior. Steps 9–10 have ordinary unmodified attention. In layers 8–20
(zero-based), the active group's key logits receive `log(2)` before SDPA. This
doubles those keys' unnormalized attention weights. Original masks are preserved;
prompt, other visual keys, and earlier latent keys remain accessible.

Prefill, Navigator, Answerer and all other layers retain ordinary SDPA. No K/V
rows are appended, moved, deleted, or relabeled. No model weights are trained.
The experiment reuses Base's realignment and cache handoff unchanged.

The shuffled control keeps one 5x and two 20x images in every group but assigns
each child to a different parent using a deterministic seed-42 derangement.
Both arms keep all 12 patches and exactly the same 10 latent steps. With the
current image preprocessing each group contains 768 visual tokens.

## Reproduction

From the project root:

```bash
bash 0912/run_spatial_pilot.sh spatial 6
bash 0912/run_spatial_pilot.sh shuffled 8
uv run --python 3.12 0912/summarize_spatial.py
```

The default pilot IDs are `0, 8, 16, ..., 120`. Output directories must be new.
An optional third launcher argument sets a new output directory name; subsequent
arguments are explicit dataset indices. A no-op audit uses `SPATIAL_GAIN=1`:

```bash
SPATIAL_GAIN=1 bash 0912/run_spatial_pilot.sh spatial 6 new_noop_case0 0
```

Inspect `spatial_traces/<id>.json` for each case's parent mapping, group tokens,
step schedule and actual hook call counts. Successful guided runs require exactly
13 biased layer calls for each of the first 8 steps and zero for both synthesis
steps. A missing hook or incompatible geometry fails explicitly.

## What this experiment can establish

The comparison can test whether real parent-child grouping outperforms an equal
attention intervention with shuffled parents at fixed input and latent budgets.
It cannot establish that latent states encode diagnostic spatial relationships
merely because a hook fired or visual attention increased. Latent-path ablation
and independent evaluation are needed before claiming mediation or general gains.
Base/Pruning-B timing references are historical and subject to machine-load noise.

Only files under `0912` are introduced. The runtime adapters are installed and
restored inside the isolated process; no shared model or pipeline source is edited.
