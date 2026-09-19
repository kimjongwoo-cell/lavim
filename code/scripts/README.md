# Canonical experiment launchers

Use one launcher only:

```bash
python run.py \
  --dataset wsi-vqa --model qwen-4b --method latent-base \
  --steps 10 --gpus 2,3
```

The CLI tags select the dataset, Qwen checkpoint, experiment method, latent-step
count, and physical GPUs. All runtime defaults and hyperparameters are centralized
in [`run.py`](../run.py) and
[`wsi_latentmas/config/experiment.py`](../wsi_latentmas/config/experiment.py).

Current final methods are `single`, `single-no-thinking`, `vlmas`, `latent-base`,
and `pruning-v3`. `reallocation-v2` and `both` remain available as experimental
methods. Use `--dry-run` before launching a new matrix.

The remaining shell files are implementation details used by the canonical runner
for the Single baseline. Historical date-stamped launchers were removed after their
outputs were preserved under `results/`.
