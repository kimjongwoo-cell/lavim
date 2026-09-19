# Final execution layout

`run.py` is the only public launcher for final experiments.

```text
wsi_latentmas/
├── config/       # Dataset/model tags and one HyperParameters object
├── data/         # Dataset availability checks and GPU shard partitioning
├── execution/    # CLI, command construction, and process lifecycle
├── pipeline/     # Canonical Text-MAS and Latent-MAS module entry points
├── backbones/    # Canonical backbone imports
└── evaluation/   # Canonical metric imports
```

The existing directories below remain private implementation cores, not entry points:

```text
backbone/          # Qwen KV-cache backbone integration
vision_text_mas/   # Single, Text-MAS, latent, pruning, and reallocation engines
models/single_v7/  # Canonical single-model inference adapter
memory/            # KV-pruning and reallocation primitives
eval/              # Metric calculation
dashboard/         # Result display only
```

All tunable values are supplied to one `HyperParameters` instance by `run.py`.
`--greedy-decoding` is enabled by default: Single, VL-MAS, and Latent Base all
use greedy final decoding and `structured_json` output by default.
For example:

```bash
python run.py --dataset wsi-vqa --model qwen-4b --method latent-base \
  --steps 10 --gpus 2,3 --patch-budget 8 --temperature 0.6 \
  --max-new-tokens 512 --io-pipeline --navigator-kv
```

Use `python run.py --help` for the complete, single hyperparameter surface.
