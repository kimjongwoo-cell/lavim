# LaVIM Nav4

Nav4 visual patch-navigation source for whole-slide pathology experiments.

The repository intentionally excludes WSI files, datasets, model weights, generated patch/evidence images, run logs, and caches.

## 1. Clone and install

Use a Linux machine with an NVIDIA GPU and CUDA-compatible PyTorch.

```bash
git clone https://github.com/kimjongwoo-cell/lavim.git
cd lavim

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Minimum runtime packages for the Nav4 path
pip install torch torchvision transformers accelerate \
  pydantic pydantic-settings pillow openslide-python openslide-bin \
  qwen-vl-utils numpy einops safetensors huggingface_hub
```

For the exact environment used during development, install the pinned package set from the source workspace's `requirements.txt` (the full environment includes CUDA/FlashAttention-related packages).

## 2. Download the Qwen3-VL model

The runner defaults to Qwen3-VL-4B-Thinking:

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Thinking \
  --local-dir "$HOME/models/Qwen3-VL-4B-Thinking"
```

If your Hugging Face account requires approval, run `huggingface-cli login` first.

## 3. Download MultiPathQA metadata and slides

The benchmark data is not stored in this repository. MultiPathQA metadata is available from the GIANT project:

```bash
git clone https://github.com/KianWeihrauch/GIANT.git /tmp/GIANT
cd /tmp/GIANT
python -m pip install -r requirements.txt
python pull_dataset.py \
  --csv MultiPathQA.csv \
  --out-root "$HOME/datasets/MultiPathQA/slides" \
  --dataset tcga
```

For GTEx or PANDA, use `--dataset gtex` or `--dataset panda`. PANDA downloads require the Kaggle API credentials. The Hugging Face metadata copy can also be downloaded with:

```bash
huggingface-cli download tbuckley/MultiPathQA \
  --repo-type dataset \
  --local-dir "$HOME/datasets/MultiPathQA/metadata"
```

Nav4's runner expects this prepared layout:

```text
datasets/MultiPathQA/ready_wsivqa/full_no_panda/
├── tcga_expert_vqa.json
├── tcga_slidebench.json
├── tcga.json
├── gtex.json
├── panda.json                 # optional; default runner excludes PANDA slides
├── slides/                    # downloaded WSI files
└── thumbnails_sanitized/      # prepared thumbnails, if your benchmark snapshot provides them
```

The JSON files must use the slide IDs referenced by the downloaded WSI files. The repository does not contain a public one-command conversion from raw GIANT downloads to every `ready_wsivqa` snapshot; keep the prepared snapshot outside Git and point the runner at it.

## 4. Configure paths

`sender_relay_exp/rtask_nav4_arm.sh` contains the paths used by the original run:

```bash
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
```

Edit these four paths for a new machine before running, or place the clone/data/model at the same locations.

## 5. Run Nav4

Run the smoke gate first:

```bash
bash sender_relay_exp/rtask_nav4_arm.sh base smoke 1 "0"
```

Then run one shard of the benchmark:

```bash
bash sender_relay_exp/rtask_nav4_arm.sh base 0 1 "0"
```

Available arms:

- `base`: Nav4 baseline
- `eovc`: Nav4 plus EOVC/pruning variant
- `pruneb`: Nav4 plus pruning-b variant

The script sets `VLMAS_NAV4=1` and writes results under `sender_relay_exp/runs/nav4/`. Keep these generated outputs outside Git.

## Main files

- `vision_text_mas/onepass_navigation_nav4.py`: independent x5/x20 visual candidate-grid selection
- `vision_text_mas/onepass_navigator.py`: Nav4 dispatch
- `sender_relay_exp/rtask_nav4_arm.sh`: smoke and shard runner
