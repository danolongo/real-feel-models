# real-feel-models · rf.v1.0.0

Twitter bot detection using a CLS + MaxPool ensemble transformer trained on the Cresci-2017 dataset.

## Architecture

Two identical transformer encoders run in parallel — one uses the **[CLS] token** (good at subtle, contextual patterns) and one uses **max-pooling** over all token representations (good at obvious spam signals). Their logits are combined via weighted average (0.7 / 0.3 by default), adaptive weights, or confidence-gating.

```
input → [CLS Transformer] → logits_cls  ─┐
      → [Max Transformer] → logits_max  ─┴─ ensemble → prediction
```

Both models share the same architecture: 9-layer encoder, d_model=512, 8 attention heads, RoBERTa tokenizer (`cardiffnlp/twitter-roberta-base-sentiment-latest`).

## Setup

```bash
uv sync
```

## Data

Downloads the [Cresci-2017](https://zenodo.org/record/1482079) bot-detection dataset (~150 MB) and outputs a single merged CSV:

```bash
uv run python rf.v1.0.0/data_pipeline/download_data.py
# → rf.v1.0.0/datasets/cresci_2017_merged.csv
```

## Train

```bash
python train.py \
  --config production \
  --data_path rf.v1.0.0/datasets/cresci_2017_merged.csv \
  --output_dir ./trained_models
```

## Test

```bash
uv run pytest   # 97 tests, ~2.5s
```

## Deploy (DigitalOcean GPU droplet)

```bash
bash deploy.sh
```
