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

## Evaluating Results

### Class imbalance warning

Cresci-2017 is heavily imbalanced (~99% bot, ~1% human). A model that predicts everything as bot scores ~99% accuracy and ~99.7% bot F1 — these numbers are meaningless on their own. **Always evaluate on per-class metrics.**

### Per-class breakdown

After training completes, a full `classification_report` is printed automatically:

```
Detailed Classification Report:
------------------------------------------------------------
              precision    recall  f1-score   support

       Human     0.xxxx    0.xxxx    0.xxxx       xxx
         Bot     0.xxxx    0.xxxx    0.xxxx     xxxxx

    accuracy                         0.xxxx     xxxxx
   macro avg     0.xxxx    0.xxxx    0.xxxx     xxxxx
weighted avg     0.xxxx    0.xxxx    0.xxxx     xxxxx
```

**The number to watch is Human recall** — the fraction of real human accounts the model correctly identified. A degenerate "predict everything bot" model has Human recall = 0.0. A well-trained model should achieve Human recall > 0.7.

### Interpreting the validation metrics during training

The per-epoch val metrics (`Val F1`, `Val Accuracy`) use binary averaging over the bot class only. They will look inflated on this dataset. Use them only to track relative improvement epoch-over-epoch, not as absolute quality scores. The final per-class report after the test phase is the authoritative evaluation.

### Quick check after training

```python
import torch
from sklearn.metrics import classification_report

state = torch.load("trained_models/best_model.pt", map_location="cpu")
# ... load model and run inference on held-out data ...
print(classification_report(y_true, y_pred, target_names=["Human", "Bot"]))
```

## Test

```bash
uv run pytest   # 97 tests, ~2.5s
```

## Deploy (DigitalOcean GPU droplet)

```bash
bash deploy.sh
```
