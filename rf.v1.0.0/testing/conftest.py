"""
Shared pytest fixtures for the rf.v1.0.0 test suite.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Root of the package (rf.v1.0.0/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from setup.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    EnsembleConfig,
    ExperimentConfig,
    get_fast_config,
)
from setup.model import CLSMaxPoolEnsemble, create_ensemble_model
from training_pipeline.trainer import EnsembleTrainer, create_ensemble_trainer


# ---------------------------------------------------------------------------
# Tiny config — keeps every test fast (< 1 s per test on CPU)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        dropout=0.0,        # deterministic outputs
        max_seq_length=32,
        vocab_size=512,
        num_classes=2,
    )


@pytest.fixture(scope="session")
def tiny_train_cfg() -> TrainingConfig:
    return TrainingConfig(
        optimizer_type="adamw",
        learning_rate=1e-3,
        weight_decay=0.01,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        scheduler_type="cosine",
        warmup_steps=2,
        gradient_clipping=True,
        clip_type="norm",
        clip_value=1.0,
        batch_size=8,
        max_epochs=1,
        use_class_weights=False,
        loss_type="weighted_ce",
        focal_alpha=0.25,
        focal_gamma=2.0,
        label_smoothing=0.1,
    )


@pytest.fixture(scope="session")
def tiny_ensemble_cfg() -> EnsembleConfig:
    return EnsembleConfig(
        primary_pool="cls",
        primary_weight=0.7,
        backup_pool="max",
        backup_weight=0.3,
        combination_method="weighted_average",
        confidence_threshold=0.8,
    )


@pytest.fixture(scope="session")
def tiny_config(tiny_model_cfg, tiny_train_cfg, tiny_ensemble_cfg) -> ExperimentConfig:
    return ExperimentConfig(
        model=tiny_model_cfg,
        training=tiny_train_cfg,
        data=DataConfig(),
        ensemble=tiny_ensemble_cfg,
        device=torch.device("cpu"),
        seed=42,
    )


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class _SyntheticDataset(Dataset):
    """Balanced binary dataset with random tokens."""

    def __init__(self, n: int, seq_len: int, vocab_size: int, seed: int = 0):
        rng = torch.Generator().manual_seed(seed)
        half = n // 2
        zeros = torch.randint(1, vocab_size // 2, (half, seq_len), generator=rng)
        ones = torch.randint(vocab_size // 2, vocab_size, (n - half, seq_len), generator=rng)
        self.input_ids = torch.cat([zeros, ones], dim=0)
        self.labels = torch.cat([torch.zeros(half), torch.ones(n - half)]).long()
        mask = torch.ones(n, seq_len, dtype=torch.long)
        mask[:, seq_len // 2 :] = 0          # simulate padding in second half
        self.attention_mask = mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


@pytest.fixture(scope="session")
def tiny_dataset(tiny_model_cfg):
    return _SyntheticDataset(n=80, seq_len=tiny_model_cfg.max_seq_length,
                             vocab_size=tiny_model_cfg.vocab_size)


@pytest.fixture(scope="session")
def tiny_loaders(tiny_dataset, tiny_train_cfg):
    n = len(tiny_dataset)
    tr, va, te = int(0.7 * n), int(0.15 * n), n - int(0.7 * n) - int(0.15 * n)
    tr_ds, va_ds, te_ds = random_split(
        tiny_dataset, [tr, va, te], generator=torch.Generator().manual_seed(0)
    )
    bs = tiny_train_cfg.batch_size
    return (
        DataLoader(tr_ds, batch_size=bs, shuffle=False),
        DataLoader(va_ds, batch_size=bs, shuffle=False),
        DataLoader(te_ds, batch_size=bs, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Model and trainer fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ensemble_model(tiny_config):
    """Fresh model for each test (avoids state bleed)."""
    torch.manual_seed(0)
    return create_ensemble_model(tiny_config.model, tiny_config.ensemble)


@pytest.fixture()
def trained_trainer(ensemble_model, tiny_loaders, tiny_config):
    """Trainer already run for 1 epoch."""
    tr, va, te = tiny_loaders
    trainer = create_ensemble_trainer(ensemble_model, tr, va, te, tiny_config)
    trainer.train()
    return trainer
