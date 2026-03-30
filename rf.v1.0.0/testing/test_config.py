"""
Tests for setup/config.py
"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from setup.config import (
    ModelConfig, TrainingConfig, DataConfig, EnsembleConfig, ExperimentConfig,
    get_default_config, get_fast_config, get_production_config,
)


class TestModelConfig:
    def test_default_attributes(self):
        cfg = ModelConfig()
        assert cfg.d_model == 512
        assert cfg.num_heads == 8
        assert cfg.num_layers == 9
        assert cfg.d_ff == cfg.d_model * 4
        assert cfg.vocab_size == 50265
        assert cfg.num_classes == 2

    def test_d_model_divisible_by_heads(self):
        cfg = ModelConfig()
        assert cfg.d_model % cfg.num_heads == 0

    def test_custom_values(self):
        cfg = ModelConfig(d_model=128, num_heads=4)
        assert cfg.d_model == 128
        assert cfg.num_heads == 4
        assert cfg.d_model % cfg.num_heads == 0


class TestTrainingConfig:
    def test_renamed_attributes_exist(self):
        """Verify bug-fix renames are in place (not old names)."""
        cfg = TrainingConfig()
        assert hasattr(cfg, "adam_eps"),         "adam_eps missing (was adam_epsilon)"
        assert hasattr(cfg, "gradient_clipping"), "gradient_clipping missing (was gradient_clippling)"
        assert not hasattr(cfg, "adam_epsilon")
        assert not hasattr(cfg, "gradient_clippling")

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.optimizer_type == "adamw"
        assert cfg.scheduler_type == "cosine"
        assert cfg.loss_type == "weighted_ce"
        assert 0.0 < cfg.learning_rate < 1.0
        assert cfg.batch_size > 0
        assert cfg.max_epochs > 0


class TestDataConfig:
    def test_renamed_attributes_exist(self):
        cfg = DataConfig()
        assert hasattr(cfg, "val_size"),     "val_size missing (was validation_size)"
        assert hasattr(cfg, "random_state"), "random_state missing (was seed)"
        assert hasattr(cfg, "pin_memory"),   "pin_memory missing (was pin_mem)"
        assert not hasattr(cfg, "validation_size")
        assert not hasattr(cfg, "pin_mem")

    def test_split_ratios_sum(self):
        cfg = DataConfig()
        assert cfg.test_size + cfg.val_size < 1.0


class TestEnsembleConfig:
    def test_renamed_attributes_exist(self):
        cfg = EnsembleConfig()
        assert hasattr(cfg, "backup_pool"),   "backup_pool missing (was secondary_pool)"
        assert hasattr(cfg, "backup_weight"), "backup_weight missing (was secondary_weight)"
        assert not hasattr(cfg, "secondary_pool")
        assert not hasattr(cfg, "secondary_weight")

    def test_weights_sum_to_one(self):
        cfg = EnsembleConfig()
        assert abs(cfg.primary_weight + cfg.backup_weight - 1.0) < 1e-6


class TestConfigFactories:
    def test_get_default_config(self):
        cfg = get_default_config()
        assert isinstance(cfg, ExperimentConfig)
        assert isinstance(cfg.device, torch.device)

    def test_get_fast_config(self):
        cfg = get_fast_config()
        assert isinstance(cfg, ExperimentConfig)
        # fast should be smaller than default
        default = get_default_config()
        assert cfg.model.d_model < default.model.d_model
        assert cfg.model.num_layers < default.model.num_layers

    def test_get_production_config(self):
        cfg = get_production_config()
        assert isinstance(cfg, ExperimentConfig)
        # production should have larger batch than default
        default = get_default_config()
        assert cfg.training.batch_size >= default.training.batch_size

    def test_fast_d_model_divisible_by_heads(self):
        cfg = get_fast_config()
        assert cfg.model.d_model % cfg.model.num_heads == 0
