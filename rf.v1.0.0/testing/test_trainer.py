"""
Tests for training_pipeline/trainer.py

Includes regression test for shallow state_dict copy (best model bug).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from setup.config import get_fast_config
from setup.model import create_ensemble_model
from training_pipeline.trainer import EnsembleTrainer, create_ensemble_trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_config():
    """1-epoch CPU config with tiny model for speed."""
    cfg = get_fast_config()
    cfg.device = torch.device("cpu")
    cfg.training.max_epochs = 1
    cfg.training.batch_size = 8
    cfg.training.use_class_weights = False
    cfg.training.scheduler_type = "cosine"
    cfg.training.warmup_steps = 1
    cfg.model.vocab_size = 256
    cfg.model.max_seq_length = 16
    return cfg


def _make_loaders(cfg, n=80):
    from torch.utils.data import DataLoader, TensorDataset, random_split
    ids = torch.randint(0, cfg.model.vocab_size, (n, cfg.model.max_seq_length))
    mask = torch.ones_like(ids)
    labels = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    ds = TensorDataset(ids, mask, labels)

    tr, va, te = int(0.7 * n), int(0.15 * n), n - int(0.7 * n) - int(0.15 * n)
    tr_ds, va_ds, te_ds = random_split(ds, [tr, va, te],
                                       generator=torch.Generator().manual_seed(0))

    def _collate(batch):
        ids_, mask_, lbl_ = zip(*batch)
        return {
            "input_ids": torch.stack(ids_),
            "attention_mask": torch.stack(mask_),
            "labels": torch.stack(lbl_),
        }

    bs = cfg.training.batch_size
    return (
        DataLoader(tr_ds, batch_size=bs, shuffle=False, collate_fn=_collate),
        DataLoader(va_ds, batch_size=bs, shuffle=False, collate_fn=_collate),
        DataLoader(te_ds, batch_size=bs, shuffle=False, collate_fn=_collate),
    )


def _make_trainer() -> EnsembleTrainer:
    torch.manual_seed(0)
    cfg = _make_tiny_config()
    model = create_ensemble_model(cfg.model, cfg.ensemble)
    tr, va, te = _make_loaders(cfg)
    return create_ensemble_trainer(model, tr, va, te, cfg)


# ---------------------------------------------------------------------------
# EnsembleTrainer construction
# ---------------------------------------------------------------------------

class TestEnsembleTrainerInit:
    def test_trainer_creates_scaler_on_cpu(self):
        trainer = _make_trainer()
        # CPU: scaler must be None (AMP not active)
        assert trainer.scaler is None

    def test_criterion_created(self):
        trainer = _make_trainer()
        assert trainer.criterion is not None

    def test_optimization_manager_created(self):
        trainer = _make_trainer()
        assert trainer.optimization_manager is not None

    def test_history_keys(self):
        trainer = _make_trainer()
        expected = {
            "train_loss", "val_loss", "val_accuracy", "val_precision",
            "val_recall", "val_f1", "val_roc_auc", "ensemble_agreement",
            "primary_accuracy", "backup_accuracy",
        }
        assert set(trainer.history.keys()) == expected


# ---------------------------------------------------------------------------
# save_best_model — regression for shallow copy bug
# ---------------------------------------------------------------------------

class TestSaveBestModel:
    def test_saved_state_is_deep_copy(self):
        """
        Regression: state_dict().copy() was a shallow copy — tensors were shared.
        After save, mutating the model must NOT change the saved snapshot.
        """
        trainer = _make_trainer()

        # Force a save
        trainer.save_best_model(0.9)
        assert trainer.best_model_state is not None

        # Snapshot values before mutation
        saved_values = {
            k: v.clone() for k, v in trainer.best_model_state.items()
        }

        # Zero out all model parameters (simulate continued training)
        with torch.no_grad():
            for p in trainer.model.parameters():
                p.zero_()

        # Saved state must be unchanged
        for k, saved_v in saved_values.items():
            assert torch.allclose(trainer.best_model_state[k], saved_v), (
                f"Parameter '{k}' in best_model_state was mutated after model update. "
                "Shallow copy bug still present."
            )

    def test_best_model_updated_only_when_improved(self):
        trainer = _make_trainer()
        trainer.save_best_model(0.5)
        first_state = {k: v.clone() for k, v in trainer.best_model_state.items()}

        # Worse F1 — should NOT update
        updated = trainer.save_best_model(0.3)
        assert not updated

        # Verify state unchanged
        for k in first_state:
            assert torch.allclose(trainer.best_model_state[k], first_state[k])

    def test_best_model_updated_when_improved(self):
        trainer = _make_trainer()
        trainer.save_best_model(0.5)
        updated = trainer.save_best_model(0.9)
        assert updated
        assert trainer.best_val_f1 == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# save_model
# ---------------------------------------------------------------------------

class TestSaveModel:
    def test_save_model_creates_file(self, tmp_path):
        trainer = _make_trainer()
        path = tmp_path / "model.pt"
        trainer.save_model(str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_saved_model_is_loadable(self, tmp_path):
        trainer = _make_trainer()
        path = tmp_path / "model.pt"
        trainer.save_model(str(path))
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        assert isinstance(state, dict)
        assert len(state) > 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def setup_method(self):
        self.trainer = _make_trainer()

    def test_evaluate_returns_expected_keys(self):
        metrics = self.trainer.evaluate(self.trainer.val_loader, "Val")
        expected = {
            "loss", "accuracy", "precision", "recall", "f1", "roc_auc",
            "primary_accuracy", "backup_accuracy", "agreement_rate",
            "predictions", "labels", "probabilities",
        }
        assert set(metrics.keys()) == expected

    def test_accuracy_in_range(self):
        metrics = self.trainer.evaluate(self.trainer.val_loader, "Val")
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_loss_is_finite(self):
        metrics = self.trainer.evaluate(self.trainer.val_loader, "Val")
        assert torch.tensor(metrics["loss"]).isfinite()

    def test_agreement_rate_in_range(self):
        metrics = self.trainer.evaluate(self.trainer.val_loader, "Val")
        assert 0.0 <= metrics["agreement_rate"] <= 1.0


# ---------------------------------------------------------------------------
# train (full loop)
# ---------------------------------------------------------------------------

class TestTrainLoop:
    def test_train_returns_model(self):
        from setup.model import CLSMaxPoolEnsemble
        trainer = _make_trainer()
        result = trainer.train()
        assert isinstance(result, CLSMaxPoolEnsemble)

    def test_history_populated_after_training(self):
        trainer = _make_trainer()
        trainer.train()
        assert len(trainer.history["train_loss"]) == trainer.config.training.max_epochs
        assert len(trainer.history["val_f1"]) == trainer.config.training.max_epochs

    def test_best_model_state_set_after_training(self):
        trainer = _make_trainer()
        trainer.train()
        assert trainer.best_model_state is not None

    def test_no_nan_in_history(self):
        trainer = _make_trainer()
        trainer.train()
        for key, values in trainer.history.items():
            for v in values:
                assert torch.tensor(v).isfinite(), f"NaN/Inf in history['{key}']"
