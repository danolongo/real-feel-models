"""
Tests for setup/optimizer.py
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from setup.optimizer import (
    AdvancedLRScheduler,
    AdvancedGradientClipper,
    OptimizationManager,
)
from setup.config import TrainingConfig


def _simple_model():
    torch.manual_seed(0)
    return nn.Linear(16, 2)


def _training_cfg(**kw) -> TrainingConfig:
    defaults = dict(
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
    defaults.update(kw)
    return TrainingConfig(**defaults)


class TestAdvancedLRScheduler:
    def _optimizer(self):
        return torch.optim.AdamW(_simple_model().parameters(), lr=1.0)

    def test_linear_warmup_zero_at_start(self):
        opt = self._optimizer()
        sched = AdvancedLRScheduler.get_linear_schedule_with_warmup(opt, 10, 100)
        # step 0: lr_lambda(0) = 0/10 = 0
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-7)

    def test_linear_warmup_peaks_at_warmup_steps(self):
        opt = self._optimizer()
        sched = AdvancedLRScheduler.get_linear_schedule_with_warmup(opt, 5, 100)
        for _ in range(5):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0, rel=1e-4)

    def test_cosine_warmup_increases_during_warmup(self):
        opt = self._optimizer()
        sched = AdvancedLRScheduler.get_cosine_schedule_with_warmup(opt, 10, 100)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        assert lrs[-1] > lrs[0]

    def test_cosine_decays_after_warmup(self):
        opt = self._optimizer()
        sched = AdvancedLRScheduler.get_cosine_schedule_with_warmup(opt, 5, 50)
        for _ in range(5):
            sched.step()
        peak_lr = opt.param_groups[0]["lr"]
        for _ in range(20):
            sched.step()
        assert opt.param_groups[0]["lr"] < peak_lr

    def test_polynomial_zero_after_total_steps(self):
        opt = self._optimizer()
        sched = AdvancedLRScheduler.get_polynomial_schedule_with_warmup(opt, 2, 10)
        for _ in range(15):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-7)


class TestAdvancedGradientClipper:
    def _model_with_large_grads(self):
        m = _simple_model()
        loss = m(torch.randn(4, 16)).sum() * 1000.0
        loss.backward()
        return m

    def test_norm_clipping_reduces_norm(self):
        m = self._model_with_large_grads()
        clipper = AdvancedGradientClipper(clip_type="norm", clip_value=1.0)
        stats = clipper.clip_gradients(m)
        assert stats["was_clipped"]
        # Verify the actual gradient norm is now within clip_value
        actual_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in m.parameters() if p.grad is not None
        ) ** 0.5
        assert actual_norm <= 1.0 + 1e-4

    def test_value_clipping_does_not_crash(self):
        m = self._model_with_large_grads()
        clipper = AdvancedGradientClipper(clip_type="value", clip_value=0.1)
        stats = clipper.clip_gradients(m)
        assert "original_norm" in stats

    def test_stats_returned_correctly(self):
        m = self._model_with_large_grads()
        clipper = AdvancedGradientClipper(clip_type="norm", clip_value=1.0)
        stats = clipper.clip_gradients(m)
        assert set(stats.keys()) == {"original_norm", "clipped_norm", "was_clipped", "param_count"}

    def test_get_statistics_empty(self):
        clipper = AdvancedGradientClipper()
        assert clipper.get_statistics() == {}

    def test_get_statistics_after_clip(self):
        m = self._model_with_large_grads()
        clipper = AdvancedGradientClipper(clip_type="norm", clip_value=1.0)
        clipper.clip_gradients(m)
        stats = clipper.get_statistics()
        assert "mean_norm" in stats
        assert stats["mean_norm"] > 0


class TestOptimizationManager:
    def _setup(self, **cfg_kw):
        model = _simple_model()
        cfg = _training_cfg(**cfg_kw)
        total_steps = 20
        manager = OptimizationManager(model, cfg, total_steps)
        return model, manager

    def test_creates_optimizer(self):
        _, mgr = self._setup()
        assert mgr.optimizer is not None

    def test_creates_scheduler(self):
        _, mgr = self._setup(scheduler_type="cosine")
        assert mgr.scheduler is not None

    def test_no_scheduler_when_none(self):
        _, mgr = self._setup(scheduler_type="none")
        assert mgr.scheduler is None

    def test_gradient_clipper_created(self):
        _, mgr = self._setup(gradient_clipping=True)
        assert mgr.gradient_clipper is not None

    def test_no_clipper_when_disabled(self):
        _, mgr = self._setup(gradient_clipping=False)
        assert mgr.gradient_clipper is None

    def test_optimization_step_reduces_loss(self):
        """Loss must decrease after several update steps."""
        torch.manual_seed(0)
        model, mgr = self._setup(gradient_clipping=False, scheduler_type="none")
        x = torch.randn(8, 16)
        targets = torch.randint(0, 2, (8,))

        losses = []
        for _ in range(10):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            mgr.optimization_step(loss)
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss did not decrease after 10 steps"

    def test_optimization_step_returns_stats(self):
        model, mgr = self._setup()
        loss = model(torch.randn(4, 16)).sum()
        stats = mgr.optimization_step(loss)
        assert set(stats.keys()) == {
            "learning_rate", "gradient_norm", "gradient_clipped", "parameter_norm"
        }

    def test_history_populated(self):
        model, mgr = self._setup()
        loss = model(torch.randn(4, 16)).sum()
        mgr.optimization_step(loss)
        assert len(mgr.optimization_history["learning_rates"]) == 1
        assert len(mgr.optimization_history["gradient_norms"]) == 1

    def test_adamw_optimizer_type(self):
        _, mgr = self._setup(optimizer_type="adamw")
        assert isinstance(mgr.optimizer, torch.optim.AdamW)

    def test_adam_optimizer_type(self):
        _, mgr = self._setup(optimizer_type="adam")
        assert isinstance(mgr.optimizer, torch.optim.Adam)

    def test_unknown_optimizer_raises(self):
        model = _simple_model()
        cfg = _training_cfg(optimizer_type="rmsprop")
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            OptimizationManager(model, cfg, 10)

    def test_unknown_scheduler_raises(self):
        model = _simple_model()
        cfg = _training_cfg(scheduler_type="cyclic")
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            OptimizationManager(model, cfg, 10)
