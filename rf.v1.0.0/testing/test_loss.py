"""
Tests for setup/loss.py

Includes regression test for focal loss p_t sign bug.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from setup.loss import AdvancedLossFunction, EnsembleLoss


def _logits(B=4, C=2):
    torch.manual_seed(0)
    return torch.randn(B, C)


def _labels(B=4, C=2):
    return torch.randint(0, C, (B,))


class TestAdvancedLossFunction:

    def test_weighted_ce_is_finite(self):
        fn = AdvancedLossFunction(loss_type="weighted_ce")
        loss = fn(_logits(), _labels())
        assert loss.isfinite()

    def test_weighted_ce_is_positive(self):
        fn = AdvancedLossFunction(loss_type="weighted_ce")
        loss = fn(_logits(), _labels())
        assert loss.item() > 0

    def test_standard_ce_matches_pytorch(self):
        fn = AdvancedLossFunction(loss_type="standard_ce")
        logits, labels = _logits(), _labels()
        expected = F.cross_entropy(logits, labels)
        assert torch.allclose(fn(logits, labels), expected)

    def test_focal_loss_is_finite(self):
        fn = AdvancedLossFunction(loss_type="focal")
        loss = fn(_logits(), _labels())
        assert loss.isfinite()

    def test_focal_loss_p_t_in_unit_interval(self):
        """
        Regression: p_t = torch.exp(ce_loss) was wrong — CE is positive so
        exp(CE) > 1.  Fixed to exp(-CE) which gives p_t in (0, 1].
        """
        logits, labels = _logits(), _labels()
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        p_t = torch.exp(-ce_loss)          # correct formula
        assert (p_t > 0).all() and (p_t <= 1.0 + 1e-6).all(), (
            f"p_t out of (0,1]: min={p_t.min():.4f}, max={p_t.max():.4f}"
        )

    def test_focal_loss_le_standard_ce(self):
        """Focal loss down-weights easy examples, so FL ≤ CE on average."""
        torch.manual_seed(42)
        fn_focal = AdvancedLossFunction(loss_type="focal", focal_gamma=2.0)
        fn_ce = AdvancedLossFunction(loss_type="standard_ce")
        logits, labels = _logits(B=64), _labels(B=64)
        assert fn_focal(logits, labels).item() <= fn_ce(logits, labels).item() + 1e-4

    def test_label_smoothed_is_finite(self):
        fn = AdvancedLossFunction(loss_type="label_smoothed", label_smoothing=0.1)
        loss = fn(_logits(), _labels())
        assert loss.isfinite()

    def test_label_smoothed_less_than_hard(self):
        """Label smoothing reduces overconfident loss."""
        logits = torch.tensor([[10.0, -10.0], [10.0, -10.0]])
        labels = torch.tensor([0, 0])
        fn_hard = AdvancedLossFunction(loss_type="standard_ce")
        fn_smooth = AdvancedLossFunction(loss_type="label_smoothed", label_smoothing=0.1)
        # Hard CE on a very confident correct prediction is near 0
        # Smoothed CE should be slightly higher
        hard = fn_hard(logits, labels).item()
        smooth = fn_smooth(logits, labels).item()
        assert smooth > hard - 1e-4

    def test_unknown_loss_type_raises(self):
        fn = AdvancedLossFunction(loss_type="unknown")
        with pytest.raises(ValueError, match="Unknown loss type"):
            fn(_logits(), _labels())

    def test_class_weights_applied(self):
        """Loss with heavy weight on minority class should differ from unweighted."""
        logits, labels = _logits(B=8), _labels(B=8)
        fn_plain = AdvancedLossFunction(loss_type="standard_ce")
        weights = torch.tensor([1.0, 100.0])
        fn_weighted = AdvancedLossFunction(
            loss_type="weighted_ce", class_weights=weights
        )
        assert fn_plain(logits, labels).item() != pytest.approx(
            fn_weighted(logits, labels).item(), rel=0.01
        )

    def test_compute_class_weights_returns_finite(self):
        fn = AdvancedLossFunction(num_classes=2)
        labels = torch.tensor([0, 0, 0, 1])
        weights = fn.compute_class_weights(labels)
        assert weights.shape == (2,)
        assert weights.isfinite().all()

    def test_compute_class_weights_zero_count_handled(self):
        """Classes with zero samples must get weight 0, not inf."""
        fn = AdvancedLossFunction(num_classes=3)
        labels = torch.tensor([0, 0, 1])   # class 2 absent
        weights = fn.compute_class_weights(labels)
        assert weights[2].item() == 0.0
        assert weights.isfinite().all()


class TestEnsembleLoss:
    def _make_outputs(self, B=4, C=2):
        torch.manual_seed(0)
        return {
            "ensemble": torch.randn(B, C),
            "primary": torch.randn(B, C),
            "backup": torch.randn(B, C),
        }

    def test_returns_tuple(self):
        fn = EnsembleLoss({"num_classes": 2, "loss_type": "standard_ce"})
        out = fn(self._make_outputs(), _labels())
        assert isinstance(out, tuple) and len(out) == 2

    def test_loss_components_keys(self):
        fn = EnsembleLoss({"num_classes": 2, "loss_type": "standard_ce"})
        _, components = fn(self._make_outputs(), _labels())
        assert set(components.keys()) == {"total", "ensemble", "primary", "backup", "individual"}

    def test_total_loss_is_finite(self):
        fn = EnsembleLoss({"num_classes": 2, "loss_type": "standard_ce"})
        total, _ = fn(self._make_outputs(), _labels())
        assert total.isfinite()

    def test_all_components_finite(self):
        fn = EnsembleLoss({"num_classes": 2, "loss_type": "standard_ce"})
        _, components = fn(self._make_outputs(), _labels())
        for k, v in components.items():
            assert v.isfinite(), f"Component '{k}' is not finite"

    def test_alpha_zero_ignores_individual(self):
        """alpha=0 → total should equal ensemble loss."""
        outputs = self._make_outputs()
        labels = _labels()
        fn = EnsembleLoss({"num_classes": 2, "loss_type": "standard_ce"}, alpha=0.0)
        total, components = fn(outputs, labels)
        assert torch.allclose(total, components["ensemble"], atol=1e-6)

    def test_alpha_one_ignores_ensemble(self):
        """alpha=1 → total should equal individual loss."""
        outputs = self._make_outputs()
        labels = _labels()
        fn = EnsembleLoss({"num_classes": 2, "loss_type": "standard_ce"}, alpha=1.0)
        total, components = fn(outputs, labels)
        assert torch.allclose(total, components["individual"], atol=1e-6)
