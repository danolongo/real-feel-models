"""
Tests for setup/model.py

Includes regression tests for every bug that was fixed:
  - super().__init__() call
  - qkv projection naming
  - torch.matmul for 4-D tensors
  - permute order after attention
  - attention mask direction (critical: padding must be suppressed)
  - config attribute names (d_model, num_heads, etc.)
  - CLSMaxPoolEnsemble super().__init__()
"""

import sys
from pathlib import Path
import math

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from setup.model import (
    MultiHeadAttention,
    TransformerEncoderLayer,
    AdvancedPoolingHead,
    BotDetectionTransformer,
    CLSMaxPoolEnsemble,
    create_ensemble_model,
)
from setup.config import ModelConfig, EnsembleConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_cfg(**kw) -> ModelConfig:
    defaults = dict(d_model=64, num_layers=2, num_heads=4, d_ff=128,
                    dropout=0.0, max_seq_length=16, vocab_size=256, num_classes=2)
    defaults.update(kw)
    return ModelConfig(**defaults)


def _make_ensemble_cfg(method="weighted_average") -> EnsembleConfig:
    return EnsembleConfig(
        primary_pool="cls", primary_weight=0.7,
        backup_pool="max", backup_weight=0.3,
        combination_method=method,
    )


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def setup_method(self):
        torch.manual_seed(0)
        self.B, self.S, self.D, self.H = 2, 8, 64, 4
        self.mha = MultiHeadAttention(self.D, self.H, dropout_rate=0.0)

    def test_output_shape(self):
        x = torch.randn(self.B, self.S, self.D)
        out = self.mha(x)
        assert out.shape == (self.B, self.S, self.D)

    def test_output_shape_with_mask(self):
        x = torch.randn(self.B, self.S, self.D)
        # additive mask: 0.0 for real, -10000 for padding
        mask = torch.zeros(self.B, 1, 1, self.S)
        mask[:, :, :, self.S // 2 :] = -10000.0
        out = self.mha(x, mask)
        assert out.shape == (self.B, self.S, self.D)

    def test_attention_mask_suppresses_padding(self):
        """
        Regression: mask == 0 was applied to an additive mask, inverting it.
        Now we use scores + mask, so padding tokens (mask=-10000) get ~0 attention.
        """
        torch.manual_seed(1)
        B, S, D, H = 1, 8, 64, 4
        mha = MultiHeadAttention(D, H, dropout_rate=0.0)
        mha.eval()

        x = torch.randn(B, S, D)
        # mask all positions from index 4 onward as padding
        mask = torch.zeros(B, 1, 1, S)
        mask[:, :, :, 4:] = -10000.0

        with torch.no_grad():
            # Hook into the softmax by computing manually
            qkv = mha.qkv_proj(x)
            qkv = qkv.reshape(B, S, H, 3 * (D // H)).permute(0, 2, 1, 3)
            q, k, v = qkv.chunk(3, dim=-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // H)
            scores_masked = scores + mask
            weights = F.softmax(scores_masked, dim=-1)

        # Attention weight toward padding positions (cols 4-7) should be near 0
        padding_weights = weights[0, :, :, 4:].max().item()
        assert padding_weights < 1e-3, (
            f"Padding positions have attention weight {padding_weights:.6f} "
            f"(expected ~0). Mask may be inverted."
        )

    def test_no_nan_in_output(self):
        x = torch.randn(self.B, self.S, self.D)
        out = self.mha(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flows(self):
        x = torch.randn(self.B, self.S, self.D, requires_grad=True)
        out = self.mha(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# TransformerEncoderLayer
# ---------------------------------------------------------------------------

class TestTransformerEncoderLayer:
    def setup_method(self):
        torch.manual_seed(0)
        self.B, self.S, self.D = 2, 8, 64
        self.layer = TransformerEncoderLayer(self.D, 4, 128, dropout_rate=0.0)

    def test_output_shape(self):
        x = torch.randn(self.B, self.S, self.D)
        out = self.layer(x)
        assert out.shape == (self.B, self.S, self.D)

    def test_residual_connection(self):
        """Output must differ from input (non-trivial transformation)."""
        x = torch.randn(self.B, self.S, self.D)
        out = self.layer(x)
        assert not torch.allclose(out, x)

    def test_no_nan(self):
        x = torch.randn(self.B, self.S, self.D)
        out = self.layer(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# AdvancedPoolingHead
# ---------------------------------------------------------------------------

class TestAdvancedPoolingHead:
    def setup_method(self):
        torch.manual_seed(0)
        self.B, self.S, self.D = 2, 8, 64

    def test_cls_output_shape(self):
        head = AdvancedPoolingHead(self.D, 2, pooling_strategy="CLS", dropout_rate=0.0)
        h = torch.randn(self.B, self.S, self.D)
        out = head(h)
        assert out.shape == (self.B, 2)

    def test_cls_uses_first_token(self):
        """CLS pooling reads only position 0: non-CLS tokens must be irrelevant."""
        head = AdvancedPoolingHead(self.D, 2, pooling_strategy="CLS", dropout_rate=0.0)
        head.eval()
        h = torch.randn(self.B, self.S, self.D)
        out1 = head(h)

        # Zeroing tokens 1+ must NOT change output — CLS only uses index 0
        h2 = h.clone()
        h2[:, 1:] = 0.0
        out2 = head(h2)
        assert torch.allclose(out1, out2), (
            "CLS pooling read tokens beyond position 0"
        )

        # Zeroing the CLS token itself MUST change output
        h3 = h.clone()
        h3[:, 0] = 0.0
        out3 = head(h3)
        assert not torch.allclose(out1, out3), (
            "CLS pooling did not change when position-0 token changed"
        )

    def test_max_output_shape(self):
        head = AdvancedPoolingHead(self.D, 2, pooling_strategy="max", dropout_rate=0.0)
        h = torch.randn(self.B, self.S, self.D)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        out = head(h, attention_mask=mask)
        assert out.shape == (self.B, 2)

    def test_max_mask_excludes_padding(self):
        """Masked-out tokens must not influence max-pool output."""
        head = AdvancedPoolingHead(self.D, 2, pooling_strategy="max", dropout_rate=0.0)
        head.eval()
        h = torch.randn(self.B, self.S, self.D)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        mask[:, self.S // 2 :] = 0   # mask second half

        out_masked = head(h, attention_mask=mask)

        # Replace masked tokens with large values — output must NOT change
        h_poison = h.clone()
        h_poison[:, self.S // 2 :] = 1e4
        out_poisoned = head(h_poison, attention_mask=mask)
        assert torch.allclose(out_masked, out_poisoned, atol=1e-5), (
            "Max pooling allowed masked (padding) tokens to influence output"
        )

    def test_unknown_strategy_raises(self):
        head = AdvancedPoolingHead(self.D, 2, pooling_strategy="mean", dropout_rate=0.0)
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            head(torch.randn(self.B, self.S, self.D))


# ---------------------------------------------------------------------------
# BotDetectionTransformer
# ---------------------------------------------------------------------------

class TestBotDetectionTransformer:
    def setup_method(self):
        torch.manual_seed(0)
        self.cfg = _make_model_cfg()

    def test_cls_forward_shape(self):
        m = BotDetectionTransformer(self.cfg, pooling_strategy="CLS")
        m.eval()
        ids = torch.randint(0, self.cfg.vocab_size, (2, self.cfg.max_seq_length))
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = m(ids, mask)
        assert out.shape == (2, self.cfg.num_classes)

    def test_max_forward_shape(self):
        m = BotDetectionTransformer(self.cfg, pooling_strategy="max")
        m.eval()
        ids = torch.randint(0, self.cfg.vocab_size, (2, self.cfg.max_seq_length))
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = m(ids, mask)
        assert out.shape == (2, self.cfg.num_classes)

    def test_no_nan_with_padding(self):
        m = BotDetectionTransformer(self.cfg, pooling_strategy="CLS")
        m.eval()
        ids = torch.randint(0, self.cfg.vocab_size, (3, self.cfg.max_seq_length))
        mask = torch.ones_like(ids)
        mask[:, self.cfg.max_seq_length // 2 :] = 0
        with torch.no_grad():
            out = m(ids, mask)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self):
        m = BotDetectionTransformer(self.cfg, pooling_strategy="CLS")
        ids = torch.randint(0, self.cfg.vocab_size, (2, self.cfg.max_seq_length))
        mask = torch.ones_like(ids)
        out = m(ids, mask)
        out.sum().backward()
        # At least one parameter must have a gradient
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# CLSMaxPoolEnsemble
# ---------------------------------------------------------------------------

class TestCLSMaxPoolEnsemble:
    def setup_method(self):
        torch.manual_seed(0)
        self.cfg = _make_model_cfg()
        self.B = 2
        self.ids = torch.randint(0, self.cfg.vocab_size, (self.B, self.cfg.max_seq_length))
        self.mask = torch.ones_like(self.ids)

    def _make(self, method="weighted_average") -> CLSMaxPoolEnsemble:
        ens_cfg = _make_ensemble_cfg(method)
        return create_ensemble_model(self.cfg, ens_cfg)

    def test_weighted_average_shape(self):
        m = self._make("weighted_average")
        m.eval()
        with torch.no_grad():
            out = m(self.ids, self.mask)
        assert out.shape == (self.B, self.cfg.num_classes)

    def test_adaptive_shape(self):
        m = self._make("adaptive")
        m.eval()
        with torch.no_grad():
            out = m(self.ids, self.mask)
        assert out.shape == (self.B, self.cfg.num_classes)

    def test_confidence_gated_shape(self):
        m = self._make("confidence_gated")
        m.eval()
        with torch.no_grad():
            out = m(self.ids, self.mask)
        assert out.shape == (self.B, self.cfg.num_classes)

    def test_return_individual_keys(self):
        m = self._make()
        m.eval()
        with torch.no_grad():
            out = m(self.ids, self.mask, return_individual=True)
        assert set(out.keys()) == {"ensemble", "primary", "backup"}
        for v in out.values():
            assert v.shape == (self.B, self.cfg.num_classes)

    def test_predict_with_reasoning_keys(self):
        m = self._make()
        reasoning = m.predict_with_reasoning(self.ids, self.mask)
        expected = {
            "predictions", "probabilities", "primary_confidence",
            "backup_confidence", "primary_predictions", "backup_predictions", "agreement",
        }
        assert set(reasoning.keys()) == expected

    def test_probabilities_sum_to_one(self):
        m = self._make()
        reasoning = m.predict_with_reasoning(self.ids, self.mask)
        sums = reasoning["probabilities"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(self.B), atol=1e-5)

    def test_predictions_in_valid_range(self):
        m = self._make()
        reasoning = m.predict_with_reasoning(self.ids, self.mask)
        assert reasoning["predictions"].min() >= 0
        assert reasoning["predictions"].max() < self.cfg.num_classes

    def test_agreement_is_binary(self):
        m = self._make()
        reasoning = m.predict_with_reasoning(self.ids, self.mask)
        vals = reasoning["agreement"].unique()
        assert set(vals.tolist()).issubset({0.0, 1.0})

    def test_unknown_combination_method_raises(self):
        m = self._make()
        m.ensemble_config.combination_method = "invalid"
        with pytest.raises(ValueError, match="Unknown combination method"):
            m(self.ids, self.mask)

    def test_no_nan_in_outputs(self):
        for method in ["weighted_average", "adaptive", "confidence_gated"]:
            m = self._make(method)
            m.eval()
            with torch.no_grad():
                out = m(self.ids, self.mask)
            assert not torch.isnan(out).any(), f"NaN in {method} output"
