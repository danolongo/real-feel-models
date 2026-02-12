"""
rf.v1.0.0.model
This is the 2nd step of the training pipeline

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Union
from .config import ModelConfig, EnsembleConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_size: int, num_heads: int, dropout_rate: float):
        super().__init__
        assert dim_size % num_heads == 0, "dimension size must be divisible by number of heads"

        self.dim_size = dim_size
        self.num_heads = num_heads
        self.head_dim = dim_size // num_heads

        self.qvk_proj = nn.Linear(dim_size, 3 * dim_size)
        self.output_proj = nn.Linear(dim_size, dim_size)
        self.droput = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, sequence_len, dim_size = x.size()

        # project to query key and val
        qkv = self.qvk_projection(x)
        qkv = qkv.reshape(batch_size, sequence_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)

        q, k, v = qkv.chunk(3, dim=-1)

        # scaled dot product attention
        scores = torch.mm(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.droput(attention_weights)

        attended_values = torch.mm(attention_weights, v)
        attended_values = attended_values.permute(0, 1, 2, 3).contiguous()
        attended_values = attended_values.reshape(batch_size, sequence_len, dim_size)

        output = self.output_proj(attended_values)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dimension_size: int, number_heads: int, feedforward_dimensions: int, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(dimension_size, number_heads, dropout_rate)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension_size, feedforward_dimensions),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dimensions, dimension_size)
        )
        self.norm1 = nn.LayerNorm(dimension_size)
        self.norm2 = nn.LayerNorm(dimension_size)
        self.dropout_rate = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pre-norm self attention
        attention_output = self.self_attention(self.norm1(x), mask)
        x = x + self.dropout_rate(attention_output)

        # pre-norm feed forward
        feed_forward_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout_rate(feed_forward_output)

        return x

class AdvancedPoolingHead(nn.Module):
    """this head supports CLS and maxpool"""
    def __init__(self, dimension_size: int, num_classes: int, pooling_strategy: str = 'CLS', dropout_rate: float = 0.1):
        super().__init__()
        self.dimension_size = dimension_size
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy

        # classification layers
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(dimension_size)
        self.classifier = nn.Linear(dimension_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def pool_representations(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """pool token representations into single sequence representation"""
        if self.pooling_strategy == 'CLS':
            # use [CLS] token (very first token)
            pooled = hidden_states[:, 0, :]

        elif self.pooling_strategy == 'max':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                masked_hidden = hidden_states.clone()
                masked_hidden[mask_expanded == 0] = -1e9
                pooled = masked_hidden.max(dim=1)[0]
            else:
                pooled = masked_hidden.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return pooled

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pool representations
        pooled = self.pool_representations(hidden_states, attention_mask)

        # apply layer norm and dropout
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)

        # classification
        logits = self.classifier(pooled)

        return logits

class BotDetectionTransformer(nn.Module):
    """single transformer model with pooling strategy"""

    def __init__(self, config: ModelConfig, pooling_strategy: str = 'cls'):
        super().__init__()
        self.config = config
        self.pooling_strategy = pooling_strategy

        # embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.dimension_size)
        self.position_embedding = nn.Embedding(config.max_sequence_len, config.dimension_size)

        # transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.dimension_size,
                config.heads,
                config.feedforward_dimensions,
                config.dropout_rate
            ) for _ in range(config.layers)
        ])

        # pooling head
        self.pooling_head = AdvancedPoolingHead(
            config.dimension_size,
            config.num_classes,
            pooling_strategy,
            config.dropout_rate
        )

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """init weights following BERT-style init"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """forward pass"""

        batch_size, sequence_length = input_ids.size()
        device = input_ids.device

        # create position ids
        position_ids = torch.arange(sequence_length, device=device).unsqueeze(0).expand(batch_size, -1)

        # embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        # scale embeddings (from BERT)
        embeddings = embeddings * math.sqrt(self.config.dimension_size)

        # create attention mask for transformer
        if attention_mask is not None:
        # convert to 4D mask for multi-head attention
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        # pass through transformer layers
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, extended_attention_mask)

        # classification
        logits = self.pooling_head(hidden_states, attention_mask)

        return logits

class CLSMaxPoolEnsemble(nn.Module):
    """CLS + MaxPool emsemble for bot detection"""

    def __init__(self, model_config: ModelConfig, ensemble_config: EnsembleConfig):
        super().__init()
        self.model_config = model_config
        self.ensemble_config = ensemble_config

        # primary model ([CLS])
        self.primary_model = BotDetectionTransformer(model_config, pooling_strategy='cls')
        # backup model (maxpool)
        self.backup_model = BotDetectionTransformer(model_config, pooling_strategy='max')

        # ensemble configuration weights
        if ensemble_config.combination_method == 'adaptive':
            self.combination_weights = nn.Parameter(torch.Tensor([
                ensemble_config.primary_weight,
                ensemble_config.backup_weight
            ]))

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                return_individual: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """forward pass thru ensamble"""

        # predictions from both models
        primary_logits = self.primary_model(input_ids, attention_mask)
        backup_logits = self.backup_model(input_ids, attention_mask)

        # combine predictions based on ensamble startegy
        if self.ensemble_config.combination_method == 'weighted_average':
            ensemble_logits = (
                self.ensemble_config.primary_weight * primary_logits +
                self.ensemble_config.backup_weight * backup_logits
            )

        elif self.ensemble_config.combination_method == 'adaptive':
            # learnable combination weights
            weights = F.softmax(self.combination_weights, dim=0)
            ensemble_logits = weights[0] * primary_logits + weights[1] * backup_logits

        elif self.ensemble_config.combination_method == 'confidence_gated':
            # use backup (maxpool) when primary ([CLS]) is unsure
            primary_probabilities = F.softmax(primary_logits, dim=-1)
            primary_confidence = torch.max(primary_probabilities, dim=1)[0]

            # use backup when primary confidence is low
            use_backup = primary_confidence < self.ensemble_config.confidence_threshold
            use_backup = use_backup.unsqueeze(-1).float()

            ensemble_logits = (
                (1 - use_backup) * primary_logits +
                use_backup * backup_logits
            )

        else:
            raise ValueError(f"Unknown combination method: {self.ensemble_config.combination_method}")

        if return_individual:
            return {
                'ensemble': ensemble_logits,
                'primary': primary_logits,
                'backup': backup_logits
            }
        else:
            return ensemble_logits

    def predict_with_reasoning(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """make prediction with reasoning about which model contributed most"""

        with torch.no_grad():
            # get individual predictions
            outputs = self.forward(input_ids, attention_mask, return_individual=True)

            # convert to probabilities
            ensemble_probabilities = F.softmax(outputs['ensemble'], dim=-1)
            primary_probabilities = F.softmax(outputs['primary'], dim=-1)
            backup_probabilities = F.softmax(outputs['backup'], dim=-1)

            # which model is the most confident
            primary_confidence = torch.max(primary_probabilities, dim=1)[0]
            backup_confidence = torch.max(backup_probabilities, dim=1)[0]

            # classification decisions
            ensemble_predictions = torch.argmax(ensemble_probabilities, dim=-1)
            primary_predictions = torch.argmax(primary_probabilities, dim=-1)
            backup_predictions = torch.argmax(backup_probabilities, dim=-1)

            return {
                'predictions': ensemble_predictions,
                'probabilities': ensemble_probabilities,
                'primary_confidence': primary_confidence,
                'backup_confidence': backup_confidence,
                'primary_predictions': primary_predictions,
                'backup_predictions': backup_predictions,
                'agreement': (primary_predictions == backup_predictions).float()
            }

def create_ensemble_model(model_config: ModelConfig, ensemble_config: EnsembleConfig) -> CLSMaxPoolEnsemble:
    """creates CLS+MaxPool ensemble model"""

    model = CLSMaxPoolEnsemble(model_config, ensemble_config)

    # print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"CLS + MaxPool Ensemble Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Primary model (CLS): Sophisticated bot detection")
    print(f"  Backup model (MaxPool): Obvious spam detection")
    print(f"  Ensemble method: {ensemble_config.combination_method}")

    return model