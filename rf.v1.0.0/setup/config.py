"""
rf.v1.0.0.setup.config
This is the 1st step of the training pipeline

"""

import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model architecture config"""
    d_model: int = 512
    num_layers: int = 9
    num_heads: int = 8 # d_model % num_heads has to == 0
    d_ff: int = d_model * 4 # 2048
    dropout: float = 0.15
    max_seq_length: int = 128
    vocab_size: int = 50265  # RoBERTa vocab size
    num_classes: int = 2 # human or bot

@dataclass
class TrainingConfig:
    # adam = Adaptive Moment Estimation
    # "where weight decay does not accumulate in the momentum nor variance'
    optimizer_type: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01 # regularizer to prevent overfitting
    adam_betas: tuple = (0.9, 0.999) # 1st moment and 2nd moment estimate, "momentum decay rate"
    adam_epsilon: float = 1e-8 # to prevent division by zero

    # learning rate scheduling
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000

    # gradient clipping. handles 'exploding' gradients
    gradient_clippling: bool = True
    clip_type: str = "norm"
    clip_value: float = 1.0

    # training parameters
    # batch size = the number of samples that will be propagated through the network
    # requires less memory to do batches of "" size at a time
    batch_size: int = 32
    max_epochs: int = 10

    # loss function
    use_class_weights: bool = True
    loss_type: str = "weighted_ce" # weighted cross entropy
    # alpha assigns different weights to different classes, balancing the loss function
    focal_alpha: float = 0.25
    # controls the rate at which well-classified (easy) examples are down-weighted
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1 # to avoid overconfidence (how?)

@dataclass
class DataConfig:
    test_size: float = 0.2
    validation_size: float = 0.1
    seed: int = 42
    max_length: int = 128
    vocab_size: int = 50265
    num_workers: int = 4
    pin_mem: bool = True

@dataclass
class EnsembleConfig:
    """CLS (0.7) + MaxPool (0.3)"""
    primary_pool: str = 'cls'
    primary_weight: float = 0.7

    secondary_pool: str = 'max'
    secondary_weight: float = 0.3

    combination_method: str = 'weighted_average'
    confidence_threshold: float = 0.8 # when to only trust primary

    obvious_bot_threshold: float = 0.9
    subtle_bot_threshold: float = 0.6

@dataclass
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    ensemble: EnsembleConfig

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    experiment_name: str = "cls_maxpool_ensemble"
    debug: bool = False

def get_default_config() -> ExperimentConfig:
    return ExperimentConfig(
        model = ModelConfig(),
        training = TrainingConfig(),
        data = DataConfig(),
        ensemble = EnsembleConfig()
    )