"""
rf.v1.0.0.loss
This is the third step of the training pipeline

Notes:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class AdvancedLossFunction(nn.Module):
    def __init__(self, num_classes: int = 2, class_weights: Optional[torch.Tensor] = None,
                loss_type: str = 'weighted_ce', focal_alpha: float = 0.25,
                focal_gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        # register class weights as a buffer
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def compute_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()
        total_samples = labels.size(0)

        # inverse frequency weighting: w_i = N / (n_classes * n_i)
        class_weights = total_samples / (self.num_classes * class_counts)

        # handle zero counts
        class_weights = torch.where(class_counts == 0, torch.zeros_like(class_weights), class_weights)

        return class_weights

    def weighted_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # weighted cross-entropy loss

        if self.class_weights is None:
            self.class_weights = self.compute_class_weights(labels)

        return F.cross_entropy(logits, labels, weight=self.class_weights)

    def focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
        where p_t is the model's estimated probability for the true class
        """

        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        p_t = torch.exp(ce_loss)

        # alpha weighting
        if isinstance(self.focal_alpha, (list, tuple, torch.Tensor)):
            alpha_t = self.focal_alpha[labels]
        else:
            alpha_t = self.focal_alpha

        focal_loss = alpha_t * (1 - p_t) ** self.focal_gamma * ce_loss

        return focal_loss.mean()

    def label_smoothed_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        cross-entropy with label smoothing

        theory: replaces hard targets with soft targets:
        y_smooth = (1-ε) * y_true + ε/K
        where ε is smoothing parameter, K is number of classes
        """

        log_probs = F.log_softmax(logits, dim=-1)

        # create smoothed layers
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.label_smoothing / self.num_classes)
        smooth_labels.scatter(1, labels.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / self.num_classes)

        loss = -torch.sum(smooth_labels * log_probs, dim=-1)

        return loss.mean()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'weighted_ce':
            return self.weighted_cross_entropy(logits, labels)

        elif self.loss_type == 'focal':
            return self.focal_loss(logits, labels)

        elif self.loss_type == 'label_smoothed':
            return self.label_smoothed_cross_entropy(logits, labels)

        elif self.loss_type == 'standard_ce':
            return F.cross_entropy(logits, labels)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")