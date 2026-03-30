"""
rf.v1.0.0.setup.optimizer
This is the 4th step of the training pipeline

"""

import torch
import torch.optim as optim
import math
import numpy as np

class AdvancedLRScheduler:
    """Advanced learning rate schedulers for transformer optimization"""

    @staticmethod
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
        """
        Linear warmup followed by linear decay

        Theory: lr = base_lr * min(step/warmup_steps, (total_steps - step)/(total_steps - warmup_steps))
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int,
                                        num_cycles: float = 0.5, last_epoch: int = -1):
        """
        Cosine warmup followed by cosine annealing

        Theory: Smooth transitions help with convergence stability
        lr = base_lr * 0.5 * (1 + cos(π * cycles * progress))
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def get_polynomial_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int,
                                            power: float = 1.0, last_epoch: int = -1):
        """
        Polynomial decay with warmup

        Theory: lr = base_lr * (1 - progress)^power
        Power controls decay rate: power=1 is linear, power>1 is faster decay
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step > num_training_steps:
                return 0.0
            else:
                lr_range = float(num_training_steps - num_warmup_steps)
                decay_steps = float(current_step - num_warmup_steps)
                pct_remaining = 1 - decay_steps / lr_range
                return pct_remaining ** power

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class AdvancedGradientClipper:
    """Advanced gradient clipping strategies for stable training"""

    def __init__(self, clip_type: str = 'norm', clip_value: float = 1.0, adaptive_factor: float = 0.99):
        """
        Args:
            clip_type: 'norm', 'value', 'adaptive_norm', or 'percentile'
            clip_value: Clipping threshold
            adaptive_factor: EMA factor for adaptive clipping
        """
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.adaptive_factor = adaptive_factor
        self.running_grad_norm = None
        self.grad_norm_history = []

    def clip_gradients(self, model) -> dict:
        """
        Apply gradient clipping to model parameters

        Returns:
            dict with clipping statistics
        """
        # Compute current gradient norm
        total_norm = 0.0
        param_count = 0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += param.numel()

        total_norm = total_norm ** (1. / 2)
        self.grad_norm_history.append(total_norm)

        # Apply clipping based on strategy
        if self.clip_type == 'norm':
            # Standard gradient norm clipping
            clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
            was_clipped = clipped_norm > self.clip_value

        elif self.clip_type == 'value':
            # Clip individual gradient values
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
            clipped_norm = total_norm
            was_clipped = False  # Hard to determine for value clipping

        elif self.clip_type == 'adaptive_norm':
            # Adaptive gradient norm clipping
            if self.running_grad_norm is None:
                self.running_grad_norm = total_norm
            else:
                self.running_grad_norm = (self.adaptive_factor * self.running_grad_norm +
                                        (1 - self.adaptive_factor) * total_norm)

            adaptive_clip_value = self.clip_value * self.running_grad_norm
            clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), adaptive_clip_value)
            was_clipped = clipped_norm > adaptive_clip_value

        elif self.clip_type == 'percentile':
            # Percentile-based clipping (clip based on gradient norm percentiles)
            if len(self.grad_norm_history) > 100:  # Need some history
                clip_threshold = np.percentile(self.grad_norm_history[-100:], 95)
                clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
                was_clipped = clipped_norm > clip_threshold
            else:
                clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
                was_clipped = clipped_norm > self.clip_value

        else:
            # No clipping
            clipped_norm = total_norm
            was_clipped = False

        return {
            'original_norm': total_norm,
            'clipped_norm': clipped_norm.item() if isinstance(clipped_norm, torch.Tensor) else clipped_norm,
            'was_clipped': was_clipped,
            'param_count': param_count
        }

    def get_statistics(self) -> dict:
        """Get gradient norm statistics"""
        if not self.grad_norm_history:
            return {}

        norms = np.array(self.grad_norm_history)
        return {
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'max_norm': np.max(norms),
            'min_norm': np.min(norms),
            'percentile_95': np.percentile(norms, 95),
            'percentile_99': np.percentile(norms, 99)
        }


class OptimizerFactory:
    """Factory for creating optimizers with consistent configurations"""

    @staticmethod
    def create_adamw(model, lr: float = 2e-4, weight_decay: float = 0.01,
                    betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        """
        Create AdamW optimizer

        Theory: AdamW decouples weight decay from gradient updates:
        θ_t = θ_{t-1} - α * (m_t / (√v_t + ε) + λ * θ_{t-1})
        where λ is weight decay coefficient
        """
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )

    @staticmethod
    def create_adam(model, lr: float = 2e-4, weight_decay: float = 0.01,
                    betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        """
        Create Adam optimizer with L2 regularization

        Theory: Adam with L2 penalty couples weight decay with gradients:
        g_t = ∇f(θ_{t-1}) + λ * θ_{t-1}
        Can interfere with adaptive learning rate computation
        """
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )

    @staticmethod
    def create_sgd(model, lr: float = 0.01, weight_decay: float = 0.01,
                    momentum: float = 0.9, nesterov: bool = True):
        """
        Create SGD with momentum and Nesterov acceleration

        Theory: Nesterov momentum looks ahead:
        v_t = μ * v_{t-1} + g_t
        θ_t = θ_{t-1} - α * (μ * v_t + g_t)
        """
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov
        )


class OptimizationManager:
    """
    Comprehensive optimization manager that coordinates optimizer, scheduler, and clipper
    Based on experiment 5 best practices
    """

    def __init__(self, model, training_config, total_steps: int):
        self.model = model
        self.config = training_config
        self.total_steps = total_steps

        # Setup components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler() if training_config.scheduler_type != 'none' else None
        self.gradient_clipper = self._create_gradient_clipper() if training_config.gradient_clipping else None

        # Monitoring
        self.optimization_history = {
            'learning_rates': [],
            'gradient_norms': [],
            'gradient_clips': [],
            'parameter_norms': []
        }

    def _create_optimizer(self):
        """Create optimizer based on configuration"""
        if self.config.optimizer_type == 'adamw':
            return OptimizerFactory.create_adamw(
                self.model,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.adam_betas,
                eps=self.config.adam_eps
            )
        elif self.config.optimizer_type == 'adam':
            return OptimizerFactory.create_adam(
                self.model,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.adam_betas,
                eps=self.config.adam_eps
            )
        elif self.config.optimizer_type == 'sgd':
            return OptimizerFactory.create_sgd(
                self.model,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

    def _create_scheduler(self):
        """Create learning rate scheduler based on configuration"""
        if self.config.scheduler_type == 'linear':
            return AdvancedLRScheduler.get_linear_schedule_with_warmup(
                self.optimizer, self.config.warmup_steps, self.total_steps
            )
        elif self.config.scheduler_type == 'cosine':
            return AdvancedLRScheduler.get_cosine_schedule_with_warmup(
                self.optimizer, self.config.warmup_steps, self.total_steps
            )
        elif self.config.scheduler_type == 'polynomial':
            return AdvancedLRScheduler.get_polynomial_schedule_with_warmup(
                self.optimizer, self.config.warmup_steps, self.total_steps, power=1.5
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

    def _create_gradient_clipper(self):
        """Create gradient clipper based on configuration"""
        return AdvancedGradientClipper(
            clip_type=self.config.clip_type,
            clip_value=self.config.clip_value
        )

    def optimization_step(self, loss: torch.Tensor, scaler=None) -> dict:
        """
        Perform complete optimization step: backward, clip, step, schedule

        Args:
            loss: Loss tensor to backpropagate
            scaler: Optional torch.cuda.amp.GradScaler for mixed-precision training

        Returns:
            Dictionary with optimization statistics
        """
        # Backward pass
        self.optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)  # unscale before clipping
        else:
            loss.backward()

        # Gradient clipping
        grad_stats = {'original_norm': 0.0, 'was_clipped': False, 'clipped_norm': 0.0}
        if self.gradient_clipper:
            grad_stats = self.gradient_clipper.clip_gradients(self.model)

        # Parameter update
        if scaler is not None:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        # Learning rate update
        if self.scheduler:
            self.scheduler.step()

        # Monitor optimization
        current_lr = self.optimizer.param_groups[0]['lr']
        param_norm = self._compute_parameter_norm()

        self.optimization_history['learning_rates'].append(current_lr)
        self.optimization_history['gradient_norms'].append(grad_stats['original_norm'])
        was_clipped = grad_stats['was_clipped']
        if hasattr(was_clipped, 'item'):
            was_clipped = was_clipped.item()
        self.optimization_history['gradient_clips'].append(was_clipped)
        self.optimization_history['parameter_norms'].append(param_norm)

        return {
            'learning_rate': current_lr,
            'gradient_norm': grad_stats['original_norm'],
            'gradient_clipped': grad_stats['was_clipped'],
            'parameter_norm': param_norm
        }

    def _compute_parameter_norm(self) -> float:
        """Compute total parameter norm"""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                param_norm = param.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def get_optimization_statistics(self) -> dict:
        """Get comprehensive optimization statistics"""
        stats = {}

        if self.optimization_history['learning_rates']:
            stats['learning_rate'] = {
                'current': self.optimization_history['learning_rates'][-1],
                'max': max(self.optimization_history['learning_rates']),
                'min': min(self.optimization_history['learning_rates'])
            }

        if self.gradient_clipper:
            stats['gradient_clipping'] = self.gradient_clipper.get_statistics()
            stats['clip_rate'] = np.mean(self.optimization_history['gradient_clips'])

        if self.optimization_history['parameter_norms']:
            stats['parameter_norm'] = {
                'current': self.optimization_history['parameter_norms'][-1],
                'max': max(self.optimization_history['parameter_norms']),
                'min': min(self.optimization_history['parameter_norms'])
            }

        return stats