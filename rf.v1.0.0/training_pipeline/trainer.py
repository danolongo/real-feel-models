"""
rf.v1.0.0.training-pipeline.trainer
This is the 6th step of the training pipeline

Notes:
    CLS + MaxPool Ensemble Trainer
    Comprehensive training pipeline incorporating all experiment findings
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from setup.config import ExperimentConfig
from setup.model import CLSMaxPoolEnsemble
from setup.loss import AdvancedLossFunction, EnsembleLoss
from setup.optimizer import OptimizationManager


class EnsembleTrainer:
    """
    Comprehensive trainer for CLS + MaxPool ensemble bot detection
    Integrates insights from experiments 3, 4, and 5
    """

    def __init__(self, model: CLSMaxPoolEnsemble, train_loader: DataLoader,
                val_loader: DataLoader, test_loader: DataLoader, config: ExperimentConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.device

        # Move model to device
        self.model.to(self.device)

        # Setup optimization
        total_steps = len(train_loader) * config.training.max_epochs
        self.optimization_manager = OptimizationManager(self.model, config.training, total_steps)

        # Setup loss function
        self.setup_loss_function()

        # Training tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_roc_auc': [],
            'ensemble_agreement': [],  # Track how often primary and backup models agree
            'primary_accuracy': [],   # Track individual model performance
            'backup_accuracy': []
        }

        self.best_model_state = None
        self.best_val_f1 = -1.0  # ensures first epoch always saves
        self.training_start_time = None

        # AMP scaler — only active on CUDA
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

    def setup_loss_function(self):
        """Setup loss function based on configuration"""
        # Base loss configuration
        loss_config = {
            'num_classes': self.config.model.num_classes,
            'loss_type': self.config.training.loss_type,
            'focal_alpha': self.config.training.focal_alpha,
            'focal_gamma': self.config.training.focal_gamma,
            'label_smoothing': self.config.training.label_smoothing
        }

        if self.config.training.use_class_weights:
            # Compute class weights from training data
            all_labels = []
            for batch in self.train_loader:
                all_labels.extend(batch['labels'].tolist())

            labels_tensor = torch.tensor(all_labels)
            loss_fn = AdvancedLossFunction(**loss_config)
            class_weights = loss_fn.compute_class_weights(labels_tensor).to(self.device)
            loss_config['class_weights'] = class_weights

            print(f"Class weights computed: {class_weights}")

        # Create ensemble loss
        self.criterion = EnsembleLoss(loss_config, alpha=0.7)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        loss_components = {'total': 0.0, 'ensemble': 0.0, 'primary': 0.0, 'backup': 0.0}
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with AMP autocast when on GPU
            with torch.autocast(device_type=self.device.type, enabled=self.scaler is not None):
                outputs = self.model(input_ids, attention_mask, return_individual=True)
                loss, batch_loss_components = self.criterion(outputs, labels)

            # Optimization step (passes scaler for AMP-aware backward/step)
            opt_stats = self.optimization_manager.optimization_step(loss, scaler=self.scaler)

            # Track metrics
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += batch_loss_components[key].item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{opt_stats["learning_rate"]:.2e}',
                'grad_norm': f'{opt_stats["gradient_norm"]:.3f}',
                'clipped': opt_stats['gradient_clipped']
            })

        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches

        return {'total_loss': avg_loss, **loss_components}

    def evaluate(self, data_loader: DataLoader, split_name: str = "Val") -> Dict[str, float]:
        """Comprehensive evaluation including ensemble analysis"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        # Individual model tracking
        all_primary_predictions = []
        all_backup_predictions = []
        agreement_count = 0
        total_count = 0

        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=f"Evaluating {split_name}", leave=False)

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Get detailed predictions
                reasoning = self.model.predict_with_reasoning(input_ids, attention_mask)

                # Get ensemble predictions for loss calculation
                outputs = self.model(input_ids, attention_mask, return_individual=True)
                loss, _ = self.criterion(outputs, labels)

                # Store results
                total_loss += loss.item()
                all_predictions.extend(reasoning['predictions'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(reasoning['probabilities'][:, 1].cpu().numpy())

                # Track individual model performance
                all_primary_predictions.extend(reasoning['primary_predictions'].cpu().numpy())
                all_backup_predictions.extend(reasoning['backup_predictions'].cpu().numpy())
                agreement_count += reasoning['agreement'].sum().item()
                total_count += len(labels)

        # Calculate ensemble metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )

        try:
            roc_auc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            roc_auc = 0.0

        # Calculate individual model accuracies
        primary_accuracy = accuracy_score(all_labels, all_primary_predictions)
        backup_accuracy = accuracy_score(all_labels, all_backup_predictions)
        agreement_rate = agreement_count / total_count

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'primary_accuracy': primary_accuracy,
            'backup_accuracy': backup_accuracy,
            'agreement_rate': agreement_rate,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }

    def print_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Print comprehensive epoch results"""
        print(f"\nEpoch {epoch + 1}/{self.config.training.max_epochs}")
        print("-" * 80)
        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
                f"(Ensemble: {train_metrics['ensemble']:.4f}, "
                f"Primary: {train_metrics['primary']:.4f}, "
                f"Backup: {train_metrics['backup']:.4f})")
        print(f"Val Loss:     {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1:       {val_metrics['f1']:.4f}")
        print(f"Val ROC-AUC:  {val_metrics['roc_auc']:.4f}")
        print()
        print(f"Individual Model Performance:")
        print(f"  Primary (CLS):     {val_metrics['primary_accuracy']:.4f}")
        print(f"  Backup (MaxPool):  {val_metrics['backup_accuracy']:.4f}")
        print(f"  Agreement Rate:    {val_metrics['agreement_rate']:.4f}")

        # Show optimization statistics
        opt_stats = self.optimization_manager.get_optimization_statistics()
        if 'learning_rate' in opt_stats:
            print(f"  Learning Rate:     {opt_stats['learning_rate']['current']:.2e}")
        if 'clip_rate' in opt_stats:
            print(f"  Gradient Clip Rate: {opt_stats['clip_rate']:.3f}")

    def save_model(self, path: str):
        """Save model state dict to path"""
        torch.save(self.model.state_dict(), path)

    def save_best_model(self, val_f1: float):
        """Save model if it's the best so far"""
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            print(f"★ New best ensemble model saved (F1: {val_f1:.4f})")
            return True
        return False

    def train(self) -> CLSMaxPoolEnsemble:
        """Complete training loop"""
        print(f"Starting CLS + MaxPool Ensemble Training")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Ensemble strategy: {self.config.ensemble.combination_method}")
        print(f"Primary weight: {self.config.ensemble.primary_weight}")
        print(f"Backup weight: {self.config.ensemble.backup_weight}")
        print("=" * 80)

        self.training_start_time = time.time()

        for epoch in range(self.config.training.max_epochs):
            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.evaluate(self.val_loader, "Val")

            # Update history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_roc_auc'].append(val_metrics['roc_auc'])
            self.history['ensemble_agreement'].append(val_metrics['agreement_rate'])
            self.history['primary_accuracy'].append(val_metrics['primary_accuracy'])
            self.history['backup_accuracy'].append(val_metrics['backup_accuracy'])

            # Print results
            self.print_epoch_results(epoch, train_metrics, val_metrics)

            # Save best model
            self.save_best_model(val_metrics['f1'])

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best ensemble model (F1: {self.best_val_f1:.4f})")

        training_time = time.time() - self.training_start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return self.model

    def test(self) -> Dict[str, float]:
        """Comprehensive testing with ensemble analysis"""
        print("\nEvaluating ensemble on test set...")
        test_metrics = self.evaluate(self.test_loader, "Test")

        print("\nTest Results:")
        print("=" * 60)
        print(f"Ensemble Performance:")
        print(f"  Test Loss:      {test_metrics['loss']:.4f}")
        print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Test Precision: {test_metrics['precision']:.4f}")
        print(f"  Test Recall:    {test_metrics['recall']:.4f}")
        print(f"  Test F1:        {test_metrics['f1']:.4f}")
        print(f"  Test ROC AUC:   {test_metrics['roc_auc']:.4f}")
        print()
        print(f"Individual Model Analysis:")
        print(f"  Primary (CLS) Accuracy:     {test_metrics['primary_accuracy']:.4f}")
        print(f"  Backup (MaxPool) Accuracy:  {test_metrics['backup_accuracy']:.4f}")
        print(f"  Model Agreement Rate:       {test_metrics['agreement_rate']:.4f}")

        # Calculate ensemble benefit
        best_individual = max(test_metrics['primary_accuracy'], test_metrics['backup_accuracy'])
        ensemble_improvement = test_metrics['accuracy'] - best_individual
        print(f"  Ensemble Improvement:       {ensemble_improvement:+.4f}")

        # Detailed classification report
        print("\nDetailed Classification Report:")
        print("-" * 60)
        report = classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=['Human', 'Bot'],
            digits=4
        )
        print(report)

        return test_metrics

    def plot_training_history(self):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CLS + MaxPool Ensemble Training History', fontsize=16)

        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Validation', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy comparison
        axes[0, 1].plot(self.history['val_accuracy'], label='Ensemble', color='green', linewidth=2)
        axes[0, 1].plot(self.history['primary_accuracy'], label='Primary (CLS)', color='blue', linestyle='--')
        axes[0, 1].plot(self.history['backup_accuracy'], label='Backup (MaxPool)', color='orange', linestyle='--')
        axes[0, 1].set_title('Accuracy Comparison')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1 Score
        axes[0, 2].plot(self.history['val_f1'], label='F1 Score', color='purple')
        axes[0, 2].set_title('F1 Score Evolution')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Model agreement
        axes[1, 0].plot(self.history['ensemble_agreement'], label='Agreement Rate', color='brown')
        axes[1, 0].set_title('Primary-Backup Model Agreement')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Agreement Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # ROC AUC
        axes[1, 1].plot(self.history['val_roc_auc'], label='ROC AUC', color='darkgreen')
        axes[1, 1].set_title('ROC AUC Evolution')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('ROC AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Learning rate (if available)
        if hasattr(self.optimization_manager, 'optimization_history'):
            lr_history = self.optimization_manager.optimization_history['learning_rates']
            if lr_history:
                # Sample learning rates at epoch boundaries
                steps_per_epoch = len(self.train_loader)
                epoch_lrs = [lr_history[i * steps_per_epoch] for i in range(len(self.history['train_loss']))
                             if i * steps_per_epoch < len(lr_history)]
                axes[1, 2].plot(epoch_lrs, label='Learning Rate', color='red')
                axes[1, 2].set_title('Learning Rate Schedule')
                axes[1, 2].set_xlabel('Epoch')
                axes[1, 2].set_ylabel('Learning Rate')
                axes[1, 2].set_yscale('log')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'No LR History', ha='center', va='center', transform=axes[1, 2].transAxes)
        else:
            axes[1, 2].text(0.5, 0.5, 'No LR History', ha='center', va='center', transform=axes[1, 2].transAxes)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, test_metrics: Dict[str, float]):
        """Plot confusion matrix for ensemble predictions"""
        cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Human', 'Bot'],
            yticklabels=['Human', 'Bot']
        )
        plt.title('CLS + MaxPool Ensemble - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


def create_ensemble_trainer(model: CLSMaxPoolEnsemble, train_loader: DataLoader,
                            val_loader: DataLoader, test_loader: DataLoader,
                            config: ExperimentConfig) -> EnsembleTrainer:
    """
    Factory function to create ensemble trainer

    Args:
        model: CLS + MaxPool ensemble model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Complete experiment configuration

    Returns:
        Initialized ensemble trainer
    """
    return EnsembleTrainer(model, train_loader, val_loader, test_loader, config)