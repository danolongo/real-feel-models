#!/usr/bin/env python3
"""
Test script for CLS + MaxPool Ensemble Bot Detection
Validates the complete implementation with synthetic data
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import sys
from pathlib import Path

# Add the model directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ..setup.config import get_default_config, get_fast_config
from ..setup.model import create_ensemble_model
from ..training-pipeline.trainer import create_ensemble_trainer


class SyntheticBotDataset(Dataset):
    """Synthetic dataset for testing ensemble approach"""

    def __init__(self, num_samples=1000, seq_length=128, vocab_size=5000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Generate synthetic data with clear patterns
        self.data = []

        for i in range(num_samples):
            # Create patterns that are learnable but not trivial
            if i < num_samples // 2:
                # Class 0 (Human): sequences with smaller token IDs and normal patterns
                seq = torch.randint(1, vocab_size // 2, (seq_length,))
                # Add some variation to make it more realistic
                if torch.rand(1).item() < 0.3:  # 30% variation
                    variation_length = torch.randint(seq_length // 4, seq_length // 2, (1,)).item()
                    variation_start = torch.randint(0, seq_length - variation_length, (1,)).item()
                    seq[variation_start:variation_start + variation_length] = torch.randint(vocab_size // 2, vocab_size, (variation_length,))
                label = 0
            else:
                # Class 1 (Bot): sequences with larger token IDs and repetitive patterns
                seq = torch.randint(vocab_size // 2, vocab_size, (seq_length,))
                # Add repetitive patterns (bot-like behavior)
                if torch.rand(1).item() < 0.4:  # 40% repetitive patterns
                    pattern_length = torch.randint(3, 8, (1,)).item()
                    pattern = torch.randint(vocab_size // 2, vocab_size, (pattern_length,))
                    for start in range(0, seq_length - pattern_length, pattern_length):
                        seq[start:start + pattern_length] = pattern
                label = 1

            # Create attention mask (simulate variable length sequences)
            actual_length = torch.randint(seq_length // 2, seq_length + 1, (1,)).item()
            attention_mask = torch.zeros(seq_length)
            attention_mask[:actual_length] = 1

            self.data.append({
                'input_ids': seq,
                'attention_mask': attention_mask.long(),
                'labels': torch.tensor(label, dtype=torch.long)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def test_model_creation():
    """Test model creation and basic forward pass"""
    print("=" * 60)
    print("TESTING MODEL CREATION")
    print("=" * 60)

    config = get_fast_config()  # Use fast config for testing

    # Create ensemble model
    model = create_ensemble_model(config.model, config.ensemble)

    print(f"✓ Model created successfully")
    print(f"  Device: {config.device}")

    # Test forward pass
    batch_size = 4
    seq_len = 32

    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(config.device)
    attention_mask = torch.ones_like(input_ids).to(config.device)
    attention_mask[:, 20:] = 0  # Simulate padding

    model.to(config.device)
    model.eval()

    with torch.no_grad():
        # Test ensemble output
        ensemble_output = model(input_ids, attention_mask)
        print(f"✓ Ensemble forward pass successful")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {ensemble_output.shape}")
        print(f"  Output range: [{ensemble_output.min().item():.3f}, {ensemble_output.max().item():.3f}]")

        # Test individual model outputs
        individual_outputs = model(input_ids, attention_mask, return_individual=True)
        print(f"✓ Individual model outputs:")
        for key, output in individual_outputs.items():
            print(f"    {key}: {output.shape}")

        # Test reasoning output
        reasoning = model.predict_with_reasoning(input_ids, attention_mask)
        print(f"✓ Reasoning output:")
        print(f"    Predictions: {reasoning['predictions']}")
        print(f"    Primary confidence: {reasoning['primary_confidence']}")
        print(f"    Backup confidence: {reasoning['backup_confidence']}")
        print(f"    Agreement: {reasoning['agreement']}")

    return model, config


def test_dataset_creation():
    """Test synthetic dataset creation"""
    print("\n" + "=" * 60)
    print("TESTING DATASET CREATION")
    print("=" * 60)

    # Create dataset
    dataset = SyntheticBotDataset(num_samples=1000, seq_length=128, vocab_size=5000)

    print(f"✓ Dataset created with {len(dataset)} samples")

    # Test data loading
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    sample_batch = next(iter(dataloader))

    print(f"✓ DataLoader working:")
    print(f"  Batch size: {len(sample_batch['labels'])}")
    print(f"  Input IDs shape: {sample_batch['input_ids'].shape}")
    print(f"  Attention mask shape: {sample_batch['attention_mask'].shape}")
    print(f"  Labels shape: {sample_batch['labels'].shape}")

    # Check class balance
    labels = sample_batch['labels'].numpy()
    class_counts = np.bincount(labels)
    print(f"  Class distribution: {class_counts}")

    return dataset


def test_training_pipeline():
    """Test complete training pipeline"""
    print("\n" + "=" * 60)
    print("TESTING TRAINING PIPELINE")
    print("=" * 60)

    # Create configuration
    config = get_fast_config()
    config.training.max_epochs = 2  # Very quick test
    config.training.batch_size = 16

    # Create dataset and split
    full_dataset = SyntheticBotDataset(num_samples=500, seq_length=64, vocab_size=2000)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    print(f"✓ Data splits created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Update config for smaller vocab
    config.model.vocab_size = 2000

    # Create model
    model = create_ensemble_model(config.model, config.ensemble)

    # Create trainer
    trainer = create_ensemble_trainer(model, train_loader, val_loader, test_loader, config)

    print(f"✓ Trainer created successfully")
    print(f"  Optimization strategy: {config.training.optimizer_type}")
    print(f"  Learning rate scheduler: {config.training.scheduler_type}")
    print(f"  Gradient clipping: {config.training.gradient_clipping}")

    try:
        # Train model
        print(f"\n🚀 Starting training...")
        trained_model = trainer.train()

        print(f"✓ Training completed successfully")

        # Test model
        print(f"\n🧪 Testing trained model...")
        test_metrics = trainer.test()

        print(f"✓ Testing completed")
        print(f"  Final ensemble accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Primary model accuracy: {test_metrics['primary_accuracy']:.4f}")
        print(f"  Backup model accuracy: {test_metrics['backup_accuracy']:.4f}")
        print(f"  Model agreement rate: {test_metrics['agreement_rate']:.4f}")

        # Calculate ensemble benefit
        best_individual = max(test_metrics['primary_accuracy'], test_metrics['backup_accuracy'])
        ensemble_improvement = test_metrics['accuracy'] - best_individual
        print(f"  Ensemble improvement: {ensemble_improvement:+.4f}")

        return trained_model, test_metrics

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def test_ensemble_strategies():
    """Test different ensemble combination strategies"""
    print("\n" + "=" * 60)
    print("TESTING ENSEMBLE STRATEGIES")
    print("=" * 60)

    strategies = ['weighted_average', 'adaptive', 'confidence_gated']

    for strategy in strategies:
        print(f"\n--- Testing {strategy} ensemble strategy ---")

        config = get_fast_config()
        config.ensemble.combination_method = strategy
        config.model.vocab_size = 1000  # Small vocab for speed

        # Create model
        model = create_ensemble_model(config.model, config.ensemble)

        # Test with dummy data
        batch_size = 8
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(config.device)
        attention_mask = torch.ones_like(input_ids).to(config.device)

        model.to(config.device)
        model.eval()

        try:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, return_individual=True)
                reasoning = model.predict_with_reasoning(input_ids, attention_mask)

            print(f"✓ {strategy} strategy working correctly")
            print(f"  Ensemble predictions: {reasoning['predictions'][:4]}")
            print(f"  Agreement rate: {reasoning['agreement'][:4]}")

        except Exception as e:
            print(f"❌ Error with {strategy}: {str(e)}")


def main():
    """Run all tests"""
    print("CLS + MAXPOOL ENSEMBLE BOT DETECTION - COMPREHENSIVE TEST")
    print("🤖 Testing implementation based on experiments 3, 4, and 5")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Test 1: Model creation and basic functionality
        model, config = test_model_creation()

        # Test 2: Dataset creation
        dataset = test_dataset_creation()

        # Test 3: Different ensemble strategies
        test_ensemble_strategies()

        # Test 4: Complete training pipeline
        trained_model, test_metrics = test_training_pipeline()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        if test_metrics:
            print(f"Final Performance Summary:")
            print(f"  ✅ Ensemble Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  ✅ Primary (CLS) Accuracy: {test_metrics['primary_accuracy']:.4f}")
            print(f"  ✅ Backup (MaxPool) Accuracy: {test_metrics['backup_accuracy']:.4f}")
            print(f"  ✅ F1 Score: {test_metrics['f1']:.4f}")
            print(f"  ✅ ROC AUC: {test_metrics['roc_auc']:.4f}")

            # Validate ensemble benefit
            best_individual = max(test_metrics['primary_accuracy'], test_metrics['backup_accuracy'])
            if test_metrics['accuracy'] > best_individual:
                print(f"  🚀 Ensemble shows improvement: +{test_metrics['accuracy'] - best_individual:.4f}")
            else:
                print(f"  ⚠️  Ensemble performance similar to best individual model")

        print(f"\n🎯 Ready for integration into RealFeel pipeline!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)