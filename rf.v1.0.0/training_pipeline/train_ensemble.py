#!/usr/bin/env python3
"""
rf.v1.0.0.training-pipeline.trainer
This is the 7th step of the training pipeline

Notes:
    Production Training Script for CLS + MaxPool Ensemble Bot Detection
    Designed for EC2 deployment with real datasets
    Usage: python train_ensemble.py --config production --data_path /path/to/dataset --output_dir ./trained_models
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import os
import torch
from typing import Dict, Any, Optional

# Ensure rf.v1.0.0/ is on the path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from setup.config import get_default_config, get_fast_config, get_production_config
from setup.model import create_ensemble_model
from training_pipeline.trainer import create_ensemble_trainer
from torch.utils.data import Dataset, DataLoader, random_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BotDetectionDataset(Dataset):
    """
    Production dataset loader for bot detection training
    Supports Cresci-2017 dataset structure and various file formats
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        """
        Initialize dataset from file or Cresci-2017 directory structure

        Args:
            data_path: Path to dataset file or Cresci-2017 directory
            tokenizer: Transformers tokenizer
            max_length: Maximum sequence length
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def _load_data(self):
        """Load data from various file formats or Cresci-2017 structure"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        # Check if this is Cresci-2017 dataset structure
        if self.data_path.is_dir() and 'datasets_full.csv' in str(self.data_path):
            return self._load_cresci_dataset()
        elif self.data_path.is_dir():
            return self._load_directory_structure()
        elif self.data_path.suffix.lower() == '.csv':
            return self._load_csv()
        elif self.data_path.suffix.lower() == '.json':
            return self._load_json()
        elif self.data_path.suffix.lower() == '.jsonl':
            return self._load_jsonl()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

    def _load_cresci_dataset(self):
        """Load Cresci-2017 bot detection dataset with proper labeling"""
        data = []

        logger.info("Loading Cresci-2017 dataset structure...")

        # Define bot categories (label=1)
        bot_categories = [
            'fake_followers.csv',
            'social_spambots_1.csv', 'social_spambots_2.csv', 'social_spambots_3.csv',
            'traditional_spambots_1.csv', 'traditional_spambots_2.csv',
            'traditional_spambots_3.csv', 'traditional_spambots_4.csv'
        ]

        # Define human categories (label=0)
        human_categories = ['genuine_accounts.csv']

        # Load bot data (label=1)
        for category in bot_categories:
            category_path = self.data_path / category
            if category_path.exists():
                tweets_file = category_path / 'tweets.csv'
                if tweets_file.exists():
                    bot_data = self._load_cresci_tweets(tweets_file, label=1)
                    data.extend(bot_data)
                    logger.info(f"Loaded {len(bot_data)} bot samples from {category}")

        # Load human data (label=0)
        for category in human_categories:
            category_path = self.data_path / category
            if category_path.exists():
                tweets_file = category_path / 'tweets.csv'
                if tweets_file.exists():
                    human_data = self._load_cresci_tweets(tweets_file, label=0)
                    data.extend(human_data)
                    logger.info(f"Loaded {len(human_data)} human samples from {category}")

        if not data:
            raise ValueError(f"No valid Cresci-2017 data found in {self.data_path}")

        # Log class distribution
        bot_count = sum(1 for item in data if item['label'] == 1)
        human_count = sum(1 for item in data if item['label'] == 0)
        logger.info(f"Dataset composition: {human_count} human, {bot_count} bot samples")

        return data

    def _load_cresci_tweets(self, tweets_file: Path, label: int):
        """Load tweets from Cresci-2017 format CSV file"""
        import pandas as pd

        try:
            df = pd.read_csv(tweets_file)

            # Check if 'text' column exists
            if 'text' not in df.columns:
                logger.warning(f"No 'text' column found in {tweets_file}")
                return []

            data = []
            for _, row in df.iterrows():
                text = str(row['text']).strip()

                # Skip empty or very short tweets
                if len(text) < 10:
                    continue

                # Skip retweets for cleaner training data
                if text.startswith('RT @'):
                    continue

                data.append({
                    'text': text,
                    'label': label
                })

            return data

        except Exception as e:
            logger.warning(f"Error loading {tweets_file}: {e}")
            return []

    def _load_directory_structure(self):
        """Load from generic directory structure with CSV files"""
        data = []
        csv_files = list(self.data_path.rglob("*.csv"))

        for csv_file in csv_files:
            try:
                file_data = self._load_csv_file(csv_file)
                data.extend(file_data)
                logger.info(f"Loaded {len(file_data)} samples from {csv_file}")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")

        return data

    def _load_csv(self):
        """Load single CSV format: text,label columns"""
        return self._load_csv_file(self.data_path)

    def _load_csv_file(self, csv_path: Path):
        """Load CSV format from specific file: text,label columns"""
        import pandas as pd
        df = pd.read_csv(csv_path)

        # Validate required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"CSV must contain 'text' and 'label' columns. Found: {list(df.columns)}")

        data = []
        for _, row in df.iterrows():
            text = str(row['text']).strip()
            if len(text) >= 10:  # Skip very short texts
                data.append({
                    'text': text,
                    'label': int(row['label'])  # 0=human, 1=bot
                })
        return data

    def _load_json(self):
        """Load JSON format: [{text: str, label: int}, ...]"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate format
        for i, item in enumerate(data[:5]):  # Check first 5 items
            if 'text' not in item or 'label' not in item:
                raise ValueError(f"JSON items must contain 'text' and 'label' keys. Item {i}: {item}")

        return data

    def _load_jsonl(self):
        """Load JSONL format: one JSON object per line"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    if 'text' not in item or 'label' not in item:
                        logger.warning(f"Skipping line {line_num}: missing text or label")
                        continue
                    data.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON at line {line_num}")
                    continue
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item['text'])
        label = int(item['label'])

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def setup_data_loaders(config, data_path: str, tokenizer):
    """
    Create train/val/test data loaders from dataset

    Args:
        config: Training configuration
        data_path: Path to dataset file
        tokenizer: Transformers tokenizer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load full dataset (supports Cresci-2017 and custom formats)
    full_dataset = BotDetectionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config.model.max_seq_length
    )

    # Log dataset information
    sample_data = full_dataset.data[:5]  # Get first 5 samples for inspection
    logger.info("Dataset sample preview:")
    for i, sample in enumerate(sample_data):
        label_text = "BOT" if sample['label'] == 1 else "HUMAN"
        preview_text = sample['text'][:100] + "..." if len(sample['text']) > 100 else sample['text']
        logger.info(f"  [{i+1}] {label_text}: {preview_text}")

    # Log class distribution
    labels = [item['label'] for item in full_dataset.data]
    bot_count = sum(labels)
    human_count = len(labels) - bot_count
    bot_ratio = bot_count / len(labels) * 100
    human_ratio = human_count / len(labels) * 100
    logger.info(f"Class distribution: {human_count} human ({human_ratio:.1f}%), {bot_count} bot ({bot_ratio:.1f}%)")

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    return train_loader, val_loader, test_loader


def save_training_results(config, metrics: Dict[str, Any], model_path: str, output_dir: str):
    """Save training results and configuration"""
    results = {
        'training_config': {
            'model_config': {
                'd_model': config.model.d_model,
                'num_layers': config.model.num_layers,
                'num_heads': config.model.num_heads,
                'max_seq_length': config.model.max_seq_length,
                'num_classes': config.model.num_classes
            },
            'training_config': {
                'optimizer_type': config.training.optimizer_type,
                'learning_rate': config.training.learning_rate,
                'batch_size': config.training.batch_size,
                'max_epochs': config.training.max_epochs,
                'loss_type': config.training.loss_type
            },
            'ensemble_config': {
                'primary_pool': config.ensemble.primary_pool,
                'backup_pool': config.ensemble.backup_pool,
                'combination_method': config.ensemble.combination_method,
                'primary_weight': config.ensemble.primary_weight,
                'backup_weight': config.ensemble.backup_weight
            }
        },
        'final_metrics': {
            k: float(v) if hasattr(v, 'item') or isinstance(v, (int, float)) else None
            for k, v in metrics.items()
            if not isinstance(v, (list, np.ndarray))
        },
        'model_path': model_path,
        'training_timestamp': datetime.now().isoformat(),
        'device': str(config.device)
    }

    # Save results
    results_path = Path(output_dir) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training results saved to {results_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train CLS + MaxPool Ensemble Bot Detection Model')
    parser.add_argument('--config', choices=['default', 'fast', 'production'],
                       default='production', help='Configuration preset')
    parser.add_argument('--data_path', required=True, help='Path to training dataset')
    parser.add_argument('--output_dir', default='./trained_models', help='Output directory for models')
    parser.add_argument('--experiment_name', help='Custom experiment name')
    parser.add_argument('--resume', help='Path to checkpoint to resume training')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    if args.config == 'fast':
        config = get_fast_config()
    elif args.config == 'production':
        config = get_production_config()
    else:
        config = get_default_config()

    # Apply overrides
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    # Set device
    if torch.cuda.is_available():
        config.device = torch.device(f'cuda:{args.gpu_id}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        config.device = torch.device('cpu')
        logger.info("Using CPU")

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = True  # faster convolution autotune for fixed input sizes

    logger.info("="*80)
    logger.info("CLS + MAXPOOL ENSEMBLE TRAINING")
    logger.info(f"🤖 Production training for bot detection")
    logger.info(f"📊 Dataset: {args.data_path}")
    logger.info(f"💾 Output: {output_dir}")
    logger.info(f"⚙️  Config: {args.config}")
    logger.info(f"🔧 Device: {config.device}")
    logger.info("="*80)

    try:
        # Import tokenizer here to avoid import issues if transformers not available
        from transformers import AutoTokenizer

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("✓ Tokenizer initialized")

        # Setup data loaders
        train_loader, val_loader, test_loader = setup_data_loaders(config, args.data_path, tokenizer)
        logger.info("✓ Data loaders created")

        # Create model
        model = create_ensemble_model(config.model, config.ensemble)
        model.to(config.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("✓ Ensemble model created")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

        # Create trainer
        trainer = create_ensemble_trainer(model, train_loader, val_loader, test_loader, config)
        logger.info("✓ Trainer initialized")

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Start training
        logger.info("🚀 Starting training...")
        trained_model = trainer.train()
        logger.info("✅ Training completed successfully")

        # Test the trained model
        logger.info("🧪 Testing trained model...")
        test_metrics = trainer.test()
        logger.info("✅ Testing completed")

        # Print final results
        logger.info("\n" + "="*80)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("Final Performance:")
        logger.info(f"  ✅ Ensemble Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  ✅ Primary (CLS) Accuracy: {test_metrics['primary_accuracy']:.4f}")
        logger.info(f"  ✅ Backup (MaxPool) Accuracy: {test_metrics['backup_accuracy']:.4f}")
        logger.info(f"  ✅ F1 Score: {test_metrics['f1']:.4f}")
        logger.info(f"  ✅ ROC AUC: {test_metrics['roc_auc']:.4f}")

        # Calculate ensemble improvement
        best_individual = max(test_metrics['primary_accuracy'], test_metrics['backup_accuracy'])
        ensemble_improvement = test_metrics['accuracy'] - best_individual
        logger.info(f"  🚀 Ensemble Improvement: {ensemble_improvement:+.4f}")

        # Save model and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{config.experiment_name}_{timestamp}_best.pt"
        model_path = output_dir / model_filename

        trainer.save_model(str(model_path))
        logger.info(f"💾 Model saved to: {model_path}")

        # Save training results
        save_training_results(config, test_metrics, str(model_path), str(output_dir))

        logger.info(f"\n🎯 Training complete! Model ready for production deployment.")
        return True

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)