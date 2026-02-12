"""
rf.v1.0.0.data-pipeline.data
This is the 5th step of the training pipeline

Notes:
    Data loading and preprocessing pipeline for CLS + MaxPool Ensemble Bot Detection
    Connects datasets from /datasets/datasets_full.csv/ to the training pipeline
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import re

from ..setup.config import DataConfig

logger = logging.getLogger(__name__)


class BotDetectionDataset(Dataset):
    """
    PyTorch Dataset for bot detection training
    Handles tweet text and user metadata preprocessing
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        """
        Initialize dataset with preprocessed texts and labels

        Args:
            texts: List of tweet texts
            labels: List of labels (0=human, 1=bot)
            tokenizer: HuggingFace tokenizer for text encoding
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        assert len(texts) == len(labels), f"Texts ({len(texts)}) and labels ({len(labels)}) must have same length"

        logger.info(f"Created BotDetectionDataset with {len(self.texts)} samples")
        logger.info(f"Label distribution: {np.bincount(labels)}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample

        Returns:
            Dictionary with input_ids, attention_mask, and labels tensors
        """
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DatasetLoader:
    """
    Handles loading and preprocessing of bot detection datasets
    Supports multiple dataset formats and sources
    """

    def __init__(self, dataset_root: str):
        """
        Initialize dataset loader

        Args:
            dataset_root: Root directory containing dataset folders
        """
        self.dataset_root = Path(dataset_root)
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root {dataset_root} does not exist")

        logger.info(f"Initialized DatasetLoader with root: {self.dataset_root}")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess tweet text for transformer input

        Args:
            text: Raw tweet text

        Returns:
            Preprocessed text
        """
        if pd.isna(text) or text is None:
            return "[EMPTY]"

        text = str(text).strip()

        # Replace URLs and mentions (similar to dataProcessing.py)
        text = self.url_pattern.sub('http://url.removed', text)
        text = self.mention_pattern.sub('@user', text)

        # Clean whitespace
        text = ' '.join(text.split())

        return text if text else "[EMPTY]"

    def load_crowdflower_data(self) -> Tuple[List[str], List[int]]:
        """
        Load and process Crowdflower dataset

        Returns:
            Tuple of (texts, labels) where labels: 0=human, 1=bot
        """
        crowdflower_path = self.dataset_root / "crowdflower_results.csv"

        if not crowdflower_path.exists():
            logger.warning(f"Crowdflower dataset not found at {crowdflower_path}")
            return [], []

        # Load detailed results
        detailed_path = crowdflower_path / "crowdflower_results_detailed.csv"

        if not detailed_path.exists():
            logger.warning(f"Detailed Crowdflower data not found at {detailed_path}")
            return [], []

        try:
            df = pd.read_csv(detailed_path)
            logger.info(f"Loaded Crowdflower data: {len(df)} samples")

            texts = []
            labels = []

            for _, row in df.iterrows():
                # Use twitter_screen_name as text (limited but available)
                text = f"User profile: @{row.get('twitter_screen_name', 'unknown')}"

                # Map class to binary label
                class_label = row.get('class', '').lower()
                if class_label in ['genuine', 'human']:
                    label = 0  # human
                elif class_label in ['bot', 'fake', 'spam']:
                    label = 1  # bot
                else:
                    continue  # skip unknown labels

                texts.append(self.preprocess_text(text))
                labels.append(label)

            logger.info(f"Processed Crowdflower data: {len(texts)} samples")
            return texts, labels

        except Exception as e:
            logger.error(f"Error loading Crowdflower data: {e}")
            return [], []

    def load_tweet_user_data(self, folder_name: str, is_bot: bool) -> Tuple[List[str], List[int]]:
        """
        Load tweet and user data from a folder

        Args:
            folder_name: Name of the dataset folder
            is_bot: Whether this folder contains bot data (True) or human data (False)

        Returns:
            Tuple of (texts, labels)
        """
        folder_path = self.dataset_root / folder_name

        if not folder_path.exists():
            logger.warning(f"Dataset folder not found: {folder_path}")
            return [], []

        texts = []
        labels = []
        label = 1 if is_bot else 0

        # Try to load tweets first
        tweets_path = folder_path / "tweets.csv"
        if tweets_path.exists():
            try:
                tweets_df = pd.read_csv(tweets_path)
                logger.info(f"Loaded {folder_name} tweets: {len(tweets_df)} samples")

                for _, row in tweets_df.iterrows():
                    text = row.get('text', '')
                    if text and not pd.isna(text):
                        texts.append(self.preprocess_text(text))
                        labels.append(label)

            except Exception as e:
                logger.error(f"Error loading tweets from {tweets_path}: {e}")

        # Also load user data as additional samples
        users_path = folder_path / "users.csv"
        if users_path.exists():
            try:
                users_df = pd.read_csv(users_path)
                logger.info(f"Loaded {folder_name} users: {len(users_df)} samples")

                for _, row in users_df.iterrows():
                    # Create synthetic text from user profile
                    screen_name = row.get('screen_name', '')
                    description = row.get('description', '')
                    name = row.get('name', '')

                    # Combine available user info into text
                    user_text_parts = []
                    if name and not pd.isna(name):
                        user_text_parts.append(f"Name: {name}")
                    if screen_name and not pd.isna(screen_name):
                        user_text_parts.append(f"Username: @{screen_name}")
                    if description and not pd.isna(description):
                        user_text_parts.append(f"Bio: {description}")

                    if user_text_parts:
                        user_text = " | ".join(user_text_parts)
                        texts.append(self.preprocess_text(user_text))
                        labels.append(label)

            except Exception as e:
                logger.error(f"Error loading users from {users_path}: {e}")

        logger.info(f"Processed {folder_name}: {len([l for l in labels if l == label])} samples")
        return texts, labels

    def load_all_datasets(self) -> Tuple[List[str], List[int]]:
        """
        Load all available datasets and combine them

        Returns:
            Combined (texts, labels) from all datasets
        """
        all_texts = []
        all_labels = []

        # Dataset mapping: folder_name -> is_bot
        dataset_mapping = {
            'genuine_accounts.csv': False,           # Human accounts
            'fake_followers.csv': True,              # Bot accounts
            'social_spambots_1.csv': True,          # Bot accounts
            'social_spambots_2.csv': True,          # Bot accounts
            'social_spambots_3.csv': True,          # Bot accounts
            'traditional_spambots_1.csv': True,     # Bot accounts
            'traditional_spambots_2.csv': True,     # Bot accounts
            'traditional_spambots_3.csv': True,     # Bot accounts
            'traditional_spambots_4.csv': True,     # Bot accounts
        }

        # Load regular datasets
        for folder_name, is_bot in dataset_mapping.items():
            texts, labels = self.load_tweet_user_data(folder_name, is_bot)
            all_texts.extend(texts)
            all_labels.extend(labels)

        # Load Crowdflower data
        cf_texts, cf_labels = self.load_crowdflower_data()
        all_texts.extend(cf_texts)
        all_labels.extend(cf_labels)

        logger.info(f"Total loaded samples: {len(all_texts)}")
        logger.info(f"Label distribution: {np.bincount(all_labels)}")

        return all_texts, all_labels


def create_data_loaders(data_config: DataConfig, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders for bot detection

    Args:
        data_config: Data configuration object
        model_name: HuggingFace model name for tokenizer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Creating data loaders for bot detection training")

    # Find dataset root
    current_dir = Path(__file__).parent
    dataset_root = current_dir.parent / "datasets" / "datasets_full.csv"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    dataset_loader = DatasetLoader(str(dataset_root))
    texts, labels = dataset_loader.load_all_datasets()

    if len(texts) == 0:
        raise ValueError("No data loaded from datasets")

    # Convert to numpy arrays for splitting
    texts = np.array(texts)
    labels = np.array(labels)

    # Split into train and temp (val + test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels,
        test_size=(data_config.test_size + data_config.val_size),
        random_state=data_config.random_state,
        stratify=labels
    )

    # Split temp into validation and test
    val_ratio = data_config.val_size / (data_config.test_size + data_config.val_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=(1 - val_ratio),
        random_state=data_config.random_state,
        stratify=temp_labels
    )

    logger.info(f"Data splits - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    logger.info(f"Train label distribution: {np.bincount(train_labels)}")
    logger.info(f"Val label distribution: {np.bincount(val_labels)}")
    logger.info(f"Test label distribution: {np.bincount(test_labels)}")

    # Create datasets
    train_dataset = BotDetectionDataset(
        train_texts.tolist(), train_labels.tolist(), tokenizer, data_config.max_length
    )
    val_dataset = BotDetectionDataset(
        val_texts.tolist(), val_labels.tolist(), tokenizer, data_config.max_length
    )
    test_dataset = BotDetectionDataset(
        test_texts.tolist(), test_labels.tolist(), tokenizer, data_config.max_length
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Will be overridden by training config
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )

    logger.info("Data loaders created successfully")
    return train_loader, val_loader, test_loader


def get_dataset_statistics(data_config: DataConfig) -> Dict[str, Union[int, Dict]]:
    """
    Get comprehensive statistics about the loaded datasets

    Args:
        data_config: Data configuration object

    Returns:
        Dictionary with dataset statistics
    """
    current_dir = Path(__file__).parent
    dataset_root = current_dir.parent / "datasets" / "datasets_full.csv"

    dataset_loader = DatasetLoader(str(dataset_root))
    texts, labels = dataset_loader.load_all_datasets()

    # Calculate statistics
    total_samples = len(texts)
    bot_samples = sum(labels)
    human_samples = total_samples - bot_samples

    # Text length statistics
    text_lengths = [len(text.split()) for text in texts]

    return {
        'total_samples': total_samples,
        'bot_samples': bot_samples,
        'human_samples': human_samples,
        'bot_ratio': bot_samples / total_samples if total_samples > 0 else 0,
        'text_length_stats': {
            'mean': np.mean(text_lengths),
            'median': np.median(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths),
            'std': np.std(text_lengths)
        }
    }


def test_data_pipeline():
    """Test the data loading pipeline - alternative to existing train_ensemble.py"""
    from .config import get_default_config

    logging.basicConfig(level=logging.INFO)

    config = get_default_config()

    print("Testing data loading pipeline...")
    print("=" * 60)

    # Get dataset statistics
    stats = get_dataset_statistics(config.data)
    print(f"Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Bot samples: {stats['bot_samples']}")
    print(f"  Human samples: {stats['human_samples']}")
    print(f"  Bot ratio: {stats['bot_ratio']:.3f}")
    print(f"  Text length - Mean: {stats['text_length_stats']['mean']:.1f}, "
          f"Max: {stats['text_length_stats']['max']}")

    # Test data loader creation
    try:
        train_loader, val_loader, test_loader = create_data_loaders(config.data)
        print(f"\nData loaders created successfully:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Test a sample batch
        sample_batch = next(iter(train_loader))
        print(f"\nSample batch shape:")
        print(f"  Input IDs: {sample_batch['input_ids'].shape}")
        print(f"  Attention mask: {sample_batch['attention_mask'].shape}")
        print(f"  Labels: {sample_batch['labels'].shape}")
        print(f"  Label distribution in batch: {torch.bincount(sample_batch['labels'])}")

        print("\n✓ Data pipeline test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Error in data pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_data_pipeline()