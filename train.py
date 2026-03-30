#!/usr/bin/env python3
"""
Top-level training entry point.
Usage:
    python train.py --data_path rf.v1.0.0/datasets/cresci_2017_merged.csv [options]

Forwards all args to train_ensemble.main().
"""
import sys
from pathlib import Path

# Make rf.v1.0.0 importable as a package root
sys.path.insert(0, str(Path(__file__).parent / "rf.v1.0.0"))

from training_pipeline.train_ensemble import main

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
