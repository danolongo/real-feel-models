DO's GPU droplets to train and run inference

# COMPLETE FLOW

# Architecture Set-up

1. ## *config.py*: 
    the model's settings
    
    - `ModelConfig`
    - `TrainingConfig`
    - `DataConfig`
    - `EnsembleConfig`
    - `ExperimentConfig`
    - `get_default_config()`

2. ## *model.py*
    forward passes – the stuff that makes the model actually work in training
    - `MultiHeadAttention`
    - `TransformerEncoderLayer`
    - `AdvancedPoolingHead`
    - `BotDetectionTransformer`
    - `CLSMaxPoolEnsemble`
    - `create_ensemble_model()`

3. ## *loss.py*
    loss funcs for the different algos used
    - `AdvancedLossFunction`
    - `EnsembleLoss`
    - `create_loss_function()`

4. ## *optimizer.py*
    - `AdvancedLRScheduler`
    - `AdvancedGradientClipper`
    - `OptimizationManager`

# Data Pipeline

5. ## *data.py*
    - `BotDetectionDataset`
    - `DatasetLoader`
    - `create_data_loaders()`

# Training Pipeline

6. ## *trainer.py*
    - `EnsembleTrainer`
    - `create_ensemble_trainer()`

7. ## *train_ensemble.py*
    - `BotDetectionDataset`
    - `setup_data_loaders()`
    - `save_training_results()`
    - `main()`

# Testing & Validation
unsure about this one

8. ## *test_ensemble.py*
    - `test_model_creation()`
    - `test_dataset_creation()`
    - `test_training_pipeline()`
    - `test_ensemble_strategies()`

