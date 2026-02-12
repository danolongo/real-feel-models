DO tickets: https://cloudsupport.digitalocean.com/s/?teamId=001QP000013YbcxYAC
create new team (after limit is increased): https://cloud.digitalocean.com/account/team/create?i=c31241

will use DO's GPU droplets to train and run inference

# COMPLETE FLOW

# Architecture Set-up

1. ## *config.py*: 
    Dataclasses
    
    - `ModelConfig`: what it has and its role...
    - `TrainingConfig`
    - `DataConfig`
    - `EnsembleConfig`
    - `ExperimentConfig`
    - `get_default_config()`

2. ## *model.py*
    Brief description of what this block does
    - `MultiHeadAttention`
    - `TransformerEncoderLayer`
    - `AdvancedPoolingHead`
    - `BotDetectionTransformer`
    - `CLSMaxPoolEnsemble`
    - `create_ensemble_model()`

3. ## *loss.py*
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

