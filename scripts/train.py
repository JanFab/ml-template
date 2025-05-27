import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import matplotlib.pyplot as plt
import argparse
import logging

from src.models.mnist_model import MNISTModel
from src.data.mnist_data_module import MNISTDataModule
from src.utils.visualization import plot_predictions, plot_confusion_matrix

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config: dict):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config['logging']['log_level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MNIST model')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Initialize data module
    data_module = MNISTDataModule(config)
    
    # Initialize model
    model = MNISTModel(config)
    
    # Initialize MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=config['logging']['experiment_name'],
        tracking_uri=config['loging']['tracking_uri']
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        logger=mlflow_logger,
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename='mnist-{epoch:02d}-{val_loss:.2f}'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=config['training']['early_stopping_patience'],
                mode='min'
            )
        ]
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Evaluate model
    results = trainer.test(model, data_module)
    logger.info(f"Test results: {results}")
    
    # Get the current MLflow run
    current_run = mlflow.active_run()
    if current_run is None:
        current_run = mlflow.start_run(run_id=mlflow_logger.run_id)
    
    # Generate and log predictions plot within the current run
    with mlflow.start_run(run_id=current_run.info.run_id, nested=True):
        predictions_fig = plot_predictions(model, data_module.test_dataloader())
        mlflow.log_figure(predictions_fig, "predictions.png")
        plt.close(predictions_fig)
        
        # Generate and log confusion matrix within the current run
        confusion_matrix_fig = plot_confusion_matrix(model, data_module.test_dataloader())
        mlflow.log_figure(confusion_matrix_fig, "confusion_matrix.png")
        plt.close(confusion_matrix_fig)

if __name__ == '__main__':
    main()