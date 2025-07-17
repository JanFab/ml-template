import yaml
import argparse
import logging
import pytorch_lightning as pl

from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from mnist_model import MNISTModel
from mnist_data_module import MNISTDataModule

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
    parser.add_argument('--config', type=str, default='train.yaml', help='Path to config file')
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
        tracking_uri=config['logging']['tracking_uri']
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

    
    
if __name__ == '__main__':
    main()