import pytorch_lightning as pl
from model import MNISTModel
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train MNIST Model')
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator to use (auto, cpu, gpu, or mps)')
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='Number of epochs to train')
    args = parser.parse_args()

    # Initialize model
    model = MNISTModel()

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        default_root_dir='./logs'
    )

    # Train model
    trainer.fit(model)

if __name__ == '__main__':
    main() 