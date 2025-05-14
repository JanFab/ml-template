# %% [markdown]
# # MNIST Classification Experiment
# 
# This notebook demonstrates how to use our PyTorch Lightning model for MNIST classification. We'll cover:
# 1. Hardware setup and verification
# 2. Model training
# 3. Visualization of results
# 4. Model evaluation
print("Hello, world!")

# %%
import torch
import pytorch_lightning as pl
from MNISTModel import MNISTModel
from MnistDataModule import MnistDataModule
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
from utils import show_predictions, plot_confusion_matrix

# Set matplotlib style
# plt.style.use('seaborn-v0_8')

def main():
    # Check available devices
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Initialize data module with reduced number of workers
    data_module = MnistDataModule(num_workers=0, batch_size=512)  # Set to 0 to avoid multiprocessing issues

    # Initialize model
    model = MNISTModel()

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='auto',  # Automatically selects the best available accelerator
        max_epochs=3,
        default_root_dir='./logs',
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename='mnist-{epoch:02d}-{val_loss:.2f}'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                mode='min'
            )
        ]
    )

    # Train model
    trainer.fit(model, data_module)

    # Evaluate model
    results = trainer.test(model, data_module)
    print(f"Test results: {results}")

    # Show predictions
    show_predictions(model, data_module.test_dataloader())

    # Plot confusion matrix
    plot_confusion_matrix(model, data_module.test_dataloader())

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Add this line for Windows support
    main()