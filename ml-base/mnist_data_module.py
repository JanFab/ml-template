import torch
import pytorch_lightning as pl

from torchvision import datasets, transforms
from typing import Dict, Any

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Get data configuration
        self.data_dir = config['data']['data_dir']
        self.batch_size = config['data']['batch_size']
        self.num_workers = config['data']['num_workers']
        

    def setup(self, stage=None):
        # Transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Download full training set
        mnist_full = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transform
        )

        # Split into train and val
        train_size = int(0.9167 * len(mnist_full))  # 55,000
        val_size = len(mnist_full) - train_size     # 5,000
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            mnist_full, [train_size, val_size]
        )

        # Test set
        self.test_dataset = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )