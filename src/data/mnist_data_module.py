import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl
from typing import Dict, Any

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Get data configuration
        data_config = config['data']
        self.batch_size = data_config['batch_size']
        self.num_workers = data_config['num_workers']
        self.data_dir = data_config['data_dir']
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.MNIST(
                self.data_dir, 
                train=True, 
                download=True,
                transform=self.transform
            )
            self.val_dataset = datasets.MNIST(
                self.data_dir, 
                train=True, 
                download=True,
                transform=self.transform
            )
            
            # Split training data into train and validation
            train_size = int(0.8 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, _ = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size]
            )
            _, self.val_dataset = torch.utils.data.random_split(
                self.val_dataset, [train_size, val_size]
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(
                self.data_dir, 
                train=False, 
                download=True,
                transform=self.transform
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