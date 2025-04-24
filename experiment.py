# %% [markdown]
# # MNIST Classification Experiment
# 
# This notebook demonstrates how to use our PyTorch Lightning model for MNIST classification. We'll cover:
# 1. Hardware setup and verification
# 2. Model training
# 3. Visualization of results
# 4. Model evaluation

# %%
import torch
import pytorch_lightning as pl
from model import MNISTModel
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Set matplotlib style
plt.style.use('seaborn')

# Check available devices
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# %% [markdown]
# ## Data Visualization
# 
# Let's take a look at some samples from the MNIST dataset.

# %%
def show_samples(dataset, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Load training data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

# Show samples
show_samples(train_dataset)

# %% [markdown]
# ## Model Training
# 
# Let's train our model with different configurations.

# %%
# Initialize model
model = MNISTModel()

# Initialize trainer
trainer = pl.Trainer(
    accelerator='auto',  # Automatically selects the best available accelerator
    max_epochs=5,
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

# %%
# Train model
trainer.fit(model)

# %% [markdown]
# ## Model Evaluation
# 
# Let's evaluate the model's performance on the test set and visualize some predictions.

# %%
# Load test data
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

# Evaluate model
results = trainer.test(model, dataloaders=test_loader)
print(f"Test results: {results}")

# %%
def show_predictions(model, dataloader, num_images=10):
    model.eval()
    images, labels = next(iter(dataloader))
    with torch.no_grad():
        predictions = model(images)
        predicted_labels = torch.argmax(predictions, dim=1)
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Pred: {predicted_labels[i].item()}\nTrue: {labels[i].item()}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Show predictions
show_predictions(model, test_loader)

# %% [markdown]
# ## Model Analysis
# 
# Let's analyze the model's performance in more detail.

# %%
def plot_confusion_matrix(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            preds = model(x)
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plot confusion matrix
plot_confusion_matrix(model, test_loader)

# %% [markdown]
# ## Model Experimentation
# 
# Let's try different model configurations and compare their performance.

# %%
class MNISTModelV2(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=64, num_workers=4)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data', train=False, transform=transform)
        return DataLoader(dataset, batch_size=64, num_workers=4)

# %%
# Train the new model
model_v2 = MNISTModelV2()
trainer_v2 = pl.Trainer(
    accelerator='auto',
    max_epochs=5,
    default_root_dir='./logs_v2'
)

trainer_v2.fit(model_v2)
results_v2 = trainer_v2.test(model_v2, dataloaders=test_loader)
print(f"Test results for V2 model: {results_v2}") 