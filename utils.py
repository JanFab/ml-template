import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def show_samples(dataset, num_samples=10):
    """Display a grid of sample images from the dataset."""
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def show_predictions(model, dataloader, num_images=10):
    """Display model predictions for a batch of images."""
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

def plot_confusion_matrix(model, dataloader):
    """Plot confusion matrix for model predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            preds = model(x)
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), 
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.show() 