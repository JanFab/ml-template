import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_predictions(model, dataloader, num_samples=10):
    """Plot model predictions on a few samples."""
    model.eval()
    images, labels = next(iter(dataloader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    with torch.no_grad():
        predictions = model(images)
        pred_labels = predictions.argmax(dim=1)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    for i in range(num_samples):
        # Plot image
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f'True: {labels[i]}\nPred: {pred_labels[i]}')
        axes[0, i].axis('off')
        
        # Plot prediction probabilities
        probs = torch.exp(predictions[i])
        axes[1, i].bar(range(10), probs.cpu().numpy())
        axes[1, i].set_xticks(range(10))
        axes[1, i].set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(model, dataloader):
    """Plot confusion matrix for model predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            predictions = model(images)
            pred_labels = predictions.argmax(dim=1)
            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig