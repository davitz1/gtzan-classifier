#training/eval loop
import torch
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np 
import seaborn as sns
from pathlib import Path

def train_epoch(model, loader, criterion, optimizer, device):
  """Train for one epoch"""
  model.train()
  running_loss = 0
  correct = 0
  total = 0

  for X, y in loader:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()

    out = model(X)
    loss = criterion(out, y)

    loss.backward()
    optimizer.step()

    running_loss += loss.item() * X.size(0)

    _, preds = torch.max(out, 1)
    correct += (preds == y).sum().item()
    total += y.size(0)

  epoch_loss = running_loss / total
  epoch_acc = correct / total
  return epoch_loss, epoch_acc

def eval_epoch(model, loader, criterion, device):
  """Evaluate for one epoch"""
  model.eval()
  running_loss = 0
  correct = 0
  total = 0

  with torch.no_grad():
    for X, y in loader:
      X, y = X.to(device), y.to(device)
      out = model(X)
      loss = criterion(out, y)

      running_loss += loss.item() * X.size(0)

      _, preds = torch.max(out, 1)
      correct += (preds == y).sum().item()
      total += y.size(0)

  epoch_loss = running_loss / total
  epoch_acc = correct / total
  return epoch_loss, epoch_acc

def train(model, train_loader, val_loader, epochs, device, lr=0.001):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model.to(device)

  train_losses = []
  val_losses = []
  train_accs = []
  val_accs = []

  for epoch in tqdm(range(epochs), desc="Training"):
    train_loss, train_acc, = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
  return train_losses, val_losses, train_accs, val_accs

def evaluate(model, loader, device):
  """Standalone evaluation"""
  criterion = nn.CrossEntropyLoss()
  loss, acc = eval_epoch(model, loader, criterion, device)
  return loss, acc * 100

def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training and validation loss/accuracy curves"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o', markersize=3)
    plt.plot(epochs, val_losses, label="Val Loss", marker='s', markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc", marker='o', markersize=3)
    plt.plot(epochs, val_accs, label="Val Acc", marker='s', markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    plt.show()

def get_predictions(model, loader, device):
    """Get all predictions and true labels from a dataloader"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, preds = torch.max(out, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix with annotations"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
    
    return cm

def plot_error_analysis(cm, class_names, save_path=None):
    """Plot error rate per class"""
    errors_per_class = cm.sum(axis=1) - np.diag(cm)
    error_rate = errors_per_class / cm.sum(axis=1)

    sorted_idx = np.argsort(-error_rate)
    
    print("\nError Rate by Class (sorted)")
    for idx in sorted_idx:
        print(f"{class_names[idx]}: {error_rate[idx]*100:.2f}% error")
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Reds(error_rate[sorted_idx])
    plt.bar(range(len(class_names)), error_rate[sorted_idx] * 100, color=colors)
    plt.xticks(range(len(class_names)), [class_names[i] for i in sorted_idx], rotation=45, ha='right')
    plt.ylabel("Error Rate (%)")
    plt.title("Error Rate per Class (Sorted)")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error analysis saved to {save_path}")
    plt.show()

def plot_per_class_accuracy(cm, class_names, save_path=None):
    """Plot per-class accuracy"""
    accuracies = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Greens(accuracies)
    plt.bar(class_names, accuracies * 100, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy")
    plt.ylim([0, 105])
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(accuracies * 100):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class accuracy saved to {save_path}")
    plt.show()

def print_classification_report(y_true, y_pred, class_names, save_path=None):
    """Print and save classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report")
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 80 + "\n")
            f.write(report)
        print(f"Classification report saved to {save_path}")

def plot_top_confused_pairs(cm, class_names, top_n=5, save_path=None):
    """Plot the top N most confused class pairs"""
    # Get off-diagonal elements
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    # Find top confused pairs
    flat_indices = np.argsort(cm_no_diag.ravel())[::-1][:top_n]
    confused_pairs = []
    
    for idx in flat_indices:
        true_idx = idx // len(class_names)
        pred_idx = idx % len(class_names)
        count = cm[true_idx, pred_idx]
        if count > 0:
            confused_pairs.append((class_names[true_idx], class_names[pred_idx], count))
    
    print(f"\nTop {top_n} Most Confused Class Pairs")
    for i, (true_class, pred_class, count) in enumerate(confused_pairs, 1):
        print(f"{i}. {true_class} → {pred_class}: {count} times")
    
    if confused_pairs:
        plt.figure(figsize=(10, 6))
        labels = [f"{t}→{p}" for t, p, _ in confused_pairs]
        counts = [c for _, _, c in confused_pairs]
        
        plt.barh(labels, counts, color='coral')
        plt.xlabel('Number of Misclassifications')
        plt.title(f'Top {top_n} Most Confused Class Pairs')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confused pairs plot saved to {save_path}")
        plt.show()