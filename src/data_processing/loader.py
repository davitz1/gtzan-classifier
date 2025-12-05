import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_split_data(data_path, batch_size=32, test_size=0.4, val_split=0.5, 
                        random_state=42, device='cuda'):
    """
    Load MFCC data and create train/val/test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader, genres (list)
    """
    print(f"Loading data from: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    X = data["mfcc"]
    y = data["labels"]
    genres = data["mapping"].tolist()
    
    print(f"Dataset: X shape={X.shape}, y shape={y.shape}")
    print(f"Genres: {genres}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=random_state, stratify=y_temp
    )
    
    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Add channel dimension
    X_train = X_train[:, None, :, :]
    X_val = X_val[:, None, :, :]
    X_test = X_test[:, None, :, :]
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    y_val = torch.from_numpy(y_val).long().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)
    
    # Create dataloaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, genres


