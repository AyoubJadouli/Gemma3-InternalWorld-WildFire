import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, confusion_matrix,
    precision_recall_curve, f1_score
)

def find_optimal_threshold(targets, outputs):
    """
    Find the optimal classification threshold that maximizes F1 score.
    
    Args:
        targets (array-like): Ground truth binary labels
        outputs (array-like): Predicted probabilities
        
    Returns:
        float: Optimal threshold value
    """
    precisions, recalls, thresholds = precision_recall_curve(targets, outputs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find best threshold (avoiding index out of range error)
    best_idx = np.argmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 and best_idx < len(thresholds) else 0.5
    return best_threshold

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, threshold=0.5):
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): PyTorch model
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        scheduler: Learning rate scheduler (optional)
        threshold (float): Decision threshold for binary classification
        
    Returns:
        tuple: (epoch_loss, epoch_acc, epoch_auc, all_targets, all_outputs)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_outputs = []
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Handle models that don't apply sigmoid (BCEWithLogitsLoss)
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            loss = criterion(outputs, targets)
            probs = torch.sigmoid(outputs)
        else:  # nn.BCELoss
            loss = criterion(outputs, targets)
            probs = outputs
        
        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step()
        
        # Track statistics
        total_loss += loss.item() * inputs.size(0)
        total += targets.size(0)
        preds = (probs > threshold).float()
        correct += (preds == targets).sum().item()
        
        # Store for AUC calculation
        all_targets.extend(targets.cpu().numpy().ravel())
        all_outputs.extend(probs.detach().cpu().numpy().ravel())
    
    # Calculate epoch metrics
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    epoch_auc = roc_auc_score(all_targets, all_outputs)
    
    return epoch_loss, epoch_acc, epoch_auc, all_targets, all_outputs

def evaluate(model, dataloader, criterion, device, threshold=0.5, find_best_threshold=False):
    """
    Evaluate model on validation data.
    
    Args:
        model (nn.Module): PyTorch model
        dataloader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
        threshold (float): Decision threshold for binary classification
        find_best_threshold (bool): Whether to find the optimal threshold
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle models that don't apply sigmoid (BCEWithLogitsLoss)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                loss = criterion(outputs, targets)
                probs = torch.sigmoid(outputs)
            else:  # nn.BCELoss
                loss = criterion(outputs, targets)
                probs = outputs
            
            # Track statistics
            total_loss += loss.item() * inputs.size(0)
            
            # Store for metrics calculation
            all_targets.extend(targets.cpu().numpy().ravel())
            all_outputs.extend(probs.cpu().numpy().ravel())
    
    # Find best threshold if requested
    if find_best_threshold:
        threshold = find_optimal_threshold(all_targets, all_outputs)
        print(f"Optimal threshold: {threshold:.4f}")
    
    # Apply threshold to get predictions
    all_preds = (np.array(all_outputs) > threshold).astype(float)
    
    # Calculate metrics
    epoch_loss = total_loss / len(all_targets)
    epoch_acc = (all_preds == all_targets).mean()
    epoch_auc = roc_auc_score(all_targets, all_outputs)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'auc': epoch_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'threshold': threshold,
        'targets': all_targets,
        'outputs': all_outputs
    }

def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def detailed_parameter_counts(model):
    """
    Get detailed parameter counts by module.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        dict: Dictionary with parameter counts by module
    """
    result = {}
    
    for name, module in model.named_modules():
        if name:  # Skip the root module
            params = sum(p.numel() for p in module.parameters(recurse=False))
            trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            
            if params > 0:
                result[name] = {
                    'total': params,
                    'trainable': trainable,
                    'frozen': params - trainable,
                    'trainable_pct': trainable / params * 100 if params > 0 else 0
                }
    
    return result

def print_model_summary(model):
    """
    Print a summary of model architecture and parameters.
    
    Args:
        model (nn.Module): PyTorch model
    """
    total, trainable = count_parameters(model)
    
    print(f"\nModel Summary:")
    print(f"  Total Parameters:      {total:,}")
    print(f"  Trainable Parameters:  {trainable:,}")
    print(f"  Frozen Parameters:     {total - trainable:,}")
    print(f"  Trainable Percentage:  {trainable / total * 100:.2f}%")
    
    # List top modules by parameter count
    details = detailed_parameter_counts(model)
    sorted_modules = sorted(details.items(), key=lambda x: x[1]['total'], reverse=True)
    
    print("\nTop 5 Modules by Parameter Count:")
    for i, (name, stats) in enumerate(sorted_modules[:5]):
        print(f"  {i+1}. {name}")
        print(f"     Total: {stats['total']:,}, Trainable: {stats['trainable']:,} ({stats['trainable_pct']:.2f}%)")
    
    print("\nFrozen Modules:")
    for name, stats in details.items():
        if stats['trainable'] == 0 and stats['total'] > 0:
            print(f"  - {name}: {stats['total']:,} parameters")
