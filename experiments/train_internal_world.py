import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, confusion_matrix,
    precision_recall_curve, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
from tqdm import tqdm

# Add parent directory to path so we can import the models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gemma3_internal_world import Gemma3InternalWorldModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train the Internal World model for wildfire prediction')
    parser.add_argument('--data_path', type=str, default="../data/Date_final_dataset_balanced_float32.parquet",
                        help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default="../models/saved",
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--gemma_path', type=str, default="google/gemma-3-1b-it",
                        help='Path to pretrained Gemma model')
    
    return parser.parse_args()

def load_morocco_wildfire_data(data_path):
    """Load the Morocco wildfire dataset or create simulated data if not available."""
    try:
        df = pd.read_parquet(data_path)
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print(f"File {data_path} not found. Creating simulated data...")
        # Create simulated data with appropriate structure
        n_samples = 10000
        n_features = 276  # Matching the real dataset feature count
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (np.random.rand(n_samples) > 0.5).astype(np.float32)
        dates = pd.date_range(start='2010-01-01', end='2022-12-31', periods=n_samples)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['is_fire'] = y
        df['acq_date'] = dates
        print(f"Created simulated dataset with shape: {df.shape}")
    
    return df

def prepare_data(df, batch_size=128):
    """Split data, balance classes, standardize, and create data loaders."""
    print("Performing time-based train/validation split...")
    df_train = df[df.acq_date < '2022-01-01']
    df_valid = df[df.acq_date >= '2022-01-01']
    
    print(f"Training set (before balancing): {df_train.shape}")
    print(f"Validation set (before balancing): {df_valid.shape}")
    
    # Balance datasets by class
    print("Balancing datasets...")
    min_samples_train = df_train['is_fire'].value_counts().min()
    min_samples_valid = df_valid['is_fire'].value_counts().min()
    
    # Balance the training dataset
    df_train_balanced = pd.concat([
        df_train[df_train['is_fire'] == 0].sample(min_samples_train, random_state=42),
        df_train[df_train['is_fire'] == 1].sample(min_samples_train, random_state=42)
    ])
    
    # Balance the validation dataset
    df_valid_balanced = pd.concat([
        df_valid[df_valid['is_fire'] == 0].sample(min_samples_valid, random_state=42),
        df_valid[df_valid['is_fire'] == 1].sample(min_samples_valid, random_state=42)
    ])
    
    # Shuffle both datasets
    df_train_balanced = df_train_balanced.sample(frac=1, random_state=42)
    df_valid_balanced = df_valid_balanced.sample(frac=1, random_state=42)
    
    print(f"Balanced training set: {df_train_balanced.shape}")
    print(f"Balanced validation set: {df_valid_balanced.shape}")
    
    # Remove acquisition date column
    print("Removing acquisition date column...")
    if 'acq_date' in df_train_balanced.columns:
        df_train_balanced = df_train_balanced.drop(columns=['acq_date'])
        df_valid_balanced = df_valid_balanced.drop(columns=['acq_date'])
    
    # Extract features and target
    print("Preparing feature sets...")
    y_train = df_train_balanced['is_fire']
    X_train = df_train_balanced.drop(columns=['is_fire'])
    
    y_valid = df_valid_balanced['is_fire']
    X_valid = df_valid_balanced.drop(columns=['is_fire'])
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    n_features = X_train_scaled.shape[1]
    print(f"Total number of features: {n_features}")
    
    # Create tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_valid_tensor = torch.FloatTensor(X_valid_scaled)
    y_valid_tensor = torch.FloatTensor(y_valid.values).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, n_features

def find_optimal_threshold(targets, outputs):
    """Find the optimal threshold for classification based on F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(targets, outputs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find best threshold (avoiding index out of range error)
    best_idx = np.argmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 and best_idx < len(thresholds) else 0.5
    return best_threshold

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, threshold=0.5):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    all_targets = []
    all_outputs = []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step()
        
        # Track statistics
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        
        # Store for metrics calculation
        all_targets.extend(targets.cpu().numpy().ravel())
        all_outputs.extend(outputs.detach().cpu().numpy().ravel())
        
        # Update progress bar
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(all_targets)
    preds = (np.array(all_outputs) > threshold).astype(float)
    epoch_acc = (preds == all_targets).mean()
    epoch_auc = roc_auc_score(all_targets, all_outputs)
    
    return epoch_loss, epoch_acc, epoch_auc, all_targets, all_outputs

def evaluate(model, dataloader, criterion, device, threshold=0.5, find_best_threshold=False):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track statistics
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            
            # Store for metrics calculation
            all_targets.extend(targets.cpu().numpy().ravel())
            all_outputs.extend(outputs.cpu().numpy().ravel())
    
    # Find best threshold if requested
    if find_best_threshold:
        threshold = find_optimal_threshold(all_targets, all_outputs)
        print(f"Optimal threshold: {threshold:.4f}")
    
    # Apply threshold to get predictions
    preds = (np.array(all_outputs) > threshold).astype(float)
    
    # Calculate metrics
    epoch_loss = total_loss / len(all_targets)
    epoch_acc = (preds == all_targets).mean()
    epoch_auc = roc_auc_score(all_targets, all_outputs)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, preds, average='binary')
    cm = confusion_matrix(all_targets, preds)
    
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

def plot_confusion_matrix(cm, output_dir, model_name="internal_world"):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fire', 'Fire'],
                yticklabels=['No Fire', 'Fire'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
    plt.close()

def plot_training_curves(history, output_dir, model_name="internal_world"):
    """Plot training and validation curves."""
    plt.figure(figsize=(16, 12))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # AUC
    plt.subplot(2, 2, 3)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('AUC Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Precision, Recall, F1
    plt.subplot(2, 2, 4)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.title('Precision, Recall, and F1 Score', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Curves - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{model_name}_training_curves.png")
    plt.close()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    df = load_morocco_wildfire_data(args.data_path)
    train_loader, valid_loader, n_features = prepare_data(df, batch_size=args.batch_size)
    
    # Create model
    print("\nCreating Gemma3 Internal World Model...")
    model = Gemma3InternalWorldModel(
        n_features=n_features,
        gemma_path=args.gemma_path,
        dropout_rate=args.dropout
    ).to(device)
    
    # Count parameters
    params = model.count_parameters()
    print(f"Model has {params['trainable']:,} trainable parameters out of {params['total']:,} total")
    print(f"Percentage of trainable parameters: {params['trainable_percentage']:.2f}%")
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (OneCycleLR)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training loop
    print("\n===== Training Gemma3 Internal World Model =====")
    
    # For storing history
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'threshold': []
    }
    
    # For early stopping
    best_val_f1 = 0.0
    best_epoch = -1
    best_threshold = 0.5
    best_model_path = f"{args.output_dir}/gemma3_internal_world_best_model.pt"
    patience_counter = 0
    
    # Record start time
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Training phase
        train_loss, train_acc, train_auc, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler, best_threshold
        )
        
        # Validation phase (finding best threshold every 2 epochs)
        find_best_threshold = (epoch % 2 == 0) or (epoch == args.epochs - 1)
        val_metrics = evaluate(
            model, valid_loader, criterion, device, 
            threshold=best_threshold, find_best_threshold=find_best_threshold
        )
        
        if find_best_threshold:
            best_threshold = val_metrics['threshold']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['threshold'].append(best_threshold)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Training:   loss={train_loss:.4f}, acc={train_acc:.4f}, auc={train_auc:.4f}")
        print(f"  Validation: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, "
              f"auc={val_metrics['auc']:.4f}")
        print(f"             precision={val_metrics['precision']:.4f}, "
              f"recall={val_metrics['recall']:.4f}, f1={val_metrics['f1']:.4f}")
        print(f"             threshold={best_threshold:.4f}")
        
        # Save best model based on F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! (F1={val_metrics['f1']:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs. "
                  f"Best F1={best_val_f1:.4f} at epoch {best_epoch+1}")
            
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"\nGemma3 Internal World Model training time: {training_time:.2f} seconds")
    
    # Load best model for final evaluation
    if best_epoch >= 0:
        print(f"Loading best model from epoch {best_epoch+1}...")
        model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation with optimal threshold
    final_metrics = evaluate(
        model, valid_loader, criterion, device, threshold=best_threshold, find_best_threshold=False
    )
    
    # Print final results
    print(f"\nGemma3 Internal World Model Final Results:")
    print(f"  Test Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Test AUC: {final_metrics['auc']:.4f}")
    print(f"  Test Precision: {final_metrics['precision']:.4f}")
    print(f"  Test Recall: {final_metrics['recall']:.4f}")
    print(f"  Test F1 Score: {final_metrics['f1']:.4f}")
    print(f"  Optimal Threshold: {best_threshold:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(final_metrics['confusion_matrix'], args.output_dir)
    
    # Plot training curves
    plot_training_curves(history, args.output_dir)
    
    # Save results to a text file
    with open(f"{args.output_dir}/results.txt", "w") as f:
        f.write(f"Gemma3 Internal World Model Final Results:\n")
        f.write(f"  Test Accuracy: {final_metrics['accuracy']:.4f}\n")
        f.write(f"  Test AUC: {final_metrics['auc']:.4f}\n")
        f.write(f"  Test Precision: {final_metrics['precision']:.4f}\n")
        f.write(f"  Test Recall: {final_metrics['recall']:.4f}\n")
        f.write(f"  Test F1 Score: {final_metrics['f1']:.4f}\n")
        f.write(f"  Optimal Threshold: {best_threshold:.4f}\n")
        f.write(f"  Training time: {training_time:.2f} seconds\n")
        f.write(f"  Total parameters: {params['total']:,}\n")
        f.write(f"  Trainable parameters: {params['trainable']:,}\n")
        f.write(f"  Frozen parameters: {params['frozen']:,}\n")
    
    print("\nResults saved to", f"{args.output_dir}/results.txt")
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
