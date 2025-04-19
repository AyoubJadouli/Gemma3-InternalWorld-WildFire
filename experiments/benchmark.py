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
    precision_recall_curve, f1_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
from tqdm import tqdm
import gc

# Add parent directory to path so we can import the models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gemma3_internal_world import Gemma3InternalWorldModel
from models.baseline_models import (
    FFNModel, CNNModel, FFNWithPosEncoding, PhysicsEmbeddedEntropyModel
)

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark different wildfire prediction models')
    parser.add_argument('--data_path', type=str, default="../data/Date_final_dataset_balanced_float32.parquet",
                        help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default="../results",
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--gemma_path', type=str, default="google/gemma-3-1b-it",
                        help='Path to pretrained Gemma model')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to use (for faster testing)')
    
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

def prepare_data(df, batch_size=128, sample_size=None):
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
    
    # If sample_size is provided, limit the number of samples per class
    if sample_size and sample_size < min_samples_train:
        min_samples_train = sample_size
    if sample_size and sample_size < min_samples_valid:
        min_samples_valid = sample_size
        
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
    correct = 0
    total = 0
    all_targets = []
    all_outputs = []
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        
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
        preds = (torch.sigmoid(logits) > threshold).float()
        correct += (preds == targets).sum().item()
        
        # Store for AUC calculation
        all_targets.extend(targets.cpu().numpy().ravel())
        all_outputs.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel())
    
    # Calculate epoch metrics
    epoch_loss = total_loss / total
    epoch_acc = correct / total
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
            logits = model(inputs)
            loss = criterion(logits, targets)
            
            # Track statistics
            total_loss += loss.item() * inputs.size(0)
            
            # Store for metrics calculation
            all_targets.extend(targets.cpu().numpy().ravel())
            all_outputs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
    
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
    
    return epoch_loss, epoch_acc, epoch_auc, precision, recall, f1, cm, threshold, all_targets, all_outputs

def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def train_and_evaluate_model(model, model_name, train_loader, valid_loader, device, 
                           num_epochs=5, lr=0.001, weight_decay=0.01, 
                           patience=3, use_scheduler=True, pos_weight=None):
    """Train and evaluate a model."""
    print(f"\n{'='*80}")
    print(f"===== Training {model_name} =====")
    print(f"{'='*80}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss - use pos_weight if provided
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create scheduler if requested
    if use_scheduler:
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
    else:
        scheduler = None
    
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
    best_model_path = f"{model_name.replace(' ', '_').lower()}_best_model.pt"
    patience_counter = 0
    
    # Record start time
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc, train_auc, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler, best_threshold
        )
        
        # Validation phase (finding best threshold every other epoch)
        find_best_threshold = (epoch % 2 == 0) or (epoch == num_epochs - 1)
        val_loss, val_acc, val_auc, val_precision, val_recall, val_f1, cm, threshold, _, _ = evaluate(
            model, valid_loader, criterion, device, 
            threshold=best_threshold, find_best_threshold=find_best_threshold
        )
        
        if find_best_threshold:
            best_threshold = threshold
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['threshold'].append(threshold)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Training:   loss={train_loss:.4f}, acc={train_acc:.4f}, auc={train_auc:.4f}")
        print(f"  Validation: loss={val_loss:.4f}, acc={val_acc:.4f}, auc={val_auc:.4f}")
        print(f"             precision={val_precision:.4f}, recall={val_recall:.4f}, f1={val_f1:.4f}")
        print(f"             threshold={threshold:.4f}")
        
        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs. Best F1={best_val_f1:.4f} at epoch {best_epoch+1}")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"\n{model_name} training time: {training_time:.2f} seconds")
    
    # Load best model for final evaluation
    if best_epoch >= 0:
        print(f"Loading best model from epoch {best_epoch+1}...")
        model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation with optimal threshold
    val_loss, val_acc, val_auc, val_precision, val_recall, val_f1, cm, _, val_targets, val_outputs = evaluate(
        model, valid_loader, criterion, device, threshold=best_threshold, find_best_threshold=False
    )
    
    # Print final results
    print(f"\n{model_name} Final Results:")
    print(f"  Test Accuracy: {val_acc:.4f}")
    print(f"  Test AUC: {val_auc:.4f}")
    print(f"  Test Precision: {val_precision:.4f}")
    print(f"  Test Recall: {val_recall:.4f}")
    print(f"  Test F1 Score: {val_f1:.4f}")
    print(f"  Optimal Threshold: {best_threshold:.4f}")
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"  Model parameters: {trainable_params:,} trainable out of {total_params:,} total")
    
    # Return results
    return {
        'model': model,
        'model_name': model_name,
        'history': history,
        'training_time': training_time,
        'test_loss': val_loss,
        'test_acc': val_acc,
        'test_auc': val_auc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'confusion_matrix': cm,
        'best_threshold': best_threshold,
        'best_epoch': best_epoch,
        'val_targets': val_targets,
        'val_outputs': val_outputs,
        'total_params': total_params,
        'trainable_params': trainable_params
    }

def plot_confusion_matrix(cm, model_name, output_dir):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fire', 'Fire'],
                yticklabels=['No Fire', 'Fire'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    return plt.gcf()

def plot_roc_curve(results_list, output_dir):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for result in results_list:
        fpr, tpr, _ = roc_curve(result['val_targets'], result['val_outputs'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, 
                 label=f"{result['model_name']} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curves_comparison.png")
    return plt.gcf()

def plot_precision_recall_curve(results_list, output_dir):
    """Plot precision-recall curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for result in results_list:
        precision, recall, _ = precision_recall_curve(result['val_targets'], result['val_outputs'])
        avg_precision = np.mean(precision)
        plt.plot(recall, precision, lw=2, 
                 label=f"{result['model_name']} (Avg Precision = {avg_precision:.3f})")
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall_curves_comparison.png")
    return plt.gcf()

def compare_metrics(results_list, output_dir):
    """Create comparison plots for multiple models."""
    # Comparison of metrics in a bar chart
    metrics = ['test_acc', 'test_auc', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score']
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(metric_names))
    width = 0.8 / len(results_list)  # Adjust bar width based on number of models
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))
    
    for i, result in enumerate(results_list):
        values = [result[metric] for metric in metrics]
        offset = width * (i - (len(results_list) - 1) / 2)
        bars = plt.bar(x + offset, values, width, label=result['model_name'], color=colors[i])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', rotation=90, fontsize=9)
    
    plt.title('Performance Metrics Comparison', fontsize=16)
    plt.xticks(x, metric_names, fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png')
    
    # Training time comparison
    plt.figure(figsize=(12, 6))
    model_names = [r['model_name'] for r in results_list]
    training_times = [r['training_time'] for r in results_list]
    
    bars = plt.bar(model_names, training_times, color=colors)
    plt.title('Training Time Comparison', fontsize=16)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xticks(fontsize=12, rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add time values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_time_comparison.png')
    
    # Parameter count comparison
    plt.figure(figsize=(12, 6))
    model_names = [r['model_name'] for r in results_list]
    param_counts = [r['trainable_params'] for r in results_list]
    
    bars = plt.bar(model_names, param_counts, color=colors)
    plt.title('Model Parameter Count Comparison', fontsize=16)
    plt.ylabel('Number of Parameters', fontsize=14)
    plt.xticks(fontsize=12, rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.yscale('log')
    
    # Add parameter count on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height:,.0f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_count_comparison.png')
    
    return None

def visualize_architecture(model_name):
    """Create ASCII diagram of model architecture."""
    if model_name == "Internal World Model":
        diagram = """
        +--------------------+
        | Input Features     |
        +----------+---------+
                   |
        +----------v---------+
        |   4 Parallel       |
        |  Feature Branches  |
        +----------+---------+
                   |
        +----------v---------+
        |    Concatenate     |
        |   (1152-dim)       |
        +----------+---------+
                   |
        +----------v---------+
        |  3-Layer FFN       |
        +----------+---------+
                   |
        +----------v---------+
        | Projection Layer   |
        +----------+---------+
                   |
        +----------v---------+
        |  Internal World    |
        | (Transformer Layers)|
        +----------+---------+
                   |
        +----------v---------+
        | Classification MLP |
        +----------+---------+
                   |
        +----------v---------+
        |    Fire Prediction |
        +--------------------+
        """
    elif model_name == "Physics-Embedded Entropy Model":
        diagram = """
        +--------------------+
        | Input Features     |
        +----------+---------+
                   |
            +------+------+------+
            |             |      |
        +---v----+    +---v----+ +---v----+
        |  FFN   |    |  CNN   | |  PMFFN  |
        | Branch |    | Branch | | Branch  |
        +---+----+    +---+----+ +---+----+
            |             |          |
            +------+------+----------+
                   |
        +----------v---------+
        | Integration Network|
        +----------+---------+
                   |
        +----------v---------+
        |  Entropy Layer     |
        | (Physics-Informed) |
        +----------+---------+
                   |
        +----------v---------+
        | Multi-path Sigmoid |
        |    Classification  |
        +----------+---------+
                   |
        +----------v---------+
        |    Fire Prediction |
        +--------------------+
        """
    elif model_name == "FFN Model":
        diagram = """
        +--------------------+
        | Input Features     |
        +----------+---------+
                   |
        +----------v---------+
        |  Dense Layer 1     |
        |  (256 units + BN)  |
        +----------+---------+
                   |
        +----------v---------+
        |  Dense Layer 2     |
        |  (128 units + BN)  |
        +----------+---------+
                   |
        +----------v---------+
        |  Dense Layer 3     |
        |  (64 units + BN)   |
        +----------+---------+
                   |
        +----------v---------+
        |    Fire Prediction |
        +--------------------+
        """
    elif model_name == "CNN Model":
        diagram = """
        +--------------------+
        | Input Features     |
        +----------+---------+
                   |
        +----------v---------+
        | Reshape to 1D      |
        +----------+---------+
                   |
        +----------v---------+
        | Conv1D Layer 1     |
        | (32 filters)       |
        +----------+---------+
                   |
        +----------v---------+
        | Conv1D Layer 2     |
        | (64 filters)       |
        +----------+---------+
                   |
        +----------v---------+
        | Flatten + Dense    |
        | (128 units)        |
        +----------+---------+
                   |
        +----------v---------+
        |    Fire Prediction |
        +--------------------+
        """
    elif model_name == "FFN with Positional Encoding":
        diagram = """
        +--------------------+
        | Input Features     |
        +----------+---------+
                   |
        +----------v---------+
        | Reshape Each Feature|
        +----------+---------+
                   |
        +----------v---------+
        | Feature-wise FFN   |
        +----------+---------+
                   |
        +----------v---------+
        | Add Positional     |
        | Encodings          |
        +----------+---------+
                   |
        +----------v---------+
        | LayerNorm + Dropout|
        +----------+---------+
                   |
        +----------v---------+
        | Flatten + Dense    |
        +----------+---------+
                   |
        +----------v---------+
        |    Fire Prediction |
        +--------------------+
        """
    else:
        diagram = "Architecture diagram not available."
    
    # Return the diagram
    return diagram

def main():
    # Parse command line arguments
    args = parse_args()
    
    print("=" * 80)
    print("===== MOROCCO WILDFIRE PREDICTION BENCHMARK =====")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    df = load_morocco_wildfire_data(args.data_path)
    train_loader, valid_loader, n_features = prepare_data(
        df, batch_size=args.batch_size, sample_size=args.sample_size
    )
    
    # Show feature information
    print(f"\nTraining with {n_features} features")
    
    # For entropy layer parameters
    n_landcover = 4  # For entropy layer - represents NDVI-related features
    m_env_factors = min(300, n_features - n_landcover)  # Environmental factors
    
    # Create all models
    print("\nInitializing models...")
    
    # Create the Internal World model
    internal_world_model = Gemma3InternalWorldModel(
        n_features=n_features, 
        dropout_rate=0.4,
        gemma_path=args.gemma_path
    )
    
    # Create the Physics-Embedded Entropy model
    entropy_model = PhysicsEmbeddedEntropyModel(
        n_features=n_features,
        n_landcover=n_landcover,
        m_env_factors=m_env_factors
    )
    
    # Create the FFN model
    ffn_model = FFNModel(n_features=n_features)
    
    # Create the CNN model
    cnn_model = CNNModel(n_features=n_features)
    
    # Create the FFN with Positional Encoding model
    ffn_pos_model = FFNWithPosEncoding(n_features=n_features)
    
    # Create a list of models and their names to train
    models = [
        (internal_world_model, "Internal World Model"),
        (entropy_model, "Physics-Embedded Entropy Model"),
        (ffn_model, "FFN Model"),
        (cnn_model, "CNN Model"),
        (ffn_pos_model, "FFN with Positional Encoding")
    ]
    
    # Train all models and collect results
    results_list = []
    for model, model_name in models:
        # Print model parameter count
        total_params, trainable_params = count_parameters(model)
        print(f"\n{model_name} has {trainable_params:,} trainable parameters out of {total_params:,} total")
        
        # Train and evaluate the model
        result = train_and_evaluate_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            use_scheduler=True,
            pos_weight=2.0  # Give higher weight to positive class (fire events)
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(result['confusion_matrix'], model_name, args.output_dir)
        
        # Visualize model architecture
        architecture_diagram = visualize_architecture(model_name)
        print(f"\nArchitecture of {model_name}:")
        print(architecture_diagram)
        
        # Collect results for comparison
        results_list.append(result)
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compare all models
    print("\n" + "="*80)
    print("===== MODEL COMPARISON =====")
    print("="*80 + "\n")
    
    # Create a comprehensive comparison table
    comparison_data = {
        'Model': [r['model_name'] for r in results_list],
        'Accuracy': [r['test_acc'] for r in results_list],
        'AUC': [r['test_auc'] for r in results_list],
        'Precision': [r['precision'] for r in results_list],
        'Recall': [r['recall'] for r in results_list],
        'F1 Score': [r['f1'] for r in results_list],
        'Training Time (s)': [r['training_time'] for r in results_list],
        'Parameters': [r['trainable_params'] for r in results_list]
    }
    
    # Create a DataFrame for better visualization
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format the numeric columns
    for col in comparison_df.columns:
        if col not in ['Model', 'Parameters']:
            comparison_df[col] = comparison_df[col].map(lambda x: f"{x:.4f}")
    
    # Format the parameters column
    comparison_df['Parameters'] = comparison_df['Parameters'].map(lambda x: f"{x:,}")
    
    print(comparison_df.to_string(index=False))
    
    # Save the comparison DataFrame to CSV
    comparison_df.to_csv(f'{args.output_dir}/model_comparison.csv', index=False)
    
    # Plot ROC and Precision-Recall curves
    plot_roc_curve(results_list, args.output_dir)
    plot_precision_recall_curve(results_list, args.output_dir)
    
    # Visualize comparison metrics
    compare_metrics(results_list, args.output_dir)
    
    print("\nAll visualizations saved. Benchmark complete!")

if __name__ == "__main__":
    main()
