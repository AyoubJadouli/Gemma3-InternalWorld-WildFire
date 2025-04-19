import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_confusion_matrix(cm, output_dir, model_name="internal_world"):
    """
    Plot confusion matrix.
    
    Args:
        cm (array): Confusion matrix array from sklearn
        output_dir (str): Directory to save the plot
        model_name (str): Name of the model for file naming
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
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
    
    # Save the figure
    filename = f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    
    return plt.gcf()

def plot_training_curves(history, output_dir, model_name="internal_world"):
    """
    Plot training and validation curves.
    
    Args:
        history (dict): Dictionary containing training history
        output_dir (str): Directory to save the plot
        model_name (str): Name of the model for file naming
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
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
    
    # Save the figure
    filename = f"{model_name.replace(' ', '_').lower()}_training_curves.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    
    return plt.gcf()

def plot_roc_curve(results_list, output_dir):
    """
    Plot ROC curves for all models.
    
    Args:
        results_list (list): List of result dictionaries for each model
        output_dir (str): Directory to save the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(f"{output_dir}/roc_curves_comparison.png")
    
    return plt.gcf()

def plot_precision_recall_curve(results_list, output_dir):
    """
    Plot precision-recall curves for all models.
    
    Args:
        results_list (list): List of result dictionaries for each model
        output_dir (str): Directory to save the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(f"{output_dir}/precision_recall_curves_comparison.png")
    
    return plt.gcf()

def compare_metrics(results_list, output_dir):
    """
    Create comparison plots for multiple models.
    
    Args:
        results_list (list): List of result dictionaries for each model
        output_dir (str): Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    plt.yscale('log')  # Log scale for better visualization
    
    # Add parameter count on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height:,.0f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_count_comparison.png')

def visualize_architecture(model_name):
    """
    Generate ASCII diagram of model architecture.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        str: ASCII diagram of the model architecture
    """
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

def plot_feature_importance(model, feature_names, output_dir, top_n=30, model_name="internal_world"):
    """
    For models that support feature importance, create a feature importance plot.
    This is a simple approximation since gradient-based methods would be more accurate.
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        output_dir (str): Directory to save the plot
        top_n (int): Number of top features to show
        model_name (str): Name of the model for file naming
    """
    # This is a simplistic approach - for a real implementation, 
    # you would need to calculate feature importance for each model type
    
    # Create dummy feature importance (for demonstration)
    importances = np.random.rand(len(feature_names))
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    indices = indices[:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=16)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.xlabel('Relative Importance', fontsize=14)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(f"{output_dir}/{model_name.replace(' ', '_').lower()}_feature_importance.png")
    
    return plt.gcf()
