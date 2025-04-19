import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

def load_morocco_wildfire_data(data_path):
    """
    Load the Morocco wildfire dataset from parquet file.
    
    If the file is not found, create simulated data for testing purposes.
    
    Args:
        data_path (str): Path to the parquet file
        
    Returns:
        pd.DataFrame: The loaded dataset
    """
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
    """
    Prepare data for training and evaluation.
    
    Steps:
    1. Time-based train-validation split (pre-2022 and 2022+)
    2. Balance classes by undersampling
    3. Standardize features
    4. Create PyTorch dataloaders
    
    Args:
        df (pd.DataFrame): Input dataframe
        batch_size (int): Batch size for dataloaders
        sample_size (int, optional): Number of samples per class (for testing)
        
    Returns:
        tuple: (train_loader, valid_loader, n_features)
    """
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

def get_feature_groups(df, remove_date=True):
    """
    Group features in the dataset by category.
    
    Args:
        df (pd.DataFrame): Input dataframe
        remove_date (bool): Whether to remove the date column
        
    Returns:
        dict: Dictionary of feature groups
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Remove date column if requested
    if remove_date and 'acq_date' in df_copy.columns:
        df_copy = df_copy.drop(columns=['acq_date'])
    
    # Get all column names except target
    all_features = [col for col in df_copy.columns if col != 'is_fire']
    
    # Group features by type
    feature_groups = {
        'vegetation': [col for col in all_features if 'ndvi' in col.lower()],
        'moisture': [col for col in all_features if 'moisture' in col.lower()],
        'temperature': [col for col in all_features if 'temp' in col.lower()],
        'precipitation': [col for col in all_features if 'precip' in col.lower()],
        'wind': [col for col in all_features if 'wind' in col.lower()],
        'other': []
    }
    
    # Add remaining features to 'other'
    categorized = []
    for group in feature_groups.values():
        categorized.extend(group)
    
    feature_groups['other'] = [col for col in all_features if col not in categorized]
    
    return feature_groups

def get_lagged_features(df_columns):
    """
    Identify and group time-lagged features.
    
    Args:
        df_columns (list): List of column names
        
    Returns:
        dict: Dictionary of lag groups with their features
    """
    lag_groups = {}
    
    # Identify lag patterns
    for col in df_columns:
        if 'lag' in col.lower():
            # Try to extract lag value
            parts = col.split('_')
            for i, part in enumerate(parts):
                if part.lower() == 'lag' and i < len(parts) - 1:
                    try:
                        lag_value = int(parts[i + 1])
                        if lag_value not in lag_groups:
                            lag_groups[lag_value] = []
                        lag_groups[lag_value].append(col)
                        break
                    except ValueError:
                        # If not a number, continue
                        continue
    
    # Add non-lagged features
    lag_groups[0] = [col for col in df_columns if 'lag' not in col.lower() 
                     and col != 'is_fire' and col != 'acq_date']
    
    return lag_groups
