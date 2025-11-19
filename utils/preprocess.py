"""
Data preprocessing module for Duolingo spaced repetition dataset.
Loads, processes, and prepares data for training and evaluation.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_duolingo_data(dataset_path):
    """
    Load Duolingo dataset from Kaggle download path.
    
    Args:
        dataset_path: Path to the downloaded dataset directory
        
    Returns:
        DataFrame with review data
    """
    # Find CSV files in the dataset directory
    csv_files = list(Path(dataset_path).rglob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")
    
    # Try to load the main data file (common names)
    data_file = None
    for f in csv_files:
        if any(name in f.name.lower() for name in ['review', 'data', 'train', 'main']):
            data_file = f
            break
    
    # If no obvious main file, use the largest CSV
    if data_file is None:
        data_file = max(csv_files, key=lambda x: x.stat().st_size)
    
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    
    return df


def extract_relevant_columns(df):
    """
    Extract and standardize relevant columns from the dataset.
    Handles various possible column name formats.
    """
    # Common column name variations
    column_mapping = {
        'word_id': ['word_id', 'word', 'item_id', 'item', 'vocab_id'],
        'user_id': ['user_id', 'user', 'learner_id', 'learner'],
        'timestamp': ['timestamp', 'time', 'review_time', 'date', 'datetime'],
        'correct': ['correct', 'success', 'recalled', 'is_correct', 'outcome']
    }
    
    # Find matching columns
    found_cols = {}
    for target, candidates in column_mapping.items():
        for col in df.columns:
            if col.lower() in [c.lower() for c in candidates]:
                found_cols[target] = col
                break
    
    # If exact matches not found, try to infer
    if 'word_id' not in found_cols:
        # Look for columns that might be IDs
        for col in df.columns:
            if 'id' in col.lower() and 'user' not in col.lower():
                found_cols['word_id'] = col
                break
    
    if 'user_id' not in found_cols:
        for col in df.columns:
            if 'user' in col.lower() or 'learner' in col.lower():
                found_cols['user_id'] = col
                break
    
    if 'timestamp' not in found_cols:
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                found_cols['timestamp'] = col
                break
    
    if 'correct' not in found_cols:
        for col in df.columns:
            if 'correct' in col.lower() or 'success' in col.lower() or 'recall' in col.lower():
                found_cols['correct'] = col
                break
    
    # Rename columns
    df_clean = df.copy()
    for target, source in found_cols.items():
        if source != target:
            df_clean[target] = df_clean[source]
    
    # Ensure we have required columns
    required = ['word_id', 'user_id', 'timestamp', 'correct']
    missing = [r for r in required if r not in df_clean.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")
    
    return df_clean[required + [c for c in df_clean.columns if c not in required]]


def create_features(df):
    """
    Create features for each review:
    - time_since_last_review
    - previous_correct_count
    - previous_attempts
    - historical_accuracy
    
    Optimized using vectorized pandas operations for speed.
    """
    print("Creating features...")
    
    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Ensure correct is boolean/numeric
    df['correct'] = df['correct'].astype(float)
    
    # Time since last review: diff within each user group (in hours)
    print("  Computing time_since_last_review...", flush=True)
    df['time_since_last_review'] = (
        df.groupby('user_id')['timestamp']
        .diff()
        .dt.total_seconds()
        .div(3600)
        .fillna(24.0)  # Default 24 hours for first review per user
    )
    
    # Previous attempts: cumulative count per user-word combination
    print("  Computing previous_attempts...", flush=True)
    df['previous_attempts'] = (
        df.groupby(['user_id', 'word_id'])
        .cumcount()  # 0-indexed, so this gives previous_attempts
    )
    
    # Previous correct count: cumulative sum of correct per user-word
    # More efficient: shift first, then groupby cumsum (avoids lambda in transform)
    print("  Computing previous_correct_count...", flush=True)
    df['correct_shifted'] = df.groupby(['user_id', 'word_id'])['correct'].shift(1).fillna(0)
    df['previous_correct_count'] = df.groupby(['user_id', 'word_id'])['correct_shifted'].cumsum()
    
    # Historical accuracy: cumulative mean of correct per user-word
    # Reuse previous_correct_count and previous_attempts for efficiency
    print("  Computing historical_accuracy...", flush=True)
    # Historical accuracy = previous_correct_count / previous_attempts
    # When previous_attempts = 0, use default 0.5
    df['historical_accuracy'] = np.where(
        df['previous_attempts'] > 0,
        df['previous_correct_count'] / df['previous_attempts'],
        0.5  # Default for first review
    )
    
    # Clean up temporary column
    df = df.drop('correct_shifted', axis=1)
    print("  Done!", flush=True)
    
    # Fill any remaining NaN values
    df['time_since_last_review'] = df['time_since_last_review'].fillna(24.0)
    df['historical_accuracy'] = df['historical_accuracy'].fillna(0.5)
    
    print("  Feature creation complete!")
    
    return df


def encode_ids(df):
    """
    Encode user_id and word_id as integers starting from 0.
    """
    print("Encoding IDs...")
    
    user_encoder = LabelEncoder()
    word_encoder = LabelEncoder()
    
    df['user_id_encoded'] = user_encoder.fit_transform(df['user_id'])
    df['word_id_encoded'] = word_encoder.fit_transform(df['word_id'])
    
    return df, user_encoder, word_encoder


def normalize_features(df, scaler=None, fit=True):
    """
    Normalize numerical features using StandardScaler.
    """
    print("Normalizing features...")
    
    feature_cols = ['time_since_last_review', 'previous_correct_count', 
                    'previous_attempts', 'historical_accuracy']
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df, scaler


def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train/validation/test sets by user.
    Ensures users don't appear in multiple splits.
    """
    print("Splitting data...")
    
    unique_users = df['user_id_encoded'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_users)
    
    n_users = len(unique_users)
    n_train = int(n_users * train_ratio)
    n_val = int(n_users * val_ratio)
    
    train_users = unique_users[:n_train]
    val_users = unique_users[n_train:n_train + n_val]
    test_users = unique_users[n_train + n_val:]
    
    train_df = df[df['user_id_encoded'].isin(train_users)].copy()
    val_df = df[df['user_id_encoded'].isin(val_users)].copy()
    test_df = df[df['user_id_encoded'].isin(test_users)].copy()
    
    print(f"Train: {len(train_df)} reviews from {len(train_users)} users")
    print(f"Val: {len(val_df)} reviews from {len(val_users)} users")
    print(f"Test: {len(test_df)} reviews from {len(test_users)} users")
    
    return train_df, val_df, test_df


def preprocess_dataset(dataset_path, output_dir='data', max_rows=None):
    """
    Main preprocessing function.
    Loads, processes, and saves the dataset.
    
    Args:
        dataset_path: Path to dataset directory
        output_dir: Output directory for processed data
        max_rows: Optional limit on number of rows to process (for faster testing)
    """
    print("=" * 60)
    print("PREPROCESSING DUOLINGO DATASET")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    df = load_duolingo_data(dataset_path)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    
    # Limit rows if specified (for faster testing)
    if max_rows is not None and max_rows > 0:
        print(f"   Limiting to {max_rows} rows for faster processing...")
        df = df.head(max_rows).copy()
    
    # Extract relevant columns
    print("\n2. Extracting relevant columns...")
    df = extract_relevant_columns(df)
    print(f"   Using columns: {list(df.columns)}")
    
    # Create features
    print("\n3. Creating features...")
    df = create_features(df)
    
    # Encode IDs
    print("\n4. Encoding IDs...")
    df, user_encoder, word_encoder = encode_ids(df)
    
    # Normalize features
    print("\n5. Normalizing features...")
    df, scaler = normalize_features(df, fit=True)
    
    # Split data
    print("\n6. Splitting data...")
    train_df, val_df, test_df = split_data(df)
    
    # Save processed data
    print("\n7. Saving processed data...")
    output_path = os.path.join(output_dir, 'processed.pkl')
    
    processed_data = {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'user_encoder': user_encoder,
        'word_encoder': word_encoder,
        'scaler': scaler,
        'num_users': len(df['user_id_encoded'].unique()),
        'num_words': len(df['word_id_encoded'].unique())
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"   Saved to {output_path}")
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return processed_data


if __name__ == '__main__':
    # Example usage
    import kagglehub
    
    path = kagglehub.dataset_download("aravinii/duolingo-spaced-repetition-data")
    preprocess_dataset(path)

