"""
Main execution script for spaced repetition optimization system.
Trains and evaluates multiple models: SM-2, Logistic Regression, Bandit, DQN.
"""

import argparse
import pickle
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import project modules
from utils.preprocess import preprocess_dataset
from environment.memory_env import MemoryEnv
from models.sm2_baseline import train_sm2_baseline
from models.logistic_model import train_logistic_model
from models.bandit_agent import train_bandit_agent
from models.dqn_agent import train_dqn_agent
from utils.evaluation import create_full_evaluation

import warnings
warnings.filterwarnings('ignore')


def load_processed_data(data_path: str = 'data/processed.pkl'):
    """
    Load preprocessed data.
    
    Args:
        data_path: Path to processed data file
        
    Returns:
        Dictionary with processed data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Processed data not found at {data_path}. "
            "Please run preprocessing first: python main.py --preprocess"
        )
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def setup_environment(processed_data, time_budget: float = 100.0, 
                     forgetting_rate: float = 0.1, seed: int = 42):
    """
    Create and configure the memory environment.
    
    Args:
        processed_data: Dictionary with processed data
        time_budget: Time budget for reviews
        forgetting_rate: Memory decay rate
        seed: Random seed
        
    Returns:
        MemoryEnv instance
    """
    train_data = processed_data['train']
    num_items = processed_data['num_words']
    
    env = MemoryEnv(
        data=train_data,
        num_items=num_items,
        time_budget=time_budget,
        forgetting_rate=forgetting_rate,
        seed=seed
    )
    
    return env


def train_all_models(processed_data, args):
    """
    Train all models and return results.
    
    Args:
        processed_data: Dictionary with processed data
        args: Command line arguments
        
    Returns:
        Dictionary mapping model names to results
    """
    all_results = {}
    
    # Setup environment
    env = setup_environment(
        processed_data,
        time_budget=args.time_budget,
        forgetting_rate=args.forgetting_rate,
        seed=args.seed
    )
    
    train_data = processed_data['train']
    val_data = processed_data.get('val', None)
    num_items = processed_data['num_words']
    
    # Train SM-2 Baseline
    if args.model in ['all', 'sm2']:
        try:
            print("\n" + "=" * 60)
            print("TRAINING SM-2 BASELINE")
            print("=" * 60)
            sm2_results = train_sm2_baseline(env, train_data, num_episodes=args.episodes)
            all_results['SM-2'] = sm2_results
            print("✓ SM-2 Baseline training completed successfully")
        except Exception as e:
            print(f"✗ SM-2 Baseline training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Train Logistic Regression
    if args.model in ['all', 'logistic']:
        try:
            print("\n" + "=" * 60)
            print("TRAINING LOGISTIC REGRESSION")
            print("=" * 60)
            logistic_output = train_logistic_model(
                env, train_data, val_data, num_episodes=args.episodes
            )
            all_results['Logistic Regression'] = logistic_output['results']
            
            # Save model
            if args.save_models:
                os.makedirs('checkpoints', exist_ok=True)
                import pickle
                with open('checkpoints/logistic_model.pkl', 'wb') as f:
                    pickle.dump(logistic_output['model'], f)
            print("✓ Logistic Regression training completed successfully")
        except Exception as e:
            print(f"✗ Logistic Regression training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Train Bandit Agent
    if args.model in ['all', 'bandit']:
        try:
            print("\n" + "=" * 60)
            print("TRAINING BANDIT AGENT")
            print("=" * 60)
            bandit_output = train_bandit_agent(
                env,
                num_episodes=args.episodes,
                num_arms=args.num_arms,
                epsilon=args.epsilon,
                learning_rate=args.learning_rate
            )
            all_results['Bandit RL'] = bandit_output['results']
            print("✓ Bandit RL training completed successfully")
        except Exception as e:
            print(f"✗ Bandit RL training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Train DQN Agent
    if args.model in ['all', 'dqn']:
        try:
            print("\n" + "=" * 60)
            print("TRAINING DQN AGENT")
            print("=" * 60)
            
            device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
            print(f"Using device: {device}")
            
            dqn_output = train_dqn_agent(
                env,
                num_episodes=args.episodes,
                num_actions=num_items,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                device=device
            )
            all_results['DQN RL'] = dqn_output['results']
            
            # Save model
            if args.save_models:
                os.makedirs('checkpoints', exist_ok=True)
                dqn_output['agent'].save('checkpoints/dqn_model.pt')
            print("✓ DQN RL training completed successfully")
        except Exception as e:
            print(f"✗ DQN RL training failed: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Spaced Repetition Optimization System'
    )
    
    # Data arguments
    parser.add_argument('--preprocess', action='store_true',
                       help='Preprocess the dataset (requires kagglehub)')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to downloaded dataset (if preprocessing)')
    parser.add_argument('--max_rows', type=int, default=None,
                       help='Limit number of rows to process (for faster testing)')
    
    # Model selection
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'sm2', 'logistic', 'bandit', 'dqn'],
                       help='Which model(s) to train')
    
    # Training arguments
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of training episodes')
    parser.add_argument('--time_budget', type=float, default=100.0,
                       help='Time budget for reviews')
    parser.add_argument('--forgetting_rate', type=float, default=0.1,
                       help='Memory forgetting rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Bandit arguments
    parser.add_argument('--num_arms', type=int, default=10,
                       help='Number of arms for bandit')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Epsilon for epsilon-greedy')
    
    # DQN arguments
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for RL agents')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for DQN')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU for DQN training')
    
    # Output arguments
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained models')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Preprocess if requested
    if args.preprocess:
        if args.dataset_path is None:
            try:
                import kagglehub
                print("Downloading dataset from Kaggle...")
                dataset_path = kagglehub.dataset_download("aravinii/duolingo-spaced-repetition-data")
            except ImportError:
                raise ImportError(
                    "kagglehub not installed. Install with: pip install kagglehub\n"
                    "Or provide --dataset_path to a local dataset directory."
                )
        else:
            dataset_path = args.dataset_path
        
        print(f"Preprocessing dataset from: {dataset_path}")
        preprocess_dataset(dataset_path, output_dir='data', max_rows=args.max_rows)
        print("Preprocessing complete!")
        return
    
    # Load processed data
    print("Loading processed data...")
    import time
    start_time = time.time()
    processed_data = load_processed_data()
    load_time = time.time() - start_time
    print(f"Loaded data in {load_time:.2f} seconds: "
          f"{processed_data['num_users']} users, "
          f"{processed_data['num_words']} words")
    
    # Train models
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    all_results = train_all_models(processed_data, args)
    
    # Evaluate and compare
    if all_results:
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        comparison_df = create_full_evaluation(all_results, output_dir=args.output_dir)
        
        # Print final comparison
        print("\n" + "=" * 60)
        print("FINAL COMPARISON")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
    else:
        print("No models were trained.")


if __name__ == '__main__':
    main()

