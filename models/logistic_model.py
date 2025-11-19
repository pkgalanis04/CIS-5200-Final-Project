"""
Logistic Regression Model for Predicting Recall Probability.
Uses supervised learning to predict if a learner will recall a word.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Dict, List, Tuple
import pandas as pd


class LogisticRecallModel:
    """
    Logistic regression model to predict recall probability.
    Scheduler selects items with highest predicted recall probability.
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        """
        Initialize logistic regression model.
        
        Args:
            C: Inverse of regularization strength
            max_iter: Maximum iterations for optimization
        """
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        self.is_trained = False
    
    def prepare_features(self, state: np.ndarray) -> np.ndarray:
        """
        Prepare features from state vector.
        
        Args:
            state: State matrix of shape (num_items, 4)
                   Columns: [attempts, successes, time_since_last_review, historical_accuracy]
        
        Returns:
            Feature matrix of shape (num_items, 4)
        """
        return state
    
    def train(self, train_data: pd.DataFrame):
        """
        Train the logistic regression model on historical data.
        
        Args:
            train_data: DataFrame with columns including features and 'correct' label
        """
        print("Training Logistic Regression model...")
        
        # Prepare features
        feature_cols = ['time_since_last_review', 'previous_correct_count',
                       'previous_attempts', 'historical_accuracy']
        
        X = train_data[feature_cols].values
        y = train_data['correct'].values
        
        # Convert to binary classification (0 or 1)
        # Handle continuous values by thresholding at 0.5
        # Replace NaN with 0, then threshold
        y = np.nan_to_num(y, nan=0.0)
        y = (y > 0.5).astype(int)  # Convert to 0 or 1 by thresholding at 0.5
        
        # Verify we have binary classes
        unique_vals = np.unique(y)
        if not all(v in [0, 1] for v in unique_vals):
            raise ValueError(f"Expected binary classification (0/1), got: {unique_vals}")
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Evaluate on training data
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        train_acc = accuracy_score(y, y_pred)
        train_auc = roc_auc_score(y, y_proba)
        
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Training AUC: {train_auc:.4f}")
    
    def predict_recall_probability(self, state: np.ndarray) -> np.ndarray:
        """
        Predict recall probability for all items.
        
        Args:
            state: State matrix of shape (num_items, 4)
        
        Returns:
            Array of recall probabilities for each item
        """
        if not self.is_trained:
            # Return default probabilities if not trained
            return np.ones(state.shape[0]) * 0.5
        
        features = self.prepare_features(state)
        probabilities = self.model.predict_proba(features)[:, 1]
        
        return probabilities
    
    def select_action(self, state: np.ndarray, available_items: np.ndarray = None) -> int:
        """
        Select item with highest predicted recall probability.
        
        Args:
            state: Current state matrix
            available_items: Optional array of available item IDs
        
        Returns:
            Item ID to review next
        """
        probabilities = self.predict_recall_probability(state)
        
        if available_items is not None:
            # Only consider available items
            mask = np.zeros(len(probabilities), dtype=bool)
            mask[available_items] = True
            probabilities[~mask] = -np.inf
        
        # Select item with highest probability
        action = np.argmax(probabilities)
        
        return action


def train_logistic_model(env, train_data: pd.DataFrame, val_data: pd.DataFrame = None,
                        num_episodes: int = 10):
    """
    Train and evaluate logistic regression model.
    
    Args:
        env: MemoryEnv instance
        train_data: Training data for supervised learning
        val_data: Optional validation data
        num_episodes: Number of episodes for evaluation
        
    Returns:
        Dictionary with training and evaluation results
    """
    # Train the model
    model = LogisticRecallModel()
    model.train(train_data)
    
    # Evaluate on environment
    results = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = []
        episode_recalls = []
        
        while not done:
            # Select action using logistic model
            available_actions = env.get_available_actions()
            action = model.select_action(state, available_actions)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Track results
            episode_rewards.append(reward)
            episode_recalls.append(info['recalled'])
            
            state = next_state
        
        # Calculate metrics
        total_reviews = len(episode_recalls)
        recall_rate = sum(episode_recalls) / total_reviews if total_reviews > 0 else 0.0
        cumulative_reward = sum(episode_rewards)
        
        results.append({
            'episode': episode,
            'recall_rate': recall_rate,
            'cumulative_reward': cumulative_reward,
            'total_reviews': total_reviews
        })
        
        if (episode + 1) % 5 == 0:
            print(f"Logistic Regression Episode {episode + 1}/{num_episodes}: "
                  f"Recall Rate = {recall_rate:.3f}, Reward = {cumulative_reward:.1f}")
    
    return {
        'model': model,
        'results': results
    }

