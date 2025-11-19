"""
Reinforcement Learning Environment for Spaced Repetition Memory.
Simulates a learner reviewing words with memory decay.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import pandas as pd


class MemoryEnv:
    """
    Environment for spaced repetition learning.
    
    State: Memory features for each word (attempts, successes, time_since_last_review, etc.)
    Action: Choose which word to review next (0 to num_items-1)
    Reward: +1 for correct recall, -1 for failed recall
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 num_items: int,
                 time_budget: float = 100.0,
                 forgetting_rate: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize the memory environment.
        
        Args:
            data: DataFrame with review history
            num_items: Number of unique words/items
            time_budget: Total time available for reviews
            forgetting_rate: Rate of memory decay (higher = faster forgetting)
            seed: Random seed
        """
        self.data = data.copy()
        self.num_items = num_items
        self.time_budget = time_budget
        self.forgetting_rate = forgetting_rate
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize memory state for each item
        self.reset()
    
    def reset(self, user_id: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Args:
            user_id: Optional user ID to simulate (if None, uses first user)
            
        Returns:
            Initial state vector
        """
        # Select user
        if user_id is None:
            available_users = self.data['user_id_encoded'].unique()
            self.current_user = np.random.choice(available_users)
        else:
            self.current_user = user_id
        
        # Get user's review history
        user_data = self.data[self.data['user_id_encoded'] == self.current_user].copy()
        user_data = user_data.sort_values('timestamp').reset_index(drop=True)
        
        self.user_data = user_data
        self.current_time = 0.0
        self.remaining_budget = self.time_budget
        self.review_index = 0  # Index into user's review history
        
        # Initialize memory state for each item
        # State vector: [attempts, successes, time_since_last_review, historical_accuracy]
        self.item_states = np.zeros((self.num_items, 4))
        
        # Track which items have been reviewed
        self.item_reviewed = np.zeros(self.num_items, dtype=bool)
        
        # Track last review time for each item
        self.item_last_review_time = np.zeros(self.num_items)
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        Returns:
            State matrix of shape (num_items, 4)
            Columns: [attempts, successes, time_since_last_review, historical_accuracy]
        """
        # Update time_since_last_review for all items
        current_time_since = self.current_time - self.item_last_review_time
        current_time_since[self.item_last_review_time == 0] = 24.0  # Default for never reviewed
        
        # Normalize time (hours to days)
        current_time_since = current_time_since / 24.0
        
        # Update state
        state = self.item_states.copy()
        state[:, 2] = current_time_since  # time_since_last_review
        
        return state
    
    def _simulate_recall(self, word_id: int, state: np.ndarray) -> bool:
        """
        Simulate whether the learner recalls the word.
        Uses exponential forgetting curve with base probability.
        
        Args:
            word_id: ID of the word being reviewed
            state: Current state vector for the word
            
        Returns:
            True if recalled correctly, False otherwise
        """
        attempts, successes, time_since, accuracy = state
        
        # Base recall probability from historical accuracy
        base_prob = max(0.1, min(0.9, accuracy))
        
        # Exponential decay based on time since last review
        # Longer time = lower probability
        time_decay = np.exp(-self.forgetting_rate * time_since)
        
        # Boost from number of successful reviews
        success_boost = min(0.3, successes * 0.05)
        
        # Final recall probability
        recall_prob = base_prob * time_decay + success_boost
        recall_prob = max(0.1, min(0.95, recall_prob))
        
        # Sample recall outcome
        recalled = np.random.random() < recall_prob
        
        return recalled
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Word ID to review (0 to num_items-1)
            
        Returns:
            next_state: New state after review
            reward: Reward from the review (+1 correct, -1 incorrect)
            done: Whether episode is finished
            info: Additional information
        """
        if action < 0 or action >= self.num_items:
            raise ValueError(f"Invalid action: {action}. Must be in [0, {self.num_items-1}]")
        
        # Get current state for the selected word
        word_state = self.item_states[action]
        
        # Simulate recall
        recalled = self._simulate_recall(action, word_state)
        
        # Calculate reward
        reward = 1.0 if recalled else -1.0
        
        # Update memory state
        self.item_states[action, 0] += 1  # attempts
        if recalled:
            self.item_states[action, 1] += 1  # successes
        
        # Update historical accuracy
        attempts = self.item_states[action, 0]
        successes = self.item_states[action, 1]
        if attempts > 0:
            self.item_states[action, 3] = successes / attempts
        
        # Update last review time
        self.item_last_review_time[action] = self.current_time
        self.item_reviewed[action] = True
        
        # Advance time (each review takes some time)
        review_time = 1.0  # 1 hour per review
        self.current_time += review_time
        self.remaining_budget -= review_time
        
        # Check if done
        done = (self.remaining_budget <= 0) or (self.review_index >= len(self.user_data) - 1)
        
        # Get next state
        next_state = self.get_state()
        
        # Increment review index
        self.review_index += 1
        
        info = {
            'word_id': action,
            'recalled': recalled,
            'time_remaining': self.remaining_budget,
            'total_reviews': self.review_index
        }
        
        return next_state, reward, done, info
    
    def get_available_actions(self) -> np.ndarray:
        """
        Get list of available actions (words that can be reviewed).
        In this implementation, all words are always available.
        
        Returns:
            Array of available action indices
        """
        return np.arange(self.num_items)
    
    def render(self, mode='human'):
        """
        Render the current state (for debugging).
        """
        if mode == 'human':
            print(f"Time: {self.current_time:.2f}, Budget: {self.remaining_budget:.2f}")
            print(f"Total reviews: {self.review_index}")
            print(f"Items reviewed: {self.item_reviewed.sum()}/{self.num_items}")
            
            # Show top 5 items by attempts
            attempts = self.item_states[:, 0]
            top_indices = np.argsort(attempts)[-5:][::-1]
            print("\nTop 5 items by attempts:")
            for idx in top_indices:
                state = self.item_states[idx]
                print(f"  Item {idx}: attempts={state[0]:.0f}, successes={state[1]:.0f}, "
                      f"accuracy={state[3]:.2f}")

