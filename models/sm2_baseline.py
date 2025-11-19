"""
SM-2 Algorithm Baseline for Spaced Repetition.
SuperMemo 2 algorithm implementation.
"""

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


class SM2Baseline:
    """
    SM-2 (SuperMemo 2) algorithm for spaced repetition scheduling.
    
    Key parameters:
    - EF (Ease Factor): Multiplier for intervals, adjusted based on performance
    - Interval: Days until next review
    - Repetition count: Number of successful consecutive reviews
    """
    
    def __init__(self, initial_ef: float = 2.5, min_ef: float = 1.3):
        """
        Initialize SM-2 algorithm.
        
        Args:
            initial_ef: Initial ease factor
            min_ef: Minimum ease factor
        """
        self.initial_ef = initial_ef
        self.min_ef = min_ef
        
        # Track state for each item
        self.item_states: Dict[int, Dict] = {}
    
    def update(self, item_id: int, quality: int, current_time: float = 0.0):
        """
        Update SM-2 state after reviewing an item.
        
        Args:
            item_id: ID of the reviewed item
            quality: Quality of recall (0-5, where 3+ is passing)
                     We'll use: 5=perfect, 4=good, 3=pass, 2=fail, 1=bad, 0=very bad
            current_time: Current time in the simulation
        """
        if item_id not in self.item_states:
            # Initialize new item
            self.item_states[item_id] = {
                'ef': self.initial_ef,
                'interval': 1.0,  # days
                'repetitions': 0,
                'next_review_time': current_time + 1.0
            }
        
        state = self.item_states[item_id]
        
        # Update ease factor based on quality
        # EF = EF + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        ef_change = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        state['ef'] = max(self.min_ef, state['ef'] + ef_change)
        
        # Update interval and repetitions based on quality
        if quality < 3:
            # Failed recall - reset
            state['interval'] = 1.0
            state['repetitions'] = 0
        else:
            # Successful recall
            if state['repetitions'] == 0:
                state['interval'] = 1.0
            elif state['repetitions'] == 1:
                state['interval'] = 6.0
            else:
                state['interval'] = state['interval'] * state['ef']
            
            state['repetitions'] += 1
        
        # Calculate next review time
        state['next_review_time'] = current_time + state['interval']
    
    def select_action(self, available_items: np.ndarray, current_time: float) -> int:
        """
        Select which item to review next.
        Chooses the item with the earliest next_review_time.
        
        Args:
            available_items: Array of available item IDs
            current_time: Current time in simulation
            
        Returns:
            Item ID to review next
        """
        if len(available_items) == 0:
            raise ValueError("No available items")
        
        # Initialize items that haven't been seen
        for item_id in available_items:
            if item_id not in self.item_states:
                self.item_states[item_id] = {
                    'ef': self.initial_ef,
                    'interval': 1.0,
                    'repetitions': 0,
                    'next_review_time': current_time  # Review immediately if never seen
                }
        
        # Find item with earliest next review time
        next_times = np.array([
            self.item_states[item_id]['next_review_time'] 
            for item_id in available_items
        ])
        
        # If multiple items are due, prefer those with lower intervals (more urgent)
        earliest_idx = np.argmin(next_times)
        selected_item = available_items[earliest_idx]
        
        return selected_item
    
    def reset(self):
        """Reset all item states."""
        self.item_states = {}
    
    def get_state(self, item_id: int) -> Dict:
        """Get current state for an item."""
        if item_id not in self.item_states:
            return {
                'ef': self.initial_ef,
                'interval': 1.0,
                'repetitions': 0,
                'next_review_time': 0.0
            }
        return self.item_states[item_id].copy()


def quality_from_recall(recalled: bool) -> int:
    """
    Convert boolean recall to SM-2 quality score.
    
    Args:
        recalled: True if word was recalled correctly
        
    Returns:
        Quality score (0-5)
    """
    return 5 if recalled else 2  # Perfect recall = 5, failure = 2


def train_sm2_baseline(env, data: pd.DataFrame, num_episodes: int = 10):
    """
    Train/evaluate SM-2 baseline on the environment.
    
    Args:
        env: MemoryEnv instance
        data: Training data
        num_episodes: Number of episodes to run
        
    Returns:
        List of episode results (rewards, recall rates, etc.)
    """
    sm2 = SM2Baseline()
    results = []
    
    for episode in range(num_episodes):
        sm2.reset()
        state = env.reset()
        done = False
        episode_rewards = []
        episode_recalls = []
        
        while not done:
            # Get available actions
            available_actions = env.get_available_actions()
            
            # Select action using SM-2
            action = sm2.select_action(available_actions, env.current_time)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Update SM-2 with result
            recalled = info['recalled']
            quality = quality_from_recall(recalled)
            sm2.update(action, quality, env.current_time)
            
            # Track results
            episode_rewards.append(reward)
            episode_recalls.append(recalled)
        
        # Calculate episode metrics
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
            print(f"SM-2 Episode {episode + 1}/{num_episodes}: "
                  f"Recall Rate = {recall_rate:.3f}, Reward = {cumulative_reward:.1f}")
    
    return results

