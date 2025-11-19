"""
Multi-Armed Bandit RL Agent for Spaced Repetition.
Groups items by difficulty and uses epsilon-greedy policy.
"""

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


class BanditAgent:
    """
    Multi-Armed Bandit agent using epsilon-greedy policy.
    Groups items by difficulty (based on historical accuracy) into k arms.
    """
    
    def __init__(self, 
                 num_arms: int = 10,
                 epsilon: float = 0.1,
                 learning_rate: float = 0.1,
                 initial_q: float = 0.0):
        """
        Initialize bandit agent.
        
        Args:
            num_arms: Number of difficulty groups (arms)
            epsilon: Exploration rate for epsilon-greedy
            learning_rate: Learning rate for Q-value updates
            initial_q: Initial Q-value for each arm
        """
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.initial_q = initial_q
        
        # Q-values for each arm
        self.q_values = np.ones(num_arms) * initial_q
        
        # Count of pulls for each arm
        self.arm_counts = np.zeros(num_arms, dtype=int)
        
        # Mapping from item_id to arm (difficulty group)
        self.item_to_arm: Dict[int, int] = {}
        
        # Reverse mapping: arm to list of items
        self.arm_to_items: Dict[int, List[int]] = {i: [] for i in range(num_arms)}
    
    def assign_items_to_arms(self, state: np.ndarray, available_items: np.ndarray):
        """
        Assign items to difficulty arms based on historical accuracy.
        
        Args:
            state: State matrix of shape (num_items, 4)
            available_items: Array of available item IDs
        """
        # Use historical_accuracy (column 3) to determine difficulty
        accuracies = state[:, 3]
        
        # Create difficulty buckets
        # Lower accuracy = harder = higher arm index
        if len(available_items) > 0:
            item_accuracies = accuracies[available_items]
            
            # Assign to arms based on accuracy percentiles
            for item_id in available_items:
                if item_id not in self.item_to_arm:
                    accuracy = accuracies[item_id]
                    
                    # Map accuracy to arm (0 = easiest, num_arms-1 = hardest)
                    # Items with higher accuracy go to lower arm indices
                    arm = int((1.0 - accuracy) * self.num_arms)
                    arm = max(0, min(self.num_arms - 1, arm))
                    
                    self.item_to_arm[item_id] = arm
                    self.arm_to_items[arm].append(item_id)
    
    def select_arm(self) -> int:
        """
        Select arm using epsilon-greedy policy.
        
        Returns:
            Selected arm index
        """
        if np.random.random() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.num_arms)
        else:
            # Exploit: arm with highest Q-value
            return np.argmax(self.q_values)
    
    def select_action(self, state: np.ndarray, available_items: np.ndarray) -> int:
        """
        Select item to review.
        First selects a difficulty arm, then picks a random item from that arm.
        
        Args:
            state: Current state matrix
            available_items: Available item IDs
        
        Returns:
            Selected item ID
        """
        # Assign items to arms if needed
        self.assign_items_to_arms(state, available_items)
        
        # Select arm
        selected_arm = self.select_arm()
        
        # Get items in this arm
        arm_items = [item for item in self.arm_to_items[selected_arm] 
                    if item in available_items]
        
        # If no items in selected arm, fall back to random from available
        if len(arm_items) == 0:
            return np.random.choice(available_items)
        
        # Select random item from the arm
        return np.random.choice(arm_items)
    
    def update(self, arm: int, reward: float):
        """
        Update Q-value for the selected arm using incremental update.
        
        Args:
            arm: Arm that was selected
            reward: Reward received (+1 for correct, -1 for incorrect)
        """
        self.arm_counts[arm] += 1
        
        # Incremental Q-value update: Q(a) = Q(a) + alpha * (R - Q(a))
        self.q_values[arm] += self.learning_rate * (reward - self.q_values[arm])
    
    def get_arm_for_item(self, item_id: int) -> int:
        """Get the arm (difficulty group) for an item."""
        return self.item_to_arm.get(item_id, 0)
    
    def reset(self):
        """Reset agent state (but keep learned Q-values)."""
        # Keep Q-values and counts for online learning
        pass


def train_bandit_agent(env, num_episodes: int = 50, num_arms: int = 10, 
                      epsilon: float = 0.1, learning_rate: float = 0.1):
    """
    Train and evaluate bandit agent.
    
    Args:
        env: MemoryEnv instance
        num_episodes: Number of training episodes
        num_arms: Number of difficulty groups
        epsilon: Exploration rate
        learning_rate: Learning rate for Q-updates
        
    Returns:
        List of episode results
    """
    agent = BanditAgent(num_arms=num_arms, epsilon=epsilon, learning_rate=learning_rate)
    results = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = []
        episode_recalls = []
        
        while not done:
            # Get available actions
            available_actions = env.get_available_actions()
            
            # Select action using bandit
            action = agent.select_action(state, available_actions)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Update bandit
            arm = agent.get_arm_for_item(action)
            agent.update(arm, reward)
            
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
        
        if (episode + 1) % 10 == 0:
            print(f"Bandit Episode {episode + 1}/{num_episodes}: "
                  f"Recall Rate = {recall_rate:.3f}, Reward = {cumulative_reward:.1f}")
            print(f"  Q-values: {agent.q_values}")
    
    return {
        'agent': agent,
        'results': results
    }

