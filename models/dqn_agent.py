"""
Deep Q-Network (DQN) Agent for Spaced Repetition.
Full model-free RL using neural network function approximation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Optional, List
import pandas as pd


class DQN(nn.Module):
    """
    Deep Q-Network for learning Q(s, a) values.
    Input: State vector for an item
    Output: Q-value for selecting that item
    """
    
    def __init__(self, state_dim: int = 4, hidden_dims: List[int] = [64, 64]):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state vector per item (4: attempts, successes, time, accuracy)
            hidden_dims: List of hidden layer dimensions
        """
        super(DQN, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer: Q-value (single scalar)
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)
        
        Returns:
            Q-value tensor of shape (batch_size, 1) or (1,)
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    Stores (state, action, reward, next_state, done) tuples.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add experience to buffer.
        
        Args:
            state: Current state (full state matrix)
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent for spaced repetition scheduling.
    """
    
    def __init__(self,
                 state_dim: int = 4,
                 num_actions: int = 100,
                 hidden_dims: List[int] = [64, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 replay_capacity: int = 10000,
                 target_update_freq: int = 10,
                 device: str = 'cpu'):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state per item
            num_actions: Number of possible actions (items)
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay per episode
            batch_size: Mini-batch size
            replay_capacity: Replay buffer capacity
            target_update_freq: Frequency of target network updates
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Networks
        self.q_network = DQN(state_dim, hidden_dims).to(device)
        self.target_network = DQN(state_dim, hidden_dims).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_capacity)
        
        # Training step counter
        self.step_count = 0
    
    def select_action(self, state: np.ndarray, available_actions: np.ndarray, 
                     training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state matrix (num_items, state_dim)
            available_actions: Available action indices
            training: Whether in training mode (affects epsilon)
        
        Returns:
            Selected action (item ID)
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(available_actions)
        
        # Exploit: select action with highest Q-value
        with torch.no_grad():
            # Get Q-values for all items
            state_tensor = torch.FloatTensor(state).to(self.device)  # (num_items, state_dim)
            q_values = self.q_network(state_tensor).squeeze()  # (num_items,)
            
            # Mask unavailable actions
            q_values_np = q_values.cpu().numpy()
            q_values_np[~np.isin(np.arange(len(q_values_np)), available_actions)] = -np.inf
            
            # Select best action
            action = np.argmax(q_values_np)
        
        return action
    
    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon * self.epsilon_decay)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step on a mini-batch from replay buffer.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Unpack batch
        states = np.array([exp[0] for exp in batch])  # (batch, num_items, state_dim)
        actions = np.array([exp[1] for exp in batch])  # (batch,)
        rewards = np.array([exp[2] for exp in batch])  # (batch,)
        next_states = np.array([exp[3] for exp in batch])  # (batch, num_items, state_dim)
        dones = np.array([exp[4] for exp in batch])  # (batch,)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        
        # Get Q-values for selected actions
        # For each experience, get Q-value of the state-action pair
        # We need to extract Q(s, a) where a is the action taken
        q_values = []
        for i in range(self.batch_size):
            state = states_t[i]  # (num_items, state_dim)
            action = actions_t[i].item()
            # Get Q-value for the selected item's state
            item_state = state[action:action+1]  # (1, state_dim)
            q_value = self.q_network(item_state)  # (1, 1)
            q_values.append(q_value.squeeze())
        q_values = torch.stack(q_values)  # (batch_size,)
        
        # Compute target Q-values
        with torch.no_grad():
            # Get max Q-value from next states
            next_q_values = []
            for i in range(self.batch_size):
                next_state = next_states_t[i]  # (num_items, state_dim)
                # Get Q-values for all items in next state
                next_q_all = self.target_network(next_state).squeeze()  # (num_items,)
                next_q = next_q_all.max()  # Max Q over all items
                next_q_values.append(next_q)
            next_q_values = torch.stack(next_q_values)  # (batch_size,)
            
            target_q = rewards_t + (self.gamma * next_q_values * ~dones_t)
        
        # Compute loss
        loss = nn.MSELoss()(q_values, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']


def train_dqn_agent(env, num_episodes: int = 100, num_actions: int = 100,
                   learning_rate: float = 0.001, batch_size: int = 32,
                   device: str = 'cpu'):
    """
    Train and evaluate DQN agent.
    
    Args:
        env: MemoryEnv instance
        num_episodes: Number of training episodes
        num_actions: Number of possible actions (items)
        learning_rate: Learning rate
        batch_size: Mini-batch size
        device: 'cpu' or 'cuda'
        
    Returns:
        Dictionary with agent and results
    """
    agent = DQNAgent(
        state_dim=4,
        num_actions=num_actions,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device
    )
    
    results = []
    losses = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = []
        episode_recalls = []
        episode_losses = []
        
        while not done:
            # Select action
            available_actions = env.get_available_actions()
            action = agent.select_action(state, available_actions, training=True)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train on mini-batch
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
                losses.append(loss)
            
            # Track results
            episode_rewards.append(reward)
            episode_recalls.append(info['recalled'])
            
            state = next_state
        
        # Update epsilon
        agent.update_epsilon()
        
        # Calculate metrics
        total_reviews = len(episode_recalls)
        recall_rate = sum(episode_recalls) / total_reviews if total_reviews > 0 else 0.0
        cumulative_reward = sum(episode_rewards)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        results.append({
            'episode': episode,
            'recall_rate': recall_rate,
            'cumulative_reward': cumulative_reward,
            'total_reviews': total_reviews,
            'avg_loss': avg_loss,
            'epsilon': agent.epsilon
        })
        
        if (episode + 1) % 10 == 0:
            print(f"DQN Episode {episode + 1}/{num_episodes}: "
                  f"Recall Rate = {recall_rate:.3f}, Reward = {cumulative_reward:.1f}, "
                  f"Loss = {avg_loss:.4f}, Epsilon = {agent.epsilon:.3f}")
    
    return {
        'agent': agent,
        'results': results,
        'losses': losses
    }

