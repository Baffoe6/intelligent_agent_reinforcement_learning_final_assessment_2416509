"""
Optimized RL Agent Implementation
Enhanced agent with composition-based design and performance optimizations
"""

import torch
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
import random

from .networks import NetworkArchitecture, OptimizedQTrainer, create_network
from .environment import Direction, Point
from ..utils.logging import get_logger


class StateProcessor:
    """Handles state preprocessing and feature extraction"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = get_logger()
    
    def process_state(self, game) -> np.ndarray:
        """Extract and preprocess state features from game"""
        head = game.snake[0]
        
        # Pre-calculate points for efficiency
        directions = {
            'left': Point(head.x - 20, head.y),
            'right': Point(head.x + 20, head.y),
            'up': Point(head.x, head.y - 20),
            'down': Point(head.x, head.y + 20)
        }
        
        # Current direction flags
        dir_flags = {
            'left': game.direction == Direction.LEFT,
            'right': game.direction == Direction.RIGHT,
            'up': game.direction == Direction.UP,
            'down': game.direction == Direction.DOWN
        }
        
        # Danger detection (vectorized)
        dangers = self._calculate_dangers(game, directions, dir_flags)
        
        # Food location relative to head
        food_features = [
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]
        
        # Combine all features
        state = dangers + list(dir_flags.values()) + food_features
        
        return np.array(state, dtype=np.float32)  # Use float32 for better performance
    
    def _calculate_dangers(self, game, directions: Dict, dir_flags: Dict) -> List[bool]:
        """Calculate danger in all directions efficiently"""
        # Danger straight
        danger_straight = (
            (dir_flags['right'] and game.is_collision_point(directions['right'])) or
            (dir_flags['left'] and game.is_collision_point(directions['left'])) or
            (dir_flags['up'] and game.is_collision_point(directions['up'])) or
            (dir_flags['down'] and game.is_collision_point(directions['down']))
        )
        
        # Danger right (relative to current direction)
        danger_right = (
            (dir_flags['up'] and game.is_collision_point(directions['right'])) or
            (dir_flags['down'] and game.is_collision_point(directions['left'])) or
            (dir_flags['left'] and game.is_collision_point(directions['up'])) or
            (dir_flags['right'] and game.is_collision_point(directions['down']))
        )
        
        # Danger left (relative to current direction)
        danger_left = (
            (dir_flags['down'] and game.is_collision_point(directions['right'])) or
            (dir_flags['up'] and game.is_collision_point(directions['left'])) or
            (dir_flags['right'] and game.is_collision_point(directions['up'])) or
            (dir_flags['left'] and game.is_collision_point(directions['down']))
        )
        
        return [danger_straight, danger_right, danger_left]


class ExperienceBuffer:
    """Optimized experience replay buffer with memory efficiency"""
    
    def __init__(self, capacity: int, device: Optional[torch.device] = None):
        self.capacity = capacity
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer = deque(maxlen=capacity)
        self.logger = get_logger()
    
    def push(self, state: np.ndarray, action: List[int], reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors efficiently
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
    
    def __len__(self):
        return len(self.buffer)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get buffer usage statistics"""
        return {
            'current_size': len(self.buffer),
            'capacity': self.capacity,
            'usage_percent': (len(self.buffer) / self.capacity) * 100
        }


class ExplorationStrategy:
    """Handles exploration vs exploitation strategy"""
    
    def __init__(self, epsilon_start: int = 80, epsilon_decay: float = 1.0, 
                 randomness_factor: int = 200):
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.randomness_factor = randomness_factor
        self.episodes = 0
    
    def get_epsilon(self) -> float:
        """Calculate current epsilon value"""
        return max(0, self.epsilon_start - (self.episodes * self.epsilon_decay))
    
    def should_explore(self) -> bool:
        """Decide whether to explore or exploit"""
        epsilon = self.get_epsilon()
        return random.randint(0, self.randomness_factor) < epsilon
    
    def update(self):
        """Update exploration parameters (call after each episode)"""
        self.episodes += 1


class OptimizedAgent:
    """Optimized RL Agent with composition-based design"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Components
        self.state_processor = StateProcessor(self.device)
        self.experience_buffer = ExperienceBuffer(
            config.get('memory_size', 100000), 
            self.device
        )
        self.exploration = ExplorationStrategy(
            config.get('epsilon_start', 80),
            config.get('epsilon_decay_rate', 1.0),
            config.get('exploration_randomness', 200)
        )
        
        # Network and trainer
        network_config = config.get('network_config', {})
        self.network = create_network(
            config.get('network_type', 'baseline'),
            network_config
        )
        
        self.trainer = OptimizedQTrainer(
            self.network,
            lr=config.get('learning_rate', 0.001),
            gamma=config.get('gamma', 0.9),
            device=self.device
        )
        
        # Training state
        self.n_games = 0
        self.training_steps = 0
        self.batch_size = config.get('batch_size', 1000)
        
        # Performance tracking
        self.performance_stats = {
            'training_losses': [],
            'episode_scores': [],
            'exploration_rates': []
        }
    
    def get_state(self, game) -> np.ndarray:
        """Get processed state from game"""
        return self.state_processor.process_state(game)
    
    def get_action(self, state: np.ndarray) -> List[int]:
        """Get action using current policy"""
        action = [0, 0, 0]
        
        if self.exploration.should_explore():
            # Random action (exploration)
            move = random.randint(0, 2)
            action[move] = 1
        else:
            # Policy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.network(state_tensor)
                move = torch.argmax(q_values).item()
                action[move] = 1
        
        return action
    
    def remember(self, state: np.ndarray, action: List[int], reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in buffer"""
        self.experience_buffer.push(state, action, reward, next_state, done)
    
    def train_short_memory(self, state: np.ndarray, action: List[int], reward: float,
                          next_state: np.ndarray, done: bool):
        """Train on single experience"""
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        self.performance_stats['training_losses'].append(loss)
        self.training_steps += 1
    
    def train_long_memory(self, batch_size: Optional[int] = None):
        """Train on batch of experiences from buffer"""
        if len(self.experience_buffer) == 0:
            return
        
        batch_size = batch_size or self.config.get('batch_size', 1000)
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.experience_buffer.sample(batch_size)
        
        # Train
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        self.performance_stats['training_losses'].append(loss)
        self.training_steps += 1
    
    def on_episode_end(self, score: int):
        """Called at the end of each episode"""
        self.n_games += 1
        self.exploration.update()
        
        # Track performance
        self.performance_stats['episode_scores'].append(score)
        self.performance_stats['exploration_rates'].append(self.exploration.get_epsilon())
        
        # Log progress
        if self.n_games % 10 == 0:
            avg_loss = np.mean(self.performance_stats['training_losses'][-100:]) if self.performance_stats['training_losses'] else 0
            self.logger.debug(f"Episode {self.n_games}: Score {score}, Avg Loss: {avg_loss:.4f}, Epsilon: {self.exploration.get_epsilon():.1f}")
    
    def save_model(self, file_path: str):
        """Save model and training state"""
        self.network.save(file_path)
        self.logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str):
        """Load model from file"""
        self.network = NetworkArchitecture.load(file_path)
        self.network.to(self.device)
        
        # Update trainer with loaded network
        self.trainer = OptimizedQTrainer(
            self.network,
            lr=self.config.get('learning_rate', 0.001),
            gamma=self.config.get('gamma', 0.9),
            device=self.device
        )
        
        self.logger.info(f"Model loaded from {file_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        network_stats = self.network.get_parameter_count()
        buffer_stats = self.experience_buffer.get_usage_stats()
        
        return {
            'episodes': self.n_games,
            'training_steps': self.training_steps,
            'exploration_rate': self.exploration.get_epsilon(),
            'network_parameters': network_stats,
            'buffer_usage': buffer_stats,
            'recent_scores': self.performance_stats['episode_scores'][-10:],
            'average_loss': np.mean(self.performance_stats['training_losses'][-100:]) if self.performance_stats['training_losses'] else 0
        }