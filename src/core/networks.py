"""
Optimized Neural Network Architectures
Unified, configurable neural network implementation with proper initialization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class NetworkArchitecture(nn.Module):
    """Unified configurable neural network architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.input_size = config.get('input_size', 11)
        self.output_size = config.get('output_size', 3)
        self.layers = config.get('layers', [256])
        self.activation = config.get('activation', 'relu')
        self.dropout_rate = config.get('dropout', 0.0)
        self.use_batch_norm = config.get('batch_norm', False)
        
        # Build network
        self.network = self._build_network()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_network(self) -> nn.Module:
        """Build neural network layers"""
        modules = []
        
        # Input layer
        prev_size = self.input_size
        
        # Hidden layers
        for i, layer_size in enumerate(self.layers):
            # Linear layer
            modules.append(nn.Linear(prev_size, layer_size))
            
            # Batch normalization (optional)
            if self.use_batch_norm:
                modules.append(nn.BatchNorm1d(layer_size))
            
            # Activation
            if self.activation.lower() == 'relu':
                modules.append(nn.ReLU())
            elif self.activation.lower() == 'tanh':
                modules.append(nn.Tanh())
            elif self.activation.lower() == 'sigmoid':
                modules.append(nn.Sigmoid())
            
            # Dropout (optional)
            if self.dropout_rate > 0:
                modules.append(nn.Dropout(self.dropout_rate))
            
            prev_size = layer_size
        
        # Output layer
        modules.append(nn.Linear(prev_size, self.output_size))
        
        return nn.Sequential(*modules)
    
    def _initialize_weights(self):
        """Initialize network weights using appropriate initialization"""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for ReLU networks
                if self.activation.lower() == 'relu':
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                
                # Initialize bias to small positive value
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)
    
    def save(self, file_path: str):
        """Save model state"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both state dict and config
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': self.config,
            'architecture': str(self.network)
        }
        
        torch.save(checkpoint, file_path)
    
    @classmethod
    def load(cls, file_path: str) -> 'NetworkArchitecture':
        """Load model from file"""
        checkpoint = torch.load(file_path, map_location='cpu')
        
        # Create network with saved config
        network = cls(checkpoint['config'])
        network.load_state_dict(checkpoint['state_dict'])
        
        return network
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }


class OptimizedQTrainer:
    """Optimized Q-Learning trainer with performance improvements"""
    
    def __init__(self, model: NetworkArchitecture, lr: float = 0.001, gamma: float = 0.9, 
                 device: Optional[str] = None):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer with better defaults
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.lr,
            weight_decay=1e-5,  # L2 regularization
            eps=1e-8
        )
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()  # Huber loss - more stable than MSE
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=10,
            factor=0.5
        )
    
    def train_step(self, state, action, reward, next_state, done):
        """Optimized training step with proper tensor handling"""
        # Convert to tensors with proper shape handling
        if (isinstance(state, torch.Tensor) and state.dim() > 1) or \
           (isinstance(state, (list, tuple)) and len(state) > 0 and isinstance(state[0], (list, tuple, np.ndarray))):
            # Multiple experiences (batch) - already tensors from experience buffer
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
                next_state = torch.FloatTensor(next_state).to(self.device)
                action = torch.LongTensor(action).to(self.device)
                reward = torch.FloatTensor(reward).to(self.device)
                done = torch.BoolTensor(done).to(self.device)
            else:
                # Already tensors from experience buffer
                state = state.to(self.device)
                next_state = next_state.to(self.device)
                action = action.to(self.device)
                reward = reward.to(self.device)
                done = done.to(self.device)
        else:
            # Single experience
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            action = torch.LongTensor(action).unsqueeze(0).to(self.device)
            
            # Handle reward conversion - check if already a tensor
            if isinstance(reward, torch.Tensor):
                reward = reward.float().to(self.device)
                if reward.dim() == 0:  # scalar tensor
                    reward = reward.unsqueeze(0)
            else:
                reward = torch.FloatTensor([reward]).to(self.device)
            
            # Handle done conversion - check if already a tensor    
            if isinstance(done, torch.Tensor):
                done = done.bool().to(self.device)
                if done.dim() == 0:  # scalar tensor
                    done = done.unsqueeze(0)
            else:
                done = torch.BoolTensor([done]).to(self.device)
        
        # Current Q values
        current_q_values = self.model(state)
        
        # Next Q values (detached for stability)
        with torch.no_grad():
            next_q_values = self.model(next_state)
            max_next_q_values = next_q_values.max(1)[0]
            
            # Target Q values
            target_q_values = reward + (self.gamma * max_next_q_values * ~done)
        
        # Get Q values for taken actions
        if len(action.shape) > 1:
            # Multi-dimensional action (one-hot)
            action_indices = action.argmax(1)
        else:
            action_indices = action
        
        predicted_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Compute loss
        loss = self.criterion(predicted_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step(loss.item())
        
        return loss.item()


def create_network(network_type: str, config: Optional[Dict[str, Any]] = None) -> NetworkArchitecture:
    """Factory function to create networks from configuration"""
    if config is None:
        # Default configurations
        configs = {
            'baseline': {
                'input_size': 11,
                'output_size': 3,
                'layers': [256],
                'activation': 'relu'
            },
            'increased': {
                'input_size': 11,
                'output_size': 3,
                'layers': [512, 256, 128, 64],
                'activation': 'relu'
            },
            'deeper': {
                'input_size': 11,
                'output_size': 3,
                'layers': [256, 256, 128, 128, 64, 32, 16],
                'activation': 'relu',
                'dropout': 0.1,
                'batch_norm': True
            },
            'wide': {
                'input_size': 11,
                'output_size': 3,
                'layers': [512, 512],
                'activation': 'relu'
            }
        }
        
        if network_type not in configs:
            raise ValueError(f"Unknown network type: {network_type}")
        
        config = configs[network_type]
    
    return NetworkArchitecture(config)