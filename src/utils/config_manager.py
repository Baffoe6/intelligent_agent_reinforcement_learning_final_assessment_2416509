"""
Configuration Management System
Centralized configuration handling with YAML/JSON support
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Centralized configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self._config = {}
        self._config_file = config_file
        
        # Load default configuration
        self._load_default_config()
        
        # Load user configuration if provided
        if config_file:
            self.load_config(config_file)
    
    def _load_default_config(self):
        """Load default configuration"""
        default_config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                self._config = yaml.safe_load(f)
    
    def load_config(self, config_file: str):
        """Load configuration from file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                user_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Merge with existing configuration
        self._deep_merge(self._config, user_config)
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_agent_config(self, memory_type: str = 'baseline') -> Dict[str, Any]:
        """Get agent configuration for a given memory type"""
        agent_config = self.get('agent', {}).copy()
        
        # Get memory size
        memory_size = self.get(f'memory.{memory_type}', 100000)
        agent_config['memory_size'] = memory_size
        
        return agent_config
    
    def get_network_config(self, network_type: str) -> Dict[str, Any]:
        """Get network configuration for a given network type"""
        network_config = self.get(f'networks.{network_type}', {}).copy()
        
        # Ensure we have the basic fields
        if 'input_size' not in network_config:
            network_config['input_size'] = 11
        if 'output_size' not in network_config:
            network_config['output_size'] = 3
        
        # Convert 'layers' to the correct format if needed
        if 'layers' in network_config and isinstance(network_config['layers'], list):
            network_config['layers'] = network_config['layers']
        
        return network_config
    
    def get_environment_config(self, has_walls: bool = False) -> Dict[str, Any]:
        """Get environment configuration"""
        env_config = self.get('environment', {}).copy()
        
        # Set wall configuration
        env_config['has_walls'] = has_walls
        if has_walls:
            env_config['wall_config'] = self.get('environment.walls', {})
        
        # Ensure headless mode for training
        env_config['headless'] = True
        
        return env_config


# Global configuration instance
_global_config = None


def get_config(config_file: Optional[str] = None) -> Config:
    """Get global configuration instance"""
    global _global_config
    
    if _global_config is None:
        _global_config = Config(config_file)
    elif config_file and _global_config._config_file != config_file:
        # Reload with new config file
        _global_config = Config(config_file)
    
    return _global_config