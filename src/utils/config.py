"""
Configuration Management System for Snake RL Assessment
Centralized configuration handling with validation and defaults
"""

import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import os

@dataclass
class NetworkConfig:
    """Neural network architecture configuration"""
    type: str = "baseline"
    input_size: int = 11
    output_size: int = 3
    hidden_sizes: List[int] = None
    dropout: float = 0.0
    batch_norm: bool = False
    activation: str = "relu"
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256]

@dataclass
class AgentConfig:
    """RL Agent configuration"""
    memory_size: int = 100_000
    batch_size: int = 1000
    learning_rate: float = 0.001
    gamma: float = 0.9
    epsilon_start: float = 80.0
    epsilon_decay: float = 1.0
    target_update_freq: int = 100

@dataclass
class EnvironmentConfig:
    """Game environment configuration"""
    width: int = 640
    height: int = 480
    block_size: int = 20
    speed: int = 40
    has_walls: bool = False
    wall_pattern: str = "default"
    max_steps_multiplier: int = 100

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    name: str = "default_experiment"
    episodes: int = 100
    seed: int = 42
    save_models: bool = True
    save_frequency: int = 50
    log_frequency: int = 20
    use_gpu: bool = False
    device: str = "auto"

@dataclass
class OutputConfig:
    """Output and logging configuration"""
    results_dir: str = "./results"
    models_dir: str = "./models"
    plots_dir: str = "./plots"
    logs_dir: str = "./logs"
    log_level: str = "INFO"
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300

@dataclass
class AssessmentConfig:
    """Complete assessment configuration"""
    network: NetworkConfig
    agent: AgentConfig
    environment: EnvironmentConfig
    experiment: ExperimentConfig
    output: OutputConfig
    
    def __post_init__(self):
        # Ensure all configs are proper dataclass instances
        if isinstance(self.network, dict):
            self.network = NetworkConfig(**self.network)
        if isinstance(self.agent, dict):
            self.agent = AgentConfig(**self.agent)
        if isinstance(self.environment, dict):
            self.environment = EnvironmentConfig(**self.environment)
        if isinstance(self.experiment, dict):
            self.experiment = ExperimentConfig(**self.experiment)
        if isinstance(self.output, dict):
            self.output = OutputConfig(**self.output)

class ConfigManager:
    """Manages configuration loading, validation, and defaults"""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def load_config(self, config_path: str) -> AssessmentConfig:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return AssessmentConfig(**config_dict)
    
    def save_config(self, config: AssessmentConfig, file_path: str):
        """Save configuration to file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(config)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    def get_default_config(self) -> AssessmentConfig:
        """Get default configuration"""
        return AssessmentConfig(
            network=NetworkConfig(),
            agent=AgentConfig(),
            environment=EnvironmentConfig(),
            experiment=ExperimentConfig(),
            output=OutputConfig()
        )
    
    def get_preset_configs(self) -> Dict[str, AssessmentConfig]:
        """Get predefined experiment configurations"""
        configs = {}
        
        # Baseline configuration
        baseline = self.get_default_config()
        baseline.experiment.name = "baseline"
        configs["baseline"] = baseline
        
        # Increased NN configuration
        increased_nn = self.get_default_config()
        increased_nn.network.type = "increased"
        increased_nn.network.hidden_sizes = [512, 256, 128, 64]
        increased_nn.experiment.name = "increased_nn"
        configs["increased_nn"] = increased_nn
        
        # Deeper NN configuration
        deeper_nn = self.get_default_config()
        deeper_nn.network.type = "deeper"
        deeper_nn.network.hidden_sizes = [256, 256, 128, 128, 64, 32, 16]
        deeper_nn.network.dropout = 0.1
        deeper_nn.experiment.name = "deeper_nn"
        configs["deeper_nn"] = deeper_nn
        
        # Wide NN configuration
        wide_nn = self.get_default_config()
        wide_nn.network.type = "wide"
        wide_nn.network.hidden_sizes = [512, 512]
        wide_nn.experiment.name = "wide_nn"
        configs["wide_nn"] = wide_nn
        
        # Small memory configuration
        small_memory = self.get_default_config()
        small_memory.agent.memory_size = 10_000
        small_memory.experiment.name = "small_memory"
        configs["small_memory"] = small_memory
        
        # Large memory configuration
        large_memory = self.get_default_config()
        large_memory.agent.memory_size = 500_000
        large_memory.experiment.name = "large_memory"
        configs["large_memory"] = large_memory
        
        # Tiny memory configuration
        tiny_memory = self.get_default_config()
        tiny_memory.agent.memory_size = 1_000
        tiny_memory.experiment.name = "tiny_memory"
        configs["tiny_memory"] = tiny_memory
        
        # Wall environment configurations
        for name, base_config in list(configs.items()):
            wall_config = AssessmentConfig(**asdict(base_config))
            wall_config.environment.has_walls = True
            wall_config.experiment.name = f"{name}_with_walls"
            configs[f"{name}_with_walls"] = wall_config
        
        return configs
    
    def create_preset_config_files(self):
        """Create preset configuration files"""
        presets = self.get_preset_configs()
        
        for name, config in presets.items():
            file_path = self.config_dir / f"{name}.yaml"
            self.save_config(config, file_path)
            print(f"Created preset config: {file_path}")
    
    def validate_config(self, config: AssessmentConfig) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate network config
        if config.network.input_size <= 0:
            issues.append("Network input_size must be positive")
        if config.network.output_size <= 0:
            issues.append("Network output_size must be positive")
        if not config.network.hidden_sizes or len(config.network.hidden_sizes) == 0:
            issues.append("Network must have at least one hidden layer")
        if any(size <= 0 for size in config.network.hidden_sizes):
            issues.append("All hidden layer sizes must be positive")
        if not (0 <= config.network.dropout < 1):
            issues.append("Network dropout must be between 0 and 1")
        
        # Validate agent config
        if config.agent.memory_size <= 0:
            issues.append("Agent memory_size must be positive")
        if config.agent.batch_size <= 0:
            issues.append("Agent batch_size must be positive")
        if config.agent.learning_rate <= 0:
            issues.append("Agent learning_rate must be positive")
        if not (0 <= config.agent.gamma <= 1):
            issues.append("Agent gamma must be between 0 and 1")
        
        # Validate environment config
        if config.environment.width <= 0 or config.environment.height <= 0:
            issues.append("Environment dimensions must be positive")
        if config.environment.block_size <= 0:
            issues.append("Environment block_size must be positive")
        if config.environment.speed <= 0:
            issues.append("Environment speed must be positive")
        
        # Validate experiment config
        if config.experiment.episodes <= 0:
            issues.append("Experiment episodes must be positive")
        if config.experiment.save_frequency <= 0:
            issues.append("Experiment save_frequency must be positive")
        if config.experiment.log_frequency <= 0:
            issues.append("Experiment log_frequency must be positive")
        
        return issues
    
    def merge_configs(self, base_config: AssessmentConfig, override_config: Dict[str, Any]) -> AssessmentConfig:
        """Merge override values into base configuration"""
        base_dict = asdict(base_config)
        
        # Deep merge dictionaries
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = deep_merge(base_dict, override_config)
        return AssessmentConfig(**merged_dict)

# Global config manager instance
config_manager = ConfigManager()

def load_config(config_path: str) -> AssessmentConfig:
    """Convenience function to load configuration"""
    return config_manager.load_config(config_path)

def get_default_config() -> AssessmentConfig:
    """Convenience function to get default configuration"""
    return config_manager.get_default_config()

def get_preset_config(preset_name: str) -> AssessmentConfig:
    """Convenience function to get preset configuration"""
    presets = config_manager.get_preset_configs()
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    return presets[preset_name]

if __name__ == "__main__":
    # Create default configuration files
    manager = ConfigManager()
    manager.create_preset_config_files()
    print(f"Configuration files created in {manager.config_dir}")