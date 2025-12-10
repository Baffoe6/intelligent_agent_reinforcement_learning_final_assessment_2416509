"""
Unified Experiment Manager
Consolidated experiment runner with configurable parameters
"""

import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..core.agent import OptimizedAgent
from ..core.environment import SnakeEnvironment
from ..utils.logging import get_logger, log_experiment_start, log_experiment_end, log_episode_progress
from ..utils.config_manager import get_config


class ExperimentConfig:
    """Configuration for a single experiment"""
    
    def __init__(self, name: str, agent_config: Dict[str, Any], 
                 env_config: Dict[str, Any], episodes: int = 100):
        self.name = name
        self.agent_config = agent_config
        self.env_config = env_config
        self.episodes = episodes


class ExperimentResult:
    """Results from a single experiment"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.scores = []
        self.mean_scores = []
        self.training_losses = []
        self.episode_times = []
        self.start_time = None
        self.end_time = None
        self.final_stats = {}
    
    def add_episode_result(self, score: int, mean_score: float, 
                          training_loss: float, episode_time: float):
        """Add results from one episode"""
        self.scores.append(score)
        self.mean_scores.append(mean_score)
        self.training_losses.append(training_loss)
        self.episode_times.append(episode_time)
    
    def finalize(self, agent_stats: Dict[str, Any]):
        """Finalize experiment results"""
        self.end_time = time.time()
        self.final_stats = {
            'config_name': self.config.name,
            'episodes': self.config.episodes,
            'final_mean_score': self.mean_scores[-1] if self.mean_scores else 0,
            'max_score': max(self.scores) if self.scores else 0,
            'final_50_avg': np.mean(self.scores[-50:]) if len(self.scores) >= 50 else np.mean(self.scores) if self.scores else 0,
            'final_20_avg': np.mean(self.scores[-20:]) if len(self.scores) >= 20 else np.mean(self.scores) if self.scores else 0,
            'total_training_time': self.end_time - self.start_time if self.start_time else 0,
            'avg_episode_time': np.mean(self.episode_times) if self.episode_times else 0,
            'avg_training_loss': np.mean(self.training_losses) if self.training_losses else 0,
            'agent_stats': agent_stats,
            'agent_config': self.config.agent_config,
            'env_config': self.config.env_config
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.final_stats


class ExperimentManager:
    """Unified experiment manager for all experiment types"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = get_config(config_file)
        self.logger = get_logger()
        self.results = {}
        self.experiment_start_time = datetime.now()
    
    def run_experiment(self, experiment_config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment"""
        self.logger.info(f"🚀 Starting experiment: {experiment_config.name}")
        log_experiment_start(experiment_config.name, experiment_config.agent_config)
        
        # Create result tracker
        result = ExperimentResult(experiment_config)
        result.start_time = time.time()
        
        # Create agent and environment
        agent = OptimizedAgent(experiment_config.agent_config)
        env = SnakeEnvironment(experiment_config.env_config)
        
        # Training metrics
        total_score = 0
        record = 0
        progress_interval = self.config.get('experiments.progress_interval', 20)
        
        try:
            for episode in range(experiment_config.episodes):
                episode_start = time.time()
                
                # Reset environment
                env.reset()
                episode_loss = 0
                loss_count = 0
                
                while True:
                    # Get current state
                    state_old = agent.get_state(env)
                    
                    # Get action
                    action = agent.get_action(state_old)
                    
                    # Perform step
                    reward, done, score = env.step(action)
                    state_new = agent.get_state(env)
                    
                    # Train short memory
                    agent.train_short_memory(state_old, action, reward, state_new, done)
                    
                    # Store experience
                    agent.remember(state_old, action, reward, state_new, done)
                    
                    if done:
                        break
                
                # Episode complete
                agent.on_episode_end(score)
                
                # Train long memory
                agent.train_long_memory()
                
                # Update metrics
                total_score += score
                mean_score = total_score / (episode + 1)
                record = max(record, score)
                
                # Calculate episode time and loss
                episode_time = time.time() - episode_start
                agent_stats = agent.get_statistics()
                avg_loss = agent_stats.get('average_loss', 0)
                
                # Store episode results
                result.add_episode_result(score, mean_score, avg_loss, episode_time)
                
                # Progress reporting
                if (episode + 1) % progress_interval == 0:
                    recent_avg = np.mean(result.scores[-progress_interval:])
                    log_episode_progress(episode + 1, experiment_config.episodes, 
                                       score, recent_avg, record)
                    
                    self.logger.info(f"Memory usage: {agent_stats['buffer_usage']['usage_percent']:.1f}%")
                
                # Save best model
                if score >= record and self.config.get('experiments.auto_save_models', True):
                    model_path = f"models/{experiment_config.name}_best.pth"
                    agent.save_model(model_path)
        
        except KeyboardInterrupt:
            self.logger.warning("Experiment interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Error during experiment: {e}")
            raise
        
        finally:
            # Clean up
            env.close()
        
        # Finalize results
        final_agent_stats = agent.get_statistics()
        result.finalize(final_agent_stats)
        
        # Log completion
        log_experiment_end(experiment_config.name, result.final_stats)
        
        # Store results
        self.results[experiment_config.name] = result
        
        return result
    
    def run_baseline_experiment(self, episodes: int = None) -> ExperimentResult:
        """Run baseline experiment"""
        episodes = episodes or self.config.get('experiments.default_episodes', 100)
        
        agent_config = self.config.get_agent_config('baseline')
        agent_config['network_type'] = 'baseline'
        agent_config['network_config'] = self.config.get_network_config('baseline')
        
        env_config = self.config.get_environment_config(has_walls=False)
        
        experiment_config = ExperimentConfig(
            "baseline",
            agent_config,
            env_config,
            episodes
        )
        
        return self.run_experiment(experiment_config)
    
    def run_neural_network_experiments(self, episodes: int = None) -> List[ExperimentResult]:
        """Run neural network architecture experiments"""
        episodes = episodes or self.config.get('experiments.default_episodes', 100)
        
        network_types = ['increased', 'deeper', 'wide']
        results = []
        
        for network_type in network_types:
            agent_config = self.config.get_agent_config('baseline')
            agent_config['network_type'] = network_type
            agent_config['network_config'] = self.config.get_network_config(network_type)
            
            env_config = self.config.get_environment_config(has_walls=False)
            
            experiment_config = ExperimentConfig(
                f"neural_network_{network_type}",
                agent_config,
                env_config,
                episodes
            )
            
            result = self.run_experiment(experiment_config)
            results.append(result)
        
        return results
    
    def run_memory_experiments(self, episodes: int = None) -> List[ExperimentResult]:
        """Run memory size experiments"""
        episodes = episodes or self.config.get('experiments.default_episodes', 100)
        
        memory_sizes = ['tiny', 'small', 'large']
        results = []
        
        for memory_size in memory_sizes:
            agent_config = self.config.get_agent_config(memory_size)
            agent_config['network_type'] = 'baseline'
            agent_config['network_config'] = self.config.get_network_config('baseline')
            
            env_config = self.config.get_environment_config(has_walls=False)
            
            experiment_config = ExperimentConfig(
                f"memory_{memory_size}",
                agent_config,
                env_config,
                episodes
            )
            
            result = self.run_experiment(experiment_config)
            results.append(result)
        
        return results
    
    def run_environment_complexity_experiments(self, episodes: int = None) -> List[ExperimentResult]:
        """Run environment complexity experiments (with walls)"""
        episodes = episodes or self.config.get('experiments.default_episodes', 100)
        
        experiments = [
            ('baseline_walls', 'baseline', 'baseline'),
            ('increased_nn_walls', 'increased', 'baseline'),
            ('large_memory_walls', 'baseline', 'large')
        ]
        
        results = []
        
        for exp_name, network_type, memory_size in experiments:
            agent_config = self.config.get_agent_config(memory_size)
            agent_config['network_type'] = network_type
            agent_config['network_config'] = self.config.get_network_config(network_type)
            
            env_config = self.config.get_environment_config(has_walls=True)
            
            experiment_config = ExperimentConfig(
                exp_name,
                agent_config,
                env_config,
                episodes
            )
            
            result = self.run_experiment(experiment_config)
            results.append(result)
        
        return results
    
    def run_full_assessment(self, episodes: int = None) -> Dict[str, ExperimentResult]:
        """Run complete assessment suite"""
        episodes = episodes or self.config.get('experiments.default_episodes', 100)
        
        self.logger.info(f"🎯 Starting full assessment with {episodes} episodes per experiment")
        
        # Part A: Baseline
        self.logger.info("📊 Part A: Baseline")
        self.run_baseline_experiment(episodes)
        
        # Part B: Neural Networks
        self.logger.info("📊 Part B: Neural Network Architectures")
        self.run_neural_network_experiments(episodes)
        
        # Part C: Memory Sizes
        self.logger.info("📊 Part C: Memory Buffer Sizes")
        self.run_memory_experiments(episodes)
        
        # Part D: Environment Complexity
        self.logger.info("📊 Part D: Environment Complexity")
        self.run_environment_complexity_experiments(episodes)
        
        self.logger.info("✅ Full assessment completed!")
        
        return self.results
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save experiment results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/experiment_results_{timestamp}.json"
        
        # Ensure directory exists
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        results_dict = {}
        for name, result in self.results.items():
            results_dict[name] = result.to_dict()
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Results saved to {filename}")
        return filename
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary"""
        if not self.results:
            return {"message": "No experiments completed yet"}
        
        summary = {}
        
        for name, result in self.results.items():
            stats = result.final_stats
            summary[name] = {
                'final_score': stats.get('final_mean_score', 0),
                'max_score': stats.get('max_score', 0),
                'training_time_minutes': stats.get('total_training_time', 0) / 60,
                'episodes': stats.get('episodes', 0)
            }
        
        return summary