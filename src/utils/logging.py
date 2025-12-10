"""
Logging System for Snake RL Assessment
Provides structured logging with configurable output formats
"""

import logging
import sys
import os
from typing import Optional
from pathlib import Path
from datetime import datetime


class CustomFormatter(logging.Formatter):
    """Custom formatter with color support"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'ENDC': '\033[0m'       # End color
    }
    
    def format(self, record):
        # Add color to levelname
        if hasattr(record, 'levelname'):
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['ENDC']}"
        
        return super().format(record)


class ExperimentLogger:
    """Enhanced logger for experiment tracking"""
    
    def __init__(self, name: str, config: Optional[dict] = None):
        self.logger = logging.getLogger(name)
        self.config = config or {}
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with handlers"""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set level
        level = getattr(logging, self.config.get('level', 'INFO').upper())
        self.logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = CustomFormatter(
            self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if self.config.get('file_logging', True):
            log_file = self.config.get('log_file', 'logs/snake_rl.log')
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_formatter = logging.Formatter(
                self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def experiment_start(self, experiment_name: str, config: dict):
        """Log experiment start"""
        self.logger.info(f"🚀 Starting experiment: {experiment_name}")
        self.logger.debug(f"Experiment config: {config}")
    
    def experiment_end(self, experiment_name: str, results: dict):
        """Log experiment completion"""
        self.logger.info(f"✅ Completed experiment: {experiment_name}")
        self.logger.info(f"Final score: {results.get('final_mean_score', 'N/A')}")
        self.logger.info(f"Max score: {results.get('max_score', 'N/A')}")
    
    def episode_progress(self, episode: int, total: int, score: int, avg_score: float, record: int):
        """Log episode progress"""
        self.logger.info(f"Episode {episode}/{total} - Score: {score} - Avg: {avg_score:.1f} - Record: {record}")
    
    def performance_warning(self, message: str):
        """Log performance warning"""
        self.logger.warning(f"⚠️ Performance: {message}")
    
    def debug(self, message: str):
        """Debug logging"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Info logging"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Warning logging"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Error logging"""
        self.logger.error(message)


# Global logger instance
_global_logger = None


def get_logger(name: str = "snake_rl", config: Optional[dict] = None) -> ExperimentLogger:
    """Get global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = ExperimentLogger(name, config)
    
    return _global_logger


def setup_logging(config: dict):
    """Setup logging with configuration"""
    global _global_logger
    _global_logger = ExperimentLogger("snake_rl", config)


# Convenience functions
def log_experiment_start(experiment_name: str, config: dict):
    """Log experiment start"""
    get_logger().experiment_start(experiment_name, config)


def log_experiment_end(experiment_name: str, results: dict):
    """Log experiment completion"""
    get_logger().experiment_end(experiment_name, results)


def log_episode_progress(episode: int, total: int, score: int, avg_score: float, record: int):
    """Log episode progress"""
    get_logger().episode_progress(episode, total, score, avg_score, record)


def log_performance_warning(message: str):
    """Log performance warning"""
    get_logger().performance_warning(message)