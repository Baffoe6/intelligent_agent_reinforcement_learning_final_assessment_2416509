"""
Comprehensive Logging System for Snake RL Assessment
Provides structured logging with proper formatting and file rotation
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json
import sys
from typing import Dict, Any, Optional
import traceback

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Create colored format string
        colored_format = f"%(asctime)s - {level_color}%(levelname)-8s{reset_color} - %(name)s - %(message)s"
        
        formatter = logging.Formatter(colored_format, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

class ExperimentLogger:
    """Centralized logging system for experiments"""
    
    def __init__(self, name: str = "snake_rl", log_dir: str = "./logs", 
                 level: str = "INFO", use_colors: bool = True):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.level = getattr(logging, level.upper())
        self.use_colors = use_colors
        
        # Create main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handlers()
        
        # Prevent duplicate logging
        self.logger.propagate = False
    
    def _setup_console_handler(self):
        """Setup console output handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        
        if self.use_colors and sys.stdout.isatty():
            formatter = ColoredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self):
        """Setup file output handlers"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main log file (rotating)
        main_log_file = self.log_dir / f"{self.name}.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file, maxBytes=10*1024*1024, backupCount=5
        )
        main_handler.setLevel(self.level)
        main_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)-8s - %(name)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(main_formatter)
        self.logger.addHandler(main_handler)
        
        # Error log file
        error_log_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=5*1024*1024, backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(main_formatter)
        self.logger.addHandler(error_handler)
        
        # JSON log file for structured data
        json_log_file = self.log_dir / f"{self.name}_{timestamp}.json"
        json_handler = logging.FileHandler(json_log_file)
        json_handler.setLevel(self.level)
        json_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(json_handler)
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance"""
        if name:
            return logging.getLogger(f"{self.name}.{name}")
        return self.logger
    
    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment start with configuration"""
        self.logger.info("="*80)
        self.logger.info("EXPERIMENT STARTING")
        self.logger.info("="*80)
        self.logger.info("Configuration:", extra={'config': config})
    
    def log_experiment_end(self, results: Dict[str, Any]):
        """Log experiment completion with results"""
        self.logger.info("="*80)
        self.logger.info("EXPERIMENT COMPLETED")
        self.logger.info("="*80)
        self.logger.info("Results:", extra={'results': results})
    
    def log_episode_progress(self, episode: int, total_episodes: int, 
                           score: int, avg_score: float, record: int,
                           training_time: float = None):
        """Log episode progress"""
        progress_pct = (episode / total_episodes) * 100
        
        extra_data = {
            'episode': episode,
            'total_episodes': total_episodes,
            'progress_percent': progress_pct,
            'score': score,
            'average_score': avg_score,
            'record': record
        }
        
        if training_time is not None:
            extra_data['training_time'] = training_time
        
        self.logger.info(
            f"Episode {episode}/{total_episodes} ({progress_pct:.1f}%) - "
            f"Score: {score} - Avg: {avg_score:.2f} - Record: {record}",
            extra=extra_data
        )
    
    def log_model_save(self, model_path: str, score: int):
        """Log model save event"""
        self.logger.info(f"New best model saved: {model_path} (score: {score})",
                        extra={'model_path': model_path, 'score': score, 'event': 'model_save'})
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        self.logger.info("Performance metrics", extra={'metrics': metrics, 'event': 'performance'})
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        self.logger.error(f"Error in {context}: {str(error)}", 
                         extra={'context': context, 'error_type': type(error).__name__},
                         exc_info=True)
    
    def log_memory_usage(self, memory_info: Dict[str, Any]):
        """Log memory usage information"""
        self.logger.debug("Memory usage", extra={'memory': memory_info, 'event': 'memory'})
    
    def log_training_step(self, episode: int, step: int, loss: float = None, 
                         reward: float = None, action: str = None):
        """Log detailed training step information"""
        extra_data = {
            'episode': episode,
            'step': step,
            'event': 'training_step'
        }
        
        if loss is not None:
            extra_data['loss'] = loss
        if reward is not None:
            extra_data['reward'] = reward
        if action is not None:
            extra_data['action'] = action
        
        self.logger.debug(f"Training step {step}", extra=extra_data)

# Global logger instance
_global_logger = None

def setup_logging(name: str = "snake_rl", log_dir: str = "./logs", 
                 level: str = "INFO", use_colors: bool = True) -> ExperimentLogger:
    """Setup global logging system"""
    global _global_logger
    _global_logger = ExperimentLogger(name, log_dir, level, use_colors)
    return _global_logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger.get_logger(name)

def log_experiment_start(config: Dict[str, Any]):
    """Convenience function to log experiment start"""
    global _global_logger
    if _global_logger:
        _global_logger.log_experiment_start(config)

def log_experiment_end(results: Dict[str, Any]):
    """Convenience function to log experiment end"""
    global _global_logger
    if _global_logger:
        _global_logger.log_experiment_end(results)

def log_episode_progress(episode: int, total_episodes: int, score: int, 
                        avg_score: float, record: int, training_time: float = None):
    """Convenience function to log episode progress"""
    global _global_logger
    if _global_logger:
        _global_logger.log_episode_progress(episode, total_episodes, score, 
                                          avg_score, record, training_time)

def log_error(error: Exception, context: str = ""):
    """Convenience function to log errors"""
    global _global_logger
    if _global_logger:
        _global_logger.log_error(error, context)

# Exception handler for unhandled exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger = get_logger("exception_handler")
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Set up global exception handler
sys.excepthook = handle_exception

if __name__ == "__main__":
    # Test logging system
    exp_logger = setup_logging(level="DEBUG")
    
    logger = get_logger("test")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test experiment logging
    log_experiment_start({"test": "config"})
    log_episode_progress(50, 100, 15, 12.5, 25, 120.5)
    log_experiment_end({"final_score": 12.5})
    
    print(f"Test logs created in {exp_logger.log_dir}")