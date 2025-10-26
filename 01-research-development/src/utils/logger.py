"""
Logging utilities for the price matrix system.

This module provides centralized logging configuration and utilities
for consistent logging across all components of the system.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        # Create base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # Add common fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'model_name'):
            log_entry['model_name'] = record.model_name
        if hasattr(record, 'experiment_id'):
            log_entry['experiment_id'] = record.experiment_id

        return json.dumps(log_entry)


class PriceMatrixLogger:
    """
    Centralized logger for the price matrix system.
    """

    _instance: Optional['PriceMatrixLogger'] = None
    _loggers: Dict[str, logging.Logger] = {}

    def __new__(cls) -> 'PriceMatrixLogger':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_root_logger()

    def _setup_root_logger(self) -> None:
        """
        Set up the root logger with default configuration.
        """
        # Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Set default level
        root_logger.setLevel(logging.INFO)

        # Add console handler with simple formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    def get_logger(self, name: str, log_file: Optional[str] = None,
                  level: str = 'INFO', use_json: bool = False) -> logging.Logger:
        """
        Get or create a logger with specified configuration.

        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
            use_json: Whether to use JSON formatting

        Returns:
            Configured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # Remove any existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create formatter
        if use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Cache logger
        self._loggers[name] = logger

        return logger

    def configure_from_config(self, config: Dict[str, Any]) -> None:
        """
        Configure logging from configuration dictionary.

        Args:
            config: Configuration dictionary
        """
        # Update root logger level
        if 'log_level' in config:
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, config['log_level'].upper()))

        # Configure specific loggers
        if 'loggers' in config:
            for logger_name, logger_config in config['loggers'].items():
                log_file = logger_config.get('log_file')
                level = logger_config.get('level', 'INFO')
                use_json = logger_config.get('use_json', False)

                self.get_logger(logger_name, log_file, level, use_json)


class ExperimentLogger:
    """
    Logger for machine learning experiments with structured logging.
    """

    def __init__(self, experiment_name: str, run_name: Optional[str] = None,
                 log_dir: str = 'logs/experiments'):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            run_name: Name of the specific run
            log_dir: Directory to store experiment logs
        """
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger_name = f"experiment.{experiment_name}.{self.run_name}"
        self.logger = PriceMatrixLogger().get_logger(
            logger_name,
            log_file=self.log_dir / f"{self.run_name}.log",
            use_json=True
        )

        # Initialize experiment log
        self._log_experiment_start()

    def _log_experiment_start(self) -> None:
        """Log experiment start with metadata."""
        self.logger.info("Experiment started", extra={
            'extra_fields': {
                'experiment_name': self.experiment_name,
                'run_name': self.run_name,
                'event_type': 'experiment_start',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
        })

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            params: Hyperparameter dictionary
        """
        self.logger.info("Hyperparameters logged", extra={
            'extra_fields': {
                'hyperparameters': params,
                'event_type': 'hyperparameters'
            }
        })

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log training/validation metrics.

        Args:
            metrics: Metrics dictionary
            step: Training step (optional)
        """
        log_data = {
            'metrics': metrics,
            'event_type': 'metrics'
        }
        if step is not None:
            log_data['step'] = step

        self.logger.info("Metrics logged", extra={
            'extra_fields': log_data
        })

    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """
        Log model information.

        Args:
            model_info: Model information dictionary
        """
        self.logger.info("Model info logged", extra={
            'extra_fields': {
                'model_info': model_info,
                'event_type': 'model_info'
            }
        })

    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """
        Log error with context.

        Args:
            error: Exception object
            context: Additional context information
        """
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'event_type': 'error'
        }
        if context:
            error_data['context'] = context

        self.logger.error("Error occurred", extra={
            'extra_fields': error_data
        }, exc_info=True)

    def log_experiment_end(self, status: str = 'completed',
                          final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Log experiment end.

        Args:
            status: Experiment status ('completed', 'failed', 'interrupted')
            final_metrics: Final metrics (optional)
        """
        end_data = {
            'status': status,
            'event_type': 'experiment_end'
        }
        if final_metrics:
            end_data['final_metrics'] = final_metrics

        self.logger.info("Experiment ended", extra={
            'extra_fields': end_data
        })


class PerformanceLogger:
    """
    Logger for performance monitoring and profiling.
    """

    def __init__(self, name: str = 'performance'):
        """
        Initialize performance logger.

        Args:
            name: Logger name
        """
        self.logger = PriceMatrixLogger().get_logger(
            f"performance.{name}",
            log_file=f"logs/performance/{name}.log",
            use_json=True
        )

    def log_timing(self, operation: str, duration: float,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log operation timing.

        Args:
            operation: Operation name
            duration: Duration in seconds
            metadata: Additional metadata
        """
        log_data = {
            'operation': operation,
            'duration_seconds': duration,
            'event_type': 'timing'
        }
        if metadata:
            log_data.update(metadata)

        self.logger.info(f"Operation '{operation}' completed in {duration:.3f}s", extra={
            'extra_fields': log_data
        })

    def log_memory_usage(self, operation: str,
                        memory_mb: float,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log memory usage.

        Args:
            operation: Operation name
            memory_mb: Memory usage in MB
            metadata: Additional metadata
        """
        log_data = {
            'operation': operation,
            'memory_mb': memory_mb,
            'event_type': 'memory'
        }
        if metadata:
            log_data.update(metadata)

        self.logger.info(f"Operation '{operation}' used {memory_mb:.2f} MB", extra={
            'extra_fields': log_data
        })

    def log_resource_usage(self, operation: str,
                          cpu_percent: float,
                          memory_mb: float,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log comprehensive resource usage.

        Args:
            operation: Operation name
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            metadata: Additional metadata
        """
        log_data = {
            'operation': operation,
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'event_type': 'resource_usage'
        }
        if metadata:
            log_data.update(metadata)

        self.logger.info(f"Resource usage for '{operation}': CPU {cpu_percent:.1f}%, Memory {memory_mb:.2f} MB", extra={
            'extra_fields': log_data
        })


# Global logger instances
logger = PriceMatrixLogger().get_logger('price_matrix')
experiment_logger = None
performance_logger = PerformanceLogger()


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Convenience function to get a logger.

    Args:
        name: Logger name
        **kwargs: Additional arguments for get_logger

    Returns:
        Logger instance
    """
    return PriceMatrixLogger().get_logger(name, **kwargs)


def setup_experiment_logging(experiment_name: str,
                           run_name: Optional[str] = None) -> ExperimentLogger:
    """
    Set up experiment logging.

    Args:
        experiment_name: Name of the experiment
        run_name: Name of the run

    Returns:
        ExperimentLogger instance
    """
    global experiment_logger
    experiment_logger = ExperimentLogger(experiment_name, run_name)
    return experiment_logger


# Context manager for timing operations
class Timer:
    """
    Context manager for timing code blocks.
    """

    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        """
        Initialize timer.

        Args:
            operation: Operation name
            logger: Logger to use (optional)
        """
        self.operation = operation
        self.logger = logger or performance_logger.logger
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        performance_logger.log_timing(self.operation, duration)

        if exc_type is None:
            self.logger.debug(f"Operation '{self.operation}' completed in {duration:.3f}s")
        else:
            self.logger.error(f"Operation '{self.operation}' failed after {duration:.3f}s")


if __name__ == "__main__":
    # Example usage
    import time

    # Basic logging
    logger = get_logger('example')
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Experiment logging
    exp_logger = setup_experiment_logging('test_experiment')
    exp_logger.log_hyperparameters({'learning_rate': 0.001, 'batch_size': 32})
    exp_logger.log_metrics({'train_loss': 0.5, 'val_accuracy': 0.85}, step=100)

    # Performance timing
    with Timer('example_operation'):
        time.sleep(0.1)  # Simulate some work

    # Resource logging
    performance_logger.log_resource_usage('test_operation', 45.2, 128.5)

    print("Logging examples completed. Check logs directory for output files.")