"""
Configuration management for the price matrix system.

This module provides centralized configuration management using
dataclasses and YAML configuration files for all system components.
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""

    # Neural Network parameters
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    activation: str = 'relu'
    output_activation: str = 'linear'
    l2_reg: float = 1e-4

    # Random Forest parameters
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = 'sqrt'

    # Ensemble parameters
    ensemble_method: str = 'weighted'
    n_base_models: int = 3

    # Training parameters
    validation_split: float = 0.2
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5

    # Model saving
    save_best_only: bool = True
    save_weights_only: bool = False
    model_checkpoint_dir: str = 'models/checkpoints'


@dataclass
class DataConfig:
    """Configuration for data generation and processing."""

    # Data generation parameters
    n_samples: int = 100000
    random_seed: int = 42

    # Yield curve parameters
    yield_tenors: List[float] = field(default_factory=lambda: [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    base_rate: float = 0.03
    rate_volatility: float = 0.005

    # Volatility surface parameters
    volatility_expiries: List[float] = field(default_factory=lambda: [0.25, 0.5, 1, 2, 5])
    volatility_strikes: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])
    vol_base_level: float = 0.2
    vol_term_structure: bool = True
    vol_skew: bool = True

    # Option parameters
    spot_range: List[float] = field(default_factory=lambda: [90.0, 110.0])
    strike_range: List[float] = field(default_factory=lambda: [85.0, 115.0])
    time_range: List[float] = field(default_factory=lambda: [0.1, 2.0])
    rate_range: List[float] = field(default_factory=lambda: [0.01, 0.08])
    vol_range: List[float] = field(default_factory=lambda: [0.1, 0.5])

    # Swaption parameters
    swap_rates_range: List[float] = field(default_factory=lambda: [0.02, 0.08])
    volatilities_range: List[float] = field(default_factory=lambda: [0.1, 0.8])
    swap_tenors: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    option_tenors: List[int] = field(default_factory=lambda: [1, 2, 3, 5])

    # Data processing
    test_size: float = 0.2
    val_size: float = 0.1
    scaling_method: str = 'standard'
    handle_outliers: bool = True
    outlier_method: str = 'iqr'
    outlier_threshold: float = 1.5


@dataclass
class PricingConfig:
    """Configuration for pricing engines."""

    # Monte Carlo parameters
    n_simulations: int = 10000
    n_steps: int = 252
    time_horizon: float = 1.0

    # Black-Scholes parameters
    dividend_yield: float = 0.0

    # SABR parameters
    sabr_alpha: float = 0.2
    sabr_beta: float = 0.7
    sabr_rho: float = -0.3
    sabr_nu: float = 0.3

    # Numerical methods
    tolerance: float = 1e-6
    max_iterations: int = 100

    # Variance reduction
    use_antithetic: bool = False
    use_control_variates: bool = False
    use_importance_sampling: bool = False


@dataclass
class RiskConfig:
    """Configuration for risk metrics calculation."""

    # VaR parameters
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    time_horizons: List[int] = field(default_factory=lambda: [1, 5, 10])
    var_methods: List[str] = field(default_factory=lambda: ['historical', 'parametric', 'monte_carlo'])

    # Risk factor shocks
    spot_shock: float = 0.01
    vol_shock: float = 0.01
    rate_shock: float = 0.0001
    time_decay: int = 1

    # Stress testing
    stress_scenarios: Dict[str, float] = field(default_factory=lambda: {
        'mild_stress': -0.05,
        'moderate_stress': -0.10,
        'severe_stress': -0.20,
        'black_swan': -0.30
    })

    # Backtesting
    backtest_window: int = 252
    var_model_update_freq: int = 21


@dataclass
class ExperimentConfig:
    """Configuration for experiments and model training."""

    # Experiment tracking
    experiment_name: str = 'default_experiment'
    run_name: Optional[str] = None
    log_to_mlflow: bool = True
    log_to_wandb: bool = False

    # Hyperparameter tuning
    tune_hyperparameters: bool = False
    tuning_method: str = 'grid'  # 'grid' or 'random'
    n_trials: int = 50
    cv_folds: int = 5

    # Model evaluation
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'rmse', 'mae', 'r2', 'mape'
    ])

    # Cross-validation
    use_cv: bool = True
    cv_method: str = 'kfold'
    n_cv_splits: int = 5

    # Feature selection
    feature_selection: bool = False
    selection_method: str = 'recursive'
    max_features: Optional[int] = None


@dataclass
class QuantumConfig:
    """Configuration for quantum computing services."""

    # Quantum service settings
    enabled: bool = True
    api_key: str = field(default_factory=lambda: os.getenv('QUANTUM_API_KEY', 'wPQOh--o2TjczKSr8xYZXZPudXBm4Ia6m__gdphs-5IR'))
    provider: str = 'ibm'  # 'ibm', 'aws', 'google', 'rigetti'
    backend: str = 'simulator'  # 'simulator' or specific quantum device
    
    # Quantum execution parameters
    shots: int = 1024
    max_circuits: int = 100
    optimization_level: int = 1
    
    # IBM Quantum specific settings
    ibm_hub: str = 'ibm-q'
    ibm_group: str = 'open'
    ibm_project: str = 'main'
    
    # AWS Braket specific settings
    aws_region: str = 'us-east-1'
    aws_s3_bucket: str = 'amazon-braket-quantum-results'
    
    # Google Quantum AI specific settings
    google_project_id: str = ''
    google_processor_id: str = 'rainbow'
    
    # Quantum algorithm settings
    use_quantum_pricing: bool = False
    quantum_monte_carlo: bool = False
    quantum_optimization: bool = False
    
    # Error mitigation
    error_mitigation: bool = True
    readout_error_mitigation: bool = True
    
    # Timeout settings
    job_timeout: int = 300  # seconds
    max_retries: int = 3


@dataclass
class SystemConfig:
    """Overall system configuration."""

    # Paths
    project_root: str = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: str = 'data'
    models_dir: str = 'models'
    logs_dir: str = 'logs'
    results_dir: str = 'results'

    # Logging
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_to_console: bool = True

    # Parallel processing
    n_jobs: int = -1
    use_gpu: bool = True

    # Memory management
    max_memory_gb: float = 8.0
    chunk_size: int = 10000

    # API settings
    api_host: str = '0.0.0.0'
    api_port: int = 8000
    api_workers: int = 4

    # Database
    db_type: str = 'sqlite'
    db_path: str = 'price_matrix.db'

    # External services
    use_mlflow: bool = True
    mlflow_tracking_uri: str = 'http://localhost:5000'
    use_wandb: bool = False
    wandb_project: str = 'price-matrix'


@dataclass
class Config:
    """Master configuration class containing all sub-configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Recursively create config objects
        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Recursively create config from dictionary."""
        config = cls()

        for section_name, section_dict in config_dict.items():
            if hasattr(config, section_name):
                section_config = getattr(config, section_name)
                for key, value in section_dict.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

        return config

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self)

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.

        Args:
            updates: Dictionary with configuration updates
        """
        for section_name, section_updates in updates.items():
            if hasattr(self, section_name):
                section_config = getattr(self, section_name)
                for key, value in section_updates.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

    def get_nested_value(self, key_path: str) -> Any:
        """
        Get nested configuration value using dot notation.

        Args:
            key_path: Dot-separated path to configuration value (e.g., 'model.learning_rate')

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self

        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            else:
                raise KeyError(f"Configuration key not found: {key_path}")

        return value

    def set_nested_value(self, key_path: str, value: Any) -> None:
        """
        Set nested configuration value using dot notation.

        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        obj = self

        for key in keys[:-1]:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            else:
                raise KeyError(f"Configuration path not found: {'.'.join(keys[:-1])}")

        if hasattr(obj, keys[-1]):
            setattr(obj, keys[-1], value)
        else:
            raise KeyError(f"Configuration key not found: {keys[-1]}")


# Global configuration instance
config = Config()


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file or return default.

    Args:
        config_path: Path to configuration file (optional)

    Returns:
        Config instance
    """
    if config_path is not None:
        return Config.from_yaml(config_path)
    else:
        return config


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save configuration to file.

    Args:
        config: Config instance to save
        config_path: Path to save configuration file
    """
    config.to_yaml(config_path)


def create_default_config_file(config_path: Union[str, Path] = 'config.yaml') -> None:
    """
    Create default configuration file.

    Args:
        config_path: Path to create configuration file
    """
    config = Config()
    config.to_yaml(config_path)
    print(f"Default configuration file created at: {config_path}")


if __name__ == "__main__":
    # Create default configuration file
    create_default_config_file()

    # Example usage
    config = load_config()

    # Access configuration values
    print(f"Learning rate: {config.model.learning_rate}")
    print(f"Batch size: {config.model.batch_size}")
    print(f"Number of samples: {config.data.n_samples}")

    # Update configuration
    config.set_nested_value('model.learning_rate', 0.0005)
    print(f"Updated learning rate: {config.model.learning_rate}")

    # Save updated configuration
    save_config(config, 'updated_config.yaml')
