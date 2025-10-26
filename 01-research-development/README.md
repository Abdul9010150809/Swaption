# Research & Development

This directory contains the research and development components for the Price Matrix system, focusing on swaption pricing using machine learning models.

## Structure

### Data (`data/`)
- **raw/**: Raw market data (yield curves, volatility surfaces, swap rates)
- **processed/**: Cleaned and preprocessed datasets for training and testing

### Source Code (`src/`)
- **data/**: Data generation, preprocessing, and feature engineering
- **models/**: Machine learning models (Random Forest, Neural Networks, Ensemble)
- **pricing/**: Pricing algorithms (Analytic, Monte Carlo)
- **utils/**: Utilities for configuration, logging, and visualization

### Notebooks (`notebooks/`)
- Data exploration and analysis
- Feature engineering experiments
- Model training and evaluation
- Results visualization

### Experiments (`experiments/`)
- Configuration files for different experiments
- Model checkpoints and saved models
- Experiment results and metrics

### Tests (`tests/`)
- Unit tests for all components
- Integration tests for pricing workflows

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run data generation:
```bash
python -m src.data.data_generator
```

3. Start with the notebooks:
- `01_data_exploration.ipynb` - Explore the generated data
- `02_feature_engineering.ipynb` - Engineer features for modeling
- `03_model_experiments.ipynb` - Train and compare models
- `04_model_evaluation.ipynb` - Evaluate model performance

## Key Features

- **Synthetic Data Generation**: Generate realistic swaption market data
- **Feature Engineering**: Create relevant features for pricing models
- **Multiple Models**: Random Forest, Neural Networks, and Ensemble methods
- **Pricing Methods**: Both analytic and Monte Carlo approaches
- **Risk Metrics**: Calculate Greeks and other risk measures
- **Experiment Tracking**: MLflow integration for experiment management

## Configuration

Use the YAML files in `experiments/experiment_configs/` to configure:
- Model hyperparameters
- Data generation parameters
- Training settings
- Evaluation metrics