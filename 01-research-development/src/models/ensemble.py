"""
Ensemble model implementation for financial pricing.

This module provides ensemble methods combining multiple pricing models
for improved accuracy and robustness in derivative pricing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import KFold
import logging

from .base_model import EnsemblePricingModel
from .random_forest import RandomForestPricingModel
from .neural_network import SwaptionPricingNN

logger = logging.getLogger(__name__)


class StackingEnsemble(EnsemblePricingModel):
    """
    Stacking ensemble for derivative pricing.

    Combines predictions from multiple base models using a meta-learner
    to achieve better performance than individual models.
    """

    def __init__(self,
                 base_models: List,
                 meta_model=None,
                 cv_folds: int = 5,
                 name: str = "StackingEnsemble"):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of base pricing models
            meta_model: Meta-learner model (if None, uses LinearRegression)
            cv_folds: Number of cross-validation folds for meta-features
            name: Ensemble name
        """
        super().__init__(base_models, name)

        if meta_model is None:
            from sklearn.linear_model import LinearRegression
            meta_model = LinearRegression()

        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.meta_features_ = None

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the stacking ensemble.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional training parameters
        """
        logger.info(f"Training stacking ensemble with {len(self.base_models)} base models")

        # Train base models
        for model in self.base_models:
            model.train(X, y, **kwargs)

        # Generate meta-features using cross-validation
        self.meta_features_ = self._generate_meta_features(X, y)

        # Train meta-model
        self.meta_model.fit(self.meta_features_, y)

        self.is_trained = True
        self.training_date = pd.Timestamp.now()

        logger.info("Stacking ensemble training completed")

    def _generate_meta_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Generate meta-features using cross-validation predictions.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            DataFrame with meta-features
        """
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        meta_features = np.zeros((len(X), len(self.base_models)))
        meta_features = pd.DataFrame(meta_features,
                                   columns=[f'model_{i}' for i in range(len(self.base_models))])

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]

            for i, model in enumerate(self.base_models):
                # Train model on fold training data
                model_copy = model.__class__(**model.hyperparameters if hasattr(model, 'hyperparameters') else {})
                model_copy.train(X_train_fold, y_train_fold)

                # Generate predictions for validation fold
                meta_features.iloc[val_idx, i] = model_copy.predict(X_val_fold)

        return meta_features

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions using stacking.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        # Generate base model predictions
        base_predictions = np.column_stack([model.predict(X) for model in self.base_models])

        # Use meta-model to combine predictions
        return self.meta_model.predict(base_predictions)


class WeightedEnsemble(EnsemblePricingModel):
    """
    Weighted ensemble with learnable weights.

    Learns optimal weights for combining base model predictions.
    """

    def __init__(self,
                 base_models: List,
                 weight_regularization: float = 0.0,
                 name: str = "WeightedEnsemble"):
        """
        Initialize weighted ensemble.

        Args:
            base_models: List of base pricing models
            weight_regularization: L2 regularization for weights
            name: Ensemble name
        """
        super().__init__(base_models, name)
        self.weight_regularization = weight_regularization
        self.weights = None

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the weighted ensemble by learning optimal weights.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional training parameters
        """
        logger.info(f"Training weighted ensemble with {len(self.base_models)} base models")

        # Train base models
        for model in self.base_models:
            model.train(X, y, **kwargs)

        # Learn optimal weights using linear regression with constraints
        from sklearn.linear_model import Ridge

        # Generate base model predictions on training data
        base_predictions = np.column_stack([model.predict(X) for model in self.base_models])

        # Fit weights using Ridge regression
        ridge = Ridge(alpha=self.weight_regularization, fit_intercept=False)
        ridge.fit(base_predictions, y)

        # Ensure weights are positive and sum to 1 (simplex constraint approximation)
        self.weights = np.maximum(ridge.coef_, 0)  # Non-negative weights
        self.weights = self.weights / np.sum(self.weights)  # Normalize to sum to 1

        self.is_trained = True
        self.training_date = pd.Timestamp.now()

        logger.info(f"Learned weights: {self.weights}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make weighted ensemble predictions.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        # Generate base model predictions
        base_predictions = np.column_stack([model.predict(X) for model in self.base_models])

        # Apply learned weights
        return np.dot(base_predictions, self.weights)


class BootstrapEnsemble(EnsemblePricingModel):
    """
    Bootstrap aggregating (bagging) ensemble.

    Trains multiple instances of the same model on bootstrap samples
    and averages predictions for improved stability.
    """

    def __init__(self,
                 base_model_class,
                 n_estimators: int = 10,
                 bootstrap_fraction: float = 0.8,
                 model_params: Dict[str, Any] = None,
                 name: str = "BootstrapEnsemble"):
        """
        Initialize bootstrap ensemble.

        Args:
            base_model_class: Class of the base model to ensemble
            n_estimators: Number of bootstrap models
            bootstrap_fraction: Fraction of data for each bootstrap sample
            model_params: Parameters for base model initialization
            name: Ensemble name
        """
        if model_params is None:
            model_params = {}

        # Create multiple instances of the base model
        base_models = [base_model_class(**model_params) for _ in range(n_estimators)]

        super().__init__(base_models, name)

        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.bootstrap_fraction = bootstrap_fraction
        self.model_params = model_params

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train bootstrap ensemble.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional training parameters
        """
        logger.info(f"Training bootstrap ensemble with {self.n_estimators} models")

        n_samples = len(X)
        sample_size = int(self.bootstrap_fraction * n_samples)

        np.random.seed(42)  # For reproducibility

        for i, model in enumerate(self.base_models):
            # Create bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=sample_size, replace=True)
            X_bootstrap = X.iloc[bootstrap_indices]
            y_bootstrap = y.iloc[bootstrap_indices]

            # Train model on bootstrap sample
            model.train(X_bootstrap, y_bootstrap, **kwargs)

        self.is_trained = True
        self.training_date = pd.Timestamp.now()

        logger.info("Bootstrap ensemble training completed")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make bootstrap ensemble predictions.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        # Average predictions from all bootstrap models
        predictions = np.array([model.predict(X) for model in self.base_models])
        return np.mean(predictions, axis=0)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates from bootstrap variance.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        predictions = np.array([model.predict(X) for model in self.base_models])
        mean_predictions = np.mean(predictions, axis=0)
        uncertainties = np.std(predictions, axis=0)

        return mean_predictions, uncertainties


class AdaptiveEnsemble(EnsemblePricingModel):
    """
    Adaptive ensemble that adjusts weights based on recent performance.

    Uses online learning to adapt model weights based on prediction accuracy.
    """

    def __init__(self,
                 base_models: List,
                 learning_rate: float = 0.01,
                 name: str = "AdaptiveEnsemble"):
        """
        Initialize adaptive ensemble.

        Args:
            base_models: List of base pricing models
            learning_rate: Learning rate for weight adaptation
            name: Ensemble name
        """
        super().__init__(base_models, name)
        self.learning_rate = learning_rate
        self.performance_weights = np.ones(len(base_models)) / len(base_models)  # Equal initial weights

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the adaptive ensemble.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional training parameters
        """
        logger.info(f"Training adaptive ensemble with {len(self.base_models)} base models")

        # Train base models
        for model in self.base_models:
            model.train(X, y, **kwargs)

        self.is_trained = True
        self.training_date = pd.Timestamp.now()

        logger.info("Adaptive ensemble training completed")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make adaptive ensemble predictions.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        # Generate base model predictions
        base_predictions = np.column_stack([model.predict(X) for model in self.base_models])

        # Apply current performance weights
        return np.dot(base_predictions, self.performance_weights)

    def update_weights(self, X: pd.DataFrame, y_true: pd.Series) -> None:
        """
        Update model weights based on recent prediction performance.

        Args:
            X: Feature matrix
            y_true: True target values
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before updating weights")

        # Get predictions from each model
        predictions = np.array([model.predict(X) for model in self.base_models])

        # Calculate prediction errors for each model
        errors = np.abs(predictions - y_true.values.reshape(1, -1))

        # Update weights based on relative performance
        mean_errors = np.mean(errors, axis=1)
        weights_update = 1.0 / (mean_errors + 1e-6)  # Inverse error weighting
        weights_update = weights_update / np.sum(weights_update)  # Normalize

        # Smooth weight updates
        self.performance_weights = (1 - self.learning_rate) * self.performance_weights + \
                                  self.learning_rate * weights_update

        logger.debug(f"Updated weights: {self.performance_weights}")


def create_financial_ensemble(model_types: List[str] = None,
                            n_models: int = 3) -> EnsemblePricingModel:
    """
    Factory function to create a financial pricing ensemble.

    Args:
        model_types: List of model types to include
        n_models: Number of models if model_types not specified

    Returns:
        Configured ensemble model
    """
    if model_types is None:
        model_types = ['random_forest', 'neural_network', 'gradient_boosting']

    base_models = []

    for model_type in model_types:
        if model_type == 'random_forest':
            base_models.append(RandomForestPricingModel(n_estimators=50))
        elif model_type == 'neural_network':
            # Placeholder - would need input_dim
            pass  # Skip for now, requires input_dim
        elif model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            from .base_model import ScikitLearnPricingModel
            gb_model = ScikitLearnPricingModel(GradientBoostingRegressor(n_estimators=50))
            base_models.append(gb_model)

    if len(base_models) < 2:
        raise ValueError("Need at least 2 base models for ensemble")

    # Create weighted ensemble by default
    return WeightedEnsemble(base_models, name="FinancialEnsemble")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression

    # Create sample financial-like data
    X, y = make_regression(n_samples=500, n_features=8, noise=0.1, random_state=42)
    feature_names = [f'feature_{i}' for i in range(8)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='price')

    # Create ensemble with Random Forest models
    rf_models = [RandomForestPricingModel(n_estimators=20 + i*10) for i in range(3)]
    ensemble = WeightedEnsemble(rf_models, name="RF_Ensemble")

    # Train ensemble
    ensemble.train(X_df, y_series)

    # Make predictions
    predictions = ensemble.predict(X_df)

    print(f"Ensemble trained with {len(rf_models)} Random Forest models")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Learned weights: {ensemble.weights}")