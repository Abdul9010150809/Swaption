"""
Base model classes for financial pricing models.

This module provides abstract base classes and common functionality
for all pricing models in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


class PricingModel(ABC):
    """
    Abstract base class for all pricing models.

    Defines the interface that all pricing models must implement.
    """

    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.is_trained = False
        self.training_date = None
        self.model_version = "1.0.0"

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the pricing model.

        Args:
            X: Feature matrix
            y: Target values (prices)
            **kwargs: Additional training parameters
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make price predictions.

        Args:
            X: Feature matrix

        Returns:
            Array of predicted prices
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available
        """
        pass

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: File path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model': self,
            'name': self.name,
            'training_date': self.training_date,
            'version': self.model_version,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str) -> 'PricingModel':
        """
        Load a saved model from disk.

        Args:
            path: File path to load the model from

        Returns:
            Loaded model instance
        """
        model_data = joblib.load(path)
        model = model_data['model']
        logger.info(f"Model loaded from {path}")
        return model

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X: Feature matrix
            y_true: True target values

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X)

        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

        return metrics

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model metadata
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'training_date': self.training_date,
            'version': self.model_version
        }


class ScikitLearnPricingModel(PricingModel, BaseEstimator, RegressorMixin):
    """
    Base class for scikit-learn based pricing models.

    Provides common functionality for models that use scikit-learn estimators.
    """

    def __init__(self, estimator: BaseEstimator, name: str = "ScikitLearnModel"):
        super().__init__(name)
        self.estimator = estimator
        self.feature_names_ = None

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the scikit-learn model.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional parameters passed to fit method
        """
        self.estimator.fit(X, y, **kwargs)
        self.feature_names_ = list(X.columns)
        self.is_trained = True
        self.training_date = datetime.now()
        logger.info(f"Model {self.name} trained successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.estimator.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the estimator.

        Returns:
            Dictionary of feature importances, or None if not available
        """
        if not hasattr(self.estimator, 'feature_importances_'):
            return None

        if self.feature_names_ is None:
            return None

        importances = self.estimator.feature_importances_
        return dict(zip(self.feature_names_, importances))


class NeuralNetworkPricingModel(PricingModel):
    """
    Base class for neural network based pricing models.

    Provides common functionality for TensorFlow/Keras models.
    """

    def __init__(self, model=None, name: str = "NeuralNetworkModel"):
        super().__init__(name)
        self.model = model
        self.history = None

    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32,
              **kwargs) -> None:
        """
        Train the neural network model.

        Args:
            X: Feature matrix
            y: Target values
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            **kwargs: Additional training parameters
        """
        if self.model is None:
            raise ValueError("Model architecture must be defined before training")

        self.history = self.model.fit(
            X.values, y.values,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )

        self.is_trained = True
        self.training_date = datetime.now()
        logger.info(f"Neural network {self.name} trained successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the neural network.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X.values).flatten()

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Feature importance for neural networks is not straightforward.
        This returns None by default.

        Returns:
            None (feature importance not implemented for NN models)
        """
        return None


class EnsemblePricingModel(PricingModel):
    """
    Base class for ensemble pricing models.

    Combines multiple base models for improved performance.
    """

    def __init__(self, base_models: list, name: str = "EnsembleModel"):
        super().__init__(name)
        self.base_models = base_models
        self.weights = None

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train all base models in the ensemble.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional training parameters
        """
        for model in self.base_models:
            model.train(X, y, **kwargs)

        self.is_trained = True
        self.training_date = datetime.now()
        logger.info(f"Ensemble {self.name} trained successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions by averaging base model predictions.

        Args:
            X: Feature matrix

        Returns:
            Array of ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        predictions = np.array([model.predict(X) for model in self.base_models])

        if self.weights is not None:
            return np.average(predictions, axis=0, weights=self.weights)
        else:
            return np.mean(predictions, axis=0)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get average feature importance across base models.

        Returns:
            Dictionary of average feature importances
        """
        importances = []
        for model in self.base_models:
            imp = model.get_feature_importance()
            if imp is not None:
                importances.append(imp)

        if not importances:
            return None

        # Average importances across models
        avg_importance = {}
        for feature in importances[0].keys():
            avg_importance[feature] = np.mean([imp.get(feature, 0) for imp in importances])

        return avg_importance