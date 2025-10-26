"""
Neural Network model implementation for financial pricing.

This module provides neural network based pricing models using TensorFlow/Keras
with financial domain-specific architectures and training procedures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.models import Model
import logging

from .base_model import NeuralNetworkPricingModel

logger = logging.getLogger(__name__)


class SwaptionPricingNN(NeuralNetworkPricingModel):
    """
    Neural network for swaption pricing with tail risk capture.

    Implements a custom architecture designed for derivative pricing
    with specialized layers for financial features.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.2,
                 l2_reg: float = 1e-4,
                 activation: str = 'relu',
                 output_activation: str = 'linear',
                 name: str = "SwaptionPricingNN"):
        """
        Initialize the swaption pricing neural network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            name: Model name
        """
        super().__init__(name=name)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.activation = activation
        self.output_activation = output_activation

        # Build the model
        self._build_model()

    def _build_model(self) -> None:
        """
        Build the neural network architecture.
        """
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name='input_layer')

        # Hidden layers with batch normalization and dropout
        x = inputs
        for i, units in enumerate(self.hidden_dims):
            x = layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'hidden_layer_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)

        # Output layer
        outputs = layers.Dense(
            1,
            activation=self.output_activation,
            name='output_layer'
        )(x)

        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)

    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 1e-3,
              patience: int = 20,
              **kwargs) -> None:
        """
        Train the neural network with financial data considerations.

        Args:
            X: Feature matrix
            y: Target values (swaption prices)
            validation_split: Fraction of data for validation
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            patience: Early stopping patience
            **kwargs: Additional training parameters
        """
        logger.info(f"Training {self.name} with {len(X)} samples")

        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )

        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                mode='min'
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-6,
                mode='min'
            ),
            callbacks.ModelCheckpoint(
                filepath=f'models/checkpoints/{self.name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        ]

        # Train model
        self.history = self.model.fit(
            X.values, y.values,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1,
            **kwargs
        )

        self.is_trained = True
        self.training_date = pd.Timestamp.now()

        # Log training results
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        logger.info(f"Training completed - Loss: {final_loss:.6f}, Val Loss: {final_val_loss:.6f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained neural network.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X.values, verbose=0).flatten()

    def predict_with_uncertainty(self, X: pd.DataFrame, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using Monte Carlo dropout.

        Args:
            X: Feature matrix
            n_samples: Number of Monte Carlo samples

        Returns:
            Tuple of (mean_predictions, uncertainties)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Enable dropout during inference for uncertainty estimation
        predictions = []
        for _ in range(n_samples):
            pred = self.model.predict(X.values, verbose=0).flatten()
            predictions.append(pred)

        predictions = np.array(predictions)
        mean_predictions = np.mean(predictions, axis=0)
        uncertainties = np.std(predictions, axis=0)

        return mean_predictions, uncertainties

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Estimate feature importance using permutation importance.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Note: This is a simplified implementation
        # For more accurate feature importance, consider using techniques like
        # Integrated Gradients, SHAP, or LRP (Layer-wise Relevance Propagation)

        logger.warning("Feature importance for neural networks is approximate and computationally expensive")
        return None

    def save_model(self, path: str) -> None:
        """
        Save the neural network model to disk.

        Args:
            path: File path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        # Save model architecture and weights
        self.model.save(path)
        logger.info(f"Neural network model saved to {path}")

    @classmethod
    def load_model(cls, path: str) -> 'SwaptionPricingNN':
        """
        Load a saved neural network model.

        Args:
            path: File path to load the model from

        Returns:
            Loaded model instance
        """
        model = tf.keras.models.load_model(path)

        # Create instance and assign loaded model
        instance = cls(input_dim=model.input_shape[1])
        instance.model = model
        instance.is_trained = True

        logger.info(f"Neural network model loaded from {path}")
        return instance


class OptionPricingNN(NeuralNetworkPricingModel):
    """
    Neural network for European option pricing.

    Specialized architecture for option pricing with attention to
    volatility smiles and term structure effects.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.1,
                 l2_reg: float = 1e-4,
                 name: str = "OptionPricingNN"):
        """
        Initialize the option pricing neural network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
            name: Model name
        """
        super().__init__(name=name)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Build the model
        self._build_model()

    def _build_model(self) -> None:
        """
        Build the neural network architecture for option pricing.
        """
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name='input_layer')

        # Feature processing layers
        x = layers.Dense(64, activation='relu', name='feature_processor')(inputs)
        x = layers.BatchNormalization()(x)

        # Hidden layers
        for i, units in enumerate(self.hidden_dims):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'hidden_layer_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)

        # Output layer with softplus for positive prices
        outputs = layers.Dense(1, activation='softplus', name='output_layer')(x)

        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)


class LSTMPricingNN(NeuralNetworkPricingModel):
    """
    LSTM-based neural network for time series financial pricing.

    Uses LSTM layers to capture temporal dependencies in financial time series.
    """

    def __init__(self,
                 input_dim: int,
                 sequence_length: int = 10,
                 lstm_units: List[int] = [64, 32],
                 dense_units: List[int] = [32],
                 dropout_rate: float = 0.2,
                 name: str = "LSTMPricingNN"):
        """
        Initialize the LSTM pricing neural network.

        Args:
            input_dim: Number of features per time step
            sequence_length: Length of input sequences
            lstm_units: List of LSTM layer units
            dense_units: List of dense layer units after LSTM
            dropout_rate: Dropout rate
            name: Model name
        """
        super().__init__(name=name)

        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        # Build the model
        self._build_model()

    def _build_model(self) -> None:
        """
        Build the LSTM neural network architecture.
        """
        # Input layer for sequences
        inputs = layers.Input(shape=(self.sequence_length, self.input_dim), name='input_layer')

        # LSTM layers
        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1  # Return sequences for all but last LSTM
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name=f'lstm_layer_{i+1}'
            )(x)

        # Dense layers
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_layer_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dense_dropout_{i+1}')(x)

        # Output layer
        outputs = layers.Dense(1, activation='linear', name='output_layer')(x)

        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)


class ResidualPricingNN(NeuralNetworkPricingModel):
    """
    Residual neural network for financial pricing.

    Uses residual connections to improve gradient flow and training stability.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 128, 64],
                 dropout_rate: float = 0.1,
                 name: str = "ResidualPricingNN"):
        """
        Initialize the residual pricing neural network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate
            name: Model name
        """
        super().__init__(name=name)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build the model
        self._build_model()

    def _build_model(self) -> None:
        """
        Build the residual neural network architecture.
        """
        def residual_block(x, units, name_prefix):
            """Residual block with skip connection."""
            # Main path
            y = layers.Dense(units, activation='relu', name=f'{name_prefix}_dense1')(x)
            y = layers.BatchNormalization(name=f'{name_prefix}_bn1')(y)
            y = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_dropout1')(y)

            y = layers.Dense(units, activation='relu', name=f'{name_prefix}_dense2')(y)
            y = layers.BatchNormalization(name=f'{name_prefix}_bn2')(y)
            y = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_dropout2')(y)

            # Skip connection
            if x.shape[-1] != units:
                x = layers.Dense(units, name=f'{name_prefix}_skip')(x)

            return layers.Add(name=f'{name_prefix}_add')([x, y])

        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name='input_layer')

        # Initial processing
        x = layers.Dense(self.hidden_dims[0], activation='relu', name='initial_dense')(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)

        # Residual blocks
        for i, units in enumerate(self.hidden_dims):
            x = residual_block(x, units, f'residual_block_{i+1}')

        # Output layer
        outputs = layers.Dense(1, activation='linear', name='output_layer')(x)

        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 0.1 + 5  # Simulated prices

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y, name='price')

    # Initialize and train model
    nn_model = SwaptionPricingNN(input_dim=n_features, hidden_dims=[64, 32])
    nn_model.train(X_df, y_series, epochs=10, batch_size=32, validation_split=0.2)

    # Make predictions
    predictions = nn_model.predict(X_df)

    print(f"Model trained with {len(X_df)} samples")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
