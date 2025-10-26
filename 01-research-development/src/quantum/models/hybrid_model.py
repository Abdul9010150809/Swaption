#!/usr/bin/env python3
"""
Hybrid Quantum-Classical Model Implementation
Combines quantum and classical machine learning for enhanced performance
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

from .vqc_regressor import VariationalQuantumRegressor
from .quantum_kernel import QuantumKernelRegressor

logger = logging.getLogger(__name__)

class HybridQuantumClassicalModel:
    """
    Hybrid quantum-classical model that combines quantum and classical approaches.

    This model uses quantum methods for feature extraction/transformation
    and classical methods for final prediction, providing a balance between
    quantum advantage and classical reliability.
    """

    def __init__(self, quantum_model_type: str = 'vqc',
                 classical_model_type: str = 'random_forest',
                 n_qubits: int = 4, n_layers: int = 2,
                 use_quantum_features: bool = True):
        """
        Initialize the hybrid model.

        Args:
            quantum_model_type: Type of quantum model ('vqc', 'kernel')
            classical_model_type: Type of classical model ('random_forest', 'linear')
            n_qubits: Number of qubits for quantum models
            n_layers: Number of layers for quantum models
            use_quantum_features: Whether to use quantum feature extraction
        """
        self.quantum_model_type = quantum_model_type
        self.classical_model_type = classical_model_type
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_quantum_features = use_quantum_features

        # Initialize components
        self.quantum_model = self._create_quantum_model()
        self.classical_model = self._create_classical_model()
        self.scaler = StandardScaler()

        # Training state
        self.is_trained = False
        self.quantum_trained = False
        self.classical_trained = False

    def _create_quantum_model(self):
        """Create the quantum model component."""
        if self.quantum_model_type == 'vqc':
            return VariationalQuantumRegressor(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers
            )
        elif self.quantum_model_type == 'kernel':
            return QuantumKernelRegressor(
                n_qubits=self.n_qubits,
                n_features=6  # Assume 6 financial features
            )
        else:
            raise ValueError(f"Unknown quantum model type: {self.quantum_model_type}")

    def _create_classical_model(self):
        """Create the classical model component."""
        if self.classical_model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.classical_model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        else:
            raise ValueError(f"Unknown classical model type: {self.classical_model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray,
            quantum_weight: float = 0.5) -> bool:
        """
        Train the hybrid model.

        Args:
            X: Training features
            y: Training targets
            quantum_weight: Weight for quantum predictions (0-1)

        Returns:
            True if training successful
        """
        try:
            self.quantum_weight = quantum_weight
            self.classical_weight = 1 - quantum_weight

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train quantum model
            self.quantum_trained = self.quantum_model.fit(X_scaled, y)

            if self.use_quantum_features:
                # Use quantum model predictions as features for classical model
                quantum_predictions = self.quantum_model.predict(X_scaled)

                # Combine original features with quantum predictions
                X_hybrid = np.column_stack([X_scaled, quantum_predictions.ravel()])
            else:
                # Use original features for classical model
                X_hybrid = X_scaled

            # Train classical model
            self.classical_model.fit(X_hybrid, y.ravel())
            self.classical_trained = True

            self.is_trained = self.quantum_trained and self.classical_trained

            if self.is_trained:
                logger.info("Hybrid quantum-classical model trained successfully")

            return self.is_trained

        except Exception as e:
            logger.error(f"Hybrid model training failed: {e}")
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the hybrid model.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get quantum predictions
        quantum_predictions = self.quantum_model.predict(X_scaled)

        if self.use_quantum_features:
            # Use quantum predictions as additional features
            X_hybrid = np.column_stack([X_scaled, quantum_predictions.ravel()])
        else:
            X_hybrid = X_scaled

        # Get classical predictions
        classical_predictions = self.classical_model.predict(X_hybrid)

        # Combine predictions using weighted average
        hybrid_predictions = (self.quantum_weight * quantum_predictions.ravel() +
                            self.classical_weight * classical_predictions)

        return hybrid_predictions.reshape(-1, 1)

    def get_model_weights(self) -> Dict[str, float]:
        """
        Get the weights used for combining quantum and classical predictions.

        Returns:
            Dictionary with model weights
        """
        return {
            'quantum_weight': self.quantum_weight,
            'classical_weight': self.classical_weight
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model details
        """
        return {
            'model_type': 'Hybrid',
            'quantum_model': str(self.quantum_model),
            'classical_model': str(self.classical_model),
            'quantum_weight': self.quantum_weight,
            'classical_weight': self.classical_weight,
            'use_quantum_features': self.use_quantum_features,
            'is_trained': self.is_trained,
            'quantum_trained': self.quantum_trained,
            'classical_trained': self.classical_trained
        }

    def __repr__(self) -> str:
        return (f"HybridQuantumClassicalModel("
                f"quantum='{self.quantum_model_type}', "
                f"classical='{self.classical_model_type}', "
                f"trained={self.is_trained})")