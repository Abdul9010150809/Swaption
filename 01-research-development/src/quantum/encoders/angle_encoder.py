#!/usr/bin/env python3
"""
Angle Encoding for Quantum Data Encoding
Encodes classical data as rotation angles in quantum circuits
"""

import numpy as np
from typing import Optional, Union
from qiskit import QuantumCircuit
import logging

logger = logging.getLogger(__name__)

class AngleEncoder:
    """
    Angle encoding strategy for quantum data encoding.

    This encoder maps classical data features to rotation angles on quantum gates,
    providing a direct and efficient way to encode continuous data into quantum states.
    """

    def __init__(self, n_qubits: int, scaling: str = 'minmax', rotation_gate: str = 'ry'):
        """
        Initialize angle encoder.

        Args:
            n_qubits: Number of qubits to use for encoding
            scaling: Scaling method ('minmax', 'standard', 'none')
            rotation_gate: Quantum gate to use for rotations ('ry', 'rz', 'rx')
        """
        self.n_qubits = n_qubits
        self.scaling = scaling
        self.rotation_gate = rotation_gate.lower()
        self.feature_min = None
        self.feature_max = None
        self.feature_mean = None
        self.feature_std = None

        if self.rotation_gate not in ['rx', 'ry', 'rz']:
            raise ValueError(f"Unsupported rotation gate: {rotation_gate}")

    def fit(self, X: np.ndarray) -> 'AngleEncoder':
        """
        Fit the encoder to the data for scaling.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self: Fitted encoder
        """
        if self.scaling == 'minmax':
            self.feature_min = np.min(X, axis=0)
            self.feature_max = np.max(X, axis=0)
        elif self.scaling == 'standard':
            self.feature_mean = np.mean(X, axis=0)
            self.feature_std = np.std(X, axis=0)

        return self

    def transform(self, X: np.ndarray) -> QuantumCircuit:
        """
        Transform classical data to quantum circuit.

        Args:
            X: Input data of shape (n_samples, n_features) or (n_features,)

        Returns:
            QuantumCircuit: Encoded quantum circuit
        """
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Scale the data
        X_scaled = self._scale_data(X)

        # Create quantum circuit
        circuit = QuantumCircuit(self.n_qubits)

        # Encode each feature as a rotation
        for i, feature in enumerate(X_scaled[0]):  # Use first sample for single encoding
            if i < self.n_qubits:
                angle = feature * np.pi  # Scale to [0, π] range

                if self.rotation_gate == 'rx':
                    circuit.rx(angle, i)
                elif self.rotation_gate == 'ry':
                    circuit.ry(angle, i)
                elif self.rotation_gate == 'rz':
                    circuit.rz(angle, i)

        return circuit

    def fit_transform(self, X: np.ndarray) -> QuantumCircuit:
        """
        Fit the encoder and transform data in one step.

        Args:
            X: Input data

        Returns:
            QuantumCircuit: Encoded quantum circuit
        """
        return self.fit(X).transform(X)

    def _scale_data(self, X: np.ndarray) -> np.ndarray:
        """Scale data according to the specified scaling method."""
        if self.scaling == 'minmax':
            if self.feature_min is None or self.feature_max is None:
                # Use current data statistics if not fitted
                feature_min = np.min(X, axis=0)
                feature_max = np.max(X, axis=0)
            else:
                feature_min = self.feature_min
                feature_max = self.feature_max

            # Avoid division by zero
            denominator = feature_max - feature_min
            denominator = np.where(denominator == 0, 1, denominator)

            X_scaled = (X - feature_min) / denominator

        elif self.scaling == 'standard':
            if self.feature_mean is None or self.feature_std is None:
                # Use current data statistics if not fitted
                feature_mean = np.mean(X, axis=0)
                feature_std = np.std(X, axis=0)
            else:
                feature_mean = self.feature_mean
                feature_std = self.feature_std

            # Avoid division by zero
            feature_std = np.where(feature_std == 0, 1, feature_std)

            X_scaled = (X - feature_mean) / feature_std

        else:  # 'none'
            X_scaled = X

        return X_scaled

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance based on encoding (if applicable).

        Returns:
            None: Angle encoding doesn't provide feature importance
        """
        return None

    def __repr__(self) -> str:
        return (f"AngleEncoder(n_qubits={self.n_qubits}, "
                f"scaling='{self.scaling}', "
                f"rotation_gate='{self.rotation_gate}')")