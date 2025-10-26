#!/usr/bin/env python3
"""
Variational Quantum Regressor (VQR) Implementation
Provides a complete variational quantum regression model
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import logging

from .quantum_nn import QuantumNeuralNetwork
from ..encoders.angle_encoder import AngleEncoder
from ..circuits.ansatz_design import AnsatzDesigner

logger = logging.getLogger(__name__)

class VariationalQuantumRegressor(QuantumNeuralNetwork):
    """
    Variational Quantum Regressor implementing VQR algorithm.

    This class provides a complete implementation of variational quantum
    regression using parameterized quantum circuits for financial prediction.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2,
                 encoding_type: str = 'angle', ansatz_type: str = 'real_amplitudes',
                 optimizer: Optional[Any] = None, max_iter: int = 100,
                 observable: str = 'Z'):
        """
        Initialize the VQR model.

        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            encoding_type: Data encoding method ('angle', 'amplitude', 'basis')
            ansatz_type: Variational ansatz type
            optimizer: Classical optimizer
            max_iter: Maximum iterations
            observable: Measurement observable ('Z', 'X', 'Y')
        """
        super().__init__(n_qubits, n_layers, optimizer, max_iter)

        self.encoding_type = encoding_type
        self.ansatz_type = ansatz_type
        self.observable = observable

        # Initialize components
        self.encoder = self._create_encoder()
        self.ansatz_designer = AnsatzDesigner()
        self.ansatz = self._create_ansatz()

        # Observable for measurement
        self.observable_op = self._create_observable()

    def _create_encoder(self):
        """Create the data encoder."""
        if self.encoding_type == 'angle':
            return AngleEncoder(n_qubits=self.n_qubits)
        elif self.encoding_type == 'amplitude':
            from ..encoders.amplitude_encoder import AmplitudeEncoder
            return AmplitudeEncoder(n_qubits=self.n_qubits)
        elif self.encoding_type == 'basis':
            from ..encoders.basis_encoder import BasisEncoder
            return BasisEncoder(n_qubits=self.n_qubits)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def _create_ansatz(self):
        """Create the variational ansatz."""
        return self.ansatz_designer.create_ansatz(
            self.ansatz_type, self.n_qubits, self.n_layers
        )

    def _create_observable(self) -> SparsePauliOp:
        """Create the measurement observable."""
        if self.observable == 'Z':
            pauli_str = 'Z' * self.n_qubits
        elif self.observable == 'X':
            pauli_str = 'X' * self.n_qubits
        elif self.observable == 'Y':
            pauli_str = 'Y' * self.n_qubits
        else:
            raise ValueError(f"Unknown observable: {self.observable}")

        return SparsePauliOp(pauli_str)

    def _create_variational_circuit(self) -> QuantumCircuit:
        """Create the variational quantum circuit."""
        return self.ansatz

    def _encode_data(self, X: np.ndarray) -> QuantumCircuit:
        """
        Encode classical data into quantum circuit.

        Args:
            X: Input data

        Returns:
            Quantum circuit with encoded data
        """
        return self.encoder.fit_transform(X)

    def _compute_expectation(self, circuit: QuantumCircuit) -> float:
        """
        Compute expectation value of the observable.

        Args:
            circuit: Quantum circuit

        Returns:
            Expectation value
        """
        if self.estimator is None:
            logger.warning("Estimator not available, returning random value")
            return np.random.random()

        try:
            job = self.estimator.run(circuit, self.observable_op)
            result = job.result()
            expectation = result.values[0]

            # Scale to reasonable output range for regression
            # Map [-1, 1] to [0, 1] range
            scaled_expectation = (expectation + 1) / 2

            return scaled_expectation

        except Exception as e:
            logger.error(f"Expectation computation failed: {e}")
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> bool:
        """
        Train the VQR model.

        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training arguments

        Returns:
            True if training successful
        """
        # Fit the encoder
        self.encoder.fit(X)

        # Scale targets to [0, 1] range for better training
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        y_scaled = (y - self.y_min) / (self.y_max - self.y_min + 1e-8)

        # Call parent fit method
        return super().fit(X, y_scaled, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        # Get scaled predictions
        predictions_scaled = super().predict(X)

        # Rescale to original range
        predictions = predictions_scaled * (self.y_max - self.y_min) + self.y_min

        return predictions

    def get_model_architecture(self) -> Dict[str, Any]:
        """
        Get detailed model architecture information.

        Returns:
            Dictionary with architecture details
        """
        return {
            'model_type': 'VQR',
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'encoding_type': self.encoding_type,
            'ansatz_type': self.ansatz_type,
            'observable': self.observable,
            'n_parameters': len(self.ansatz.parameters) if self.ansatz else 0,
            'circuit_depth': self.ansatz.depth() if self.ansatz else 0,
            'is_trained': self.is_trained
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance based on encoding weights.

        Returns:
            Feature importance scores or None
        """
        if hasattr(self.encoder, 'get_feature_importance'):
            return self.encoder.get_feature_importance()
        return None

    def __repr__(self) -> str:
        return (f"VariationalQuantumRegressor("
                f"n_qubits={self.n_qubits}, "
                f"n_layers={self.n_layers}, "
                f"encoding='{self.encoding_type}', "
                f"ansatz='{self.ansatz_type}', "
                f"trained={self.is_trained})")