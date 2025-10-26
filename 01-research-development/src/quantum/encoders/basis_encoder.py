#!/usr/bin/env python3
"""
Basis Encoding for Quantum Data Encoding
Encodes discrete or categorical data using computational basis states
"""

import numpy as np
from typing import Optional, Union, Dict, List
from qiskit import QuantumCircuit
import logging

logger = logging.getLogger(__name__)

class BasisEncoder:
    """
    Basis encoding strategy for quantum data encoding.

    This encoder maps discrete or categorical data to computational basis states,
    suitable for classification tasks or discrete feature encoding.
    """

    def __init__(self, n_qubits: Optional[int] = None, encoding_scheme: str = 'binary'):
        """
        Initialize basis encoder.

        Args:
            n_qubits: Number of qubits (if None, calculated from data)
            encoding_scheme: Encoding scheme ('binary', 'one_hot', 'thermometer')
        """
        self.n_qubits = n_qubits
        self.encoding_scheme = encoding_scheme.lower()
        self.value_to_basis = {}
        self.basis_to_value = {}
        self.max_value = None
        self.fitted = False

        if self.encoding_scheme not in ['binary', 'one_hot', 'thermometer']:
            raise ValueError(f"Unsupported encoding scheme: {encoding_scheme}")

    def fit(self, X: np.ndarray) -> 'BasisEncoder':
        """
        Fit the encoder to the data.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self: Fitted encoder
        """
        # For simplicity, assume single feature encoding
        if X.ndim > 1 and X.shape[1] > 1:
            logger.warning("BasisEncoder currently supports single feature encoding, using first feature")

        data = X if X.ndim == 1 else X[:, 0]

        # Get unique values and sort them
        unique_values = np.unique(data)
        self.max_value = np.max(unique_values)

        # Calculate required qubits
        if self.encoding_scheme == 'binary':
            if self.n_qubits is None:
                self.n_qubits = int(np.ceil(np.log2(len(unique_values))))
        elif self.encoding_scheme == 'one_hot':
            self.n_qubits = len(unique_values)
        elif self.encoding_scheme == 'thermometer':
            self.n_qubits = int(self.max_value) + 1

        # Create mapping
        for i, value in enumerate(unique_values):
            if self.encoding_scheme == 'binary':
                # Convert to binary representation
                binary_str = format(i, f'0{self.n_qubits}b')
                basis_state = [int(bit) for bit in binary_str]
            elif self.encoding_scheme == 'one_hot':
                # One-hot encoding
                basis_state = [1 if j == i else 0 for j in range(self.n_qubits)]
            elif self.encoding_scheme == 'thermometer':
                # Thermometer encoding (unary)
                int_value = int(value)
                basis_state = [1 if j <= int_value else 0 for j in range(self.n_qubits)]

            self.value_to_basis[value] = basis_state
            self.basis_to_value[tuple(basis_state)] = value

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> QuantumCircuit:
        """
        Transform classical data to quantum circuit.

        Args:
            X: Input data of shape (n_samples,) or (n_samples, 1)

        Returns:
            QuantumCircuit: Encoded quantum circuit
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        # Handle input shape
        if X.ndim > 1 and X.shape[1] > 1:
            logger.warning("Using first feature for basis encoding")
        data = X if X.ndim == 1 else X[:, 0]

        # Use first sample for encoding
        value = data[0]

        if value not in self.value_to_basis:
            logger.warning(f"Unknown value {value}, using default encoding")
            # Use first basis state as default
            basis_state = list(self.value_to_basis.values())[0]
        else:
            basis_state = self.value_to_basis[value]

        # Create quantum circuit
        circuit = QuantumCircuit(self.n_qubits)

        # Apply X gates for |1⟩ states
        for i, bit in enumerate(basis_state):
            if bit == 1:
                circuit.x(i)

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

    def inverse_transform(self, circuit: QuantumCircuit) -> Optional[np.ndarray]:
        """
        Attempt to recover classical data from quantum circuit by measuring.

        Args:
            circuit: Quantum circuit

        Returns:
            Recovered classical data or None if not possible
        """
        # This would require actual quantum execution
        # For now, return None as it's not straightforward
        logger.warning("Inverse transform requires quantum execution")
        return None

    def get_basis_states(self) -> Dict:
        """
        Get the mapping of values to basis states.

        Returns:
            Dictionary mapping values to basis state lists
        """
        return self.value_to_basis.copy()

    def get_num_classes(self) -> int:
        """
        Get the number of unique classes/values.

        Returns:
            Number of unique values
        """
        return len(self.value_to_basis)

    def __repr__(self) -> str:
        return (f"BasisEncoder(n_qubits={self.n_qubits}, "
                f"encoding_scheme='{self.encoding_scheme}', "
                f"fitted={self.fitted})")