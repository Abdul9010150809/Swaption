#!/usr/bin/env python3
"""
Amplitude Encoding for Quantum Data Encoding
Encodes classical data as quantum state amplitudes
"""

import numpy as np
from typing import Optional, Union
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
import logging

logger = logging.getLogger(__name__)

class AmplitudeEncoder:
    """
    Amplitude encoding strategy for quantum data encoding.

    This encoder maps classical data vectors to quantum state amplitudes,
    providing exponential compression of data into quantum states.
    """

    def __init__(self, n_qubits: Optional[int] = None, padding: str = 'zero'):
        """
        Initialize amplitude encoder.

        Args:
            n_qubits: Number of qubits (if None, calculated from data size)
            padding: Padding method for non-power-of-2 data ('zero', 'repeat', 'normalize')
        """
        self.n_qubits = n_qubits
        self.padding = padding
        self.data_dimension = None
        self.amplitude_vector = None

    def fit(self, X: np.ndarray) -> 'AmplitudeEncoder':
        """
        Fit the encoder to the data.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self: Fitted encoder
        """
        self.data_dimension = X.shape[1]

        # Calculate required qubits if not specified
        if self.n_qubits is None:
            self.n_qubits = int(np.ceil(np.log2(self.data_dimension)))

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

        # Use first sample for encoding
        data_vector = X[0]

        # Prepare amplitude vector
        amplitude_vector = self._prepare_amplitude_vector(data_vector)

        # Create quantum circuit
        circuit = QuantumCircuit(self.n_qubits)

        # Use Qiskit's Initialize instruction for amplitude encoding
        init_gate = Initialize(amplitude_vector)
        circuit.append(init_gate, range(self.n_qubits))

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

    def _prepare_amplitude_vector(self, data_vector: np.ndarray) -> np.ndarray:
        """
        Prepare the amplitude vector for quantum state initialization.

        Args:
            data_vector: Input data vector

        Returns:
            Normalized amplitude vector of size 2^n_qubits
        """
        target_size = 2 ** self.n_qubits

        if len(data_vector) > target_size:
            # Truncate if too large
            logger.warning(f"Data dimension {len(data_vector)} > 2^{self.n_qubits} = {target_size}, truncating")
            amplitude_vector = data_vector[:target_size]
        else:
            # Pad if necessary
            amplitude_vector = np.zeros(target_size, dtype=complex)

            if self.padding == 'zero':
                amplitude_vector[:len(data_vector)] = data_vector
            elif self.padding == 'repeat':
                # Repeat the data to fill the vector
                repeats = target_size // len(data_vector)
                remainder = target_size % len(data_vector)
                amplitude_vector[:len(data_vector) * repeats] = np.tile(data_vector, repeats)
                amplitude_vector[len(data_vector) * repeats:len(data_vector) * repeats + remainder] = data_vector[:remainder]
            elif self.padding == 'normalize':
                # Just place data at the beginning
                amplitude_vector[:len(data_vector)] = data_vector
            else:
                raise ValueError(f"Unknown padding method: {self.padding}")

        # Normalize to ensure it's a valid quantum state
        norm = np.linalg.norm(amplitude_vector)
        if norm > 0:
            amplitude_vector = amplitude_vector / norm
        else:
            # Handle zero vector case
            amplitude_vector[0] = 1.0

        return amplitude_vector

    def inverse_transform(self, circuit: QuantumCircuit) -> Optional[np.ndarray]:
        """
        Attempt to recover classical data from quantum circuit.
        Note: This is generally not possible for amplitude encoding.

        Args:
            circuit: Quantum circuit

        Returns:
            None: Inverse transform not supported for amplitude encoding
        """
        logger.warning("Inverse transform not supported for amplitude encoding")
        return None

    def get_encoding_efficiency(self) -> float:
        """
        Get the encoding efficiency (data dimension / quantum dimension).

        Returns:
            Efficiency ratio
        """
        if self.data_dimension is None or self.n_qubits is None:
            return 0.0

        quantum_dimension = 2 ** self.n_qubits
        return self.data_dimension / quantum_dimension

    def __repr__(self) -> str:
        return (f"AmplitudeEncoder(n_qubits={self.n_qubits}, "
                f"padding='{self.padding}', "
                f"data_dimension={self.data_dimension})")