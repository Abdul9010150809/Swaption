#!/usr/bin/env python3
"""
Feature Maps Module
Provides quantum feature maps for kernel methods and data encoding
"""

import numpy as np
from typing import List, Optional, Dict, Any, Callable
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
import logging

logger = logging.getLogger(__name__)

class FeatureMapDesigner:
    """
    Designer class for creating various quantum feature maps.

    Feature maps encode classical data into quantum Hilbert space,
    enabling quantum kernel methods and quantum machine learning.
    """

    def __init__(self):
        self.feature_map_types = {
            'zz': self._create_zz_feature_map,
            'pauli': self._create_pauli_feature_map,
            'hardware_efficient': self._create_hardware_efficient_map,
            'finance_correlation': self._create_finance_correlation_map,
            'temporal': self._create_temporal_feature_map,
            'custom': self._create_custom_feature_map
        }

    def create_feature_map(self, map_type: str, n_qubits: int, n_features: int,
                          reps: int = 1, **kwargs) -> QuantumCircuit:
        """
        Create a quantum feature map.

        Args:
            map_type: Type of feature map
            n_qubits: Number of qubits
            n_features: Number of input features
            reps: Number of repetitions
            **kwargs: Additional parameters

        Returns:
            Quantum feature map circuit
        """
        if map_type not in self.feature_map_types:
            available = list(self.feature_map_types.keys())
            raise ValueError(f"Unknown feature map type: {map_type}. Available: {available}")

        return self.feature_map_types[map_type](n_qubits, n_features, reps, **kwargs)

    def _create_zz_feature_map(self, n_qubits: int, n_features: int, reps: int,
                              **kwargs) -> QuantumCircuit:
        """Create ZZ feature map."""
        feature_map = ZZFeatureMap(feature_dimension=n_features, reps=reps)
        return feature_map

    def _create_pauli_feature_map(self, n_qubits: int, n_features: int, reps: int,
                                 pauli_string: str = 'Z', **kwargs) -> QuantumCircuit:
        """Create Pauli feature map."""
        feature_map = PauliFeatureMap(feature_dimension=n_features,
                                     paulis=[pauli_string], reps=reps)
        return feature_map

    def _create_hardware_efficient_map(self, n_qubits: int, n_features: int, reps: int,
                                      **kwargs) -> QuantumCircuit:
        """Create hardware-efficient feature map."""
        circuit = QuantumCircuit(n_qubits)
        params = ParameterVector('phi', n_features * reps)

        param_idx = 0
        for rep in range(reps):
            # Single qubit rotations
            for qubit in range(min(n_qubits, n_features)):
                circuit.ry(params[param_idx], qubit)
                param_idx += 1

            # Entangling gates
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)

        return circuit

    def _create_finance_correlation_map(self, n_qubits: int, n_features: int, reps: int,
                                       **kwargs) -> QuantumCircuit:
        """
        Create finance-specific feature map that captures correlation structures.

        This feature map is designed to encode financial time series data
        with emphasis on correlation patterns between assets.
        """
        circuit = QuantumCircuit(n_qubits)
        n_params = n_features * reps + (n_features * (n_features - 1) // 2) * reps
        params = ParameterVector('finance_phi', n_params)

        param_idx = 0

        for rep in range(reps):
            # Encode individual asset features
            for i in range(min(n_qubits, n_features)):
                circuit.ry(params[param_idx], i)
                param_idx += 1

            # Encode pairwise correlations
            correlation_idx = 0
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    if correlation_idx < (n_features * (n_features - 1) // 2):
                        # Use ZZ interaction to encode correlations
                        circuit.cx(i, j)
                        circuit.rz(params[param_idx + correlation_idx], j)
                        circuit.cx(i, j)
                        correlation_idx += 1

            param_idx += correlation_idx

        return circuit

    def _create_temporal_feature_map(self, n_qubits: int, n_features: int, reps: int,
                                    **kwargs) -> QuantumCircuit:
        """
        Create temporal feature map for time series data.

        This feature map encodes temporal dependencies and patterns
        in sequential financial data.
        """
        circuit = QuantumCircuit(n_qubits)
        n_params = n_features * reps * 2  # Two parameters per feature per rep
        params = ParameterVector('temporal_phi', n_params)

        param_idx = 0

        for rep in range(reps):
            # Encode features with temporal phase
            for i in range(min(n_qubits, n_features)):
                # Amplitude encoding
                circuit.ry(params[param_idx], i)
                # Phase encoding for temporal information
                circuit.rz(params[param_idx + 1], i)
                param_idx += 2

            # Temporal correlations via controlled operations
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)
                # Add temporal phase shift
                circuit.rz(params[param_idx - 1] * 0.1, i + 1)  # Small phase for temporal correlation

        return circuit

    def _create_custom_feature_map(self, n_qubits: int, n_features: int, reps: int,
                                 gates: List[str] = None, **kwargs) -> QuantumCircuit:
        """Create custom feature map with specified gates."""
        circuit = QuantumCircuit(n_qubits)
        gates = gates or ['ry', 'rz']

        n_params = len(gates) * n_features * reps
        params = ParameterVector('custom_phi', n_params)

        param_idx = 0

        for rep in range(reps):
            for feature in range(min(n_qubits, n_features)):
                for gate in gates:
                    if gate.lower() == 'rx':
                        circuit.rx(params[param_idx], feature)
                    elif gate.lower() == 'ry':
                        circuit.ry(params[param_idx], feature)
                    elif gate.lower() == 'rz':
                        circuit.rz(params[param_idx], feature)
                    param_idx += 1

            # Entangling layer
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)

        return circuit

    def get_feature_map_info(self, map_type: str) -> Dict[str, Any]:
        """
        Get information about a specific feature map type.

        Args:
            map_type: Type of feature map

        Returns:
            Dictionary with feature map information
        """
        info = {
            'zz': {
                'description': 'ZZ feature map with Z rotations and ZZ interactions',
                'parameters': 'n_features * reps',
                'complexity': 'Medium',
                'best_for': 'General QML tasks'
            },
            'pauli': {
                'description': 'Pauli feature map with customizable Pauli operators',
                'parameters': 'n_features * reps',
                'complexity': 'Configurable',
                'best_for': 'Specific quantum observables'
            },
            'hardware_efficient': {
                'description': 'Hardware-efficient feature map for NISQ devices',
                'parameters': 'n_features * reps',
                'complexity': 'Low',
                'best_for': 'Near-term quantum hardware'
            },
            'finance_correlation': {
                'description': 'Finance-specific map capturing asset correlations',
                'parameters': 'Complex (includes correlation terms)',
                'complexity': 'High',
                'best_for': 'Financial portfolio optimization'
            },
            'temporal': {
                'description': 'Temporal feature map for time series data',
                'parameters': '2 * n_features * reps',
                'complexity': 'Medium',
                'best_for': 'Financial time series analysis'
            },
            'custom': {
                'description': 'Custom feature map with user-defined gates',
                'parameters': 'Variable',
                'complexity': 'Configurable',
                'best_for': 'Research and experimentation'
            }
        }

        return info.get(map_type, {'description': 'Unknown feature map type'})

    def list_available_maps(self) -> List[str]:
        """List all available feature map types."""
        return list(self.feature_map_types.keys())

    def evaluate_kernel_matrix(self, feature_map: QuantumCircuit,
                             X: np.ndarray, backend=None) -> np.ndarray:
        """
        Evaluate the quantum kernel matrix for given data.

        Args:
            feature_map: Quantum feature map circuit
            X: Input data matrix
            backend: Quantum backend for execution

        Returns:
            Kernel matrix
        """
        # This would require quantum kernel evaluation
        # For now, return identity matrix as placeholder
        n_samples = X.shape[0]
        logger.warning("Kernel evaluation requires quantum execution - returning identity matrix")
        return np.eye(n_samples)