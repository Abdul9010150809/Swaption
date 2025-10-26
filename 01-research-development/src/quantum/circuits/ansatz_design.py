#!/usr/bin/env python3
"""
Ansatz Design Module
Provides various parameterized quantum circuit architectures for variational algorithms
"""

import numpy as np
from typing import List, Optional, Dict, Any
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal
import logging

logger = logging.getLogger(__name__)

class AnsatzDesigner:
    """
    Designer class for creating various quantum ansatz circuits.

    Provides a unified interface for creating different types of parameterized
    quantum circuits used in variational quantum algorithms.
    """

    def __init__(self):
        self.ansatz_types = {
            'real_amplitudes': self._create_real_amplitudes,
            'efficient_su2': self._create_efficient_su2,
            'two_local': self._create_two_local,
            'hardware_efficient': self._create_hardware_efficient,
            'variational': self._create_variational,
            'finance_optimized': self._create_finance_optimized
        }

    def create_ansatz(self, ansatz_type: str, n_qubits: int, n_layers: int = 1,
                     **kwargs) -> QuantumCircuit:
        """
        Create a parameterized quantum circuit (ansatz).

        Args:
            ansatz_type: Type of ansatz ('real_amplitudes', 'efficient_su2', etc.)
            n_qubits: Number of qubits
            n_layers: Number of layers/repetitions
            **kwargs: Additional parameters for specific ansatz types

        Returns:
            Parameterized quantum circuit
        """
        if ansatz_type not in self.ansatz_types:
            available = list(self.ansatz_types.keys())
            raise ValueError(f"Unknown ansatz type: {ansatz_type}. Available: {available}")

        return self.ansatz_types[ansatz_type](n_qubits, n_layers, **kwargs)

    def _create_real_amplitudes(self, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        """Create Real Amplitudes ansatz."""
        ansatz = RealAmplitudes(num_qubits=n_qubits, reps=n_layers)
        return ansatz

    def _create_efficient_su2(self, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        """Create EfficientSU2 ansatz."""
        ansatz = EfficientSU2(num_qubits=n_qubits, reps=n_layers)
        return ansatz

    def _create_two_local(self, n_qubits: int, n_layers: int,
                         rotation_gates: List[str] = None,
                         entanglement_gates: List[str] = None, **kwargs) -> QuantumCircuit:
        """Create Two-Local ansatz."""
        rotation_gates = rotation_gates or ['ry', 'rz']
        entanglement_gates = entanglement_gates or ['cz']

        ansatz = TwoLocal(num_qubits=n_qubits,
                         rotation_blocks=rotation_gates,
                         entanglement_blocks=entanglement_gates,
                         reps=n_layers)
        return ansatz

    def _create_hardware_efficient(self, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        """Create hardware-efficient ansatz."""
        circuit = QuantumCircuit(n_qubits)
        params = ParameterVector('theta', n_qubits * n_layers * 2)

        param_idx = 0
        for layer in range(n_layers):
            # Single qubit rotations
            for qubit in range(n_qubits):
                circuit.ry(params[param_idx], qubit)
                circuit.rz(params[param_idx + 1], qubit)
                param_idx += 2

            # Entangling gates
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)

        return circuit

    def _create_variational(self, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        """Create general variational ansatz."""
        circuit = QuantumCircuit(n_qubits)
        n_params = n_qubits * n_layers * 3  # 3 parameters per qubit per layer
        params = ParameterVector('theta', n_params)

        param_idx = 0
        for layer in range(n_layers):
            # Layer of single qubit rotations
            for qubit in range(n_qubits):
                circuit.rx(params[param_idx], qubit)
                circuit.ry(params[param_idx + 1], qubit)
                circuit.rz(params[param_idx + 2], qubit)
                param_idx += 3

            # Entangling layer
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)

        return circuit

    def _create_finance_optimized(self, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        """
        Create finance-optimized ansatz with correlation-aware entanglement.

        This ansatz is designed specifically for financial data with patterns
        that capture correlation structures common in financial time series.
        """
        circuit = QuantumCircuit(n_qubits)
        n_params = n_qubits * n_layers * 2 + (n_qubits * (n_qubits - 1) // 2) * n_layers
        params = ParameterVector('finance_theta', n_params)

        param_idx = 0

        for layer in range(n_layers):
            # Single qubit rotations (representing individual assets/features)
            for qubit in range(n_qubits):
                circuit.ry(params[param_idx], qubit)
                circuit.rz(params[param_idx + 1], qubit)
                param_idx += 2

            # Correlation-based entanglement (representing asset correlations)
            # Use controlled rotations instead of CNOT for more expressiveness
            entanglement_idx = 0
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    # Controlled RY rotation to capture pairwise correlations
                    circuit.cry(params[param_idx + entanglement_idx], i, j)
                    entanglement_idx += 1

            param_idx += entanglement_idx

        return circuit

    def get_ansatz_info(self, ansatz_type: str) -> Dict[str, Any]:
        """
        Get information about a specific ansatz type.

        Args:
            ansatz_type: Type of ansatz

        Returns:
            Dictionary with ansatz information
        """
        info = {
            'real_amplitudes': {
                'description': 'Real Amplitudes ansatz with Y and Z rotations',
                'parameters': '2 * n_qubits * n_layers',
                'gates': ['RY', 'RZ', 'CNOT'],
                'complexity': 'Medium'
            },
            'efficient_su2': {
                'description': 'Efficient SU(2) ansatz for VQE applications',
                'parameters': '2 * n_qubits * n_layers',
                'gates': ['RY', 'RZ', 'CNOT'],
                'complexity': 'Medium'
            },
            'two_local': {
                'description': 'Two-local ansatz with customizable rotation and entanglement gates',
                'parameters': 'Variable',
                'gates': ['Custom rotation gates', 'Custom entanglement gates'],
                'complexity': 'Configurable'
            },
            'hardware_efficient': {
                'description': 'Hardware-efficient ansatz optimized for NISQ devices',
                'parameters': '2 * n_qubits * n_layers',
                'gates': ['RY', 'RZ', 'CNOT'],
                'complexity': 'Low'
            },
            'variational': {
                'description': 'General variational ansatz with RX, RY, RZ rotations',
                'parameters': '3 * n_qubits * n_layers',
                'gates': ['RX', 'RY', 'RZ', 'CNOT'],
                'complexity': 'High'
            },
            'finance_optimized': {
                'description': 'Finance-specific ansatz with correlation-aware entanglement',
                'parameters': 'Complex (includes correlation parameters)',
                'gates': ['RY', 'RZ', 'CRY'],
                'complexity': 'High'
            }
        }

        return info.get(ansatz_type, {'description': 'Unknown ansatz type'})

    def list_available_ansatz(self) -> List[str]:
        """List all available ansatz types."""
        return list(self.ansatz_types.keys())