#!/usr/bin/env python3
"""
Parameter Shift Rule Implementation
Provides gradient computation for quantum variational circuits
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import logging

logger = logging.getLogger(__name__)

class ParameterShiftRule:
    """
    Parameter shift rule for computing gradients of quantum circuits.

    The parameter shift rule provides a way to compute exact gradients of
    expectation values with respect to variational parameters, which is
    essential for training quantum neural networks.
    """

    def __init__(self, shift: float = np.pi / 2):
        """
        Initialize parameter shift rule.

        Args:
            shift: Shift angle (default π/2 for standard parameter shift)
        """
        self.shift = shift

    def compute_gradient(self, circuit: QuantumCircuit, observable: SparsePauliOp,
                        parameter_index: int, estimator=None) -> float:
        """
        Compute gradient with respect to a single parameter.

        Args:
            circuit: Parameterized quantum circuit
            observable: Observable to measure
            parameter_index: Index of parameter to differentiate
            estimator: Quantum estimator

        Returns:
            Gradient value
        """
        if estimator is None:
            logger.warning("No estimator provided, returning zero gradient")
            return 0.0

        try:
            # Get parameter
            param = list(circuit.parameters)[parameter_index]

            # Create shifted circuits
            circuit_plus = circuit.assign_parameters({param: param + self.shift})
            circuit_minus = circuit.assign_parameters({param: param - self.shift})

            # Compute expectation values
            job_plus = estimator.run(circuit_plus, observable)
            job_minus = estimator.run(circuit_minus, observable)

            result_plus = job_plus.result()
            result_minus = job_minus.result()

            expectation_plus = result_plus.values[0]
            expectation_minus = result_minus.values[0]

            # Apply parameter shift rule
            gradient = (expectation_plus - expectation_minus) / (2 * np.sin(self.shift))

            return gradient

        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            return 0.0

    def compute_full_gradient(self, circuit: QuantumCircuit, observable: SparsePauliOp,
                            estimator=None) -> np.ndarray:
        """
        Compute gradient with respect to all parameters.

        Args:
            circuit: Parameterized quantum circuit
            observable: Observable to measure
            estimator: Quantum estimator

        Returns:
            Gradient vector
        """
        n_parameters = len(circuit.parameters)
        gradients = np.zeros(n_parameters)

        for i in range(n_parameters):
            grad = self.compute_gradient(circuit, observable, i, estimator)
            gradients[i] = grad

        return gradients

    def compute_finite_difference_gradient(self, circuit: QuantumCircuit,
                                         observable: SparsePauliOp,
                                         epsilon: float = 1e-7,
                                         estimator=None) -> np.ndarray:
        """
        Compute gradient using finite differences (for comparison/debugging).

        Args:
            circuit: Parameterized quantum circuit
            observable: Observable to measure
            epsilon: Finite difference step size
            estimator: Quantum estimator

        Returns:
            Gradient vector
        """
        if estimator is None:
            return np.zeros(len(circuit.parameters))

        n_parameters = len(circuit.parameters)
        gradients = np.zeros(n_parameters)
        parameters = list(circuit.parameters)

        # Get baseline expectation
        job_base = estimator.run(circuit, observable)
        base_expectation = job_base.result().values[0]

        for i, param in enumerate(parameters):
            # Positive shift
            circuit_plus = circuit.assign_parameters({param: param + epsilon})
            job_plus = estimator.run(circuit_plus, observable)
            exp_plus = job_plus.result().values[0]

            # Negative shift
            circuit_minus = circuit.assign_parameters({param: param - epsilon})
            job_minus = estimator.run(circuit_minus, observable)
            exp_minus = job_minus.result().values[0]

            # Central difference
            gradients[i] = (exp_plus - exp_minus) / (2 * epsilon)

        return gradients

class QuantumGradientDescent:
    """
    Quantum-aware gradient descent optimizer using parameter shift rule.
    """

    def __init__(self, learning_rate: float = 0.01, shift_rule: Optional[ParameterShiftRule] = None):
        """
        Initialize quantum gradient descent.

        Args:
            learning_rate: Learning rate
            shift_rule: Parameter shift rule instance
        """
        self.learning_rate = learning_rate
        self.shift_rule = shift_rule or ParameterShiftRule()

    def step(self, circuit: QuantumCircuit, observable: SparsePauliOp,
             current_params: np.ndarray, estimator=None) -> np.ndarray:
        """
        Perform one optimization step.

        Args:
            circuit: Current parameterized circuit
            observable: Observable to optimize
            current_params: Current parameter values
            estimator: Quantum estimator

        Returns:
            Updated parameter values
        """
        # Bind current parameters
        param_dict = dict(zip(circuit.parameters, current_params))
        circuit_bound = circuit.assign_parameters(param_dict)

        # Compute gradient
        gradient = self.shift_rule.compute_full_gradient(circuit_bound, observable, estimator)

        # Update parameters
        new_params = current_params - self.learning_rate * gradient

        return new_params

class QuantumAdam:
    """
    Quantum-aware Adam optimizer using parameter shift rule.
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 shift_rule: Optional[ParameterShiftRule] = None):
        """
        Initialize quantum Adam optimizer.

        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            shift_rule: Parameter shift rule instance
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.shift_rule = shift_rule or ParameterShiftRule()

        # Adam state
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Timestep

    def step(self, circuit: QuantumCircuit, observable: SparsePauliOp,
             current_params: np.ndarray, estimator=None) -> np.ndarray:
        """
        Perform one Adam optimization step.

        Args:
            circuit: Current parameterized circuit
            observable: Observable to optimize
            current_params: Current parameter values
            estimator: Quantum estimator

        Returns:
            Updated parameter values
        """
        self.t += 1

        # Initialize moments
        if self.m is None:
            self.m = np.zeros_like(current_params)
            self.v = np.zeros_like(current_params)

        # Bind current parameters
        param_dict = dict(zip(circuit.parameters, current_params))
        circuit_bound = circuit.assign_parameters(param_dict)

        # Compute gradient
        gradient = self.shift_rule.compute_full_gradient(circuit_bound, observable, estimator)

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters
        new_params = current_params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return new_params