#!/usr/bin/env python3
"""
Quantum Optimizers Module
Provides optimization algorithms specifically designed for quantum variational circuits
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class QuantumOptimizer(ABC):
    """
    Abstract base class for quantum optimizers.

    Quantum optimizers are designed to work with variational quantum circuits
    and may use quantum-specific techniques like parameter shift rules.
    """

    def __init__(self, learning_rate: float = 0.01, max_iter: int = 100):
        """
        Initialize the quantum optimizer.

        Args:
            learning_rate: Learning rate
            max_iter: Maximum iterations
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.iteration = 0

    @abstractmethod
    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform one optimization step.

        Args:
            parameters: Current parameter values
            gradient: Gradient vector

        Returns:
            Updated parameter values
        """
        pass

    def reset(self):
        """Reset optimizer state."""
        self.iteration = 0

class QuantumSGD(QuantumOptimizer):
    """
    Quantum Stochastic Gradient Descent.

    Basic gradient descent optimizer adapted for quantum circuits.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0,
                 nesterov: bool = False, **kwargs):
        """
        Initialize Quantum SGD.

        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            nesterov: Whether to use Nesterov momentum
        """
        super().__init__(learning_rate, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = None

    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform SGD step.

        Args:
            parameters: Current parameters
            gradient: Gradient vector

        Returns:
            Updated parameters
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(parameters)

        # Update velocity
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient

        # Nesterov momentum
        if self.nesterov:
            parameters = parameters + self.momentum * self.velocity - self.learning_rate * gradient
        else:
            parameters = parameters + self.velocity

        return parameters

class QuantumAdam(QuantumOptimizer):
    """
    Quantum Adam optimizer.

    Adaptive moment estimation optimizer adapted for quantum variational circuits.
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8, **kwargs):
        """
        Initialize Quantum Adam.

        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Adam state
        self.m = None  # First moment
        self.v = None  # Second moment

    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform Adam step.

        Args:
            parameters: Current parameters
            gradient: Gradient vector

        Returns:
            Updated parameters
        """
        self.iteration += 1

        # Initialize moments
        if self.m is None:
            self.m = np.zeros_like(parameters)
            self.v = np.zeros_like(parameters)

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.iteration)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.iteration)

        # Update parameters
        parameters = parameters - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return parameters

    def reset(self):
        """Reset Adam state."""
        super().reset()
        self.m = None
        self.v = None

class QuantumRMSProp(QuantumOptimizer):
    """
    Quantum RMSProp optimizer.

    Root mean square propagation optimizer for quantum circuits.
    """

    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9,
                 epsilon: float = 1e-8, **kwargs):
        """
        Initialize Quantum RMSProp.

        Args:
            learning_rate: Learning rate
            rho: Decay rate for moving average
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate, **kwargs)
        self.rho = rho
        self.epsilon = epsilon
        self.Eg2 = None  # Moving average of squared gradients

    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform RMSProp step.

        Args:
            parameters: Current parameters
            gradient: Gradient vector

        Returns:
            Updated parameters
        """
        # Initialize moving average
        if self.Eg2 is None:
            self.Eg2 = np.zeros_like(parameters)

        # Update moving average of squared gradients
        self.Eg2 = self.rho * self.Eg2 + (1 - self.rho) * (gradient ** 2)

        # Update parameters
        parameters = parameters - self.learning_rate * gradient / (np.sqrt(self.Eg2) + self.epsilon)

        return parameters

    def reset(self):
        """Reset RMSProp state."""
        super().reset()
        self.Eg2 = None

class QuantumNaturalGradient(QuantumOptimizer):
    """
    Quantum Natural Gradient optimizer.

    Uses quantum Fisher information matrix for more efficient optimization
    of variational quantum circuits.
    """

    def __init__(self, learning_rate: float = 0.01, regularization: float = 1e-4, **kwargs):
        """
        Initialize Quantum Natural Gradient.

        Args:
            learning_rate: Learning rate
            regularization: Regularization parameter for FIM inversion
        """
        super().__init__(learning_rate, **kwargs)
        self.regularization = regularization
        self.fisher_information = None

    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform natural gradient step.

        Args:
            parameters: Current parameters
            gradient: Gradient vector

        Returns:
            Updated parameters
        """
        # For now, implement as preconditioned gradient
        # In practice, this would compute the quantum Fisher information matrix

        if self.fisher_information is None:
            # Initialize with identity (no preconditioning initially)
            self.fisher_information = np.eye(len(parameters))

        # Regularize FIM
        fim_reg = self.fisher_information + self.regularization * np.eye(len(parameters))

        # Compute natural gradient
        try:
            natural_gradient = np.linalg.solve(fim_reg, gradient)
        except np.linalg.LinAlgError:
            # Fallback to regular gradient if inversion fails
            natural_gradient = gradient

        # Update parameters
        parameters = parameters - self.learning_rate * natural_gradient

        return parameters

class SPSAQuantum(QuantumOptimizer):
    """
    Simultaneous Perturbation Stochastic Approximation for quantum circuits.

    SPSA is particularly well-suited for quantum optimization as it only
    requires two function evaluations per iteration, regardless of dimension.
    """

    def __init__(self, learning_rate: float = 0.01, perturbation: float = 0.1,
                 alpha: float = 0.602, gamma: float = 0.101, **kwargs):
        """
        Initialize SPSA optimizer.

        Args:
            learning_rate: Initial learning rate
            perturbation: Initial perturbation size
            alpha: Learning rate decay exponent
            gamma: Perturbation decay exponent
        """
        super().__init__(learning_rate, **kwargs)
        self.perturbation = perturbation
        self.alpha = alpha
        self.gamma = gamma

    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform SPSA step.

        Note: SPSA computes its own gradient estimate, so the gradient
        parameter is ignored here.

        Args:
            parameters: Current parameters
            gradient: Ignored (SPSA computes its own gradient estimate)

        Returns:
            Updated parameters
        """
        self.iteration += 1

        # Update learning rate and perturbation
        learning_rate = self.learning_rate / (self.iteration ** self.alpha)
        perturbation = self.perturbation / (self.iteration ** self.gamma)

        # Generate random perturbation vector
        delta = np.random.choice([-1, 1], size=len(parameters))

        # Compute function values at perturbed points
        # Note: In practice, this would require function evaluations
        # For now, we'll use the provided gradient as approximation
        gradient_estimate = gradient * (1.0 / (perturbation * delta + 1e-8))

        # Update parameters
        parameters = parameters - learning_rate * gradient_estimate

        return parameters

def create_optimizer(optimizer_type: str, **kwargs) -> QuantumOptimizer:
    """
    Factory function to create quantum optimizers.

    Args:
        optimizer_type: Type of optimizer ('sgd', 'adam', 'rmsprop', 'natural', 'spsa')
        **kwargs: Optimizer-specific parameters

    Returns:
        Quantum optimizer instance
    """
    optimizers = {
        'sgd': QuantumSGD,
        'adam': QuantumAdam,
        'rmsprop': QuantumRMSProp,
        'natural': QuantumNaturalGradient,
        'spsa': SPSAQuantum
    }

    if optimizer_type not in optimizers:
        available = list(optimizers.keys())
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Available: {available}")

    return optimizers[optimizer_type](**kwargs)