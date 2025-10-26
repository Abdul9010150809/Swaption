#!/usr/bin/env python3
"""
Core Quantum Neural Network Implementation
Provides the foundation for quantum machine learning models
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
import logging
import time

# Quantum imports with fallbacks
try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import Estimator, Sampler
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.algorithms.optimizers import SPSA, COBYLA, ADAM
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False

logger = logging.getLogger(__name__)

class QuantumNeuralNetwork(ABC):
    """
    Abstract base class for Quantum Neural Networks.

    This class provides the foundation for implementing various quantum
    machine learning algorithms including variational quantum circuits,
    quantum kernel methods, and hybrid quantum-classical models.
    """

    def __init__(self, n_qubits: int, n_layers: int = 1,
                 optimizer: Optional[Any] = None, max_iter: int = 100):
        """
        Initialize the quantum neural network.

        Args:
            n_qubits: Number of qubits in the circuit
            n_layers: Number of variational layers
            optimizer: Classical optimizer for training
            max_iter: Maximum training iterations
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.optimizer = optimizer or SPSA(maxiter=max_iter)
        self.max_iter = max_iter

        # Training state
        self.is_trained = False
        self.parameters = None
        self.training_history = []
        self.best_parameters = None
        self.best_loss = float('inf')

        # Quantum backends
        if HAS_QUANTUM:
            try:
                self.estimator = Estimator()
                self.sampler = Sampler()
            except Exception as e:
                logger.warning(f"Failed to initialize quantum primitives: {e}")
                self.estimator = None
                self.sampler = None
        else:
            self.estimator = None
            self.sampler = None

    @abstractmethod
    def _create_variational_circuit(self) -> QuantumCircuit:
        """
        Create the variational quantum circuit.

        Returns:
            Parameterized quantum circuit
        """
        pass

    @abstractmethod
    def _encode_data(self, X: np.ndarray) -> QuantumCircuit:
        """
        Encode classical data into quantum states.

        Args:
            X: Input data

        Returns:
            Quantum circuit with encoded data
        """
        pass

    @abstractmethod
    def _compute_expectation(self, circuit: QuantumCircuit) -> float:
        """
        Compute expectation value from quantum circuit.

        Args:
            circuit: Quantum circuit to evaluate

        Returns:
            Expectation value
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_split: float = 0.2, early_stopping: bool = True) -> bool:
        """
        Train the quantum neural network.

        Args:
            X: Training features
            y: Training targets
            validation_split: Fraction of data for validation
            early_stopping: Whether to use early stopping

        Returns:
            True if training successful
        """
        if not HAS_QUANTUM:
            logger.error("Quantum computing not available")
            return False

        start_time = time.time()

        try:
            # Split data for validation
            n_samples = len(X)
            n_val = int(n_samples * validation_split)
            X_train = X[:-n_val] if n_val > 0 else X
            y_train = y[:-n_val] if n_val > 0 else y
            X_val = X[-n_val:] if n_val > 0 else X
            y_val = y[-n_val:] if n_val > 0 else y

            # Initialize parameters
            circuit = self._create_variational_circuit()
            n_params = len(circuit.parameters)
            initial_params = np.random.random(n_params) * 2 * np.pi

            logger.info(f"Starting QNN training with {n_params} parameters")

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 10

            for iteration in range(self.max_iter):
                # Objective function for optimizer
                def objective(params):
                    return self._objective_function(params, X_train, y_train)

                # Optimization step
                result = self.optimizer.optimize(n_params, objective, initial_point=initial_params)
                current_params = result.x
                train_loss = result.fun

                # Validation
                if len(X_val) > 0:
                    val_loss = self._objective_function(current_params, X_val, y_val)
                else:
                    val_loss = train_loss

                # Update best parameters
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_parameters = current_params.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Record history
                self.training_history.append({
                    'iteration': iteration,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'parameters': current_params.copy()
                })

                # Early stopping
                if early_stopping and patience_counter >= patience:
                    logger.info(f"Early stopping at iteration {iteration}")
                    break

                initial_params = current_params

            self.parameters = self.best_parameters
            self.is_trained = True

            training_time = time.time() - start_time
            logger.info(f"QNN training completed in {training_time:.2f}s")

            return True

        except Exception as e:
            logger.error(f"QNN training failed: {e}")
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_trained or self.parameters is None:
            raise ValueError("Model must be trained before making predictions")

        predictions = []
        for x in X:
            circuit = self._encode_data(x.reshape(1, -1))
            variational_circuit = self._create_variational_circuit()

            # Bind parameters
            param_dict = dict(zip(variational_circuit.parameters, self.parameters))
            full_circuit = circuit.compose(variational_circuit.assign_parameters(param_dict))

            # Compute prediction
            prediction = self._compute_expectation(full_circuit)
            predictions.append(prediction)

        return np.array(predictions)

    def _objective_function(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the objective function (loss) for given parameters.

        Args:
            params: Current parameter values
            X: Input features
            y: Target values

        Returns:
            Loss value
        """
        predictions = []
        for x in X:
            circuit = self._encode_data(x.reshape(1, -1))
            variational_circuit = self._create_variational_circuit()

            # Bind parameters
            param_dict = dict(zip(variational_circuit.parameters, params))
            full_circuit = circuit.compose(variational_circuit.assign_parameters(param_dict))

            # Compute prediction
            prediction = self._compute_expectation(full_circuit)
            predictions.append(prediction)

        predictions = np.array(predictions)

        # Mean squared error
        loss = np.mean((predictions - y.ravel()) ** 2)
        return loss

    def get_training_history(self) -> List[Dict]:
        """Get the training history."""
        return self.training_history.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'is_trained': self.is_trained,
            'optimizer': str(self.optimizer),
            'max_iter': self.max_iter,
            'training_iterations': len(self.training_history),
            'best_loss': self.best_loss if self.is_trained else None
        }

    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model

        Returns:
            True if successful
        """
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return False

        try:
            model_data = {
                'n_qubits': self.n_qubits,
                'n_layers': self.n_layers,
                'parameters': self.parameters.tolist() if self.parameters is not None else None,
                'training_history': self.training_history,
                'model_info': self.get_model_info()
            }

            import json
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)

            logger.info(f"Model saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            True if successful
        """
        try:
            import json
            with open(filepath, 'r') as f:
                model_data = json.load(f)

            self.n_qubits = model_data['n_qubits']
            self.n_layers = model_data['n_layers']
            self.parameters = np.array(model_data['parameters']) if model_data['parameters'] else None
            self.training_history = model_data['training_history']
            self.is_trained = True

            logger.info(f"Model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False