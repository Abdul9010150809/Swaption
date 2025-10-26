#!/usr/bin/env python3
"""
Advanced Quantum Neural Network Components for Financial Pricing
Implements complete QNN pipeline with data encoding, ansatz circuits, and training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
import time
from abc import ABC, abstractmethod

# Quantum imports with fallbacks
try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2, TwoLocal
    from qiskit.primitives import Sampler, Estimator
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
    from qiskit import transpile
    from qiskit_algorithms.optimizers import SPSA, COBYLA, ADAM
    from qiskit_algorithms import VQEResult

    # Qiskit ML components
    try:
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.algorithms.regressors import VQR
        QISKIT_ML_AVAILABLE = True
    except ImportError:
        QISKIT_ML_AVAILABLE = False
        logging.warning("Qiskit ML not available - using custom QNN implementations")

    HAS_QUANTUM = True
except ImportError as e:
    logging.error(f"Quantum computing libraries not available: {e}")
    HAS_QUANTUM = False
    QISKIT_ML_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Quantum Data Encoding ---
class QuantumDataEncoder(ABC):
    """Abstract base class for quantum data encoding strategies"""

    def __init__(self, n_qubits: int, n_features: int):
        self.n_qubits = n_qubits
        self.n_features = n_features

    @abstractmethod
    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode classical data into quantum state"""
        pass

class AngleEncoding(QuantumDataEncoder):
    """Angle encoding: encode features as rotation angles"""

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode data using RY rotations"""
        circuit = QuantumCircuit(self.n_qubits)

        # Normalize data to [0, 2π]
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        angles = normalized_data * 2 * np.pi

        for i in range(min(len(angles), self.n_qubits)):
            circuit.ry(angles[i], i)

        return circuit

class AmplitudeEncoding(QuantumDataEncoder):
    """Amplitude encoding: encode data as quantum state amplitudes"""

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode data using amplitude encoding"""
        circuit = QuantumCircuit(self.n_qubits)

        # Normalize data
        normalized_data = data / np.linalg.norm(data)

        # Pad to power of 2 if necessary
        n_amplitudes = 2 ** self.n_qubits
        if len(normalized_data) < n_amplitudes:
            padded_data = np.zeros(n_amplitudes)
            padded_data[:len(normalized_data)] = normalized_data
            normalized_data = padded_data

        # Create state preparation circuit
        state = Statevector(normalized_data[:n_amplitudes])
        circuit.prepare_state(state, list(range(self.n_qubits)))

        return circuit

class ZZFeatureEncoding(QuantumDataEncoder):
    """ZZ feature map encoding for quantum kernels"""

    def __init__(self, n_qubits: int, n_features: int, reps: int = 1):
        super().__init__(n_qubits, n_features)
        self.reps = reps

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Create ZZ feature map circuit"""
        feature_map = ZZFeatureMap(feature_dimension=self.n_features, reps=self.reps)
        circuit = QuantumCircuit(self.n_qubits)

        # Bind parameters
        param_dict = {}
        for i, param in enumerate(feature_map.parameters):
            if i < len(data):
                param_dict[param] = data[i]

        circuit.compose(feature_map.assign_parameters(param_dict), inplace=True)
        return circuit

# --- Parameterized Quantum Circuits (Ansatz) ---
class QuantumAnsatz(ABC):
    """Abstract base class for quantum ansatz circuits"""

    def __init__(self, n_qubits: int, n_layers: int = 1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.parameters = []

    @abstractmethod
    def create_circuit(self) -> QuantumCircuit:
        """Create the parameterized quantum circuit"""
        pass

    def get_parameter_count(self) -> int:
        """Get number of parameters in the ansatz"""
        return len(self.parameters)

class RealAmplitudesAnsatz(QuantumAnsatz):
    """Real Amplitudes ansatz for variational circuits"""

    def create_circuit(self) -> QuantumCircuit:
        """Create Real Amplitudes ansatz circuit"""
        ansatz = RealAmplitudes(num_qubits=self.n_qubits, reps=self.n_layers)
        self.parameters = list(ansatz.parameters)
        return ansatz

class EfficientSU2Ansatz(QuantumAnsatz):
    """Efficient SU(2) ansatz for quantum circuits"""

    def create_circuit(self) -> QuantumCircuit:
        """Create EfficientSU2 ansatz circuit"""
        ansatz = EfficientSU2(num_qubits=self.n_qubits, reps=self.n_layers)
        self.parameters = list(ansatz.parameters)
        return ansatz

class TwoLocalAnsatz(QuantumAnsatz):
    """Two-local ansatz with customizable gates"""

    def __init__(self, n_qubits: int, n_layers: int = 1, rotation_gates: List[str] = None,
                 entanglement_gates: List[str] = None):
        super().__init__(n_qubits, n_layers)
        self.rotation_gates = rotation_gates or ['ry', 'rz']
        self.entanglement_gates = entanglement_gates or ['cz']

    def create_circuit(self) -> QuantumCircuit:
        """Create Two-Local ansatz circuit"""
        ansatz = TwoLocal(num_qubits=self.n_qubits, rotation_blocks=self.rotation_gates,
                         entanglement_blocks=self.entanglement_gates, reps=self.n_layers)
        self.parameters = list(ansatz.parameters)
        return ansatz

# --- Quantum Neural Networks ---
class QuantumNeuralNetwork(ABC):
    """Abstract base class for quantum neural networks"""

    def __init__(self, encoder: QuantumDataEncoder, ansatz: QuantumAnsatz):
        self.encoder = encoder
        self.ansatz = ansatz
        self.is_trained = False
        self.training_history = []

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train the QNN"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

class EstimatorQNNRegressor(QuantumNeuralNetwork):
    """QNN Regressor using Estimator primitive"""

    def __init__(self, encoder: QuantumDataEncoder, ansatz: QuantumAnsatz,
                 optimizer=None, max_iter: int = 100):
        super().__init__(encoder, ansatz)
        self.optimizer = optimizer or SPSA(maxiter=max_iter)
        self.optimal_parameters = None
        self.estimator = Estimator()

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train QNN using variational optimization"""
        if not HAS_QUANTUM:
            logger.error("Quantum computing not available")
            return False

        start_time = time.time()

        try:
            # Define objective function
            def objective_function(params):
                predictions = []
                for x in X:
                    pred = self._predict_single(x, params)
                    predictions.append(pred)
                predictions = np.array(predictions)
                mse = np.mean((predictions - y.ravel()) ** 2)
                return mse

            # Initial parameters
            initial_params = np.random.random(self.ansatz.get_parameter_count())

            # Optimize using minimize method for compatibility
            from scipy.optimize import minimize
            result = minimize(
                objective_function,
                initial_params,
                method='SLSQP',
                options={'maxiter': 50, 'ftol': 1e-6}
            )

            self.optimal_parameters = result.x
            self.is_trained = True

            training_time = time.time() - start_time
            self.training_history.append({
                'method': 'estimator_qnn',
                'time': training_time,
                'samples': len(X),
                'final_loss': result.fun
            })

            logger.info(f"QNN training completed in {training_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"QNN training failed: {e}")
            return False

    def _predict_single(self, x: np.ndarray, params: np.ndarray) -> float:
        """Predict single sample"""
        # Create full circuit
        circuit = QuantumCircuit(self.encoder.n_qubits)

        # Encode data
        encoding_circuit = self.encoder.encode(x)
        circuit.compose(encoding_circuit, inplace=True)

        # Apply ansatz
        ansatz_circuit = self.ansatz.create_circuit()
        # Ensure we have the right number of parameters
        n_params_needed = len(ansatz_circuit.parameters)
        if len(params) != n_params_needed:
            logger.warning(f"Parameter mismatch: got {len(params)}, need {n_params_needed}")
            # Pad or truncate parameters as needed
            if len(params) < n_params_needed:
                params = np.pad(params, (0, n_params_needed - len(params)), mode='constant')
            else:
                params = params[:n_params_needed]

        param_dict = dict(zip(ansatz_circuit.parameters, params))
        ansatz_circuit = ansatz_circuit.assign_parameters(param_dict)
        circuit.compose(ansatz_circuit, inplace=True)

        # Measure expectation value
        observable = SparsePauliOp("Z" * self.encoder.n_qubits)
        job = self.estimator.run(circuit, observable)
        result = job.result()
        expectation = result.values[0]

        # Convert to prediction (scale to reasonable range)
        prediction = (expectation + 1) / 2  # Map [-1, 1] to [0, 1]
        return prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for multiple samples"""
        if not self.is_trained or self.optimal_parameters is None:
            raise ValueError("Model not trained")

        predictions = []
        for x in X:
            pred = self._predict_single(x, self.optimal_parameters)
            predictions.append(pred)

        return np.array(predictions)

class VQRRegressor(QuantumNeuralNetwork):
    """Variational Quantum Regressor using Qiskit ML"""

    def __init__(self, encoder: QuantumDataEncoder, ansatz: QuantumAnsatz,
                 optimizer=None, max_iter: int = 100):
        super().__init__(encoder, ansatz)
        self.optimizer = optimizer or SPSA(maxiter=max_iter)
        self.vqr = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train VQR model"""
        if not QISKIT_ML_AVAILABLE:
            logger.error("Qiskit ML not available for VQR")
            return False

        if not HAS_QUANTUM:
            logger.error("Quantum computing not available")
            return False

        start_time = time.time()

        try:
            # Create feature map and ansatz
            feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=1)
            ansatz = RealAmplitudes(num_qubits=min(4, X.shape[1]), reps=1)

            # Create QNN
            qnn = EstimatorQNN(
                circuit=QuantumCircuit(min(4, X.shape[1])).compose(feature_map).compose(ansatz),
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters,
                estimator=Estimator()
            )

            # Create VQR
            self.vqr = VQR(
                neural_network=qnn,
                optimizer=self.optimizer,
                initial_point=np.random.random(ansatz.num_parameters)
            )

            # Train
            self.vqr.fit(X, y.ravel())
            self.is_trained = True

            training_time = time.time() - start_time
            self.training_history.append({
                'method': 'vqr',
                'time': training_time,
                'samples': len(X)
            })

            logger.info(f"VQR training completed in {training_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"VQR training failed: {e}")
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained or self.vqr is None:
            raise ValueError("Model not trained")

        return self.vqr.predict(X)

# --- Quantum Feature Maps ---
class QuantumFeatureMap(ABC):
    """Abstract base class for quantum feature maps"""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    @abstractmethod
    def map(self, x: np.ndarray, y: np.ndarray = None) -> QuantumCircuit:
        """Create feature map circuit"""
        pass

class ZZFeatureMapCircuit(QuantumFeatureMap):
    """ZZ feature map for kernel methods"""

    def __init__(self, n_qubits: int, reps: int = 1):
        super().__init__(n_qubits)
        self.reps = reps

    def map(self, x: np.ndarray, y: np.ndarray = None) -> QuantumCircuit:
        """Create ZZ feature map"""
        feature_map = ZZFeatureMap(feature_dimension=len(x), reps=self.reps)

        # Bind parameters
        param_dict = dict(zip(feature_map.parameters, x))
        circuit = feature_map.assign_parameters(param_dict)

        return circuit

# --- Quantum Training Utilities ---
class QuantumTrainer:
    """Utilities for quantum model training"""

    @staticmethod
    def train_with_early_stopping(model: QuantumNeuralNetwork, X: np.ndarray, y: np.ndarray,
                                patience: int = 10, min_delta: float = 1e-4) -> Dict:
        """Train with early stopping"""
        best_loss = float('inf')
        patience_counter = 0
        best_params = None

        history = []

        # Custom training loop with early stopping
        for epoch in range(100):  # Max epochs
            try:
                success = model.fit(X, y)
                if not success:
                    break

                # Calculate current loss
                predictions = model.predict(X)
                current_loss = np.mean((predictions - y.ravel()) ** 2)

                history.append({
                    'epoch': epoch,
                    'loss': current_loss
                })

                # Check early stopping
                if current_loss < best_loss - min_delta:
                    best_loss = current_loss
                    best_params = model.optimal_parameters.copy() if hasattr(model, 'optimal_parameters') else None
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            except Exception as e:
                logger.error(f"Training failed at epoch {epoch}: {e}")
                break

        return {
            'history': history,
            'best_loss': best_loss,
            'best_params': best_params,
            'epochs_trained': len(history)
        }

# --- Quantum Measurement Utilities ---
class QuantumMeasurement:
    """Utilities for quantum measurements"""

    @staticmethod
    def measure_expectation(circuit: QuantumCircuit, observable: SparsePauliOp,
                          shots: int = 8192) -> float:
        """Measure expectation value"""
        if not HAS_QUANTUM:
            return 0.0

        try:
            estimator = Estimator()
            job = estimator.run(circuit, observable)
            result = job.result()
            return result.values[0]
        except Exception as e:
            logger.error(f"Expectation measurement failed: {e}")
            return 0.0

    @staticmethod
    def measure_statevector(circuit: QuantumCircuit) -> np.ndarray:
        """Get statevector measurement"""
        if not HAS_QUANTUM:
            return np.array([])

        try:
            statevector = Statevector.from_instruction(circuit)
            return statevector.data
        except Exception as e:
            logger.error(f"Statevector measurement failed: {e}")
            return np.array([])

    @staticmethod
    def sample_circuit(circuit: QuantumCircuit, shots: int = 8192) -> Dict:
        """Sample from quantum circuit"""
        if not HAS_QUANTUM:
            return {}

        try:
            sampler = Sampler()
            job = sampler.run(circuit, shots=shots)
            result = job.result()
            return result.quasi_dists[0]
        except Exception as e:
            logger.error(f"Circuit sampling failed: {e}")
            return {}

# --- Main QNN Factory ---
class QuantumNNFactory:
    """Factory for creating quantum neural network components"""

    @staticmethod
    def create_encoder(encoding_type: str, n_qubits: int, n_features: int) -> QuantumDataEncoder:
        """Create data encoder"""
        encoders = {
            'angle': AngleEncoding,
            'amplitude': AmplitudeEncoding,
            'zz_feature': ZZFeatureEncoding
        }

        if encoding_type not in encoders:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        return encoders[encoding_type](n_qubits, n_features)

    @staticmethod
    def create_ansatz(ansatz_type: str, n_qubits: int, n_layers: int = 1) -> QuantumAnsatz:
        """Create ansatz circuit"""
        ansatz_classes = {
            'real_amplitudes': RealAmplitudesAnsatz,
            'efficient_su2': EfficientSU2Ansatz,
            'two_local': TwoLocalAnsatz
        }

        if ansatz_type not in ansatz_classes:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")

        return ansatz_classes[ansatz_type](n_qubits, n_layers)

    @staticmethod
    def create_qnn(qnn_type: str, encoder: QuantumDataEncoder, ansatz: QuantumAnsatz) -> QuantumNeuralNetwork:
        """Create quantum neural network"""
        qnn_classes = {
            'estimator_qnn': EstimatorQNNRegressor,
            'vqr': VQRRegressor
        }

        if qnn_type not in qnn_classes:
            raise ValueError(f"Unknown QNN type: {qnn_type}")

        return qnn_classes[qnn_type](encoder, ansatz)

# --- Example Usage and Testing ---
def test_qnn_components():
    """Test QNN components"""
    if not HAS_QUANTUM:
        logger.warning("Quantum computing not available - skipping tests")
        return

    logger.info("Testing QNN components...")

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(20, 4)
    y = np.random.randn(20, 1)

    try:
        # Test encoder
        encoder = AngleEncoding(n_qubits=4, n_features=4)
        circuit = encoder.encode(X[0])
        logger.info("✅ Angle encoding test passed")

        # Test ansatz
        ansatz = RealAmplitudesAnsatz(n_qubits=4, n_layers=1)
        ansatz_circuit = ansatz.create_circuit()
        logger.info(f"✅ Ansatz test passed - {ansatz.get_parameter_count()} parameters")

        # Test QNN
        qnn = EstimatorQNNRegressor(encoder, ansatz)
        success = qnn.fit(X, y)
        if success:
            predictions = qnn.predict(X[:5])
            logger.info(f"✅ QNN training test passed - predictions: {predictions[:3]}")
        else:
            logger.warning("⚠️ QNN training test failed")

    except Exception as e:
        logger.error(f"❌ QNN component test failed: {e}")

if __name__ == "__main__":
    test_qnn_components()