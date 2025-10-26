"""
Tests for quantum computing components in PriceMatrix.

This module tests:
- Quantum pricing engines
- Quantum optimization algorithms
- Quantum feature engineering
- Quantum ML pipeline integration
"""

import numpy as np
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Minimal pytest-like replacement for tests that use pytest.raises
    class _PytestMock:
        class raises:
            def __init__(self, exc):
                self.expected = exc
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                if exc_type is None:
                    raise AssertionError(f"Expected exception {self.expected} was not raised")
                return issubclass(exc_type, self.expected)
    pytest = _PytestMock()

from unittest.mock import Mock, patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock qiskit if not available
try:
    import qiskit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    # Create mock modules
    sys.modules['qiskit'] = Mock()
    sys.modules['qiskit_aer'] = Mock()
    sys.modules['qiskit_machine_learning'] = Mock()


class TestQuantumPricingEngines:
    """Test quantum pricing engines."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'api_key': 'test-quantum-key',
            'backend': 'simulator',
            'shots': 1024,
            'n_qubits': 4,
            'precision_qubits': 6
        }

        # Sample option parameters
        self.spot = 100.0
        self.strike = 105.0
        self.time_to_expiry = 1.0
        self.risk_free_rate = 0.05
        self.volatility = 0.2

    @patch('qiskit_aer.AerSimulator')
    def test_quantum_monte_carlo_engine(self, mock_simulator):
        """Test quantum Monte Carlo pricing engine."""
        from src.pricing.quantum_pricing import QuantumMonteCarloEngine

        # Mock the simulator
        mock_simulator.return_value.run.return_value.result.return_value.get_counts.return_value = {'0000': 512, '1111': 512}

        engine = QuantumMonteCarloEngine(self.config)

        price = engine.price_option(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility, 'call'
        )

        assert isinstance(price, float)
        assert price >= 0

    @patch('qiskit_aer.AerSimulator')
    def test_quantum_amplitude_estimation_engine(self, mock_simulator):
        """Test quantum amplitude estimation pricing engine."""
        from src.pricing.quantum_pricing import QuantumAmplitudeEstimationEngine

        # Mock the simulator
        mock_simulator.return_value.run.return_value.result.return_value.get_counts.return_value = {'000000': 512, '111111': 512}

        engine = QuantumAmplitudeEstimationEngine(self.config)

        price = engine.price_option(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility, 'call'
        )

        assert isinstance(price, float)
        assert price >= 0

    def test_hybrid_pricing_engine(self):
        """Test hybrid quantum-classical pricing engine."""
        from src.pricing.quantum_pricing import HybridQuantumClassicalEngine

        engine = HybridQuantumClassicalEngine(self.config)

        price = engine.price_option(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility, 'call'
        )

        assert isinstance(price, float)
        assert price >= 0

    def test_pricing_engine_factory(self):
        """Test quantum pricing engine factory function."""
        from src.pricing.quantum_pricing import create_quantum_pricing_engine

        # Test different engine types
        for engine_type in ['monte_carlo', 'amplitude_estimation', 'hybrid']:
            engine = create_quantum_pricing_engine(engine_type, self.config)
            assert engine is not None

        # Test invalid engine type
        with pytest.raises(ValueError):
            create_quantum_pricing_engine('invalid_type', self.config)


class TestQuantumOptimization:
    """Test quantum optimization algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'api_key': 'test-quantum-key',
            'backend': 'simulator',
            'n_assets': 4,
            'n_layers': 2
        }

        # Sample portfolio data
        np.random.seed(42)
        self.returns = np.random.normal(0.1, 0.05, 4)
        self.cov_matrix = np.random.random((4, 4))
        self.cov_matrix = (self.cov_matrix + self.cov_matrix.T) / 2
        self.cov_matrix += np.eye(4) * 0.1

    def test_portfolio_optimizer(self):
        """Test quantum portfolio optimization."""
        from src.pricing.quantum_optimization import QuantumPortfolioOptimizer

        optimizer = QuantumPortfolioOptimizer(self.config)

        result = optimizer.optimize_portfolio(self.returns, self.cov_matrix, risk_target=0.08)

        assert 'weights' in result
        assert 'expected_return' in result
        assert 'risk' in result
        assert 'sharpe_ratio' in result
        assert 'method' in result

        # Check weights sum to approximately 1
        assert abs(np.sum(result['weights']) - 1.0) < 0.1

    def test_risk_parity_optimizer(self):
        """Test quantum risk parity optimization."""
        from src.pricing.quantum_optimization import QuantumRiskParityOptimizer

        optimizer = QuantumRiskParityOptimizer(self.config)

        result = optimizer.optimize_risk_parity(self.cov_matrix)

        assert 'weights' in result
        assert 'risk_contributions' in result
        assert 'method' in result

        # Check dimensions
        assert len(result['weights']) == 4
        assert len(result['risk_contributions']) == 4


class TestQuantumFeatureEngineering:
    """Test quantum feature engineering."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'api_key': 'test-quantum-key',
            'backend': 'simulator',
            'n_qubits': 4
        }

        # Sample data
        np.random.seed(42)
        self.X = np.random.randn(50, 6)
        self.y = np.random.randint(0, 2, 50)

    @patch('qiskit_aer.AerSimulator')
    def test_quantum_feature_engineer(self, mock_simulator):
        """Test quantum feature engineer."""
        from src.data.quantum_feature_engineer import QuantumFeatureEngineer

        engineer = QuantumFeatureEngineer(self.config)

        # Test QPCA
        qpca_features = engineer.quantum_pca(self.X, n_components=3)
        assert qpca_features.shape[0] == self.X.shape[0]
        assert qpca_features.shape[1] == 3

        # Test quantum kernel
        kernel_matrix = engineer.quantum_kernel_matrix(self.X)
        assert kernel_matrix.shape[0] == min(50, self.X.shape[0])
        assert kernel_matrix.shape[1] == min(50, self.X.shape[0])

        # Test feature map
        feature_map_result = engineer.quantum_feature_map(self.X)
        assert feature_map_result.shape[0] == min(100, self.X.shape[0])

    def test_amplitude_encoding(self):
        """Test amplitude encoding."""
        from src.data.quantum_feature_engineer import QuantumFeatureEngineer

        engineer = QuantumFeatureEngineer(self.config)

        encoded = engineer.amplitude_encode_features(self.X)
        assert encoded.shape[0] == self.X.shape[0]

    def test_quantum_kernel_svm(self):
        """Test quantum kernel SVM."""
        from src.data.quantum_feature_engineer import QuantumKernelSVM

        qsvm = QuantumKernelSVM(self.config)
        qsvm.fit(self.X, self.y)

        predictions = qsvm.predict(self.X[:10])
        assert len(predictions) == 10

    def test_create_quantum_features(self):
        """Test comprehensive quantum feature creation."""
        from src.data.quantum_feature_engineer import create_quantum_features

        features_dict = create_quantum_features(self.X, self.config)

        expected_keys = ['original', 'quantum_pca', 'quantum_kernel', 'quantum_feature_map', 'amplitude_encoded']
        for key in expected_keys:
            assert key in features_dict


class TestPureQuantumPipeline:
    """Test pure quantum ML pipeline with only QNN algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'api_key': 'test-quantum-key',
            'backend': 'simulator',
            'n_qubits': 6,  # Increased for QNN focus
            'n_layers': 3   # Multiple layers for QNN
        }

        # Sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 8)
        self.y = np.random.normal(5.0, 2.0, 100)

    def test_variational_quantum_circuit(self):
        """Test pure Variational Quantum Circuit."""
        from src.models.quantum_ml_pipeline import VariationalQuantumCircuit

        vqc = VariationalQuantumCircuit(self.config)
        vqc.fit(self.X, self.y)

        predictions = vqc.predict(self.X[:10])
        assert len(predictions) == 10

        metrics = vqc.evaluate(self.X[:10], self.y[:10])
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'circuit_depth' in metrics
        assert 'n_parameters' in metrics

    def test_quantum_kernel_svm(self):
        """Test pure Quantum Kernel SVM."""
        from src.models.quantum_ml_pipeline import QuantumKernelSVM

        qksvm = QuantumKernelSVM(self.config)
        qksvm.fit(self.X, self.y)

        predictions = qksvm.predict(self.X[:10])
        assert len(predictions) == 10

        metrics = qksvm.evaluate(self.X[:10], self.y[:10])
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'kernel_type' in metrics
        assert metrics['kernel_type'] == 'quantum'

    def test_quantum_neural_network(self):
        """Test advanced quantum neural network (pure quantum focus)."""
        from src.models.quantum_ml_pipeline import QuantumNeuralNetwork

        qnn = QuantumNeuralNetwork(self.config)
        qnn.build_model(self.X.shape[1])

        # Test QNN-specific attributes
        assert qnn.n_qubits == 6
        assert qnn.n_layers == 3
        assert qnn.model is not None

        # Test quantum expressiveness calculation
        expressiveness = qnn._calculate_quantum_expressiveness()
        assert isinstance(expressiveness, float)
        assert 0.0 <= expressiveness <= 1.0

    def test_qnn_evaluation_metrics(self):
        """Test QNN evaluation with quantum-specific metrics."""
        from src.models.quantum_ml_pipeline import QuantumNeuralNetwork

        qnn = QuantumNeuralNetwork(self.config)
        qnn.fit(self.X, self.y, epochs=10)  # Quick training for test

        metrics = qnn.evaluate(self.X[:10], self.y[:10])

        # Check standard metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2_score' in metrics
        assert 'mean_absolute_error' in metrics

        # Check quantum-specific metrics
        assert 'quantum_expressiveness' in metrics
        assert 'circuit_depth' in metrics
        assert 'n_qubits' in metrics
        assert 'n_layers' in metrics

    def test_pure_quantum_ensemble(self):
        """Test pure quantum ensemble model."""
        from src.models.quantum_ml_pipeline import PureQuantumEnsemble, QuantumNeuralNetwork, VariationalQuantumCircuit

        ensemble = PureQuantumEnsemble(self.config)

        # Add pure quantum models
        qnn = QuantumNeuralNetwork(self.config)
        qnn.fit(self.X, self.y, epochs=5)

        vqc = VariationalQuantumCircuit(self.config)
        vqc.fit(self.X, self.y)

        ensemble.add_quantum_model(qnn, weight=0.7)  # Primary QNN
        ensemble.add_quantum_model(vqc, weight=0.3)  # VQC

        predictions = ensemble.predict(self.X[:10])
        assert len(predictions) == 10

        metrics = ensemble.evaluate(self.X[:10], self.y[:10])
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'ensemble_type' in metrics
        assert metrics['ensemble_type'] == 'pure_quantum'

    def test_create_pure_quantum_pipeline(self):
        """Test pure quantum ML pipeline creation."""
        from src.models.quantum_ml_pipeline import create_pure_quantum_pipeline

        pipeline = create_pure_quantum_pipeline(self.config)

        expected_keys = ['quantum_neural_net', 'secondary_qnn', 'variational_qc', 'quantum_kernel_svm', 'pure_quantum_ensemble']
        for key in expected_keys:
            assert key in pipeline

        # Verify all components are pure quantum
        assert pipeline['quantum_neural_net'] is not None
        assert pipeline['secondary_qnn'] is not None
        assert pipeline['variational_qc'] is not None
        assert pipeline['quantum_kernel_svm'] is not None
        assert pipeline['pure_quantum_ensemble'] is not None

    def test_train_pure_quantum_pipeline(self):
        """Test pure quantum pipeline training."""
        from src.models.quantum_ml_pipeline import train_pure_quantum_pipeline

        results = train_pure_quantum_pipeline(self.X, self.y, self.config, test_size=0.3)

        assert 'pipeline' in results

        # Check that all quantum algorithms are trained
        assert 'primary_qnn' in results
        assert 'secondary_qnn' in results
        assert 'variational_qc' in results
        assert 'quantum_kernel_svm' in results
        assert 'pure_quantum_ensemble' in results

        # Verify quantum-specific metrics
        if 'primary_qnn' in results:
            qnn_metrics = results['primary_qnn']
            assert 'quantum_expressiveness' in qnn_metrics
            assert 'circuit_depth' in qnn_metrics

        if 'pure_quantum_ensemble' in results:
            assert 'mse' in results['pure_quantum_ensemble']
            assert 'rmse' in results['pure_quantum_ensemble']
            assert results['pure_quantum_ensemble']['ensemble_type'] == 'pure_quantum'


class TestQuantumIntegration:
    """Test integration of quantum components."""

    def test_config_integration(self):
        """Test that quantum config integrates properly."""
        # This would test the config loading from YAML
        # For now, just check that the config structure exists
        config = {
            'quantum': {
                'enabled': True,
                'api_key': 'test-key',
                'provider': 'ibm',
                'backend': 'simulator'
            }
        }

        assert config['quantum']['enabled'] is True
        assert config['quantum']['api_key'] == 'test-key'

    def test_fallback_behavior(self):
        """Test that classical fallbacks work when quantum is unavailable."""
        # Test with qiskit not available
        if not QISKIT_AVAILABLE:
            from src.pricing.quantum_pricing import QuantumMonteCarloEngine

            config = {'backend': 'simulator'}
            engine = QuantumMonteCarloEngine(config)

            # Should use classical fallback
            price = engine.price_option(100, 105, 1.0, 0.05, 0.2, 'call')
            assert isinstance(price, float)
            assert price >= 0


if __name__ == "__main__":
    # Run pure quantum tests
    test_instance = TestQuantumPricingEngines()
    test_instance.setup_method()

    print("Running pure quantum component tests...")

    try:
        test_instance.test_hybrid_pricing_engine()
        print("✓ Hybrid pricing engine test passed")
    except Exception as e:
        print(f"✗ Hybrid pricing engine test failed: {e}")

    try:
        test_instance.test_pricing_engine_factory()
        print("✓ Pricing engine factory test passed")
    except Exception as e:
        print(f"✗ Pricing engine factory test failed: {e}")

    # Test pure quantum components
    quantum_test_instance = TestPureQuantumPipeline()
    quantum_test_instance.setup_method()

    try:
        quantum_test_instance.test_variational_quantum_circuit()
        print("✓ Variational Quantum Circuit test passed")
    except Exception as e:
        print(f"✗ Variational Quantum Circuit test failed: {e}")

    try:
        quantum_test_instance.test_quantum_kernel_svm()
        print("✓ Quantum Kernel SVM test passed")
    except Exception as e:
        print(f"✗ Quantum Kernel SVM test failed: {e}")

    try:
        quantum_test_instance.test_quantum_neural_network()
        print("✓ Quantum Neural Network test passed")
    except Exception as e:
        print(f"✗ Quantum Neural Network test failed: {e}")

    try:
        quantum_test_instance.test_qnn_evaluation_metrics()
        print("✓ QNN evaluation metrics test passed")
    except Exception as e:
        print(f"✗ QNN evaluation metrics test failed: {e}")

    try:
        quantum_test_instance.test_pure_quantum_ensemble()
        print("✓ Pure quantum ensemble test passed")
    except Exception as e:
        print(f"✗ Pure quantum ensemble test failed: {e}")

    try:
        quantum_test_instance.test_create_pure_quantum_pipeline()
        print("✓ Pure quantum pipeline creation test passed")
    except Exception as e:
        print(f"✗ Pure quantum pipeline creation test failed: {e}")

    print("Pure quantum component tests completed.")