"""
Tests for machine learning models.

This module contains unit tests for the pricing models including
Random Forest, Neural Networks, and Ensemble methods.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_regression

from src.models.base_model import ScikitLearnPricingModel, NeuralNetworkPricingModel
from src.models.random_forest import RandomForestPricingModel, ExtraTreesPricingModel
from src.models.neural_network import SwaptionPricingNN, OptionPricingNN
from src.models.ensemble import WeightedEnsemble, BootstrapEnsemble


class TestBaseModels(unittest.TestCase):
    """Test base model classes."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        self.y = pd.Series(y, name='price')

    def test_scikit_learn_pricing_model(self):
        """Test ScikitLearnPricingModel base class."""
        from sklearn.linear_model import LinearRegression

        model = ScikitLearnPricingModel(LinearRegression(), name="TestModel")

        # Test training
        model.train(self.X, self.y)
        self.assertTrue(model.is_trained)

        # Test prediction
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

        # Test evaluation
        metrics = model.evaluate(self.X, self.y)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)

        # Test feature importance (should be None for LinearRegression)
        importance = model.get_feature_importance()
        self.assertIsNone(importance)

    def test_neural_network_pricing_model(self):
        """Test NeuralNetworkPricingModel base class."""
        # Mock TensorFlow/Keras
        with patch('tensorflow.keras.models.Model') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            # Mock model methods
            mock_model.fit.return_value = {'loss': [0.5, 0.3], 'val_loss': [0.6, 0.4]}
            mock_model.predict.return_value = np.random.randn(len(self.X), 1)

            model = NeuralNetworkPricingModel(name="TestNN")
            model.model = mock_model

            # Test training
            model.train(self.X, self.y, epochs=2, verbose=0)
            self.assertTrue(model.is_trained)

            # Test prediction
            predictions = model.predict(self.X)
            self.assertEqual(len(predictions), len(self.X))

            # Test feature importance (should be None for NN)
            importance = model.get_feature_importance()
            self.assertIsNone(importance)


class TestRandomForestModels(unittest.TestCase):
    """Test Random Forest based models."""

    def setUp(self):
        """Set up test fixtures."""
        X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        self.y = pd.Series(y, name='price')

    def test_random_forest_pricing_model(self):
        """Test RandomForestPricingModel."""
        model = RandomForestPricingModel(n_estimators=10, random_state=42)

        # Test training
        model.train(self.X, self.y)
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.feature_names_)

        # Test prediction
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

        # Test feature importance
        importance = model.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), self.X.shape[1])

        # Test tree statistics
        stats = model.get_tree_statistics()
        self.assertIn('n_trees', stats)
        self.assertIn('mean_depth', stats)
        self.assertEqual(stats['n_trees'], 10)

        # Test uncertainty estimation
        pred_mean, pred_std = model.predict_with_uncertainty(self.X)
        self.assertEqual(len(pred_mean), len(self.X))
        self.assertEqual(len(pred_std), len(self.X))
        self.assertTrue(np.all(pred_std >= 0))

    def test_extra_trees_pricing_model(self):
        """Test ExtraTreesPricingModel."""
        model = ExtraTreesPricingModel(n_estimators=10, random_state=42)

        # Test training
        model.train(self.X, self.y)
        self.assertTrue(model.is_trained)

        # Test prediction
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

        # Test feature importance
        importance = model.get_feature_importance()
        self.assertIsNotNone(importance)

    @patch('sklearn.model_selection.GridSearchCV')
    def test_hyperparameter_tuning(self, mock_grid_search):
        """Test hyperparameter tuning functionality."""
        # Mock GridSearchCV
        mock_gs = MagicMock()
        mock_gs.best_params_ = {'n_estimators': 50, 'max_depth': 10}
        mock_gs.best_score_ = -0.5
        mock_gs.cv_results_ = {'mean_test_score': [-0.5]}
        mock_grid_search.return_value = mock_gs

        model = RandomForestPricingModel()

        # Test tuning
        results = model.tune_hyperparameters(self.X, self.y, method='grid')

        self.assertIn('best_params', results)
        self.assertIn('best_score', results)
        self.assertEqual(results['best_params']['n_estimators'], 50)

        # Check that model was updated
        self.assertEqual(model.hyperparameters['n_estimators'], 50)


class TestNeuralNetworkModels(unittest.TestCase):
    """Test Neural Network based models."""

    def setUp(self):
        """Set up test fixtures."""
        X, y = make_regression(n_samples=100, n_features=8, noise=0.1, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
        self.y = pd.Series(y, name='price')

    @patch('tensorflow.keras.layers.Input')
    @patch('tensorflow.keras.layers.Dense')
    @patch('tensorflow.keras.layers.BatchNormalization')
    @patch('tensorflow.keras.layers.Dropout')
    @patch('tensorflow.keras.models.Model')
    @patch('tensorflow.keras.optimizers.Adam')
    @patch('tensorflow.keras.callbacks.EarlyStopping')
    @patch('tensorflow.keras.callbacks.ReduceLROnPlateau')
    def test_swaption_pricing_nn(self, mock_reduce_lr, mock_early_stop,
                                mock_adam, mock_model, mock_dropout,
                                mock_batch_norm, mock_dense, mock_input):
        """Test SwaptionPricingNN."""
        # Mock TensorFlow components
        mock_input_layer = MagicMock()
        mock_input.return_value = mock_input_layer

        mock_dense_layer = MagicMock()
        mock_dense.return_value = mock_dense_layer

        mock_batch_norm_layer = MagicMock()
        mock_batch_norm.return_value = mock_batch_norm_layer

        mock_dropout_layer = MagicMock()
        mock_dropout.return_value = mock_dropout_layer

        mock_keras_model = MagicMock()
        mock_model.return_value = mock_keras_model

        # Mock model methods
        mock_keras_model.fit.return_value = MagicMock(history={
            'loss': [0.5, 0.3, 0.2],
            'val_loss': [0.6, 0.4, 0.3],
            'mae': [0.4, 0.3, 0.2],
            'val_mae': [0.5, 0.4, 0.3]
        })
        mock_keras_model.predict.return_value = np.random.randn(len(self.X), 1).flatten()

        model = SwaptionPricingNN(input_dim=self.X.shape[1])

        # Test training
        model.train(self.X, self.y, epochs=3, verbose=0)
        self.assertTrue(model.is_trained)

        # Test prediction
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

        # Test uncertainty estimation
        pred_mean, pred_std = model.predict_with_uncertainty(self.X)
        self.assertEqual(len(pred_mean), len(self.X))
        self.assertEqual(len(pred_std), len(self.X))

    def test_option_pricing_nn(self):
        """Test OptionPricingNN initialization."""
        # Similar mocking would be needed for full test
        # For now, just test that it can be instantiated
        try:
            model = OptionPricingNN(input_dim=10)
            self.assertIsInstance(model, OptionPricingNN)
        except ImportError:
            # Skip if TensorFlow not available
            self.skipTest("TensorFlow not available")


class TestEnsembleModels(unittest.TestCase):
    """Test ensemble model classes."""

    def setUp(self):
        """Set up test fixtures."""
        X, y = make_regression(n_samples=150, n_features=5, noise=0.1, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        self.y = pd.Series(y, name='price')

        # Create base models
        self.base_models = [
            RandomForestPricingModel(n_estimators=5, random_state=42),
            RandomForestPricingModel(n_estimators=5, random_state=43)
        ]

    def test_weighted_ensemble(self):
        """Test WeightedEnsemble."""
        ensemble = WeightedEnsemble(self.base_models)

        # Test training
        ensemble.train(self.X, self.y)
        self.assertTrue(ensemble.is_trained)
        self.assertIsNotNone(ensemble.weights)
        self.assertAlmostEqual(np.sum(ensemble.weights), 1.0, places=5)

        # Test prediction
        predictions = ensemble.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

        # Test feature importance
        importance = ensemble.get_feature_importance()
        self.assertIsNotNone(importance)

    def test_bootstrap_ensemble(self):
        """Test BootstrapEnsemble."""
        ensemble = BootstrapEnsemble(
            RandomForestPricingModel,
            n_estimators=3,
            bootstrap_fraction=0.8,
            model_params={'n_estimators': 5, 'random_state': 42}
        )

        # Test training
        ensemble.train(self.X, self.y)
        self.assertTrue(ensemble.is_trained)
        self.assertEqual(len(ensemble.base_models), 3)

        # Test prediction
        predictions = ensemble.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

        # Test uncertainty estimation
        pred_mean, pred_std = ensemble.predict_with_uncertainty(self.X)
        self.assertEqual(len(pred_mean), len(self.X))
        self.assertEqual(len(pred_std), len(self.X))

    def test_ensemble_feature_importance(self):
        """Test that ensemble properly aggregates feature importance."""
        ensemble = WeightedEnsemble(self.base_models)
        ensemble.train(self.X, self.y)

        importance = ensemble.get_feature_importance()

        # Should have importance for each feature
        self.assertEqual(len(importance), self.X.shape[1])

        # All importances should be non-negative
        for imp in importance.values():
            self.assertGreaterEqual(imp, 0)


class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading."""

    def setUp(self):
        """Set up test fixtures."""
        X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(3)])
        self.y = pd.Series(y, name='price')

    @patch('joblib.dump')
    @patch('joblib.load')
    def test_model_save_load(self, mock_load, mock_dump):
        """Test model saving and loading."""
        model = RandomForestPricingModel(n_estimators=5)
        model.train(self.X, self.y)

        # Mock the save/load operations
        mock_load.return_value = {'model': model, 'name': model.name, 'training_date': model.training_date}

        # Test save
        model.save_model('test_model.pkl')
        mock_dump.assert_called_once()

        # Test load
        loaded_model = RandomForestPricingModel.load_model('test_model.pkl')
        self.assertIsInstance(loaded_model, RandomForestPricingModel)


class TestModelValidation(unittest.TestCase):
    """Test model validation and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(3)])
        self.y = pd.Series(y, name='price')

    def test_untrained_model_prediction(self):
        """Test that untrained model raises error on prediction."""
        model = RandomForestPricingModel()

        with self.assertRaises(ValueError):
            model.predict(self.X)

    def test_untrained_model_evaluation(self):
        """Test that untrained model raises error on evaluation."""
        model = RandomForestPricingModel()

        with self.assertRaises(ValueError):
            model.evaluate(self.X, self.y)

    def test_empty_feature_importance(self):
        """Test feature importance on untrained model."""
        model = RandomForestPricingModel()

        importance = model.get_feature_importance()
        self.assertIsNone(importance)


if __name__ == '__main__':
    unittest.main()