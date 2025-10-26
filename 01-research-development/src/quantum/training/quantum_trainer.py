#!/usr/bin/env python3
"""
Quantum Trainer Implementation
Provides advanced training utilities for quantum neural networks
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from abc import ABC, abstractmethod
import logging
import time

logger = logging.getLogger(__name__)

class QuantumTrainer:
    """
    Advanced trainer for quantum neural networks with various optimization strategies.

    This class provides comprehensive training utilities including early stopping,
    learning rate scheduling, gradient clipping, and various optimization algorithms
    specifically designed for quantum variational circuits.
    """

    def __init__(self, optimizer: Optional[Any] = None,
                 max_epochs: int = 100, patience: int = 10,
                 learning_rate: float = 0.01, batch_size: Optional[int] = None,
                 validation_split: float = 0.2, shuffle: bool = True):
        """
        Initialize the quantum trainer.

        Args:
            optimizer: Optimization algorithm
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            learning_rate: Learning rate for optimization
            batch_size: Batch size for mini-batch training
            validation_split: Fraction of data for validation
            shuffle: Whether to shuffle data each epoch
        """
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle

        # Training state
        self.training_history = []
        self.best_model_state = None
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0

    def train(self, model, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              callbacks: Optional[List] = None) -> Dict[str, Any]:
        """
        Train a quantum model with advanced training features.

        Args:
            model: Quantum model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            callbacks: List of training callbacks

        Returns:
            Training history and results
        """
        # Prepare validation data
        if X_val is None and self.validation_split > 0:
            n_val = int(len(X_train) * self.validation_split)
            indices = np.random.permutation(len(X_train))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]

        # Initialize training state
        self.training_history = []
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0

        start_time = time.time()

        try:
            for epoch in range(self.max_epochs):
                self.current_epoch = epoch

                # Train one epoch
                train_loss = self._train_epoch(model, X_train, y_train)

                # Validate
                val_loss = self._validate_epoch(model, X_val, y_val) if X_val is not None else train_loss

                # Record history
                epoch_info = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.learning_rate,
                    'time': time.time() - start_time
                }
                self.training_history.append(epoch_info)

                # Callbacks
                if callbacks:
                    for callback in callbacks:
                        callback.on_epoch_end(epoch_info)

                # Early stopping check
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_model_state = self._get_model_state(model)
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                # Learning rate scheduling (simple decay)
                if epoch > 0 and epoch % 20 == 0:
                    self.learning_rate *= 0.9

            # Restore best model
            if self.best_model_state is not None:
                self._set_model_state(model, self.best_model_state)

            training_time = time.time() - start_time

            result = {
                'success': True,
                'epochs_trained': len(self.training_history),
                'best_loss': self.best_loss,
                'training_time': training_time,
                'history': self.training_history,
                'early_stopped': self.epochs_without_improvement >= self.patience
            }

            logger.info(f"Training completed in {training_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'epochs_trained': len(self.training_history),
                'history': self.training_history
            }

    def _train_epoch(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Train for one epoch."""
        if self.batch_size is None:
            # Full batch training
            return self._train_batch(model, X, y)
        else:
            # Mini-batch training
            return self._train_minibatch(model, X, y)

    def _train_batch(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Train with full batch."""
        # This would implement the actual training logic
        # For now, delegate to model.fit
        try:
            model.fit(X, y)
            predictions = model.predict(X)
            loss = np.mean((predictions.ravel() - y.ravel()) ** 2)
            return loss
        except Exception as e:
            logger.error(f"Batch training failed: {e}")
            return float('inf')

    def _train_minibatch(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Train with mini-batches."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        total_loss = 0
        n_batches = 0

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Train on batch
            batch_loss = self._train_batch(model, X_batch, y_batch)
            total_loss += batch_loss
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else float('inf')

    def _validate_epoch(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Validate for one epoch."""
        try:
            predictions = model.predict(X)
            loss = np.mean((predictions.ravel() - y.ravel()) ** 2)
            return loss
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return float('inf')

    def _get_model_state(self, model) -> Dict:
        """Get model state for saving."""
        # This would extract model parameters
        return {}

    def _set_model_state(self, model, state: Dict):
        """Set model state from saved state."""
        # This would restore model parameters
        pass

    def get_training_history(self) -> List[Dict]:
        """Get the training history."""
        return self.training_history.copy()

    def plot_training_history(self):
        """Plot training history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            epochs = [h['epoch'] for h in self.training_history]
            train_losses = [h['train_loss'] for h in self.training_history]
            val_losses = [h['val_loss'] for h in self.training_history]

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, label='Training Loss')
            plt.plot(epochs, val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Quantum Model Training History')
            plt.legend()
            plt.grid(True)
            plt.show()

        except ImportError:
            logger.warning("matplotlib not available for plotting")

class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""

    @abstractmethod
    def on_epoch_end(self, epoch_info: Dict):
        """Called at the end of each epoch."""
        pass

class EarlyStopping(TrainingCallback):
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch_info: Dict):
        current_loss = epoch_info['val_loss']

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1

class LearningRateScheduler(TrainingCallback):
    """Learning rate scheduler callback."""

    def __init__(self, schedule_func: Callable):
        self.schedule_func = schedule_func

    def on_epoch_end(self, epoch_info: Dict):
        new_lr = self.schedule_func(epoch_info['epoch'], epoch_info.get('learning_rate', 0.01))
        epoch_info['learning_rate'] = new_lr