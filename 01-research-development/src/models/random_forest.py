"""
Random Forest model implementation for financial pricing.

This module provides Random Forest based pricing models with
financial domain-specific configurations and feature importance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import logging

from .base_model import ScikitLearnPricingModel

logger = logging.getLogger(__name__)


class RandomForestPricingModel(ScikitLearnPricingModel):
    """
    Random Forest model for derivative pricing.

    Extends the base scikit-learn pricing model with Random Forest specific
    configurations and financial domain optimizations.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 random_state: int = 42,
                 name: str = "RandomForestModel"):
        """
        Initialize Random Forest pricing model.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap sampling
            random_state: Random state for reproducibility
            name: Model name
        """
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )

        super().__init__(estimator, name)

        # Store hyperparameters
        self.hyperparameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'random_state': random_state
        }

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the Random Forest model with financial data considerations.

        Args:
            X: Feature matrix
            y: Target values (option/swaption prices)
            **kwargs: Additional training parameters
        """
        logger.info(f"Training Random Forest model with {len(X)} samples")

        # Call parent train method
        super().train(X, y, **kwargs)

        # Log training completion with feature importance summary
        feature_importance = self.get_feature_importance()
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Top 5 features: {top_features}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores from the Random Forest.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.feature_names_ is None:
            return None

        importances = self.estimator.feature_importances_
        return dict(zip(self.feature_names_, importances))

    def get_tree_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the individual trees in the forest.

        Returns:
            Dictionary with tree statistics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting tree statistics")

        trees = self.estimator.estimators_
        depths = [tree.get_depth() for tree in trees]
        n_leaves = [tree.get_n_leaves() for tree in trees]

        return {
            'n_trees': len(trees),
            'mean_depth': np.mean(depths),
            'max_depth': np.max(depths),
            'min_depth': np.min(depths),
            'mean_n_leaves': np.mean(n_leaves),
            'max_n_leaves': np.max(n_leaves),
            'min_n_leaves': np.min(n_leaves)
        }

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using tree variance.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predictions, standard_deviations)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X.values) for tree in self.estimator.estimators_])

        # Calculate mean and standard deviation
        predictions = np.mean(tree_predictions, axis=0)
        uncertainties = np.std(tree_predictions, axis=0)

        return predictions, uncertainties

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                           param_grid: Optional[Dict[str, List[Any]]] = None,
                           method: str = 'grid',
                           cv: int = 5,
                           n_iter: int = 50) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search or random search.

        Args:
            X: Feature matrix
            y: Target values
            param_grid: Parameter grid for search
            method: Search method ('grid' or 'random')
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search

        Returns:
            Dictionary with best parameters and scores
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }

        base_estimator = RandomForestRegressor(random_state=self.hyperparameters['random_state'], n_jobs=-1)

        if method == 'grid':
            search = GridSearchCV(
                base_estimator,
                param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                base_estimator,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_squared_error',
                random_state=self.hyperparameters['random_state'],
                n_jobs=-1
            )
        else:
            raise ValueError("Method must be 'grid' or 'random'")

        logger.info(f"Starting hyperparameter tuning with {method} search")
        search.fit(X, y)

        # Update model with best parameters
        self.estimator = search.best_estimator_
        self.hyperparameters.update(search.best_params_)
        self.is_trained = True
        self.training_date = pd.Timestamp.now()

        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.6f}")

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }

    def analyze_feature_interactions(self, X: pd.DataFrame,
                                   feature_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Analyze feature interactions by training separate models.

        Args:
            X: Feature matrix
            feature_pairs: List of feature pairs to analyze

        Returns:
            Dictionary with interaction strengths
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing interactions")

        interactions = {}

        for feat1, feat2 in feature_pairs:
            if feat1 not in self.feature_names_ or feat2 not in self.feature_names_:
                continue

            # Create interaction feature
            interaction_name = f"{feat1}_{feat2}_interaction"
            X_with_interaction = X.copy()
            X_with_interaction[interaction_name] = X[feat1] * X[feat2]

            # Train model with interaction
            temp_model = RandomForestRegressor(
                n_estimators=50,
                random_state=self.hyperparameters['random_state'],
                n_jobs=-1
            )
            temp_model.fit(X_with_interaction, y)  # Use actual target

            # Get interaction importance
            all_features = list(X_with_interaction.columns)
            interaction_idx = all_features.index(interaction_name)
            interactions[f"{feat1}_{feat2}"] = temp_model.feature_importances_[interaction_idx]

        return interactions

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Random Forest model.

        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            'hyperparameters': self.hyperparameters,
            'tree_statistics': self.get_tree_statistics() if self.is_trained else None,
            'feature_importance': self.get_feature_importance()
        })
        return info


class ExtraTreesPricingModel(RandomForestPricingModel):
    """
    Extra Trees model for derivative pricing.

    Uses Extremely Randomized Trees algorithm which can provide
    better generalization for noisy financial data.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = False,
                 random_state: int = 42,
                 name: str = "ExtraTreesModel"):
        """
        Initialize Extra Trees pricing model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples at leaf node
            max_features: Number of features to consider
            bootstrap: Whether to use bootstrap sampling
            random_state: Random state
            name: Model name
        """
        from sklearn.ensemble import ExtraTreesRegressor

        estimator = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=-1
        )

        super().__init__(estimator, name)

        self.hyperparameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'random_state': random_state
        }


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression

    # Create sample financial-like data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='price')

    # Initialize and train model
    rf_model = RandomForestPricingModel(n_estimators=50, max_depth=10)
    rf_model.train(X_df, y_series)

    # Make predictions
    predictions = rf_model.predict(X_df)

    # Get feature importance
    importance = rf_model.get_feature_importance()

    # Get model statistics
    stats = rf_model.get_tree_statistics()

    print(f"Model trained with {len(X_df)} samples")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Top 3 important features: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
    print(f"Tree statistics: Mean depth = {stats['mean_depth']:.1f}")