"""
Data preprocessing module for financial pricing models.

This module provides data cleaning, normalization, and preprocessing
utilities specific to financial time series and derivative pricing data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class FinancialDataPreprocessor:
    """
    Comprehensive data preprocessor for financial pricing data.

    Handles missing values, outliers, scaling, and data splitting
    with financial domain-specific considerations.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scalers = {}
        self.imputers = {}
        self.feature_stats = {}
        self.outlier_bounds = {}

    def handle_missing_values(self, df: pd.DataFrame,
                            strategy: str = 'median',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('median', 'mean', 'most_frequent', 'knn')
            columns: Specific columns to impute (if None, impute all numeric columns)

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            self.imputers['knn'] = imputer
        else:
            imputer = SimpleImputer(strategy=strategy)
            self.imputers[strategy] = imputer

        # Fit and transform
        df[columns] = imputer.fit_transform(df[columns])

        logger.info(f"Missing values handled using {strategy} strategy for {len(columns)} columns")
        return df

    def detect_outliers(self, df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict[str, np.ndarray]:
        """
        Detect outliers in specified columns.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection

        Returns:
            Dictionary mapping column names to boolean arrays of outlier indicators
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = {}

        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers[col] = outlier_mask

                self.outlier_bounds[col] = (lower_bound, upper_bound)

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col]))
                outlier_mask = z_scores > threshold
                outliers[col] = outlier_mask

            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=self.random_state)
                outlier_mask = iso_forest.fit_predict(df[[col]]) == -1
                outliers[col] = outlier_mask

        return outliers

    def remove_outliers(self, df: pd.DataFrame,
                       outlier_masks: Dict[str, np.ndarray],
                       strategy: str = 'cap') -> pd.DataFrame:
        """
        Remove or handle outliers in the dataset.

        Args:
            df: Input DataFrame
            outlier_masks: Dictionary of outlier masks from detect_outliers
            strategy: Strategy for handling outliers ('remove', 'cap', 'median')

        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()

        for col, mask in outlier_masks.items():
            if strategy == 'remove':
                df = df[~mask]
                logger.info(f"Removed {mask.sum()} outliers from {col}")

            elif strategy == 'cap':
                if col in self.outlier_bounds:
                    lower_bound, upper_bound = self.outlier_bounds[col]
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    logger.info(f"Capped outliers in {col} to bounds [{lower_bound:.3f}, {upper_bound:.3f}]")

            elif strategy == 'median':
                median_val = df[col].median()
                df.loc[mask, col] = median_val
                logger.info(f"Replaced {mask.sum()} outliers in {col} with median {median_val:.3f}")

        return df

    def scale_features(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      method: str = 'standard',
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            df: Input DataFrame
            columns: Columns to scale (if None, scale all numeric columns)
            method: Scaling method ('standard', 'robust', 'minmax')
            fit: Whether to fit the scaler (True for training, False for test data)

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        scaler_key = f"{method}_scaler"

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard', 'robust', or 'minmax'")

        if fit:
            scaled_values = scaler.fit_transform(df[columns])
            self.scalers[scaler_key] = scaler
        else:
            if scaler_key not in self.scalers:
                raise ValueError(f"Scaler {scaler_key} not fitted. Call with fit=True first.")
            scaled_values = self.scalers[scaler_key].transform(df[columns])

        # Create new scaled columns
        scaled_df = pd.DataFrame(scaled_values, columns=[f"{col}_scaled" for col in columns], index=df.index)
        df = pd.concat([df, scaled_df], axis=1)

        logger.info(f"Scaled {len(columns)} features using {method} scaling")
        return df

    def handle_skewness(self, df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle skewed distributions using transformations.

        Args:
            df: Input DataFrame
            columns: Columns to transform (if None, transform all numeric columns)
            threshold: Skewness threshold for transformation

        Returns:
            DataFrame with transformed features
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            skewness = df[col].skew()

            if abs(skewness) > threshold:
                # Log transformation for positive skewed data
                if skewness > 0 and (df[col] > 0).all():
                    df[f"{col}_log"] = np.log(df[col] + 1e-6)
                    logger.info(f"Applied log transformation to {col} (skewness: {skewness:.3f})")

                # Square root transformation
                elif skewness > 0 and (df[col] >= 0).all():
                    df[f"{col}_sqrt"] = np.sqrt(df[col])
                    logger.info(f"Applied sqrt transformation to {col} (skewness: {skewness:.3f})")

                # Box-Cox transformation
                else:
                    try:
                        transformed, lambda_val = stats.boxcox(df[col] + abs(df[col].min()) + 1e-6)
                        df[f"{col}_boxcox"] = transformed
                        logger.info(f"Applied Box-Cox transformation to {col} (lambda: {lambda_val:.3f})")
                    except:
                        logger.warning(f"Could not apply Box-Cox transformation to {col}")

        return df

    def create_financial_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply financial domain-specific data filters.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        df = df.copy()

        # Filter out unrealistic values
        if 'volatility' in df.columns:
            df = df[df['volatility'].between(0.001, 2.0)]  # Reasonable vol range

        if 'risk_free_rate' in df.columns:
            df = df[df['risk_free_rate'].between(-0.01, 0.15)]  # Reasonable rate range

        if 'time_to_expiry' in df.columns:
            df = df[df['time_to_expiry'] > 0]  # Positive time to expiry

        if 'spot_price' in df.columns:
            df = df[df['spot_price'] > 0]  # Positive spot prices

        logger.info(f"Applied financial filters, remaining samples: {len(df)}")
        return df

    def split_data(self, df: pd.DataFrame,
                  target_column: str,
                  test_size: float = 0.2,
                  val_size: float = 0.1,
                  stratify: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            stratify: Column to stratify on

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        stratify_col = df[stratify] if stratify else None

        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_col if stratify_col is not None else None
        )

        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_time_series_splits(self, df: pd.DataFrame,
                                target_column: str,
                                time_column: str,
                                test_periods: int = 1,
                                val_periods: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time series aware train/val/test splits.

        Args:
            df: Input DataFrame (sorted by time)
            target_column: Name of target column
            time_column: Name of time column
            test_periods: Number of time periods for testing
            val_periods: Number of time periods for validation

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        df = df.sort_values(time_column).reset_index(drop=True)

        # Get unique time periods
        time_periods = df[time_column].unique()
        n_periods = len(time_periods)

        # Split indices
        test_start = n_periods - test_periods
        val_start = test_start - val_periods

        train_mask = df[time_column].isin(time_periods[:val_start])
        val_mask = df[time_column].isin(time_periods[val_start:test_start])
        test_mask = df[time_column].isin(time_periods[test_start:])

        X_train = df[train_mask].drop(columns=[target_column])
        y_train = df[train_mask][target_column]

        X_val = df[val_mask].drop(columns=[target_column])
        y_val = df[val_mask][target_column]

        X_test = df[test_mask].drop(columns=[target_column])
        y_test = df[test_mask][target_column]

        logger.info(f"Time series split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the preprocessing steps applied.

        Returns:
            Dictionary with preprocessing statistics
        """
        return {
            'scalers': list(self.scalers.keys()),
            'imputers': list(self.imputers.keys()),
            'outlier_bounds': self.outlier_bounds,
            'feature_stats': self.feature_stats
        }

    def save_preprocessor(self, path: str) -> None:
        """
        Save the preprocessor state for later use.

        Args:
            path: Path to save the preprocessor
        """
        import joblib

        state = {
            'scalers': self.scalers,
            'imputers': self.imputers,
            'feature_stats': self.feature_stats,
            'outlier_bounds': self.outlier_bounds,
            'random_state': self.random_state
        }

        joblib.dump(state, path)
        logger.info(f"Preprocessor saved to {path}")

    def load_preprocessor(self, path: str) -> None:
        """
        Load a saved preprocessor state.

        Args:
            path: Path to load the preprocessor from
        """
        import joblib

        state = joblib.load(path)

        self.scalers = state.get('scalers', {})
        self.imputers = state.get('imputers', {})
        self.feature_stats = state.get('feature_stats', {})
        self.outlier_bounds = state.get('outlier_bounds', {})
        self.random_state = state.get('random_state', 42)

        logger.info(f"Preprocessor loaded from {path}")


if __name__ == "__main__":
    # Example usage
    preprocessor = FinancialDataPreprocessor()

    # Create sample financial data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'spot_price': np.random.normal(100, 10, 1000),
        'strike_price': np.random.normal(100, 5, 1000),
        'volatility': np.random.beta(2, 5, 1000) * 0.5,  # Skewed distribution
        'risk_free_rate': np.random.normal(0.03, 0.01, 1000),
        'time_to_expiry': np.random.exponential(1, 1000),
        'option_price': np.random.normal(5, 2, 1000)
    })

    # Add some missing values and outliers
    sample_data.loc[np.random.choice(1000, 50), 'volatility'] = np.nan
    sample_data.loc[np.random.choice(1000, 20), 'spot_price'] = 1000  # Outliers

    print(f"Original data shape: {sample_data.shape}")
    print(f"Missing values: {sample_data.isnull().sum().sum()}")

    # Apply preprocessing
    processed_data = preprocessor.handle_missing_values(sample_data)
    outliers = preprocessor.detect_outliers(processed_data, ['spot_price', 'volatility'])
    processed_data = preprocessor.remove_outliers(processed_data, outliers, strategy='cap')
    processed_data = preprocessor.handle_skewness(processed_data, ['volatility'])
    processed_data = preprocessor.scale_features(processed_data, ['spot_price', 'strike_price'])

    print(f"Processed data shape: {processed_data.shape}")
    print(f"Missing values after processing: {processed_data.isnull().sum().sum()}")