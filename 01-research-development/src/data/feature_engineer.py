"""
Feature engineering module for financial pricing models.

This module provides feature engineering techniques specific to
derivative pricing and financial time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class FinancialFeatureEngineer:
    """
    Feature engineering class for financial pricing data.

    Provides methods for creating domain-specific features for
    options, swaps, and other derivative instruments.
    """

    def __init__(self):
        self.scalers = {}
        self.pca_models = {}
        self.feature_stats = {}

    def create_option_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specific to option pricing.

        Args:
            df: DataFrame with basic option data (spot, strike, time, rate, vol)

        Returns:
            DataFrame with additional option-specific features
        """
        df = df.copy()

        # Basic greeks and ratios
        df['moneyness'] = df['spot_price'] / df['strike_price']
        df['log_moneyness'] = np.log(df['moneyness'])
        df['time_value'] = df['time_to_expiry']

        # Volatility-adjusted features
        df['vol_time'] = df['volatility'] * np.sqrt(df['time_to_expiry'])
        df['vol_sqrt_time'] = df['volatility'] / np.sqrt(df['time_to_expiry'])

        # Rate-adjusted features
        df['rate_time'] = df['risk_free_rate'] * df['time_to_expiry']

        # Non-linear transformations
        df['moneyness_squared'] = df['moneyness'] ** 2
        df['vol_squared'] = df['volatility'] ** 2
        df['time_sqrt'] = np.sqrt(df['time_to_expiry'])

        # Interaction features
        df['moneyness_vol'] = df['moneyness'] * df['volatility']
        df['time_vol'] = df['time_to_expiry'] * df['volatility']
        df['rate_vol'] = df['risk_free_rate'] * df['volatility']

        return df

    def create_yield_curve_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from yield curve data.

        Args:
            df: DataFrame with yield curve maturities

        Returns:
            DataFrame with yield curve features
        """
        df = df.copy()

        # Extract maturity columns
        maturity_cols = [col for col in df.columns if col.startswith('maturity_')]
        maturities = np.array([float(col.split('_')[1]) for col in maturity_cols])

        # Curve shape features
        rates = df[maturity_cols].values

        # Level (average rate)
        df['curve_level'] = np.mean(rates, axis=1)

        # Slope (long - short rates)
        df['curve_slope'] = rates[:, -1] - rates[:, 0]

        # Curvature (difference between mid and average of short/long)
        mid_idx = len(maturities) // 2
        df['curve_curvature'] = rates[:, mid_idx] - (rates[:, 0] + rates[:, -1]) / 2

        # Rate changes (first differences)
        rate_changes = np.diff(rates, axis=1)
        for i, change in enumerate(rate_changes.T):
            df[f'rate_change_{i+1}'] = change

        # Volatility of rates
        df['rate_volatility'] = np.std(rates, axis=1)

        return df

    def create_volatility_surface_features(self, df: pd.DataFrame,
                                         n_strikes: int = 10,
                                         n_expiries: int = 8) -> pd.DataFrame:
        """
        Create features from volatility surface data.

        Args:
            df: DataFrame with volatility surface data
            n_strikes: Number of strike points
            n_expiries: Number of expiry points

        Returns:
            DataFrame with volatility surface features
        """
        df = df.copy()

        # Reshape flat volatility data back to surface
        vol_cols = [col for col in df.columns if col.startswith('vol_')]
        vol_data = df[vol_cols].values

        surfaces = vol_data.reshape(-1, n_expiries, n_strikes)

        # Surface statistics
        df['vol_mean'] = np.mean(surfaces, axis=(1, 2))
        df['vol_std'] = np.std(surfaces, axis=(1, 2))
        df['vol_min'] = np.min(surfaces, axis=(1, 2))
        df['vol_max'] = np.max(surfaces, axis=(1, 2))

        # Term structure features (average across strikes for each expiry)
        for i in range(n_expiries):
            df[f'vol_term_{i}'] = np.mean(surfaces[:, i, :], axis=1)

        # Smile features (average across expiries for each strike)
        for j in range(n_strikes):
            df[f'vol_smile_{j}'] = np.mean(surfaces[:, :, j], axis=1)

        # Volatility skew (difference between OTM calls and puts)
        atm_idx = n_strikes // 2
        if atm_idx > 0:
            df['vol_skew'] = (np.mean(surfaces[:, :, atm_idx-1], axis=1) -
                            np.mean(surfaces[:, :, atm_idx+1], axis=1))

        return df

    def create_swaption_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specific to swaption pricing.

        Args:
            df: DataFrame with swaption data

        Returns:
            DataFrame with swaption-specific features
        """
        df = df.copy()

        # Basic features
        df['swap_tenor_years'] = df['swap_tenor']
        df['option_tenor_years'] = df['option_tenor']

        # Moneyness
        df['swap_moneyness'] = df['swap_rate'] / df['strike']

        # Volatility-adjusted features
        df['vol_swap_tenor'] = df['volatility'] * np.sqrt(df['swap_tenor'])
        df['vol_option_tenor'] = df['volatility'] * np.sqrt(df['option_tenor'])

        # Rate products
        df['rate_vol_product'] = df['risk_free_rate'] * df['volatility']
        df['swap_vol_product'] = df['swap_rate'] * df['volatility']

        # Non-linear transformations
        df['swap_rate_squared'] = df['swap_rate'] ** 2
        df['volatility_squared'] = df['volatility'] ** 2
        df['log_swap_rate'] = np.log(df['swap_rate'].clip(lower=1e-6))

        # Time features
        df['total_tenor'] = df['swap_tenor'] + df['option_tenor']
        df['tenor_ratio'] = df['option_tenor'] / df['swap_tenor'].clip(lower=1)

        return df

    def add_polynomial_features(self, df: pd.DataFrame,
                              features: List[str],
                              degree: int = 2) -> pd.DataFrame:
        """
        Add polynomial features to specified columns.

        Args:
            df: Input DataFrame
            features: List of feature names to polynomialize
            degree: Degree of polynomial features

        Returns:
            DataFrame with polynomial features added
        """
        df = df.copy()

        for feature in features:
            if feature in df.columns:
                for d in range(2, degree + 1):
                    df[f'{feature}_pow_{d}'] = df[feature] ** d

        return df

    def add_interaction_features(self, df: pd.DataFrame,
                               feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Add interaction features between pairs of features.

        Args:
            df: Input DataFrame
            feature_pairs: List of tuples with feature pairs

        Returns:
            DataFrame with interaction features added
        """
        df = df.copy()

        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]

        return df

    def scale_features(self, df: pd.DataFrame,
                      features: List[str],
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale specified features using StandardScaler or MinMaxScaler.

        Args:
            df: Input DataFrame
            features: List of features to scale
            method: Scaling method ('standard' or 'minmax')

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")

        # Fit and transform
        scaled_values = scaler.fit_transform(df[features])
        scaled_df = pd.DataFrame(scaled_values, columns=features, index=df.index)

        # Update DataFrame
        for feature in features:
            df[f'{feature}_scaled'] = scaled_df[feature]

        # Store scaler for later use
        self.scalers[f'{method}_scaler'] = scaler

        return df

    def apply_pca(self, df: pd.DataFrame,
                 features: List[str],
                 n_components: Optional[int] = None,
                 variance_ratio: float = 0.95) -> pd.DataFrame:
        """
        Apply PCA to reduce dimensionality of features.

        Args:
            df: Input DataFrame
            features: List of features for PCA
            n_components: Number of PCA components (if None, use variance_ratio)
            variance_ratio: Explained variance ratio threshold

        Returns:
            DataFrame with PCA components added
        """
        df = df.copy()

        pca = PCA(n_components=n_components if n_components else variance_ratio)
        pca_components = pca.fit_transform(df[features])

        # Add PCA components to DataFrame
        n_comps = pca_components.shape[1]
        for i in range(n_comps):
            df[f'pca_component_{i+1}'] = pca_components[:, i]

        # Store PCA model
        self.pca_models['pca'] = pca

        logger.info(f"PCA applied: {n_comps} components explain "
                   f"{pca.explained_variance_ratio_.sum():.3f} variance")

        return df

    def create_rolling_features(self, df: pd.DataFrame,
                              features: List[str],
                              windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create rolling statistics features.

        Args:
            df: Input DataFrame (must be time-series)
            features: Features to create rolling stats for
            windows: Rolling window sizes

        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()

        for feature in features:
            if feature in df.columns:
                for window in windows:
                    df[f'{feature}_rolling_mean_{window}'] = df[feature].rolling(window).mean()
                    df[f'{feature}_rolling_std_{window}'] = df[feature].rolling(window).std()
                    df[f'{feature}_rolling_min_{window}'] = df[feature].rolling(window).min()
                    df[f'{feature}_rolling_max_{window}'] = df[feature].rolling(window).max()

        return df

    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about created features.

        Returns:
            Dictionary with feature statistics
        """
        return self.feature_stats

    def save_feature_engineer(self, path: str) -> None:
        """
        Save the feature engineer state for later use.

        Args:
            path: Path to save the feature engineer
        """
        import joblib

        state = {
            'scalers': self.scalers,
            'pca_models': self.pca_models,
            'feature_stats': self.feature_stats
        }

        joblib.dump(state, path)
        logger.info(f"Feature engineer saved to {path}")

    def load_feature_engineer(self, path: str) -> None:
        """
        Load a saved feature engineer state.

        Args:
            path: Path to load the feature engineer from
        """
        import joblib

        state = joblib.load(path)

        self.scalers = state.get('scalers', {})
        self.pca_models = state.get('pca_models', {})
        self.feature_stats = state.get('feature_stats', {})

        logger.info(f"Feature engineer loaded from {path}")


if __name__ == "__main__":
    # Example usage
    engineer = FinancialFeatureEngineer()

    # Create sample option data
    sample_data = pd.DataFrame({
        'spot_price': [100, 105, 95],
        'strike_price': [100, 100, 100],
        'time_to_expiry': [1.0, 0.5, 2.0],
        'risk_free_rate': [0.03, 0.025, 0.035],
        'volatility': [0.2, 0.25, 0.15]
    })

    # Create option features
    featured_data = engineer.create_option_features(sample_data)

    print("Original features:", sample_data.shape[1])
    print("After feature engineering:", featured_data.shape[1])
    print("New features created:", list(set(featured_data.columns) - set(sample_data.columns)))