"""
Data generation module for financial pricing models.

This module provides synthetic data generation for training and testing
machine learning models used in derivative pricing.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from scipy.stats import norm, lognorm
from datetime import datetime, timedelta


class FinancialDataGenerator:
    """
    Generates synthetic financial data for pricing models.

    Supports generation of:
    - Interest rate curves
    - Volatility surfaces
    - Option prices
    - Market data features
    """

    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        np.random.seed(seed)
        self.seed = seed

    def generate_yield_curve(self,
                           n_samples: int = 1000,
                           n_maturities: int = 10,
                           base_rate: float = 0.03) -> pd.DataFrame:
        """
        Generate synthetic yield curve data.

        Args:
            n_samples: Number of yield curves to generate
            n_maturities: Number of maturity points per curve
            base_rate: Base interest rate level

        Returns:
            DataFrame with yield curve data
        """
        maturities = np.linspace(0.25, 30, n_maturities)  # 3 months to 30 years

        # Generate base yield curves with some structure
        base_curves = []
        for i in range(n_samples):
            # Add random perturbations
            noise = np.random.normal(0, 0.005, n_maturities)
            # Create upward sloping curve with some curvature
            curve = base_rate + 0.001 * maturities + 0.0001 * maturities**2 + noise
            base_curves.append(curve)

        # Create DataFrame
        columns = [f'maturity_{i+1}' for i in range(n_maturities)]
        df = pd.DataFrame(base_curves, columns=columns)

        # Add metadata
        df['sample_id'] = range(n_samples)
        df['generation_date'] = datetime.now()

        return df

    def generate_volatility_surface(self,
                                  n_samples: int = 1000,
                                  strikes: np.ndarray = None,
                                  expiries: np.ndarray = None) -> pd.DataFrame:
        """
        Generate synthetic volatility surface data.

        Args:
            n_samples: Number of volatility surfaces
            strikes: Array of strike prices (as moneyness)
            expiries: Array of time to expiry

        Returns:
            DataFrame with volatility surface data
        """
        if strikes is None:
            strikes = np.linspace(0.8, 1.2, 10)  # 80% to 120% moneyness
        if expiries is None:
            expiries = np.linspace(0.25, 5, 8)  # 3 months to 5 years

        surfaces = []
        for i in range(n_samples):
            # Generate base volatility surface
            surface = np.zeros((len(expiries), len(strikes)))

            for j, expiry in enumerate(expiries):
                for k, strike in enumerate(strikes):
                    # Base volatility with term and moneyness structure
                    vol = 0.2 + 0.1 * np.exp(-expiry/2) + 0.05 * abs(np.log(strike))
                    # Add random noise
                    vol += np.random.normal(0, 0.02)
                    surface[j, k] = max(vol, 0.01)  # Ensure positive volatility

            surfaces.append(surface.flatten())

        # Create DataFrame
        columns = [f'vol_{j}_{k}' for j in range(len(expiries))
                  for k in range(len(strikes))]
        df = pd.DataFrame(surfaces, columns=columns)

        df['sample_id'] = range(n_samples)
        df['n_strikes'] = len(strikes)
        df['n_expiries'] = len(expiries)

        return df

    def generate_option_prices(self,
                             n_samples: int = 10000,
                             spot_range: Tuple[float, float] = (90, 110),
                             strike_range: Tuple[float, float] = (85, 115),
                             time_range: Tuple[float, float] = (0.1, 2.0)) -> pd.DataFrame:
        """
        Generate synthetic European option prices using Black-Scholes.

        Args:
            n_samples: Number of option prices to generate
            spot_range: Range for spot prices
            strike_range: Range for strike prices
            time_range: Range for time to expiry (years)

        Returns:
            DataFrame with option data and prices
        """
        # Generate underlying parameters
        spots = np.random.uniform(spot_range[0], spot_range[1], n_samples)
        strikes = np.random.uniform(strike_range[0], strike_range[1], n_samples)
        times = np.random.uniform(time_range[0], time_range[1], n_samples)
        rates = np.random.normal(0.03, 0.01, n_samples)
        vols = np.random.uniform(0.1, 0.5, n_samples)

        # Calculate Black-Scholes prices
        call_prices = []
        put_prices = []

        for spot, strike, time, rate, vol in zip(spots, strikes, times, rates, vols):
            d1 = (np.log(spot/strike) + (rate + vol**2/2)*time) / (vol*np.sqrt(time))
            d2 = d1 - vol*np.sqrt(time)

            call = spot*norm.cdf(d1) - strike*np.exp(-rate*time)*norm.cdf(d2)
            put = strike*np.exp(-rate*time)*norm.cdf(-d2) - spot*norm.cdf(-d1)

            call_prices.append(call)
            put_prices.append(put)

        # Create DataFrame
        df = pd.DataFrame({
            'spot_price': spots,
            'strike_price': strikes,
            'time_to_expiry': times,
            'risk_free_rate': rates,
            'volatility': vols,
            'call_price': call_prices,
            'put_price': put_prices,
            'moneyness': spots/strikes
        })

        return df

    def generate_swaption_data(self,
                             n_samples: int = 5000,
                             swap_rates_range: Tuple[float, float] = (0.02, 0.08),
                             volatilities_range: Tuple[float, float] = (0.1, 0.8)) -> pd.DataFrame:
        """
        Generate synthetic swaption pricing data.

        Args:
            n_samples: Number of swaption samples
            swap_rates_range: Range for underlying swap rates
            volatilities_range: Range for swaption volatilities

        Returns:
            DataFrame with swaption pricing data
        """
        # Generate basic parameters
        swap_rates = np.random.uniform(swap_rates_range[0], swap_rates_range[1], n_samples)
        volatilities = np.random.uniform(volatilities_range[0], volatilities_range[1], n_samples)
        swap_tenors = np.random.choice([1, 2, 3, 5, 10], n_samples)  # years
        option_tenors = np.random.choice([1, 2, 3, 5], n_samples)  # years

        # Generate additional features
        atm_strikes = swap_rates + np.random.normal(0, 0.005, n_samples)
        risk_free_rates = np.random.normal(0.025, 0.005, n_samples)

        # Simple swaption price approximation (simplified model)
        # In practice, this would use more sophisticated pricing
        prices = []
        for rate, vol, swap_tenor, opt_tenor, strike in zip(
            swap_rates, volatilities, swap_tenors, option_tenors, atm_strikes):

            # Simplified pricing formula
            time = opt_tenor
            forward_rate = rate
            annuity_factor = sum([np.exp(-risk_free_rates[0] * t) for t in range(1, swap_tenor + 1)])

            d1 = (np.log(forward_rate/strike) + (vol**2/2)*time) / (vol*np.sqrt(time))
            d2 = d1 - vol*np.sqrt(time)

            price = annuity_factor * (forward_rate*norm.cdf(d1) - strike*norm.cdf(d2))
            prices.append(max(price, 0))

        df = pd.DataFrame({
            'swap_rate': swap_rates,
            'volatility': volatilities,
            'swap_tenor': swap_tenors,
            'option_tenor': option_tenors,
            'strike': atm_strikes,
            'risk_free_rate': risk_free_rates,
            'swaption_price': prices
        })

        return df

    def save_data(self, data: pd.DataFrame, filename: str, path: str = "data/raw/"):
        """
        Save generated data to CSV file.

        Args:
            data: DataFrame to save
            filename: Name of the file
            path: Directory path to save to
        """
        full_path = f"{path.rstrip('/')}/{filename}"
        data.to_csv(full_path, index=False)
        print(f"Data saved to {full_path}")


if __name__ == "__main__":
    # Example usage
    generator = FinancialDataGenerator()

    # Generate sample data
    yield_curves = generator.generate_yield_curve(n_samples=100)
    option_data = generator.generate_option_prices(n_samples=1000)
    swaption_data = generator.generate_swaption_data(n_samples=500)

    print("Generated data shapes:")
    print(f"Yield curves: {yield_curves.shape}")
    print(f"Option data: {option_data.shape}")
    print(f"Swaption data: {swaption_data.shape}")
