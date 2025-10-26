"""
Monte Carlo pricing methods for financial derivatives.

This module provides Monte Carlo simulation methods for pricing
complex derivatives that don't have closed-form solutions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union, Callable
from scipy.stats import norm, multivariate_normal
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MonteCarloPricer:
    """
    Base Monte Carlo pricer for derivative pricing.

    Provides framework for pricing derivatives using Monte Carlo simulation.
    """

    def __init__(self, n_simulations: int = 10000, n_steps: int = 252,
                 random_seed: int = 42):
        """
        Initialize Monte Carlo pricer.

        Args:
            n_simulations: Number of simulation paths
            n_steps: Number of time steps per path
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def simulate_paths(self, spot: float, time: float, rate: float,
                      vol: float, dividend: float = 0.0,
                      model: str = 'gbm') -> np.ndarray:
        """
        Simulate asset price paths.

        Args:
            spot: Initial spot price
            time: Time horizon
            rate: Risk-free rate
            vol: Volatility
            dividend: Dividend yield
            model: Stochastic model ('gbm', 'cev', 'heston')

        Returns:
            Array of simulated price paths (n_simulations x n_steps)
        """
        dt = time / self.n_steps
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = spot

        if model == 'gbm':
            return self._simulate_gbm(paths, dt, rate, vol, dividend)
        elif model == 'cev':
            return self._simulate_cev(paths, dt, rate, vol)
        elif model == 'heston':
            return self._simulate_heston(paths, dt, rate, vol)
        else:
            raise ValueError(f"Unknown model: {model}")

    def _simulate_gbm(self, paths: np.ndarray, dt: float,
                     rate: float, vol: float, dividend: float) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion paths.
        """
        drift = (rate - dividend - 0.5 * vol**2) * dt
        diffusion = vol * np.sqrt(dt)

        for t in range(1, self.n_steps + 1):
            z = np.random.normal(0, 1, self.n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * z)

        return paths

    def _simulate_cev(self, paths: np.ndarray, dt: float,
                     rate: float, vol: float, beta: float = 0.5) -> np.ndarray:
        """
        Simulate Constant Elasticity of Variance (CEV) model paths.
        """
        for t in range(1, self.n_steps + 1):
            z = np.random.normal(0, 1, self.n_simulations)
            diffusion = vol * (paths[:, t-1] ** beta) * np.sqrt(dt)
            drift = rate * paths[:, t-1] * dt

            paths[:, t] = paths[:, t-1] + drift + diffusion * z

        return paths

    def _simulate_heston(self, paths: np.ndarray, dt: float, rate: float,
                        vol: float, kappa: float = 2.0, theta: float = 0.04,
                        sigma: float = 0.3, rho: float = -0.7) -> np.ndarray:
        """
        Simulate Heston stochastic volatility model paths.
        """
        # Initialize variance paths
        v = np.full(self.n_simulations, vol**2)  # Initial variance

        for t in range(1, self.n_steps + 1):
            # Correlated random numbers
            z1 = np.random.normal(0, 1, self.n_simulations)
            z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, self.n_simulations)

            # Variance process (CIR process)
            v = np.maximum(v, 0)  # Ensure non-negative variance
            v_new = v + kappa * (theta - v) * dt + sigma * np.sqrt(v) * np.sqrt(dt) * z2
            v = np.maximum(v_new, 0)

            # Price process
            diffusion = np.sqrt(v) * np.sqrt(dt) * z1
            drift = (rate - 0.5 * v) * dt

            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion)

        return paths

    def price_european_option(self, spot: float, strike: float, time: float,
                            rate: float, vol: float, option_type: str = 'call',
                            model: str = 'gbm', **kwargs) -> Dict[str, float]:
        """
        Price European option using Monte Carlo simulation.

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to expiry
            rate: Risk-free rate
            vol: Volatility
            option_type: 'call' or 'put'
            model: Stochastic model
            **kwargs: Additional model parameters

        Returns:
            Dictionary with price and confidence intervals
        """
        # Simulate paths
        paths = self.simulate_paths(spot, time, rate, vol, model=model, **kwargs)

        # Calculate payoffs
        final_prices = paths[:, -1]

        if option_type == 'call':
            payoffs = np.maximum(final_prices - strike, 0)
        else:
            payoffs = np.maximum(strike - final_prices, 0)

        # Discount to present value
        price = np.exp(-rate * time) * np.mean(payoffs)

        # Calculate confidence intervals
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        confidence_interval = 1.96 * std_error * np.exp(-rate * time)

        return {
            'price': price,
            'standard_error': std_error,
            'confidence_interval': confidence_interval,
            'lower_bound': price - confidence_interval,
            'upper_bound': price + confidence_interval
        }

    def price_asian_option(self, spot: float, strike: float, time: float,
                         rate: float, vol: float, option_type: str = 'call',
                         averaging_type: str = 'arithmetic') -> Dict[str, float]:
        """
        Price Asian option using Monte Carlo simulation.

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to expiry
            rate: Risk-free rate
            vol: Volatility
            option_type: 'call' or 'put'
            averaging_type: 'arithmetic' or 'geometric'

        Returns:
            Dictionary with price and statistics
        """
        paths = self.simulate_paths(spot, time, rate, vol)

        # Calculate average prices
        if averaging_type == 'arithmetic':
            averages = np.mean(paths[:, 1:], axis=1)  # Exclude initial price
        else:  # geometric
            averages = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))

        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(averages - strike, 0)
        else:
            payoffs = np.maximum(strike - averages, 0)

        price = np.exp(-rate * time) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)

        return {
            'price': price,
            'standard_error': std_error,
            'averaging_type': averaging_type
        }

    def price_barrier_option(self, spot: float, strike: float, time: float,
                           rate: float, vol: float, barrier: float,
                           option_type: str = 'call', barrier_type: str = 'up_and_out',
                           monitoring: str = 'continuous') -> Dict[str, float]:
        """
        Price barrier option using Monte Carlo simulation.

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to expiry
            rate: Risk-free rate
            vol: Volatility
            barrier: Barrier level
            option_type: 'call' or 'put'
            barrier_type: Barrier type ('up_and_out', 'down_and_out', etc.)
            monitoring: 'continuous' or 'discrete'

        Returns:
            Dictionary with price and statistics
        """
        paths = self.simulate_paths(spot, time, rate, vol)

        # Check barrier conditions
        if barrier_type == 'up_and_out':
            # Option knocks out if price >= barrier
            barrier_breached = np.any(paths >= barrier, axis=1)
        elif barrier_type == 'down_and_out':
            # Option knocks out if price <= barrier
            barrier_breached = np.any(paths <= barrier, axis=1)
        elif barrier_type == 'up_and_in':
            # Option activates if price >= barrier
            barrier_breached = np.any(paths >= barrier, axis=1)
        elif barrier_type == 'down_and_in':
            # Option activates if price <= barrier
            barrier_breached = np.any(paths <= barrier, axis=1)
        else:
            raise ValueError(f"Unknown barrier type: {barrier_type}")

        # Calculate payoffs
        final_prices = paths[:, -1]

        if option_type == 'call':
            vanilla_payoffs = np.maximum(final_prices - strike, 0)
        else:
            vanilla_payoffs = np.maximum(strike - final_prices, 0)

        # Apply barrier logic
        if barrier_type.endswith('_and_out'):
            payoffs = vanilla_payoffs * (~barrier_breached)
        else:  # _and_in
            payoffs = vanilla_payoffs * barrier_breached

        price = np.exp(-rate * time) * np.mean(payoffs)

        return {
            'price': price,
            'barrier_type': barrier_type,
            'barrier_level': barrier,
            'knockout_probability': np.mean(barrier_breached)
        }

    def price_basket_option(self, spots: np.ndarray, strikes: np.ndarray,
                          time: float, rates: np.ndarray, vols: np.ndarray,
                          weights: np.ndarray, correlation_matrix: np.ndarray,
                          option_type: str = 'call') -> Dict[str, float]:
        """
        Price basket option using Monte Carlo simulation.

        Args:
            spots: Array of spot prices for basket assets
            strikes: Strike prices (can be single value or array)
            time: Time to expiry
            rates: Risk-free rates for each asset
            vols: Volatilities for each asset
            weights: Weights for basket calculation
            correlation_matrix: Correlation matrix between assets
            option_type: 'call' or 'put'

        Returns:
            Dictionary with price and statistics
        """
        n_assets = len(spots)

        # Cholesky decomposition for correlated random numbers
        chol = np.linalg.cholesky(correlation_matrix)

        # Simulate correlated paths for each asset
        basket_paths = np.zeros((self.n_simulations, n_assets))

        for i in range(n_assets):
            # Generate correlated random numbers
            z = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
            correlated_z = z @ chol

            # Simulate individual asset paths
            paths = np.zeros((self.n_simulations, self.n_steps + 1))
            paths[:, 0] = spots[i]

            dt = time / self.n_steps
            drift = (rates[i] - 0.5 * vols[i]**2) * dt
            diffusion = vols[i] * np.sqrt(dt)

            for t in range(1, self.n_steps + 1):
                paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * correlated_z[:, t-1])

            basket_paths[:, i] = paths[:, -1]

        # Calculate basket price
        basket_prices = basket_paths @ weights

        # Handle strike
        if np.isscalar(strikes):
            strike = strikes
        else:
            strike = strikes @ weights

        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(basket_prices - strike, 0)
        else:
            payoffs = np.maximum(strike - basket_prices, 0)

        price = np.exp(-np.mean(rates) * time) * np.mean(payoffs)

        return {
            'price': price,
            'basket_price_mean': np.mean(basket_prices),
            'basket_price_std': np.std(basket_prices)
        }

    def calculate_greeks_mc(self, spot: float, strike: float, time: float,
                          rate: float, vol: float, option_type: str = 'call',
                          greek: str = 'delta', epsilon: float = 0.01) -> float:
        """
        Calculate Greeks using finite differences and Monte Carlo.

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to expiry
            rate: Risk-free rate
            vol: Volatility
            option_type: 'call' or 'put'
            greek: Greek to calculate ('delta', 'gamma', 'vega', 'theta', 'rho')
            epsilon: Finite difference step size

        Returns:
            Greek value
        """
        if greek == 'delta':
            # dPrice/dSpot
            price_up = self.price_european_option(spot + epsilon, strike, time, rate, vol, option_type)['price']
            price_down = self.price_european_option(spot - epsilon, strike, time, rate, vol, option_type)['price']
            return (price_up - price_down) / (2 * epsilon)

        elif greek == 'gamma':
            # d²Price/dSpot²
            price_up = self.price_european_option(spot + epsilon, strike, time, rate, vol, option_type)['price']
            price_base = self.price_european_option(spot, strike, time, rate, vol, option_type)['price']
            price_down = self.price_european_option(spot - epsilon, strike, time, rate, vol, option_type)['price']
            return (price_up - 2*price_base + price_down) / (epsilon**2)

        elif greek == 'vega':
            # dPrice/dVol
            price_up = self.price_european_option(spot, strike, time, rate, vol + epsilon, option_type)['price']
            price_down = self.price_european_option(spot, strike, time, rate, vol - epsilon, option_type)['price']
            return (price_up - price_down) / (2 * epsilon)

        elif greek == 'theta':
            # -dPrice/dTime
            price_now = self.price_european_option(spot, strike, time, rate, vol, option_type)['price']
            price_later = self.price_european_option(spot, strike, time - epsilon, rate, vol, option_type)['price']
            return -(price_later - price_now) / epsilon

        elif greek == 'rho':
            # dPrice/dRate
            price_up = self.price_european_option(spot, strike, time, rate + epsilon, vol, option_type)['price']
            price_down = self.price_european_option(spot, strike, time, rate - epsilon, vol, option_type)['price']
            return (price_up - price_down) / (2 * epsilon)

        else:
            raise ValueError(f"Unknown Greek: {greek}")

    def price_american_option(self, spot: float, strike: float, time: float,
                            rate: float, vol: float, option_type: str = 'call',
                            n_exercise_dates: int = 50) -> Dict[str, float]:
        """
        Price American option using Least Squares Monte Carlo (LSM).

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to expiry
            rate: Risk-free rate
            vol: Volatility
            option_type: 'call' or 'put'
            n_exercise_dates: Number of exercise dates

        Returns:
            Dictionary with price and statistics
        """
        paths = self.simulate_paths(spot, time, rate, vol)

        # Exercise dates
        exercise_times = np.linspace(time/n_exercise_dates, time, n_exercise_dates)
        dt = time / self.n_steps

        # Initialize cash flows
        cash_flows = np.zeros(self.n_simulations)

        # Work backwards from maturity
        for i in range(len(exercise_times) - 1, -1, -1):
            t = exercise_times[i]
            time_index = int(t / dt)

            # Current prices at this exercise date
            current_prices = paths[:, time_index]

            # Intrinsic value
            if option_type == 'call':
                intrinsic = np.maximum(current_prices - strike, 0)
            else:
                intrinsic = np.maximum(strike - current_prices, 0)

            # For the last exercise date, exercise value is intrinsic
            if i == len(exercise_times) - 1:
                cash_flows = intrinsic
            else:
                # Expected continuation value using regression
                itm = intrinsic > 0
                if np.sum(itm) > 0:
                    # Regression variables (price and price squared)
                    X = np.column_stack([current_prices[itm], current_prices[itm]**2])

                    # Future cash flows (discounted)
                    future_cf = cash_flows[itm] * np.exp(-rate * (exercise_times[i+1] - t))

                    # Fit regression
                    from sklearn.linear_model import LinearRegression
                    reg = LinearRegression()
                    reg.fit(X, future_cf)

                    # Expected continuation value
                    continuation = reg.predict(X)

                    # Exercise decision
                    exercise = intrinsic[itm] > continuation
                    cash_flows[itm] = np.where(exercise, intrinsic[itm], cash_flows[itm])

        # Discount final cash flows to present
        price = np.exp(-rate * time) * np.mean(cash_flows)

        return {
            'price': price,
            'exercise_probability': np.mean(cash_flows > 0)
        }


class VarianceReduction:
    """
    Variance reduction techniques for Monte Carlo simulation.
    """

    @staticmethod
    def antithetic_variables(paths: np.ndarray) -> np.ndarray:
        """
        Apply antithetic variables variance reduction.

        Args:
            paths: Original simulated paths

        Returns:
            Combined paths with antithetic variables
        """
        # Create antithetic paths by negating random numbers
        antithetic_paths = paths.copy()
        # This is a simplified implementation - would need access to random numbers
        return np.concatenate([paths, antithetic_paths], axis=0)

    @staticmethod
    def control_variates(paths: np.ndarray, control_values: np.ndarray,
                        true_values: np.ndarray) -> np.ndarray:
        """
        Apply control variates variance reduction.

        Args:
            paths: Simulated paths
            control_values: Values of control variate
            true_values: True values of control variate

        Returns:
            Adjusted paths
        """
        # Adjust payoffs using control variate
        adjustment = np.mean(control_values - true_values)
        return paths - adjustment

    @staticmethod
    def importance_sampling(paths: np.ndarray, likelihood_ratios: np.ndarray) -> np.ndarray:
        """
        Apply importance sampling variance reduction.

        Args:
            paths: Simulated paths
            likelihood_ratios: Likelihood ratios for importance sampling

        Returns:
            Adjusted paths
        """
        return paths * likelihood_ratios


if __name__ == "__main__":
    # Example usage
    mc_pricer = MonteCarloPricer(n_simulations=10000, n_steps=252)

    # Price European call option
    result = mc_pricer.price_european_option(
        spot=100, strike=105, time=1.0, rate=0.05, vol=0.2, option_type='call'
    )

    print(f"Monte Carlo Call Price: ${result['price']:.4f}")
    print(f"95% Confidence Interval: [${result['lower_bound']:.4f}, ${result['upper_bound']:.4f}]")

    # Price Asian option
    asian_result = mc_pricer.price_asian_option(
        spot=100, strike=105, time=1.0, rate=0.05, vol=0.2
    )

    print(f"Asian Call Price: ${asian_result['price']:.4f}")

    # Calculate Greeks
    delta = mc_pricer.calculate_greeks_mc(
        spot=100, strike=105, time=1.0, rate=0.05, vol=0.2, greek='delta'
    )

    print(f"Monte Carlo Delta: {delta:.6f}")