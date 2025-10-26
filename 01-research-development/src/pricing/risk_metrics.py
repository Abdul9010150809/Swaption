"""
Risk metrics calculation for financial derivatives.

This module provides comprehensive risk analysis tools including
VaR, CVaR, Greeks, stress testing, and scenario analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union, List
from scipy.stats import norm, t
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RiskMetricsCalculator:
    """
    Comprehensive risk metrics calculator for financial portfolios.

    Provides Value at Risk (VaR), Conditional VaR (CVaR), Greeks analysis,
    stress testing, and scenario analysis.
    """

    def __init__(self, confidence_level: float = 0.95, time_horizon: int = 1):
        """
        Initialize risk metrics calculator.

        Args:
            confidence_level: Confidence level for VaR calculations (e.g., 0.95, 0.99)
            time_horizon: Time horizon in days for risk calculations
        """
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.alpha = 1 - confidence_level

    def calculate_var_historical(self, returns: pd.Series,
                               portfolio_value: float = 1.0) -> Dict[str, float]:
        """
        Calculate Historical Value at Risk (VaR).

        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value

        Returns:
            Dictionary with VaR metrics
        """
        # Calculate portfolio losses (negative returns)
        losses = -returns * portfolio_value

        # Sort losses in ascending order
        sorted_losses = np.sort(losses)

        # Find the loss at the confidence level
        var_index = int(self.alpha * len(sorted_losses))
        var = sorted_losses[var_index]

        # Calculate Expected Shortfall (CVaR)
        cvar = np.mean(sorted_losses[:var_index])

        return {
            'VaR': var,
            'CVaR': cvar,
            'confidence_level': self.confidence_level,
            'time_horizon': self.time_horizon,
            'method': 'historical'
        }

    def calculate_var_parametric(self, returns: pd.Series,
                               portfolio_value: float = 1.0,
                               distribution: str = 'normal') -> Dict[str, float]:
        """
        Calculate Parametric Value at Risk (VaR) assuming normal or t-distribution.

        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            distribution: Distribution assumption ('normal' or 't')

        Returns:
            Dictionary with VaR metrics
        """
        mu = returns.mean()
        sigma = returns.std()

        # Scale for time horizon
        scaled_mu = mu * self.time_horizon
        scaled_sigma = sigma * np.sqrt(self.time_horizon)

        if distribution == 'normal':
            # Normal distribution VaR
            z_score = norm.ppf(self.alpha)
            var = -(scaled_mu + z_score * scaled_sigma) * portfolio_value

            # CVaR for normal distribution
            z_cvar = norm.pdf(z_score) / self.alpha
            cvar = -(scaled_mu + scaled_sigma * z_cvar) * portfolio_value

        elif distribution == 't':
            # Fit t-distribution
            params = t.fit(returns)
            df, loc, scale = params

            # t-distribution VaR
            t_score = t.ppf(self.alpha, df, loc, scale)
            var = -t_score * portfolio_value * np.sqrt(self.time_horizon)

            # CVaR approximation for t-distribution
            cvar = var * (t.pdf(t_score, df) / (self.alpha * (df + t_score**2) / (df - 1)))

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return {
            'VaR': var,
            'CVaR': cvar,
            'confidence_level': self.confidence_level,
            'time_horizon': self.time_horizon,
            'method': f'parametric_{distribution}',
            'mean_return': scaled_mu,
            'volatility': scaled_sigma
        }

    def calculate_var_monte_carlo(self, returns: pd.Series,
                                portfolio_value: float = 1.0,
                                n_simulations: int = 10000) -> Dict[str, float]:
        """
        Calculate Monte Carlo Value at Risk (VaR).

        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with VaR metrics
        """
        # Fit distribution to historical returns
        mu = returns.mean()
        sigma = returns.std()

        # Generate simulated returns
        simulated_returns = np.random.normal(mu, sigma, (n_simulations, self.time_horizon))
        simulated_portfolio_returns = np.prod(1 + simulated_returns, axis=1) - 1

        # Calculate portfolio losses
        losses = -simulated_portfolio_returns * portfolio_value
        sorted_losses = np.sort(losses)

        # VaR and CVaR
        var_index = int(self.alpha * n_simulations)
        var = sorted_losses[var_index]
        cvar = np.mean(sorted_losses[:var_index])

        return {
            'VaR': var,
            'CVaR': cvar,
            'confidence_level': self.confidence_level,
            'time_horizon': self.time_horizon,
            'method': 'monte_carlo',
            'n_simulations': n_simulations
        }

    def calculate_portfolio_var(self, weights: np.ndarray,
                              returns: pd.DataFrame,
                              portfolio_value: float = 1.0,
                              method: str = 'parametric') -> Dict[str, float]:
        """
        Calculate portfolio-level Value at Risk.

        Args:
            weights: Portfolio weights array
            returns: Asset returns DataFrame (assets as columns)
            portfolio_value: Portfolio value
            method: VaR calculation method

        Returns:
            Dictionary with portfolio VaR metrics
        """
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)

        if method == 'parametric':
            return self.calculate_var_parametric(portfolio_returns, portfolio_value)
        elif method == 'historical':
            return self.calculate_var_historical(portfolio_returns, portfolio_value)
        elif method == 'monte_carlo':
            return self.calculate_var_monte_carlo(portfolio_returns, portfolio_value)
        else:
            raise ValueError(f"Unknown method: {method}")

    def calculate_greeks_risk(self, greeks: Dict[str, float],
                            spot_prices: Dict[str, float],
                            volatilities: Dict[str, float],
                            risk_factors: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate risk metrics from option Greeks.

        Args:
            greeks: Dictionary of option Greeks (delta, gamma, vega, theta, rho)
            spot_prices: Current spot prices for underlying assets
            volatilities: Current volatilities
            risk_factors: Risk factor shocks (optional)

        Returns:
            Dictionary with risk metrics
        """
        if risk_factors is None:
            risk_factors = {
                'spot_shock': 0.01,  # 1% spot price change
                'vol_shock': 0.01,   # 1% volatility change
                'rate_shock': 0.0001,  # 1bp rate change
                'time_decay': 1      # 1 day time decay
            }

        # Delta risk (PV01 equivalent)
        delta_risk = abs(greeks.get('delta', 0)) * risk_factors['spot_shock']

        # Gamma risk (convexity)
        gamma_risk = 0.5 * abs(greeks.get('gamma', 0)) * (risk_factors['spot_shock'] ** 2)

        # Vega risk
        vega_risk = abs(greeks.get('vega', 0)) * risk_factors['vol_shock']

        # Theta risk (time decay)
        theta_risk = abs(greeks.get('theta', 0)) * risk_factors['time_decay']

        # Rho risk (interest rate risk)
        rho_risk = abs(greeks.get('rho', 0)) * risk_factors['rate_shock']

        return {
            'delta_risk': delta_risk,
            'gamma_risk': gamma_risk,
            'vega_risk': vega_risk,
            'theta_risk': theta_risk,
            'rho_risk': rho_risk,
            'total_greeks_risk': delta_risk + gamma_risk + vega_risk + theta_risk + rho_risk
        }

    def stress_test(self, portfolio_returns: pd.Series,
                  stress_scenarios: Dict[str, float],
                  portfolio_value: float = 1.0) -> Dict[str, float]:
        """
        Perform stress testing under various market scenarios.

        Args:
            portfolio_returns: Historical portfolio returns
            stress_scenarios: Dictionary of stress scenarios with return shocks
            portfolio_value: Current portfolio value

        Returns:
            Dictionary with stress test results
        """
        results = {}

        for scenario_name, shock in stress_scenarios.items():
            # Apply shock to returns
            stressed_returns = portfolio_returns + shock

            # Calculate stressed portfolio value
            stressed_value = portfolio_value * (1 + stressed_returns).prod()

            # Calculate loss
            loss = portfolio_value - stressed_value
            loss_percentage = (loss / portfolio_value) * 100

            results[scenario_name] = {
                'stressed_value': stressed_value,
                'loss': loss,
                'loss_percentage': loss_percentage
            }

        return results

    def scenario_analysis(self, portfolio_weights: np.ndarray,
                        asset_scenarios: Dict[str, np.ndarray],
                        portfolio_value: float = 1.0) -> Dict[str, Any]:
        """
        Perform scenario analysis with multiple asset return scenarios.

        Args:
            portfolio_weights: Portfolio weights
            asset_scenarios: Dictionary of scenarios with asset return arrays
            portfolio_value: Portfolio value

        Returns:
            Dictionary with scenario analysis results
        """
        results = {}

        for scenario_name, asset_returns in asset_scenarios.items():
            # Calculate portfolio returns for this scenario
            portfolio_returns = asset_returns.dot(portfolio_weights)

            # Calculate portfolio value changes
            final_value = portfolio_value * (1 + portfolio_returns).prod()
            pnl = final_value - portfolio_value
            pnl_percentage = (pnl / portfolio_value) * 100

            results[scenario_name] = {
                'final_value': final_value,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'portfolio_returns': portfolio_returns,
                'worst_month': portfolio_returns.min(),
                'best_month': portfolio_returns.max(),
                'volatility': portfolio_returns.std() * np.sqrt(12)  # Annualized
            }

        return results

    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.

        Args:
            portfolio_values: Time series of portfolio values

        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate cumulative returns
        cumulative = portfolio_values / portfolio_values.iloc[0]

        # Calculate running maximum
        running_max = cumulative.expanding().max()

        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Current drawdown
        current_drawdown = drawdown.iloc[-1]

        # Drawdown duration
        drawdown_periods = drawdown < 0
        if drawdown_periods.any():
            drawdown_durations = []
            start_idx = None

            for i, is_drawdown in enumerate(drawdown_periods):
                if is_drawdown and start_idx is None:
                    start_idx = i
                elif not is_drawdown and start_idx is not None:
                    drawdown_durations.append(i - start_idx)
                    start_idx = None

            if start_idx is not None:
                drawdown_durations.append(len(drawdown_periods) - start_idx)

            max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0
            avg_drawdown_duration = np.mean(drawdown_durations) if drawdown_durations else 0
        else:
            max_drawdown_duration = 0
            avg_drawdown_duration = 0

        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'drawdown_series': drawdown
        }

    def calculate_risk_parity_weights(self, returns: pd.DataFrame,
                                    target_risk: float = None) -> np.ndarray:
        """
        Calculate risk parity portfolio weights.

        Args:
            returns: Asset returns DataFrame
            target_risk: Target portfolio risk (if None, uses equal risk contribution)

        Returns:
            Risk parity weights array
        """
        cov_matrix = returns.cov()
        n_assets = len(returns.columns)

        # Initial equal weights
        weights = np.ones(n_assets) / n_assets

        # Risk parity optimization (simplified)
        for _ in range(100):  # Iterative optimization
            # Calculate portfolio volatility contribution of each asset
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            asset_contributions = (weights * (cov_matrix @ weights)) / portfolio_vol

            # Adjust weights to equalize risk contributions
            weights = weights * (1 / asset_contributions)
            weights = weights / np.sum(weights)

        return weights

    def calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate tail risk metrics like kurtosis, skewness, and tail ratios.

        Args:
            returns: Returns series

        Returns:
            Dictionary with tail risk metrics
        """
        # Basic moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Tail ratios
        gain_threshold = returns[returns > 0].quantile(0.95) if len(returns[returns > 0]) > 0 else 0
        loss_threshold = returns[returns < 0].quantile(0.05) if len(returns[returns < 0]) > 0 else 0

        tail_ratio = abs(gain_threshold / loss_threshold) if loss_threshold != 0 else np.inf

        # Value at Risk ratio (annualized)
        var_95 = self.calculate_var_parametric(returns, method='parametric')['VaR']
        expected_return = returns.mean() * 252  # Annualized
        var_ratio = abs(expected_return / var_95) if var_95 != 0 else np.inf

        # Conditional VaR ratio
        cvar_95 = self.calculate_var_parametric(returns, method='parametric')['CVaR']
        cvar_ratio = abs(expected_return / cvar_95) if cvar_95 != 0 else np.inf

        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            'var_ratio': var_ratio,
            'cvar_ratio': cvar_ratio,
            'gain_threshold_95': gain_threshold,
            'loss_threshold_5': loss_threshold
        }

    def backtest_var(self, returns: pd.Series, var_series: pd.Series,
                   portfolio_value: float = 1.0) -> Dict[str, float]:
        """
        Backtest VaR model performance.

        Args:
            returns: Actual returns series
            var_series: Predicted VaR series (positive values = loss thresholds)
            portfolio_value: Portfolio value

        Returns:
            Dictionary with backtest metrics
        """
        # Calculate actual losses
        losses = -returns * portfolio_value

        # Count violations (losses exceeding VaR)
        violations = losses > var_series
        n_violations = violations.sum()
        n_observations = len(losses)

        # Violation rate
        violation_rate = n_violations / n_observations

        # Expected violation rate
        expected_rate = 1 - self.confidence_level

        # Kupiec test (unconditional coverage)
        if n_violations == 0:
            kupiec_p_value = 1.0
        else:
            log_likelihood = (n_observations - n_violations) * np.log(1 - expected_rate) + \
                           n_violations * np.log(expected_rate)
            log_likelihood_alt = (n_observations - n_violations) * np.log(1 - violation_rate) + \
                               n_violations * np.log(violation_rate)
            test_statistic = -2 * (log_likelihood - log_likelihood_alt)
            kupiec_p_value = 1 - chi2.cdf(test_statistic, 1)

        # Average violation size
        if n_violations > 0:
            avg_violation = np.mean(losses[violations] - var_series[violations])
        else:
            avg_violation = 0

        return {
            'n_violations': n_violations,
            'n_observations': n_observations,
            'violation_rate': violation_rate,
            'expected_violation_rate': expected_rate,
            'kupiec_p_value': kupiec_p_value,
            'avg_violation_size': avg_violation,
            'backtest_passed': abs(violation_rate - expected_rate) < 0.02  # Within 2%
        }


class PortfolioRiskManager:
    """
    Portfolio risk management system integrating multiple risk metrics.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.risk_calculator = RiskMetricsCalculator(confidence_level)

    def comprehensive_risk_report(self, portfolio_weights: np.ndarray,
                                asset_returns: pd.DataFrame,
                                portfolio_value: float = 1.0) -> Dict[str, Any]:
        """
        Generate comprehensive risk report for a portfolio.

        Args:
            portfolio_weights: Portfolio weights
            asset_returns: Historical asset returns
            portfolio_value: Portfolio value

        Returns:
            Comprehensive risk report
        """
        # Portfolio returns
        portfolio_returns = asset_returns.dot(portfolio_weights)

        # VaR calculations
        var_historical = self.risk_calculator.calculate_var_historical(portfolio_returns, portfolio_value)
        var_parametric = self.risk_calculator.calculate_var_parametric(portfolio_returns, portfolio_value)
        var_mc = self.risk_calculator.calculate_var_monte_carlo(portfolio_returns, portfolio_value)

        # Risk metrics
        tail_risk = self.risk_calculator.calculate_tail_risk_metrics(portfolio_returns)

        # Portfolio statistics
        portfolio_stats = {
            'expected_return': portfolio_returns.mean() * 252,  # Annualized
            'volatility': portfolio_returns.std() * np.sqrt(252),  # Annualized
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
            'max_drawdown': self.risk_calculator.calculate_max_drawdown(
                (1 + portfolio_returns).cumprod() * portfolio_value
            )['max_drawdown']
        }

        return {
            'portfolio_stats': portfolio_stats,
            'VaR_historical': var_historical,
            'VaR_parametric': var_parametric,
            'VaR_monte_carlo': var_mc,
            'tail_risk_metrics': tail_risk,
            'risk_parity_weights': self.risk_calculator.calculate_risk_parity_weights(asset_returns)
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample returns data
    n_days = 252
    returns = pd.Series(np.random.normal(0.001, 0.02, n_days), name='portfolio_returns')

    # Initialize risk calculator
    risk_calc = RiskMetricsCalculator(confidence_level=0.95, time_horizon=1)

    # Calculate VaR
    var_hist = risk_calc.calculate_var_historical(returns, portfolio_value=1000000)
    var_param = risk_calc.calculate_var_parametric(returns, portfolio_value=1000000)

    print("Historical VaR (1-day, 95%): ${:,.2f}".format(var_hist['VaR']))
    print("Parametric VaR (1-day, 95%): ${:,.2f}".format(var_param['VaR']))

    # Tail risk metrics
    tail_risk = risk_calc.calculate_tail_risk_metrics(returns)
    print(f"Skewness: {tail_risk['skewness']:.3f}")
    print(f"Kurtosis: {tail_risk['kurtosis']:.3f}")

    # Maximum drawdown
    portfolio_values = (1 + returns).cumprod() * 1000000
    drawdown = risk_calc.calculate_max_drawdown(portfolio_values)
    print("Maximum Drawdown: {:.2%}".format(drawdown['max_drawdown']))