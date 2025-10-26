"""
Analytic pricing methods for financial derivatives.

This module provides closed-form pricing formulas for various
financial instruments including options, swaps, and swaptions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from scipy.stats import norm
from scipy.optimize import brentq
import logging

logger = logging.getLogger(__name__)


class BlackScholesPricer:
    """
    Black-Scholes option pricing model.

    Provides pricing and risk metrics for European options.
    """

    def __init__(self):
        """Initialize Black-Scholes pricer."""
        pass

    @staticmethod
    def d1(spot: float, strike: float, time: float,
          rate: float, vol: float, dividend: float = 0.0) -> float:
        """
        Calculate d1 parameter for Black-Scholes formula.

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to expiry
            rate: Risk-free rate
            vol: Volatility
            dividend: Dividend yield

        Returns:
            d1 value
        """
        return (np.log(spot/strike) + (rate - dividend + vol**2/2) * time) / (vol * np.sqrt(time))

    @staticmethod
    def d2(spot: float, strike: float, time: float,
          rate: float, vol: float, dividend: float = 0.0) -> float:
        """
        Calculate d2 parameter for Black-Scholes formula.

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to expiry
            rate: Risk-free rate
            vol: Volatility
            dividend: Dividend yield

        Returns:
            d2 value
        """
        return BlackScholesPricer.d1(spot, strike, time, rate, vol, dividend) - vol * np.sqrt(time)

    def call_price(self, spot: float, strike: float, time: float,
                  rate: float, vol: float, dividend: float = 0.0) -> float:
        """
        Calculate European call option price.

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to expiry
            rate: Risk-free rate
            vol: Volatility
            dividend: Dividend yield

        Returns:
            Call option price
        """
        if time <= 0:
            return max(spot - strike, 0)

        d1 = self.d1(spot, strike, time, rate, vol, dividend)
        d2 = self.d2(spot, strike, time, rate, vol, dividend)

        return spot * np.exp(-dividend * time) * norm.cdf(d1) - \
               strike * np.exp(-rate * time) * norm.cdf(d2)

    def put_price(self, spot: float, strike: float, time: float,
                 rate: float, vol: float, dividend: float = 0.0) -> float:
        """
        Calculate European put option price.

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to expiry
            rate: Risk-free rate
            vol: Volatility
            dividend: Dividend yield

        Returns:
            Put option price
        """
        if time <= 0:
            return max(strike - spot, 0)

        d1 = self.d1(spot, strike, time, rate, vol, dividend)
        d2 = self.d2(spot, strike, time, rate, vol, dividend)

        return strike * np.exp(-rate * time) * norm.cdf(-d2) - \
               spot * np.exp(-dividend * time) * norm.cdf(-d1)

    def option_greeks(self, spot: float, strike: float, time: float,
                     rate: float, vol: float, dividend: float = 0.0,
                     option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks.

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to expiry
            rate: Risk-free rate
            vol: Volatility
            dividend: Dividend yield
            option_type: 'call' or 'put'

        Returns:
            Dictionary of Greeks
        """
        if time <= 0:
            return {'delta': 1.0 if option_type == 'call' else -1.0,
                   'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

        d1 = self.d1(spot, strike, time, rate, vol, dividend)
        d2 = d1 - vol * np.sqrt(time)

        # Common terms
        pdf_d1 = norm.pdf(d1)
        exp_rt = np.exp(-rate * time)
        exp_dt = np.exp(-dividend * time)
        sqrt_t = np.sqrt(time)

        if option_type == 'call':
            delta = exp_dt * norm.cdf(d1)
            rho = strike * time * exp_rt * norm.cdf(d2)
        else:  # put
            delta = -exp_dt * norm.cdf(-d1)
            rho = -strike * time * exp_rt * norm.cdf(-d2)

        gamma = exp_dt * pdf_d1 / (spot * vol * sqrt_t)
        theta = (-spot * exp_dt * pdf_d1 * vol / (2 * sqrt_t) -
                rate * strike * exp_rt * norm.cdf(d2) +
                dividend * spot * exp_dt * norm.cdf(d1))
        vega = spot * exp_dt * pdf_d1 * sqrt_t

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def implied_volatility(self, market_price: float, spot: float, strike: float,
                          time: float, rate: float, dividend: float = 0.0,
                          option_type: str = 'call', tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Args:
            market_price: Market option price
            spot: Spot price
            strike: Strike price
            time: Time to expiry
            rate: Risk-free rate
            dividend: Dividend yield
            option_type: 'call' or 'put'
            tolerance: Convergence tolerance

        Returns:
            Implied volatility
        """
        def objective(vol):
            if option_type == 'call':
                return self.call_price(spot, strike, time, rate, vol, dividend) - market_price
            else:
                return self.put_price(spot, strike, time, rate, vol, dividend) - market_price

        def derivative(vol):
            # Vega is the derivative of price w.r.t. volatility
            greeks = self.option_greeks(spot, strike, time, rate, vol, dividend, option_type)
            return greeks['vega'] / 100  # Vega is usually quoted per 1% vol change

        # Initial guess
        vol = 0.2

        # Newton-Raphson iteration
        for _ in range(100):
            price_diff = objective(vol)
            if abs(price_diff) < tolerance:
                return vol

            vega = derivative(vol)
            if abs(vega) < 1e-8:
                break

            vol = vol - price_diff / vega

            # Ensure volatility stays positive
            vol = max(vol, 0.001)

        # Fallback to bisection if Newton-Raphson fails
        try:
            return brentq(objective, 0.001, 5.0, xtol=tolerance)
        except:
            logger.warning("Implied volatility calculation failed")
            return np.nan


class SABRPricer:
    """
    SABR (Stochastic Alpha Beta Rho) model for volatility smile.

    Used for pricing options with stochastic volatility and correlation.
    """

    def __init__(self, alpha: float = 0.2, beta: float = 0.7,
                 rho: float = -0.3, nu: float = 0.3):
        """
        Initialize SABR model parameters.

        Args:
            alpha: Initial volatility
            beta: Elasticity parameter (0 < beta <= 1)
            rho: Correlation between forward and volatility
            nu: Volatility of volatility
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def volatility(self, forward: float, strike: float, time: float) -> float:
        """
        Calculate SABR implied volatility.

        Args:
            forward: Forward price
            strike: Strike price
            time: Time to expiry

        Returns:
            Implied volatility
        """
        if abs(forward - strike) < 1e-8:
            # ATM case
            return self.alpha / (forward ** (1 - self.beta))

        # Log moneyness
        log_fk = np.log(forward / strike)

        # SABR volatility formula
        z = (self.nu / self.alpha) * (forward * strike) ** ((1 - self.beta) / 2) * log_fk

        if abs(z) < 1e-6:
            x_z = 1
        else:
            x_z = z / np.log((np.sqrt(1 - 2 * self.rho * z + z**2) + z - self.rho) /
                           (1 - self.rho))

        term1 = self.alpha / ((forward * strike) ** ((1 - self.beta) / 2))
        term2 = (1 + ((1 - self.beta)**2 / 24) * log_fk**2 +
                ((1 - self.beta)**4 / 1920) * log_fk**4)
        term3 = 1 + (self.nu**2 / 24) * time

        return term1 * x_z * term2 * term3


class SwaptionPricer:
    """
    Analytic swaption pricing using Black model.

    Prices European swaptions using the Black formula for swaptions.
    """

    def __init__(self):
        """Initialize swaption pricer."""
        self.bs_pricer = BlackScholesPricer()

    def annuity_factor(self, swap_rate: float, tenor: float,
                      discount_factors: np.ndarray) -> float:
        """
        Calculate annuity factor for swap pricing.

        Args:
            swap_rate: Swap rate
            tenor: Swap tenor in years
            discount_factors: Array of discount factors

        Returns:
            Annuity factor
        """
        # Simplified annuity calculation
        n_payments = int(tenor * 2)  # Semi-annual payments
        payment_times = np.linspace(0.5, tenor, n_payments)

        # Approximate discount factors (would need yield curve in practice)
        df = np.exp(-swap_rate * payment_times)

        return np.sum(df)

    def swaption_price(self, swap_rate: float, strike: float,
                      option_tenor: float, swap_tenor: float,
                      volatility: float, risk_free_rate: float = 0.03) -> float:
        """
        Price European swaption using Black model.

        Args:
            swap_rate: Current swap rate (forward rate)
            strike: Strike swap rate
            option_tenor: Time to option expiry
            swap_tenor: Underlying swap tenor
            volatility: Swaption volatility
            risk_free_rate: Risk-free rate

        Returns:
            Swaption price
        """
        # Simplified Black model for swaptions
        # In practice, would need proper annuity factor calculation

        # Forward swap rate (simplified)
        forward_rate = swap_rate

        # Annuity factor approximation
        annuity = self.annuity_factor(swap_rate, swap_tenor, None)

        # Black formula for swaptions
        d1 = (np.log(forward_rate/strike) + (volatility**2/2) * option_tenor) / \
             (volatility * np.sqrt(option_tenor))
        d2 = d1 - volatility * np.sqrt(option_tenor)

        price = annuity * (forward_rate * norm.cdf(d1) - strike * norm.cdf(d2))

        return price

    def swaption_greeks(self, swap_rate: float, strike: float,
                       option_tenor: float, swap_tenor: float,
                       volatility: float) -> Dict[str, float]:
        """
        Calculate swaption Greeks.

        Args:
            swap_rate: Current swap rate
            strike: Strike swap rate
            option_tenor: Time to option expiry
            swap_tenor: Underlying swap tenor
            volatility: Swaption volatility

        Returns:
            Dictionary of Greeks
        """
        # Simplified Greeks calculation
        forward_rate = swap_rate
        annuity = self.annuity_factor(swap_rate, swap_tenor, None)

        d1 = (np.log(forward_rate/strike) + (volatility**2/2) * option_tenor) / \
             (volatility * np.sqrt(option_tenor))
        d2 = d1 - volatility * np.sqrt(option_tenor)

        # Delta approximation
        delta = annuity * norm.cdf(d1)

        # Vega
        vega = annuity * forward_rate * np.sqrt(option_tenor) * norm.pdf(d1)

        return {
            'delta': delta,
            'vega': vega,
            'gamma': 0.0,  # Simplified
            'theta': 0.0   # Simplified
        }


class BondPricer:
    """
    Analytic bond pricing and yield calculations.
    """

    def __init__(self):
        """Initialize bond pricer."""
        pass

    def bond_price(self, face_value: float, coupon_rate: float,
                  maturity: float, ytm: float, frequency: int = 2) -> float:
        """
        Calculate bond price using present value of cash flows.

        Args:
            face_value: Bond face value
            coupon_rate: Annual coupon rate
            maturity: Time to maturity in years
            ytm: Yield to maturity
            frequency: Coupon payment frequency per year

        Returns:
            Bond price
        """
        coupon = face_value * coupon_rate / frequency
        n_periods = int(maturity * frequency)

        price = 0
        for t in range(1, n_periods + 1):
            price += coupon / (1 + ytm/frequency) ** t

        price += face_value / (1 + ytm/frequency) ** n_periods

        return price

    def yield_to_maturity(self, price: float, face_value: float,
                         coupon_rate: float, maturity: float,
                         frequency: int = 2, tolerance: float = 1e-6) -> float:
        """
        Calculate yield to maturity using numerical methods.

        Args:
            price: Bond price
            face_value: Bond face value
            coupon_rate: Annual coupon rate
            maturity: Time to maturity in years
            frequency: Coupon payment frequency per year
            tolerance: Convergence tolerance

        Returns:
            Yield to maturity
        """
        def objective(ytm):
            return self.bond_price(face_value, coupon_rate, maturity, ytm, frequency) - price

        # Initial guess
        ytm_guess = coupon_rate

        try:
            return brentq(objective, 0.001, 0.5, xtol=tolerance)
        except:
            logger.warning("YTM calculation failed")
            return np.nan

    def bond_duration(self, face_value: float, coupon_rate: float,
                     maturity: float, ytm: float, frequency: int = 2) -> float:
        """
        Calculate Macaulay duration.

        Args:
            face_value: Bond face value
            coupon_rate: Annual coupon rate
            maturity: Time to maturity in years
            ytm: Yield to maturity
            frequency: Coupon payment frequency per year

        Returns:
            Macaulay duration
        """
        coupon = face_value * coupon_rate / frequency
        n_periods = int(maturity * frequency)
        ytm_period = ytm / frequency

        duration = 0
        pv_total = 0

        for t in range(1, n_periods + 1):
            pv_cf = coupon / (1 + ytm_period) ** t
            duration += t * pv_cf
            pv_total += pv_cf

        # Face value payment
        pv_face = face_value / (1 + ytm_period) ** n_periods
        duration += n_periods * pv_face
        pv_total += pv_face

        return duration / pv_total / frequency  # Convert to years


if __name__ == "__main__":
    # Example usage
    bs = BlackScholesPricer()

    # Price a call option
    call_price = bs.call_price(spot=100, strike=105, time=1.0, rate=0.05, vol=0.2)
    put_price = bs.put_price(spot=100, strike=105, time=1.0, rate=0.05, vol=0.2)

    print(f"Call price: ${call_price:.4f}")
    print(f"Put price: ${put_price:.4f}")

    # Calculate Greeks
    greeks = bs.option_greeks(spot=100, strike=105, time=1.0, rate=0.05, vol=0.2)
    print(f"Greeks: {greeks}")

    # Swaption pricing
    swaption_pricer = SwaptionPricer()
    swaption_price = swaption_pricer.swaption_price(
        swap_rate=0.03, strike=0.035, option_tenor=1.0,
        swap_tenor=5.0, volatility=0.15
    )
    print(f"Swaption price: ${swaption_price:.6f}")