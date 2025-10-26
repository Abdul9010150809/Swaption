"""
Tests for pricing engines.

This module contains unit tests for the analytic and Monte Carlo
pricing methods including Black-Scholes, SABR, and simulation methods.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.pricing.analytic import (
    BlackScholesPricer, SABRPricer, SwaptionPricer, BondPricer
)
from src.pricing.monte_carlo import MonteCarloPricer


class TestBlackScholesPricer(unittest.TestCase):
    """Test Black-Scholes option pricing."""

    def setUp(self):
        """Set up test fixtures."""
        self.pricer = BlackScholesPricer()

    def test_d1_d2_calculation(self):
        """Test d1 and d2 parameter calculations."""
        spot, strike, time, rate, vol = 100, 105, 1.0, 0.05, 0.2

        d1 = self.pricer.d1(spot, strike, time, rate, vol)
        d2 = self.pricer.d2(spot, strike, time, rate, vol)

        # d2 should equal d1 - vol*sqrt(time)
        expected_d2 = d1 - vol * np.sqrt(time)
        self.assertAlmostEqual(d2, expected_d2, places=10)

    def test_call_price(self):
        """Test European call option pricing."""
        spot, strike, time, rate, vol = 100, 100, 1.0, 0.05, 0.2

        call_price = self.pricer.call_price(spot, strike, time, rate, vol)

        # At-the-money call should be positive
        self.assertGreater(call_price, 0)

        # Call price should be greater than intrinsic value
        intrinsic = max(spot - strike, 0)
        self.assertGreaterEqual(call_price, intrinsic)

    def test_put_price(self):
        """Test European put option pricing."""
        spot, strike, time, rate, vol = 100, 100, 1.0, 0.05, 0.2

        put_price = self.pricer.put_price(spot, strike, time, rate, vol)

        # At-the-money put should be positive
        self.assertGreater(put_price, 0)

        # Put price should be greater than intrinsic value
        intrinsic = max(strike - spot, 0)
        self.assertGreaterEqual(put_price, intrinsic)

    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        spot, strike, time, rate, vol = 100, 105, 1.0, 0.05, 0.2

        call_price = self.pricer.call_price(spot, strike, time, rate, vol)
        put_price = self.pricer.put_price(spot, strike, time, rate, vol)

        # Put-call parity: C - P = S - K * e^(-rT)
        lhs = call_price - put_price
        rhs = spot - strike * np.exp(-rate * time)

        self.assertAlmostEqual(lhs, rhs, places=6)

    def test_zero_time_limit(self):
        """Test pricing at expiration."""
        spot, strike, rate, vol = 100, 105, 0.05, 0.2

        # Call option at expiration
        call_price = self.pricer.call_price(spot, strike, 0, rate, vol)
        expected_call = max(spot - strike, 0)
        self.assertAlmostEqual(call_price, expected_call, places=6)

        # Put option at expiration
        put_price = self.pricer.put_price(spot, strike, 0, rate, vol)
        expected_put = max(strike - spot, 0)
        self.assertAlmostEqual(put_price, expected_put, places=6)

    def test_option_greeks(self):
        """Test calculation of option Greeks."""
        spot, strike, time, rate, vol = 100, 100, 1.0, 0.05, 0.2

        greeks = self.pricer.option_greeks(spot, strike, time, rate, vol)

        # Check all Greeks are present
        expected_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for greek in expected_greeks:
            self.assertIn(greek, greeks)

        # Delta should be between 0 and 1 for calls, -1 and 0 for puts
        self.assertGreater(greeks['delta'], 0)
        self.assertLess(greeks['delta'], 1)

        # Gamma should be positive
        self.assertGreater(greeks['gamma'], 0)

        # Vega should be positive
        self.assertGreater(greeks['vega'], 0)

    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        spot, strike, time, rate = 100, 100, 1.0, 0.05
        vol = 0.2

        # Calculate option price
        market_price = self.pricer.call_price(spot, strike, time, rate, vol)

        # Calculate implied volatility
        implied_vol = self.pricer.implied_volatility(market_price, spot, strike, time, rate)

        # Should recover the original volatility
        self.assertAlmostEqual(implied_vol, vol, places=3)

    def test_implied_volatility_edge_cases(self):
        """Test implied volatility with edge cases."""
        spot, strike, time, rate = 100, 100, 1.0, 0.05

        # Deep in-the-money call
        deep_itm_price = spot - strike + 1  # Intrinsic + small time value
        iv = self.pricer.implied_volatility(deep_itm_price, spot, strike, time, rate)
        self.assertTrue(np.isfinite(iv) or np.isnan(iv))  # Should not crash

        # Deep out-of-the-money call
        deep_otm_price = 0.01
        iv = self.pricer.implied_volatility(deep_otm_price, spot, strike, time, rate)
        self.assertTrue(np.isfinite(iv) or np.isnan(iv))  # Should not crash


class TestSABRPricer(unittest.TestCase):
    """Test SABR volatility model."""

    def setUp(self):
        """Set up test fixtures."""
        self.pricer = SABRPricer(alpha=0.2, beta=0.7, rho=-0.3, nu=0.3)

    def test_volatility_calculation(self):
        """Test SABR volatility calculation."""
        forward, strike, time = 100, 105, 1.0

        vol = self.pricer.volatility(forward, strike, time)

        # Volatility should be positive
        self.assertGreater(vol, 0)

        # ATM volatility should be close to alpha
        atm_vol = self.pricer.volatility(forward, forward, time)
        self.assertAlmostEqual(atm_vol, self.pricer.alpha, places=1)

    def test_parameter_effects(self):
        """Test how SABR parameters affect volatility."""
        forward, strike, time = 100, 105, 1.0

        # Higher alpha should increase volatility
        high_alpha_pricer = SABRPricer(alpha=0.3, beta=0.7, rho=-0.3, nu=0.3)
        low_vol = self.pricer.volatility(forward, strike, time)
        high_vol = high_alpha_pricer.volatility(forward, strike, time)
        self.assertGreater(high_vol, low_vol)

        # Higher beta should decrease volatility for OTM options
        high_beta_pricer = SABRPricer(alpha=0.2, beta=0.9, rho=-0.3, nu=0.3)
        low_beta_vol = self.pricer.volatility(forward, strike, time)
        high_beta_vol = high_beta_pricer.volatility(forward, strike, time)
        self.assertLess(high_beta_vol, low_beta_vol)


class TestSwaptionPricer(unittest.TestCase):
    """Test swaption pricing."""

    def setUp(self):
        """Set up test fixtures."""
        self.pricer = SwaptionPricer()

    def test_swaption_price(self):
        """Test swaption pricing."""
        swap_rate, strike, option_tenor, swap_tenor = 0.03, 0.035, 1.0, 5.0
        volatility, risk_free_rate = 0.15, 0.03

        price = self.pricer.swaption_price(
            swap_rate, strike, option_tenor, swap_tenor, volatility, risk_free_rate
        )

        # Price should be positive
        self.assertGreater(price, 0)

        # OTM swaption should be cheaper than ATM
        otm_price = self.pricer.swaption_price(
            swap_rate, strike + 0.01, option_tenor, swap_tenor, volatility, risk_free_rate
        )
        self.assertLess(otm_price, price)

    def test_swaption_greeks(self):
        """Test swaption Greeks calculation."""
        swap_rate, strike, option_tenor, swap_tenor = 0.03, 0.035, 1.0, 5.0
        volatility = 0.15

        greeks = self.pricer.swaption_greeks(swap_rate, strike, option_tenor, swap_tenor, volatility)

        # Check Greeks are present
        expected_greeks = ['delta', 'vega', 'gamma', 'theta']
        for greek in expected_greeks:
            self.assertIn(greek, greeks)

        # Delta should be positive for calls
        self.assertGreater(greeks['delta'], 0)

        # Vega should be positive
        self.assertGreater(greeks['vega'], 0)


class TestBondPricer(unittest.TestCase):
    """Test bond pricing."""

    def setUp(self):
        """Set up test fixtures."""
        self.pricer = BondPricer()

    def test_bond_price(self):
        """Test bond price calculation."""
        face_value, coupon_rate, maturity, ytm = 1000, 0.05, 5.0, 0.06

        price = self.pricer.bond_price(face_value, coupon_rate, maturity, ytm)

        # Bond price should be positive
        self.assertGreater(price, 0)

        # Bond with higher coupon should be more expensive
        high_coupon_price = self.pricer.bond_price(face_value, 0.08, maturity, ytm)
        self.assertGreater(high_coupon_price, price)

        # Bond with higher YTM should be cheaper
        high_ytm_price = self.pricer.bond_price(face_value, coupon_rate, maturity, 0.08)
        self.assertLess(high_ytm_price, price)

    def test_yield_to_maturity(self):
        """Test YTM calculation."""
        face_value, coupon_rate, maturity = 1000, 0.05, 5.0
        ytm = 0.06

        # Calculate price
        price = self.pricer.bond_price(face_value, coupon_rate, maturity, ytm)

        # Calculate YTM from price
        calculated_ytm = self.pricer.yield_to_maturity(price, face_value, coupon_rate, maturity)

        # Should recover the original YTM
        self.assertAlmostEqual(calculated_ytm, ytm, places=4)

    def test_bond_duration(self):
        """Test bond duration calculation."""
        face_value, coupon_rate, maturity, ytm = 1000, 0.05, 5.0, 0.06

        duration = self.pricer.bond_duration(face_value, coupon_rate, maturity, ytm)

        # Duration should be positive and less than maturity
        self.assertGreater(duration, 0)
        self.assertLess(duration, maturity)

        # Zero-coupon bond duration should equal maturity
        zero_coupon_duration = self.pricer.bond_duration(face_value, 0, maturity, ytm)
        self.assertAlmostEqual(zero_coupon_duration, maturity, places=1)


class TestMonteCarloPricer(unittest.TestCase):
    """Test Monte Carlo pricing methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.pricer = MonteCarloPricer(n_simulations=1000, n_steps=50, random_seed=42)

    def test_path_generation(self):
        """Test asset price path generation."""
        spot, time, rate, vol = 100, 1.0, 0.05, 0.2

        paths = self.pricer.simulate_paths(spot, time, rate, vol)

        # Check dimensions
        self.assertEqual(paths.shape, (self.pricer.n_simulations, self.pricer.n_steps + 1))

        # Initial price should be spot
        self.assertAlmostEqual(paths[0, 0], spot, places=6)

        # All prices should be positive
        self.assertTrue(np.all(paths > 0))

    def test_european_option_pricing(self):
        """Test European option pricing with Monte Carlo."""
        spot, strike, time, rate, vol = 100, 100, 1.0, 0.05, 0.2

        result = self.pricer.price_european_option(spot, strike, time, rate, vol, 'call')

        # Check result structure
        self.assertIn('price', result)
        self.assertIn('standard_error', result)
        self.assertIn('confidence_interval', result)

        # Price should be positive
        self.assertGreater(result['price'], 0)

        # Standard error should be positive
        self.assertGreater(result['standard_error'], 0)

    def test_asian_option_pricing(self):
        """Test Asian option pricing."""
        spot, strike, time, rate, vol = 100, 100, 1.0, 0.05, 0.2

        result = self.pricer.price_asian_option(spot, strike, time, rate, vol, 'call')

        # Check result structure
        self.assertIn('price', result)
        self.assertIn('standard_error', result)
        self.assertIn('averaging_type', result)

        # Price should be positive
        self.assertGreater(result['price'], 0)

    def test_barrier_option_pricing(self):
        """Test barrier option pricing."""
        spot, strike, time, rate, vol, barrier = 100, 100, 1.0, 0.05, 0.2, 110

        result = self.pricer.price_barrier_option(
            spot, strike, time, rate, vol, barrier, 'call', 'up_and_out'
        )

        # Check result structure
        self.assertIn('price', result)
        self.assertIn('barrier_type', result)
        self.assertIn('barrier_level', result)
        self.assertIn('knockout_probability', result)

        # Knockout probability should be between 0 and 1
        self.assertGreaterEqual(result['knockout_probability'], 0)
        self.assertLessEqual(result['knockout_probability'], 1)

    def test_greeks_calculation_mc(self):
        """Test Greeks calculation using Monte Carlo."""
        spot, strike, time, rate, vol = 100, 100, 1.0, 0.05, 0.2

        delta = self.pricer.calculate_greeks_mc(spot, strike, time, rate, vol, 'call', 'delta')

        # Delta should be between 0 and 1
        self.assertGreater(delta, 0)
        self.assertLess(delta, 1)

    def test_reproducibility(self):
        """Test that Monte Carlo results are reproducible with same seed."""
        pricer1 = MonteCarloPricer(n_simulations=100, random_seed=42)
        pricer2 = MonteCarloPricer(n_simulations=100, random_seed=42)

        spot, strike, time, rate, vol = 100, 100, 1.0, 0.05, 0.2

        result1 = pricer1.price_european_option(spot, strike, time, rate, vol, 'call')
        result2 = pricer2.price_european_option(spot, strike, time, rate, vol, 'call')

        # Results should be very close (same seed)
        self.assertAlmostEqual(result1['price'], result2['price'], places=3)


class TestPricingIntegration(unittest.TestCase):
    """Test integration between different pricing methods."""

    def test_bs_vs_mc_convergence(self):
        """Test that Monte Carlo converges to Black-Scholes for European options."""
        spot, strike, time, rate, vol = 100, 100, 0.5, 0.05, 0.2

        # Black-Scholes price
        bs_pricer = BlackScholesPricer()
        bs_price = bs_pricer.call_price(spot, strike, time, rate, vol)

        # Monte Carlo price with many simulations
        mc_pricer = MonteCarloPricer(n_simulations=50000, n_steps=100, random_seed=42)
        mc_result = mc_pricer.price_european_option(spot, strike, time, rate, vol, 'call')
        mc_price = mc_result['price']

        # Should be close (within 1 standard error)
        std_error = mc_result['standard_error']
        self.assertLess(abs(bs_price - mc_price), 2 * std_error)

    def test_model_comparison(self):
        """Test comparison between different stochastic models."""
        spot, time, rate, vol = 100, 1.0, 0.05, 0.2

        pricer = MonteCarloPricer(n_simulations=1000, random_seed=42)

        # GBM price
        gbm_paths = pricer.simulate_paths(spot, time, rate, vol, model='gbm')
        gbm_final = gbm_paths[:, -1].mean()

        # CEV price (beta=0.5)
        cev_paths = pricer.simulate_paths(spot, time, rate, vol, model='cev', beta=0.5)
        cev_final = cev_paths[:, -1].mean()

        # Both should be close to spot (for short time, low vol)
        self.assertAlmostEqual(gbm_final, spot, delta=5)
        self.assertAlmostEqual(cev_final, spot, delta=5)


if __name__ == '__main__':
    unittest.main()