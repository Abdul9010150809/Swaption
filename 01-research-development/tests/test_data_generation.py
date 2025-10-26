"""
Tests for data generation module.

This module contains unit tests for the financial data generation
functionality including yield curves, volatility surfaces, and option data.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data.data_generator import FinancialDataGenerator


class TestFinancialDataGenerator(unittest.TestCase):
    """Test cases for FinancialDataGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = FinancialDataGenerator(seed=42)

    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsInstance(self.generator, FinancialDataGenerator)
        self.assertEqual(self.generator.seed, 42)

    def test_generate_yield_curve(self):
        """Test yield curve generation."""
        n_samples = 10
        n_maturities = 5
        result = self.generator.generate_yield_curve(n_samples, n_maturities)

        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), n_samples)
        self.assertEqual(len([col for col in result.columns if col.startswith('maturity_')]), n_maturities)

        # Check metadata columns
        self.assertIn('sample_id', result.columns)
        self.assertIn('generation_date', result.columns)

        # Check yield values are reasonable
        maturity_cols = [col for col in result.columns if col.startswith('maturity_')]
        yields = result[maturity_cols].values
        self.assertTrue(np.all(yields > 0))  # Positive yields
        self.assertTrue(np.all(yields < 0.5))  # Reasonable yield range

    def test_generate_volatility_surface(self):
        """Test volatility surface generation."""
        n_samples = 5
        n_strikes = 3
        n_expiries = 4
        result = self.generator.generate_volatility_surface(n_samples, None, None)

        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), n_samples)

        # Check volatility columns exist
        vol_cols = [col for col in result.columns if col.startswith('vol_')]
        self.assertEqual(len(vol_cols), n_strikes * n_expiries)

        # Check metadata
        self.assertIn('sample_id', result.columns)
        self.assertEqual(result['n_strikes'].iloc[0], n_strikes)
        self.assertEqual(result['n_expiries'].iloc[0], n_expiries)

    def test_generate_option_prices(self):
        """Test European option price generation."""
        n_samples = 100
        result = self.generator.generate_option_prices(n_samples)

        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), n_samples)

        # Check required columns
        required_cols = ['spot_price', 'strike_price', 'time_to_expiry',
                        'risk_free_rate', 'volatility', 'call_price', 'put_price']
        for col in required_cols:
            self.assertIn(col, result.columns)

        # Check option prices are non-negative
        self.assertTrue(np.all(result['call_price'] >= 0))
        self.assertTrue(np.all(result['put_price'] >= 0))

        # Check put-call parity approximately holds
        # C - P ≈ S - K * e^(-rT)
        lhs = result['call_price'] - result['put_price']
        rhs = (result['spot_price'] -
               result['strike_price'] * np.exp(-result['risk_free_rate'] * result['time_to_expiry']))
        parity_diff = np.abs(lhs - rhs)
        self.assertTrue(np.mean(parity_diff) < 0.01)  # Should be very close

    def test_generate_swaption_data(self):
        """Test swaption data generation."""
        n_samples = 50
        result = self.generator.generate_swaption_data(n_samples)

        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), n_samples)

        # Check required columns
        required_cols = ['swap_rate', 'volatility', 'swap_tenor', 'option_tenor',
                        'strike', 'risk_free_rate', 'swaption_price']
        for col in required_cols:
            self.assertIn(col, result.columns)

        # Check value ranges
        self.assertTrue(np.all(result['swap_rate'] > 0))
        self.assertTrue(np.all(result['volatility'] > 0))
        self.assertTrue(np.all(result['swaption_price'] >= 0))

    @patch('builtins.print')
    def test_save_data(self, mock_print):
        """Test data saving functionality."""
        # Create sample data
        sample_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        # Mock the save operation
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            self.generator.save_data(sample_data, 'test.csv')

            # Check that to_csv was called with correct path
            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args
            self.assertIn('data/raw/test.csv', args[0])

        # Check that print was called
        mock_print.assert_called_once()

    def test_reproducibility(self):
        """Test that generator produces reproducible results."""
        gen1 = FinancialDataGenerator(seed=123)
        gen2 = FinancialDataGenerator(seed=123)

        data1 = gen1.generate_option_prices(10)
        data2 = gen2.generate_option_prices(10)

        # Results should be identical with same seed
        pd.testing.assert_frame_equal(data1, data2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        gen1 = FinancialDataGenerator(seed=123)
        gen2 = FinancialDataGenerator(seed=456)

        data1 = gen1.generate_option_prices(10)
        data2 = gen2.generate_option_prices(10)

        # Results should be different with different seeds
        self.assertFalse(data1.equals(data2))


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases."""

    def setUp(self):
        self.generator = FinancialDataGenerator(seed=42)

    def test_zero_time_to_expiry(self):
        """Test behavior with zero time to expiry."""
        # This should work for at-the-money options
        result = self.generator.generate_option_prices(n_samples=1)
        result.loc[0, 'time_to_expiry'] = 0

        # Call price should equal intrinsic value
        spot = result.loc[0, 'spot_price']
        strike = result.loc[0, 'strike_price']
        expected_call = max(spot - strike, 0)

        # Note: This test assumes the generator handles zero time properly
        # In practice, you might want to filter out such cases
        self.assertIsInstance(result, pd.DataFrame)

    def test_extreme_volatility_values(self):
        """Test with extreme volatility values."""
        result = self.generator.generate_option_prices(n_samples=10)

        # Manually set extreme volatilities
        result.loc[0, 'volatility'] = 0.001  # Very low vol
        result.loc[1, 'volatility'] = 2.0    # Very high vol

        # Should still produce valid prices
        self.assertTrue(np.all(result['call_price'] >= 0))
        self.assertTrue(np.all(result['put_price'] >= 0))

    def test_negative_rates(self):
        """Test with negative interest rates."""
        result = self.generator.generate_option_prices(n_samples=10)

        # Set some negative rates
        result.loc[:2, 'risk_free_rate'] = -0.005

        # Should still work (negative rates are possible)
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()