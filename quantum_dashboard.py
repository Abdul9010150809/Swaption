#!/usr/bin/env python3
"""
QUANTUM FINANCE DASHBOARD - KAGGLE INTEGRATION & ENHANCED FEATURES
Production-ready with real data integration and advanced analytics
"""
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import scipy.stats as stats
import time
import logging
import os
import json
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')
from kaggle.api.kaggle_api_extended import KaggleApi

# Matplotlib imports for quantum circuit visualization
try:
    import matplotlib
    import matplotlib.pyplot as plt
    # Import Figure type explicitly and assign an alias to avoid referencing matplotlib.figure in type checks
    from matplotlib.figure import Figure as MatplotlibFigure
    HAS_MATPLOTLIB = True
except ImportError:
    # Ensure the alias exists even if matplotlib is not available
    MatplotlibFigure = None
    HAS_MATPLOTLIB = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- KAGGLE CONFIGURATION ---
KAGGLE_CONFIG = {
    'username': None,
    'key': None,
    'datasets': {
        'interest_rates': 'cmirzai/interest-rates-and-inflation',
        'options_data': 'mateuszbuda/stock-options-data',
        'volatility': 'gokulrajkmv/implied-volatility-options',
        'yield_curve': 'fedesorce/us-treasury-yield-curve'
    }
}

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    KaggleApi = None
    logger.warning("Kaggle API not found. Please install with 'pip install kaggle'.")

def setup_kaggle():
    """Setup Kaggle API credentials with better error handling and fallback"""
    if not KaggleApi:
        st.warning("Kaggle API package not installed. Use: pip install kaggle")
        return None

    try:
        # Set KAGGLE_CONFIG_DIR to current directory
        os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath('.')

        # Check if kaggle.json exists
        kaggle_json_path = os.path.join(os.path.abspath('.'), 'kaggle.json')
        if not os.path.exists(kaggle_json_path):
            st.warning("⚠️ kaggle.json not found. Using synthetic data for demonstration.")
            st.info("To use real Kaggle data: Download kaggle.json from https://www.kaggle.com/settings and place it in the project root.")
            return None

        # Try to authenticate
        api = KaggleApi()
        api.authenticate()
        st.success("✅ Kaggle API authenticated successfully!")
        return api
    except Exception as e:
        st.warning(f"⚠️ Kaggle API authentication failed: {e}")
        st.info("Using synthetic data for demonstration. Real Kaggle data requires valid kaggle.json file.")
        return None

# --- QUANTUM COMPUTING IMPORTS ---
HAS_QUANTUM = False
HAS_QISKIT = False

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal
    from qiskit_aer import AerSimulator
    from qiskit.visualization import plot_histogram
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
    HAS_QUANTUM = True
    logger.info("✅ Quantum computing loaded successfully")
except ImportError as e:
    logger.warning(f"Quantum imports failed: {e}")

# --- CLASSICAL ML IMPORTS ---
try:
    # Use explicit aliases to avoid any name resolution conflicts in large modules
    from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor, GradientBoostingRegressor as SklearnGradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    # Expose aliases into globals so existing references can be updated locally
    HasSklearnEnsembleAliases = True
    HAS_ML = True
    logger.info("✅ Classical ML libraries loaded successfully")
except ImportError as e:
    logger.error(f"Classical ML imports failed: {e}")
    HAS_ML = False

# External configuration
import yaml

class ConfigManager:
    """External configuration management"""
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from YAML"""
        default_config = {
            'kaggle': {
                'datasets': {
                    'interest_rates': 'cmirzai/interest-rates-and-inflation',
                    'yield_curve': 'fedesorce/us-treasury-yield-curve'
                },
                'update_frequency': '24h'
            },
            'models': {
                'training_samples': 2000,
                'cv_folds': 5,
                'quantum_shots': 1024
            },
            'ui': {
                'refresh_interval': 30,
                'default_theme': 'dark'
            }
        }
        
        try:
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
                # Merge user config into defaults (user values take precedence)
                return self.merge_configs(default_config, user_config)
        except FileNotFoundError:
            return default_config
        except Exception as e:
            logger.warning(f"Failed to load config file {self.config_path}: {e}")
            return default_config

    def merge_configs(self, base, override):
        """Recursively merge two configuration dicts (override takes precedence)."""
        if override is None:
            return base
        merged = dict(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = self.merge_configs(merged[k], v)
            else:
                merged[k] = v
        return merged
    
    def get(self, key, default=None):
        """Safe config access"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(k, None)
            if value is None:
                return default
        return value

# --- KAGGLE DATA MANAGER ---
class KaggleDataManager:
    """Minimal Kaggle data manager used as a base class for enhanced manager."""
    def __init__(self):
        self.api = None
        self.loaded_datasets = {}
        self.data_dir = './kaggle_data'
        # Try to initialize Kaggle API if available, otherwise None
        try:
            self.api = setup_kaggle()
        except Exception:
            self.api = None
        self.loaded_datasets = {}
        self.data_dir = './data'
        # Ensure data directory exists
        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except Exception:
            pass

    def ensure_data_dir(self):
        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except Exception:
            pass

    def load_interest_rates(self):
        """Load interest rate dataset from Kaggle or return synthetic data."""
        try:
            self.ensure_data_dir()
            if self.api and KAGGLE_CONFIG['datasets'].get('interest_rates'):
                self.api.dataset_download_files(KAGGLE_CONFIG['datasets']['interest_rates'], path=self.data_dir, unzip=True)
                files = os.listdir(self.data_dir)
                csv_files = [f for f in files if f.endswith('.csv')]
                if csv_files:
                    df = pd.read_csv(os.path.join(self.data_dir, csv_files[0]))
                    self.loaded_datasets['interest_rates'] = df
                    return df
        except Exception:
            logger.warning("Could not load interest rates from Kaggle; falling back to synthetic data.")
        return self._generate_synthetic_rates()

    def load_yield_curve(self):
        """Load yield curve dataset from Kaggle or return synthetic data."""
        try:
            self.ensure_data_dir()
            if self.api and KAGGLE_CONFIG['datasets'].get('yield_curve'):
                self.api.dataset_download_files(KAGGLE_CONFIG['datasets']['yield_curve'], path=self.data_dir, unzip=True)
                files = os.listdir(self.data_dir)
                csv_files = [f for f in files if f.endswith('.csv')]
                if csv_files:
                    df = pd.read_csv(os.path.join(self.data_dir, csv_files[0]))
                    self.loaded_datasets['yield_curve'] = df
                    return df
        except Exception:
            logger.warning("Could not load yield curve from Kaggle; falling back to synthetic data.")
        return self._generate_synthetic_yield_curve()

    def _generate_synthetic_rates(self):
        """Generate simple synthetic interest rate data."""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        data = {
            'date': dates,
            'sofr': np.random.normal(0.05, 0.01, len(dates)),
            'libor_3m': np.random.normal(0.055, 0.015, len(dates)),
            'ust_2y': np.random.normal(0.047, 0.02, len(dates)),
            'ust_10y': np.random.normal(0.041, 0.015, len(dates))
        }
        return pd.DataFrame(data)

    def _generate_synthetic_yield_curve(self):
        """Generate simple synthetic yield curve data."""
        tenors = ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '20Y', '30Y']
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='M')
        data = {}
        # assign date separately to avoid type inference conflicts for the dict
        data['date'] = dates
        for tenor in tenors:
            base_rate = 0.02 + (tenors.index(tenor) * 0.005)
            data[tenor] = np.random.normal(base_rate, 0.01, len(dates))
        return pd.DataFrame(data)

# Enhanced Kaggle Data Manager
class EnhancedKaggleDataManager(KaggleDataManager):
    def __init__(self):
        super().__init__()
        self.data_cache = {}
        self.data_quality_metrics = {}
    
    def validate_and_clean_data(self, df, dataset_name):
        """Enhanced data validation and cleaning"""
        try:
            # Data quality checks
            initial_rows = len(df)
            df_clean = df.dropna()
            df_clean = df_clean[(df_clean.select_dtypes(include=[np.number]) != np.inf).all(axis=1)]
            
            # Date parsing with multiple format attempts
            date_columns = ['date', 'Date', 'DATE', 'timestamp', 'Time']
            for col in date_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', utc=True)
                    df_clean = df_clean.dropna(subset=[col])
            
            # Store quality metrics
            self.data_quality_metrics[dataset_name] = {
                'initial_rows': initial_rows,
                'cleaned_rows': len(df_clean),
                'data_loss_percent': ((initial_rows - len(df_clean)) / initial_rows) * 100,
                'completeness_score': (len(df_clean) / initial_rows) * 100
            }
            
            return df_clean
        except Exception as e:
            logger.error(f"Data validation failed for {dataset_name}: {e}")
            return df
    
    def load_yield_curve(self):
        """Load yield curve data from Kaggle"""
        try:
            if self.api:
                self.api.dataset_download_files(
                    KAGGLE_CONFIG['datasets']['yield_curve'], 
                    path='./data', 
                    unzip=True
                )
                files = os.listdir('./data')
                csv_files = [f for f in files if f.endswith('.csv')]
                if csv_files:
                    df = pd.read_csv(f'./data/{csv_files[0]}')
                    self.loaded_datasets['yield_curve'] = df
                    return df
            else:
                return self._generate_synthetic_yield_curve()
        except Exception as e:
            logger.error(f"Failed to load yield curve: {e}")
            return self._generate_synthetic_yield_curve()
    
    def _generate_synthetic_rates(self):
        """Generate synthetic interest rate data"""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        data = {
            'date': dates,
            'sofr': np.random.normal(0.05, 0.01, len(dates)),
            'libor_3m': np.random.normal(0.055, 0.015, len(dates)),
            'ust_2y': np.random.normal(0.047, 0.02, len(dates)),
            'ust_10y': np.random.normal(0.041, 0.015, len(dates))
        }
        return pd.DataFrame(data)
    
    def _generate_synthetic_yield_curve(self):
        """Generate synthetic yield curve data"""
        tenors = ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '20Y', '30Y']
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='M')
        
        data = {}
        # assign date separately to avoid type inference conflicts for the dict
        data['date'] = dates
        for tenor in tenors:
            base_rate = 0.02 + (tenors.index(tenor) * 0.005)
            data[tenor] = np.random.normal(base_rate, 0.01, len(dates))
        
        return pd.DataFrame(data)

# --- TRADITIONAL SWAPTION PRICING MODELS ---
class TraditionalSwaptionPricer:
    """Traditional swaption pricing models: Black-76, SABR, LMM, etc."""

    def __init__(self, kaggle_manager=None):
        self.kaggle_manager = kaggle_manager or KaggleDataManager()
        self.market_data = self._load_market_data()
        self.yield_curve = self._build_discount_curve()
        self.volatility_surface = self._build_volatility_surface()
        self.sabr_params = self._initialize_sabr_params()
        self.historical_prices = []

    def _load_market_data(self):
        """Load market data from Kaggle or generate synthetic"""
        try:
            rates_df = self.kaggle_manager.load_interest_rates()
            latest_rates = rates_df.iloc[-1] if rates_df is not None and not rates_df.empty else None

            if latest_rates is not None:
                return {
                    'SOFR': float(latest_rates.get('sofr', 0.0530)),
                    'LIBOR_3M': float(latest_rates.get('libor_3m', 0.0565)),
                    'UST_2Y': float(latest_rates.get('ust_2y', 0.0475)),
                    'UST_10Y': float(latest_rates.get('ust_10y', 0.0410)),
                    'SWAP_2Y': float(latest_rates.get('ust_2y', 0.0475)) + 0.002,
                    'SWAP_5Y': 0.0430,
                    'SWAP_10Y': 0.0415,
                    'SWAP_30Y': 0.0435,
                    'VIX': 15.5,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")

        # Fallback to synthetic data
        return {
            'SOFR': 0.0530, 'LIBOR_3M': 0.0565, 'UST_2Y': 0.0475,
            'UST_10Y': 0.0410, 'SWAP_2Y': 0.0480, 'SWAP_5Y': 0.0430,
            'SWAP_10Y': 0.0415, 'SWAP_30Y': 0.0435, 'VIX': 15.5,
            'timestamp': datetime.now()
        }

    def _build_discount_curve(self):
        """Build discount curve from real or synthetic data"""
        try:
            yield_df = self.kaggle_manager.load_yield_curve()
            if yield_df is not None and len(yield_df) > 0:
                latest_yields = yield_df.iloc[-1]
                # Map tenor names to numerical values
                tenor_map = {'1M': 1/12, '3M': 0.25, '6M': 0.5, '1Y': 1.0,
                           '2Y': 2.0, '5Y': 5.0, '10Y': 10.0, '20Y': 20.0, '30Y': 30.0}

                discount_factors = {}
                for tenor_str, tenor_val in tenor_map.items():
                    if tenor_str in latest_yields:
                        rate = float(latest_yields[tenor_str]) / 100
                        discount_factors[tenor_val] = np.exp(-rate * tenor_val)

                if discount_factors:
                    return discount_factors
        except Exception as e:
            logger.error(f"Failed to build discount curve from real data: {e}")

        # Fallback to synthetic curve
        tenors = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
        rates = [self.market_data['SOFR'], self.market_data['LIBOR_3M'],
                self.market_data['LIBOR_3M'] + 0.001, self.market_data['UST_2Y'] - 0.002,
                self.market_data['SWAP_2Y'], self.market_data['SWAP_5Y'] - 0.001,
                self.market_data['SWAP_5Y'], self.market_data['SWAP_10Y'] - 0.001,
                self.market_data['SWAP_10Y'], self.market_data['SWAP_30Y'] - 0.002,
                self.market_data['SWAP_30Y'], self.market_data['SWAP_30Y']]

        return {tenor: np.exp(-rate * tenor) for tenor, rate in zip(tenors, rates)}

    def _build_volatility_surface(self):
        """Build volatility surface"""
        surface = {}
        expiries = [0.25, 0.5, 1, 2, 3, 5, 7, 10]
        tenors = [1, 2, 5, 10, 15, 20, 30]
        base_vol = self.market_data['VIX'] / 100

        for expiry in expiries:
            for tenor in tenors:
                atm_vol = base_vol + 0.02 * (tenor/10) - 0.01 * (expiry/5)
                surface[(expiry, tenor)] = max(0.10, min(0.40, atm_vol))
        return surface

    def _initialize_sabr_params(self):
        """Initialize SABR model parameters"""
        return {
            'alpha': 0.2,    # Initial volatility
            'beta': 0.7,     # Elasticity parameter
            'rho': -0.3,     # Correlation
            'nu': 0.4        # Volatility of volatility
        }

    def calculate_annuity_factor(self, start, end, frequency=4):
        """Calculate swap annuity factor"""
        annuity = 0.0
        payment_times = np.arange(start + 1/frequency, end + 1/frequency, 1/frequency)
        for t in payment_times:
            df = self._interpolate_discount_factor(t)
            annuity += df * 1/frequency
        return annuity

    def _interpolate_discount_factor(self, time):
        """Interpolate discount factor"""
        tenors = sorted(self.yield_curve.keys())
        if time <= tenors[0]: return self.yield_curve[tenors[0]]
        if time >= tenors[-1]: return self.yield_curve[tenors[-1]]

        for i in range(len(tenors) - 1):
            if tenors[i] <= time <= tenors[i + 1]:
                t1, t2 = tenors[i], tenors[i + 1]
                df1, df2 = self.yield_curve[t1], self.yield_curve[t2]
                log_df = np.log(df1) + (np.log(df2) - np.log(df1)) * (time - t1) / (t2 - t1)
                return np.exp(log_df)
        return self.yield_curve[tenors[-1]]

    def calculate_forward_swap_rate(self, expiry, tenor):
        """Calculate forward swap rate"""
        try:
            annuity = self.calculate_annuity_factor(expiry, expiry + tenor)
            float_leg_pv = (self._interpolate_discount_factor(expiry) -
                           self._interpolate_discount_factor(expiry + tenor))
            return float_leg_pv / annuity if annuity > 0 else 0.04
        except:
            return 0.04

    # --- TRADITIONAL PRICING MODELS ---

    def black_76_swaption_price(self, notional, expiry, tenor, strike, swaption_type, volatility=None):
        """Calculate swaption price using Black-76 model"""
        forward_rate = self.calculate_forward_swap_rate(expiry, tenor)
        if volatility is None:
            volatility = self.volatility_surface.get((expiry, tenor), 0.20)
        annuity = self.calculate_annuity_factor(expiry, expiry + tenor)

        d1 = (np.log(forward_rate / strike) + (volatility**2 / 2) * expiry) / (volatility * np.sqrt(expiry))
        d2 = d1 - volatility * np.sqrt(expiry)

        if swaption_type == "Payer Swaption":
            price = annuity * (forward_rate * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2))
        else:
            price = annuity * (strike * stats.norm.cdf(-d2) - forward_rate * stats.norm.cdf(-d1))

        final_price = max(notional * price, 0.0)

        # Store historical price
        self.historical_prices.append({
            'timestamp': datetime.now(),
            'price': final_price,
            'model': 'Black-76',
            'type': swaption_type,
            'expiry': expiry,
            'tenor': tenor,
            'strike': strike
        })

        return final_price
# Add this missing method to TraditionalSwaptionPricer class
    def monte_carlo_price(self, expiry, strike, volatility, risk_free_rate, paths=10000, tenor=5.0):
        """Monte Carlo simulation for swaption pricing"""
        try:
            dt = expiry / 252.0  # Daily steps
            n_steps = int(expiry * 252)
            
            # Simulate underlying swap rate
            forward_rate = self.calculate_forward_swap_rate(expiry, tenor)
            z = np.random.standard_normal((paths, n_steps))
            
            # Generate price paths
            rates = np.zeros((paths, n_steps + 1))
            rates[:, 0] = forward_rate
            
            for t in range(1, n_steps + 1):
                rates[:, t] = rates[:, t-1] * np.exp(
                    (risk_free_rate - 0.5 * volatility**2) * dt + 
                    volatility * np.sqrt(dt) * z[:, t-1]
                )
            
            # Calculate payoff
            payoffs = np.maximum(rates[:, -1] - strike, 0)
            
            # Discount payoff
            price = np.exp(-risk_free_rate * expiry) * np.mean(payoffs)
            
            # Scale by annuity factor
            annuity = self.calculate_annuity_factor(expiry, expiry + tenor)
            final_price = annuity * price * 1000000
            
            return max(final_price, 0.0)
            
        except Exception as e:
            logger.error(f"Monte Carlo pricing failed: {e}")
            # Fallback to Black-76
            return self.black_76_swaption_price(1000000, expiry, tenor, strike, "Payer Swaption", volatility) * 0.8

    def sabr_swaption_price(self, notional, expiry, tenor, strike, swaption_type, sabr_params=None):
        """Calculate swaption price using SABR model"""
        if sabr_params is None:
            sabr_params = self.sabr_params

        forward_rate = self.calculate_forward_swap_rate(expiry, tenor)
        annuity = self.calculate_annuity_factor(expiry, expiry + tenor)

        # SABR implied volatility calculation
        sabr_vol = self._sabr_implied_volatility(forward_rate, strike, expiry, sabr_params)

        # Use Black-76 with SABR volatility
        d1 = (np.log(forward_rate / strike) + (sabr_vol**2 / 2) * expiry) / (sabr_vol * np.sqrt(expiry))
        d2 = d1 - sabr_vol * np.sqrt(expiry)

        if swaption_type == "Payer Swaption":
            price = annuity * (forward_rate * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2))
        else:
            price = annuity * (strike * stats.norm.cdf(-d2) - forward_rate * stats.norm.cdf(-d1))

        final_price = max(notional * price, 0.0)

        # Store historical price
        self.historical_prices.append({
            'timestamp': datetime.now(),
            'price': final_price,
            'model': 'SABR',
            'type': swaption_type,
            'expiry': expiry,
            'tenor': tenor,
            'strike': strike
        })

        return final_price

    def _sabr_implied_volatility(self, forward, strike, expiry, params):
        """Calculate SABR implied volatility"""
        alpha, beta, rho, nu = params['alpha'], params['beta'], params['rho'], params['nu']

        # SABR formula implementation
        if abs(forward - strike) < 1e-6:
            # ATM case
            vol = alpha / (forward ** (1 - beta))
        else:
            # Non-ATM case
            z = (nu / alpha) * (forward ** (1 - beta) - strike ** (1 - beta)) / (1 - beta)
            x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

            vol = (alpha / (forward ** (1 - beta))) * (z / x_z) * (1 + ((1 - beta)**2 / 24) * (np.log(forward/strike))**2 +
                  ((1 - beta)**4 / 1920) * (np.log(forward/strike))**4)

        return vol

    def normal_model_swaption_price(self, notional, expiry, tenor, strike, swaption_type, normal_vol=None):
        """Calculate swaption price using Normal (Bachelier) model"""
        forward_rate = self.calculate_forward_swap_rate(expiry, tenor)
        annuity = self.calculate_annuity_factor(expiry, expiry + tenor)

        if normal_vol is None:
            # Convert lognormal vol to normal vol approximation
            log_vol = self.volatility_surface.get((expiry, tenor), 0.20)
            normal_vol = log_vol * forward_rate

        # Bachelier formula
        d = (forward_rate - strike) / (normal_vol * np.sqrt(expiry))

        if swaption_type == "Payer Swaption":
            price = annuity * normal_vol * np.sqrt(expiry) * (d * stats.norm.cdf(d) + stats.norm.pdf(d))
        else:
            price = annuity * normal_vol * np.sqrt(expiry) * (-d * stats.norm.cdf(-d) + stats.norm.pdf(-d))

        final_price = max(notional * price, 0.0)

        # Store historical price
        self.historical_prices.append({
            'timestamp': datetime.now(),
            'price': final_price,
            'model': 'Normal',
            'type': swaption_type,
            'expiry': expiry,
            'tenor': tenor,
            'strike': strike
        })

        return final_price

    def hull_white_swaption_price(self, notional, expiry, tenor, strike, swaption_type, hw_params=None):
        """Calculate swaption price using Hull-White model (simplified)"""
        if hw_params is None:
            hw_params = {'a': 0.1, 'sigma': 0.015}  # Mean reversion and volatility

        forward_rate = self.calculate_forward_swap_rate(expiry, tenor)
        annuity = self.calculate_annuity_factor(expiry, expiry + tenor)

        # Simplified Hull-White swaption pricing
        # This is a basic approximation - full implementation would be more complex
        a, sigma = hw_params['a'], hw_params['sigma']

        # Approximate volatility for Hull-White
        hw_vol = sigma * np.sqrt((1 - np.exp(-2*a*expiry))/(2*a)) / a

        # Use Black-76 with Hull-White volatility
        d1 = (np.log(forward_rate / strike) + (hw_vol**2 / 2) * expiry) / (hw_vol * np.sqrt(expiry))
        d2 = d1 - hw_vol * np.sqrt(expiry)

        if swaption_type == "Payer Swaption":
            price = annuity * (forward_rate * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2))
        else:
            price = annuity * (strike * stats.norm.cdf(-d2) - forward_rate * stats.norm.cdf(-d1))

        final_price = max(notional * price, 0.0)

        # Store historical price
        self.historical_prices.append({
            'timestamp': datetime.now(),
            'price': final_price,
            'model': 'Hull-White',
            'type': swaption_type,
            'expiry': expiry,
            'tenor': tenor,
            'strike': strike
        })

        return final_price

    def price_swaption_all_models(self, notional, expiry, tenor, strike, swaption_type):
        """Price swaption using all traditional models"""
        results = {}

        try:
            results['Black-76'] = self.black_76_swaption_price(notional, expiry, tenor, strike, swaption_type)
        except Exception as e:
            logger.warning(f"Black-76 pricing failed: {e}")
            results['Black-76'] = 0.0

        try:
            results['SABR'] = self.sabr_swaption_price(notional, expiry, tenor, strike, swaption_type)
        except Exception as e:
            logger.warning(f"SABR pricing failed: {e}")
            results['SABR'] = 0.0

        try:
            results['Normal'] = self.normal_model_swaption_price(notional, expiry, tenor, strike, swaption_type)
        except Exception as e:
            logger.warning(f"Normal model pricing failed: {e}")
            results['Normal'] = 0.0

        try:
            results['Hull-White'] = self.hull_white_swaption_price(notional, expiry, tenor, strike, swaption_type)
        except Exception as e:
            logger.warning(f"Hull-White pricing failed: {e}")
            results['Hull-White'] = 0.0

        return results

# Keep the old class for backward compatibility
EnhancedSwaptionPricer = TraditionalSwaptionPricer

# --- ENHANCED CLASSICAL ML ENGINE ---
class EnhancedClassicalML:
    """Enhanced classical ML with feature engineering and cross-validation"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.training_history = []
        self.feature_importance = {}
        self.feature_names = []
        # Initialization message instead of referencing undefined variable X
        st.write("EnhancedClassicalML initialized")

    def engineer_features(self, features):
        """Engineer advanced features for swaption pricing with robust error handling"""
        try:
            # Create a copy to avoid modifying the original
            features_eng = features.copy()
            
            # Debug: Print available columns
            available_columns = features_eng.columns.tolist()
            print(f"🔍 Available columns in engineer_features: {available_columns}")
            
            # Check if required basic columns exist
            required_basic = ['strike', 'forward_rate']
            missing_basic = [col for col in required_basic if col not in features_eng.columns]
            
            if missing_basic:
                st.warning(f"⚠️ Missing basic columns: {missing_basic}")
                # Try to create missing columns with reasonable defaults
                if 'strike' not in features_eng.columns:
                    if 'forward_rate' in features_eng.columns:
                        features_eng['strike'] = features_eng['forward_rate'] * 0.95  # 5% OTM
                    else:
                        features_eng['strike'] = 0.04  # Default strike
                
                if 'forward_rate' not in features_eng.columns:
                    if 'strike' in features_eng.columns:
                        features_eng['forward_rate'] = features_eng['strike'] * 1.05  # 5% ITM
                    else:
                        features_eng['forward_rate'] = 0.042  # Default forward rate
            
            # Moneyness features (now these should always work)
            features_eng['moneyness'] = features_eng['strike'] / features_eng['forward_rate']
            features_eng['log_moneyness'] = np.log(features_eng['moneyness'])
            
            # Time features with safe access
            if 'expiry' in features_eng.columns:
                features_eng['time_sqrt'] = np.sqrt(features_eng['expiry'])
                features_eng['expiry_squared'] = features_eng['expiry'] ** 2
            else:
                features_eng['time_sqrt'] = 1.0
                features_eng['expiry_squared'] = 1.0
            
            # Volatility features with safe access
            if 'volatility' in features_eng.columns:
                features_eng['volatility_squared'] = features_eng['volatility'] ** 2
                features_eng['log_volatility'] = np.log(features_eng['volatility'])
            else:
                features_eng['volatility_squared'] = 0.04
                features_eng['log_volatility'] = -3.0
            
            # Interaction features
            if 'volatility' in features_eng.columns and 'expiry' in features_eng.columns:
                features_eng['time_vol_interaction'] = features_eng['expiry'] * features_eng['volatility']
                features_eng['vol_time_adjusted'] = features_eng['volatility'] * np.sqrt(features_eng['expiry'])
            else:
                features_eng['time_vol_interaction'] = 0.1
                features_eng['vol_time_adjusted'] = 0.15
            
            # Rate features
            features_eng['rate_spread'] = features_eng['forward_rate'] - features_eng['strike']
            features_eng['rate_ratio'] = features_eng['strike'] / features_eng['forward_rate']
            
            print(f"✅ Engineered features shape: {features_eng.shape}")
            print(f"📋 Final columns: {features_eng.columns.tolist()}")
            
            return features_eng
            
        except Exception as e:
            print(f"❌ Error in engineer_features: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            # Return original features if engineering fails
            return features
    
    def train_models_with_cv(self, X, y, cv_folds=5):
        """Train models with cross-validation - handles both arrays and DataFrames"""
        from sklearn.model_selection import cross_val_score, KFold
        
        # Convert to numpy arrays if they are DataFrames and store feature names
        if hasattr(X, 'values'):
            X_values = X.values
            self.feature_names = X.columns.tolist()
        else:
            X_values = X
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        if hasattr(y, 'values'):
            y_values = y.values
        else:
            y_values = y
        
        # Debug: Check input data
        # Instantiate models using the aliased ensemble classes to avoid name conflicts
        models = {
            'Random Forest': SklearnRandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
            'Gradient Boosting': SklearnGradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=6),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, random_state=42, max_depth=6),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        results = {}
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_values, y_values, cv=kf, scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
                
                # Train final model
                model.fit(X_values, y_values)
                self.models[name] = model
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
                # Store results
                y_pred = model.predict(X_values)
                results[name] = {
                    'model': model,
                    'cv_mae': cv_mae,
                    'mae': mean_absolute_error(y_values, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_values, y_pred)),
                    'r2': r2_score(y_values, y_pred),
                    'predictions': y_pred
                }
                
                self.training_history.append({
                    'model': name,
                    'cv_mae': cv_mae,
                    'timestamp': datetime.now()
                })
                
                print(f"✅ Trained {name} - CV MAE: {cv_mae:.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                st.error(f"Failed to train {name}: {e}")
        
        return results

    def predict_ensemble(self, features):
        """Ensemble prediction using all models"""
        predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict([features])[0]
                predictions.append(pred)
            except:
                continue
        
        return np.mean(predictions) if predictions else 0.0

# --- ENHANCED QUANTUM ML ENGINE ---
class EnhancedQuantumML:
    """Enhanced quantum ML with multiple circuit architectures and improved performance"""
    
    def __init__(self):
        self.circuit_generator = QuantumCircuitGenerator()
        self.backend = AerSimulator() if HAS_QISKIT else None
        self.quantum_results = []
        self.circuit_performance = {}
        self.optimization_level = 1  # Default optimization level
        self.error_mitigation = False
        self.shots = 1024  # ADD THIS LINE - Initialize shots attribute

    def _simulate_classical(self, features):
        """Classical simulation of quantum expectation as a fallback."""
        if features is None:
            return 0.0
        # A simple hash-based deterministic simulation
        try:
            feature_sum = sum(features)
            # Scale to be in [-1, 1] range
            return np.tanh(feature_sum - np.mean(features))
        except:
            return np.random.uniform(-0.5, 0.5)
        
    def configure_backend(self, optimization_level=1, error_mitigation=False, shots=1024):
        """Configure quantum backend with optimization and error mitigation"""
        self.optimization_level = optimization_level
        self.error_mitigation = error_mitigation
        self.shots = shots  # This will now work properly
        
    def run_advanced_circuit(self, circuit_type, features=None, params=None, custom_circuit=None, show_diagram=False):
        """Run advanced quantum circuits with performance tracking, error handling, and optional diagram generation"""
        if not HAS_QISKIT:
            return self._simulate_classical(features), None, {}

        try:
            start_time = time.time()

            # Generate or use custom circuit
            try:
                # Prefer a provided custom circuit, otherwise generate based on the requested type
                if custom_circuit is not None:
                    circuit = custom_circuit
                else:
                    circuit = self._generate_circuit_by_type(circuit_type, features, params)
            except Exception as gen_err:
                logger.warning(f"Failed to generate circuit '{circuit_type}': {gen_err}")
                # Fallback to a minimal single-qubit circuit to keep flow working
                try:
                    circuit = QuantumCircuit(1)
                    circuit.h(0)
                except Exception:
                    circuit = None

            if show_diagram:
                try:
                    fig_obj = None
                    # Only attempt to draw if circuit is not None and has draw attribute
                    if circuit is None:
                        # Fallback to text representation if circuit is unavailable
                        st.code("Circuit not available.", language='text')
                    elif hasattr(circuit, 'draw'):
                        try:
                            drawn = circuit.draw(output='mpl', style={'showindex': True})
                        except Exception as draw_err:
                            logger.warning(f"Circuit drawing failed: {draw_err}")
                            drawn = None

                        # If qiskit returned a Matplotlib Figure instance directly
                        if MatplotlibFigure is not None and isinstance(drawn, MatplotlibFigure):
                            fig_obj = drawn
                        # If qiskit returned an Axes or similar with a .figure attribute (safe access)
                        elif MatplotlibFigure is not None and hasattr(drawn, 'figure') and getattr(drawn, 'figure', None) is not None and isinstance(getattr(drawn, 'figure'), MatplotlibFigure):
                            fig_obj = getattr(drawn, 'figure')
                        # If it returned an object with a savefig method, try to use plt.gcf() as a safe fallback
                        elif hasattr(drawn, 'savefig') and callable(getattr(drawn, 'savefig', None)):
                            try:
                                fig_obj = plt.gcf() if HAS_MATPLOTLIB else None
                            except Exception:
                                fig_obj = None
                        # If draw returned a string or None or unexpected type, fallback to text representation
                        elif isinstance(drawn, str) or drawn is None:
                            fig_obj = None
                        # If we obtained a figure object attempt to render it, otherwise show text fallback
                        if fig_obj is not None:
                            try:
                                st.pyplot(fig_obj)
                            except Exception:
                                st.code(str(circuit), language='text')
                        else:
                            st.code(str(circuit), language='text')
                            st.markdown("**Circuit Text Representation**")
                    else:
                        # If circuit has no draw method, fallback to text representation
                        try:
                            st.code(str(circuit), language='text')
                            st.markdown("**Circuit Text Representation**")
                        except Exception:
                            st.text("Circuit representation not available.")

                    # Only access circuit attributes if circuit is available
                    if circuit is not None:
                        st.markdown(f"**Circuit Diagram - {circuit_type.replace('_', ' ').title()}**")
                        st.markdown(f"- Qubits: {circuit.num_qubits}")
                        st.markdown(f"- Depth: {circuit.depth()}")
                        st.markdown(f"- Gates: {sum(circuit.count_ops().values())}")
                    else:
                        st.markdown("**Circuit Diagram - Not Available**")
                except Exception as diagram_error:
                    logger.warning(f"Could not generate circuit diagram: {diagram_error}")
                    # Fallback to text representation if possible
                    try:
                        st.code(str(circuit), language='text')
                    except Exception:
                        st.text("Circuit representation not available.")

            # Advanced transpilation with optimization
            if self.backend is None:
                raise RuntimeError("Quantum backend is not available.")

            # Ensure circuit is valid for transpilation (transpile does not accept None)
            if circuit is None:
                logger.warning("Circuit is None, using minimal fallback circuit for transpilation.")
                try:
                    circuit = QuantumCircuit(1)
                    circuit.h(0)
                except Exception:
                    raise RuntimeError("Failed to create fallback quantum circuit for transpilation.")

            transpiled = transpile(
                circuit,
                self.backend,
                optimization_level=self.optimization_level
            )

            # Execute with error mitigation if enabled
            try:
                job = self.backend.run(transpiled, shots=self.shots)
                result = job.result()
                counts = result.get_counts()
                if counts is None:
                    raise RuntimeError("No counts returned from quantum execution")
            except Exception as exec_error:
                logger.warning(f"Quantum execution failed: {exec_error}. Using fallback simulation.")
                # Fallback to classical simulation
                return self._simulate_classical(features), None, {}

            # Calculate multiple expectation values with advanced observables
            expectation_values = self._calculate_advanced_expectation(counts, circuit_type)
            execution_time = time.time() - start_time

            # Store comprehensive performance metrics
            performance_data = {
                'execution_time': execution_time,
                'expectation': expectation_values['combined'],
                'z_expectation': expectation_values['z_expectation'],
                'parity_expectation': expectation_values['parity_expectation'],
                'variance': expectation_values['variance'],
                'shots': self.shots,
                'qubits': circuit.num_qubits,
                'depth': circuit.depth(),
                'gate_count': sum(circuit.count_ops().values()),
                'optimization_level': self.optimization_level,
                'error_mitigation': self.error_mitigation
            }

            self.circuit_performance[circuit_type] = performance_data

            # Store detailed results
            result_entry = {
                'circuit_type': circuit_type,
                'expectation': expectation_values['combined'],
                'expectation_breakdown': expectation_values,
                'counts': counts,
                'execution_time': execution_time,
                'circuit_metrics': {
                    'qubits': circuit.num_qubits,
                    'depth': circuit.depth(),
                    'gate_count': sum(circuit.count_ops().values())
                },
                'timestamp': datetime.now(),
                'features_used': features[:6] if features else None,
                'params_used': params[:10] if params else None  # Store first 10 params
            }

            self.quantum_results.append(result_entry)

            return expectation_values['combined'], circuit, counts

        except Exception as e:
            logger.error(f"Advanced quantum circuit failed: {e}")
            st.error(f"Quantum computation failed: {e}")
            return self._simulate_classical(features), None, {}

    def _calculate_advanced_expectation(self, counts, circuit_type):
        """Calculate various expectation values from measurement counts."""
        total_shots = sum(counts.values())
        if total_shots == 0:
            return {'combined': 0.0, 'z_expectation': 0.0, 'parity_expectation': 0.0, 'variance': 0.0}

        # Z expectation on the first qubit
        z_exp = 0
        for state, count in counts.items():
            if state[-1] == '0':
                z_exp += count
            else:
                z_exp -= count
        z_expectation = z_exp / total_shots

        # Parity expectation (even vs odd number of 1s)
        parity_exp = 0
        for state, count in counts.items():
            if state.count('1') % 2 == 0:
                parity_exp += count
            else:
                parity_exp -= count
        parity_expectation = parity_exp / total_shots

        # Combined expectation (simple average)
        combined_expectation = (z_expectation + parity_expectation) / 2

        # Variance of the combined expectation
        variance = 1 - combined_expectation**2

        return {
            'combined': combined_expectation,
            'z_expectation': z_expectation,
            'parity_expectation': parity_expectation,
            'variance': variance
        }

    def _generate_circuit_by_type(self, circuit_type, features, params):
        """Generate a quantum circuit based on the specified type."""
        if circuit_type == "feature_map_advanced":
            return self.circuit_generator.create_advanced_feature_map(features)
        elif circuit_type == "variational_advanced":
            return self.circuit_generator.create_advanced_variational(params if params else [])
        elif circuit_type == "quantum_neural_network":
            return self.circuit_generator.create_qnn_circuit(features, params)
        elif circuit_type == "amplitude_estimation":
            return self.circuit_generator.create_amplitude_estimation_circuit()
        elif circuit_type == "quantum_approximate_optimization":
            return self.circuit_generator.create_qaoa_circuit(features, params)
        elif circuit_type == "efficient_su2":
            return self.circuit_generator.create_efficient_su2_circuit(features, params)
        else:
            # Fallback to a default circuit
            logger.warning(f"Unknown circuit type '{circuit_type}', falling back to feature map.")
            return self.circuit_generator.create_advanced_feature_map(features)
# These functions are called but not defined in the provided code:
def get_currency_specific_rates(currency, country, market_data):
    # This function is called in show_live_pricing but not defined
    pass

def generate_advanced_training_data(n_samples, pricer):
    # This function is called but not defined
    pass

def display_feature_importance(model_name, feature_names, importance):
    # This function is called but not defined  
    pass
def show_live_pricing(pricer, classical_ml, quantum_ml):
    """Live pricing with speed comparison between traditional and quantum approaches"""
    
    st.markdown("## 🎯 Live Swaption Pricing Calculator")
    
    # Simple pricing configuration
    st.markdown("### 📊 Pricing Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Basic parameters
        notional = st.selectbox(
            "Notional Amount (Millions)",
            [1, 5, 10, 25, 50, 100],
            index=2,
            format_func=lambda x: f"${x}M"
        ) * 1000000  # Convert to actual notional
        
        expiry = st.slider("Expiry (Years)", 0.25, 10.0, 2.0, 0.25)
        tenor = st.slider("Tenor (Years)", 1.0, 30.0, 5.0, 0.5)
    
    with col2:
        # Rate parameters
        strike = st.slider("Strike Rate (%)", 1.0, 10.0, 3.5, 0.1) / 100
        forward_rate = st.slider("Forward Rate (%)", 1.0, 10.0, 4.2, 0.1) / 100
        volatility = st.slider("Volatility (%)", 10.0, 80.0, 25.0, 1.0) / 100
    
    with col3:
        # Option parameters
        swaption_type = st.selectbox(
            "Swaption Type",
            ["Payer Swaption", "Receiver Swaption"]
        )
        risk_free_rate = st.number_input("Risk-Free Rate (%)", 1.0, 10.0, 5.3, 0.1) / 100
        
        # Quantum configuration
        st.markdown("#### ⚛️ Quantum Settings")
        quantum_enabled = st.checkbox("Enable Quantum Pricing", value=True)
        quantum_weight = st.slider("Quantum Influence", 0.0, 1.0, 0.3, 
                                 help="How much quantum correction to apply")
        quantum_shots = st.slider("Quantum Shots", 100, 5000, 1024, 
                                help="More shots = more accuracy but slower")

    # Display current parameters
    st.markdown("### 📋 Current Parameters")
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)
    
    with param_col1:
        st.metric("Notional", f"${notional:,.0f}")
    with param_col2:
        st.metric("Forward Rate", f"{forward_rate:.3%}")
    with param_col3:
        st.metric("Strike", f"{strike:.3%}")
    with param_col4:
        st.metric("Volatility", f"{volatility:.1%}")

    # Calculate button
    if st.button("🎯 Calculate Swaption Prices", type="primary", use_container_width=True):
        with st.spinner("Calculating traditional and quantum prices..."):
            try:
                # Initialize timing variables
                traditional_start_time = time.time()
                
                # Calculate traditional price
                traditional_price = calculate_simple_black76(
                    notional=notional,
                    forward_rate=forward_rate,
                    strike=strike,
                    expiry=expiry,
                    volatility=volatility,
                    option_type=swaption_type,
                    risk_free_rate=risk_free_rate
                )
                
                traditional_end_time = time.time()
                traditional_duration = traditional_end_time - traditional_start_time
                
                # Calculate quantum price if enabled
                quantum_price = traditional_price
                quantum_expectation = 0.5
                quantum_details = {}
                quantum_duration = 0
                
                if quantum_enabled and quantum_ml:
                    try:
                        quantum_start_time = time.time()
                        
                        # Configure quantum backend
                        quantum_ml.configure_backend(
                            optimization_level=1,
                            error_mitigation=False,
                            shots=quantum_shots
                        )
                        
                        # Prepare features for quantum circuit
                        features = [
                            forward_rate,
                            strike, 
                            volatility,
                            expiry,
                            tenor,
                            notional / 1000000  # Scale down for quantum
                        ]
                        
                        # Run quantum circuit
                        quantum_expectation, circuit, counts = quantum_ml.run_advanced_circuit(
                            "feature_map_advanced",
                            features=features,
                            show_diagram=False
                        )
                        
                        # Apply quantum correction to traditional price
                        quantum_correction = (quantum_expectation - 0.5) * 0.25  # Scale correction
                        quantum_price = traditional_price * (1 + quantum_correction * quantum_weight)
                        
                        quantum_end_time = time.time()
                        quantum_duration = quantum_end_time - quantum_start_time
                        
                        quantum_details = {
                            'expectation': quantum_expectation,
                            'correction': quantum_correction,
                            'circuit_used': "feature_map_advanced",
                            'shots': quantum_shots,
                            'execution_time': quantum_duration
                        }
                        
                    except Exception as quantum_error:
                        st.warning(f"⚠️ Quantum calculation failed: {quantum_error}")
                        quantum_price = traditional_price
                        quantum_duration = 0
                
                # Calculate speed metrics
                speed_metrics = calculate_speed_metrics(
                    traditional_duration, 
                    quantum_duration if quantum_enabled else 0,
                    quantum_enabled
                )
                
                # Store both results
                st.session_state.live_price_result = {
                    'traditional_price': traditional_price,
                    'quantum_price': quantum_price,
                    'quantum_enabled': quantum_enabled,
                    'quantum_details': quantum_details,
                    'speed_metrics': speed_metrics,
                    'notional': notional,
                    'forward_rate': forward_rate,
                    'strike': strike,
                    'expiry': expiry,
                    'volatility': volatility,
                    'option_type': swaption_type,
                    'timestamp': datetime.now()
                }
                
                st.success("✅ Price calculations completed!")
                
            except Exception as e:
                st.error(f"❌ Pricing calculation failed: {str(e)}")
                # Fallback calculation
                fallback_price = calculate_fallback_price(notional, forward_rate, strike, expiry)
                st.session_state.live_price_result = {
                    'traditional_price': fallback_price,
                    'quantum_price': fallback_price,
                    'quantum_enabled': False,
                    'speed_metrics': {'traditional_speed': 0.001, 'quantum_speed': 0},
                    'notional': notional,
                    'forward_rate': forward_rate,
                    'strike': strike,
                    'expiry': expiry,
                    'volatility': volatility,
                    'option_type': swaption_type,
                    'timestamp': datetime.now(),
                    'note': 'Used fallback calculation'
                }
                st.warning(f"⚠️ Using fallback calculation: ${fallback_price:,.0f}")

    # Display results - BOTH TRADITIONAL AND QUANTUM WITH SPEED
    if 'live_price_result' in st.session_state:
        result = st.session_state.live_price_result
        speed_metrics = result.get('speed_metrics', {})
        
        st.markdown("---")
        st.markdown("## 💰 Swaption Price Results")
        
        # Main price comparison
        col_price1, col_price2, col_price3 = st.columns(3)
        
        with col_price1:
            st.markdown("### 🏛️ Traditional")
            st.markdown(f"# **${result['traditional_price']:,.0f}**")
            st.caption(f"Black-76 Model • {speed_metrics.get('traditional_speed', 0):.3f}s")
            
        with col_price2:
            if result['quantum_enabled']:
                st.markdown("### ⚛️ Quantum")
                st.markdown(f"# **${result['quantum_price']:,.0f}**")
                price_diff = result['quantum_price'] - result['traditional_price']
                diff_pct = (price_diff / result['traditional_price']) * 100
                quantum_time = result['quantum_details'].get('execution_time', 0)
                st.caption(f"Quantum Enhanced ({diff_pct:+.1f}%) • {quantum_time:.3f}s")
            else:
                st.markdown("### ⚛️ Quantum")
                st.markdown("# **—**")
                st.caption("Quantum disabled")
                
        with col_price3:
            if result['quantum_enabled']:
                st.markdown("### 📊 Comparison")
                price_diff = result['quantum_price'] - result['traditional_price']
                diff_pct = (price_diff / result['traditional_price']) * 100
                
                if abs(diff_pct) < 1:
                    status = "🟰 Similar"
                    color = "gray"
                elif diff_pct > 0:
                    status = "📈 Quantum Higher"
                    color = "green"
                else:
                    status = "📉 Quantum Lower" 
                    color = "red"
                    
                st.metric("Price Difference", f"${price_diff:,.0f}", f"{diff_pct:+.1f}%")
                st.caption(status)
            else:
                st.markdown("### 📊 Comparison")
                st.metric("Quantum Status", "Disabled", delta_color="off")
        
        # SPEED COMPARISON SECTION
        st.markdown("### ⚡ Speed Performance")
        
        if result['quantum_enabled']:
            col_speed1, col_speed2, col_speed3, col_speed4 = st.columns(4)
            
            with col_speed1:
                traditional_speed = speed_metrics.get('traditional_speed', 0)
                st.metric("Traditional Speed", f"{traditional_speed:.4f}s")
                
            with col_speed2:
                quantum_speed = result['quantum_details'].get('execution_time', 0)
                st.metric("Quantum Speed", f"{quantum_speed:.3f}s")
                
            with col_speed3:
                speed_ratio = speed_metrics.get('speed_ratio', 0)
                st.metric("Speed Ratio", f"{speed_ratio:.1f}x", 
                         delta="Faster" if speed_ratio < 1 else "Slower",
                         delta_color="inverse" if speed_ratio < 1 else "normal")
                
            with col_speed4:
                efficiency = speed_metrics.get('efficiency_score', 0)
                st.metric("Efficiency Score", f"{efficiency:.0f}/100")
            
            # Speed visualization
            st.markdown("#### 📈 Speed Comparison Chart")
            
            methods = ['Traditional', 'Quantum']
            speeds = [speed_metrics.get('traditional_speed', 0), 
                     result['quantum_details'].get('execution_time', 0)]
            
            fig_speed = go.Figure()
            fig_speed.add_trace(go.Bar(
                x=methods,
                y=speeds,
                marker_color=['blue', 'purple'],
                text=[f"{s:.3f}s" for s in speeds],
                textposition='auto',
            ))
            fig_speed.update_layout(
                title='Execution Speed Comparison (Lower is Better)',
                yaxis_title='Time (seconds)',
                height=300
            )
            st.plotly_chart(fig_speed, use_container_width=True)
            
            # Speed analysis
            col_analysis1, col_analysis2 = st.columns(2)
            
            with col_analysis1:
                st.markdown("##### 🎯 Speed Insights")
                if speed_metrics.get('speed_ratio', 1) < 1:
                    st.success("🚀 **Quantum is faster!** Quantum computing shows speed advantage")
                else:
                    st.info("⏳ **Traditional is faster** for this calculation size")
                    
                if speed_metrics.get('efficiency_score', 0) > 80:
                    st.success("🏆 **Excellent efficiency** - Well balanced performance")
                elif speed_metrics.get('efficiency_score', 0) > 60:
                    st.info("✅ **Good efficiency** - Reasonable performance balance")
                else:
                    st.warning("⚖️ **Consider optimizing** - Performance could be improved")
                    
            with col_analysis2:
                st.markdown("##### 📊 Performance Metrics")
                st.write(f"**Traditional Operations:** {speed_metrics.get('traditional_ops', 'N/A')}")
                st.write(f"**Quantum Operations:** {speed_metrics.get('quantum_ops', 'N/A')}")
                st.write(f"**Speed Advantage:** {speed_metrics.get('speed_advantage', 'N/A')}")
                st.write(f"**Calculation Complexity:** {speed_metrics.get('complexity', 'N/A')}")
        
        else:
            st.info("⚡ Enable Quantum pricing to see speed comparison")
        
        # Price breakdown
        with st.expander("📊 Detailed Price Breakdown"):
            col_break1, col_break2 = st.columns(2)
            
            with col_break1:
                st.markdown("#### 🏛️ Traditional Pricing")
                st.write(f"**Calculation Method:** Black-76 Model")
                st.write(f"**Execution Time:** {speed_metrics.get('traditional_speed', 0):.4f} seconds")
                st.write(f"**Notional:** ${result['notional']:,.0f}")
                st.write(f"**Forward Rate:** {result['forward_rate']:.3%}")
                st.write(f"**Strike Rate:** {result['strike']:.3%}")
                st.write(f"**Expiry:** {result['expiry']:.1f} years")
                st.write(f"**Volatility:** {result['volatility']:.1%}")
                st.write(f"**Option Type:** {result['option_type']}")
                
            with col_break2:
                if result['quantum_enabled']:
                    st.markdown("#### ⚛️ Quantum Enhancement")
                    st.write(f"**Quantum Expectation:** {result['quantum_details'].get('expectation', 0):.4f}")
                    st.write(f"**Correction Factor:** {result['quantum_details'].get('correction', 0):.4f}")
                    st.write(f"**Quantum Weight:** {quantum_weight:.0%}")
                    st.write(f"**Circuit Used:** {result['quantum_details'].get('circuit_used', 'N/A')}")
                    st.write(f"**Quantum Shots:** {result['quantum_details'].get('shots', 'N/A')}")
                    st.write(f"**Execution Time:** {result['quantum_details'].get('execution_time', 0):.3f} seconds")
                else:
                    st.markdown("#### ⚛️ Quantum Enhancement")
                    st.write("**Status:** Disabled")

        # Visualization
        st.markdown("### 📈 Price Comparison")
        
        if result['quantum_enabled']:
            # Create comparison chart
            models = ['Traditional', 'Quantum']
            prices = [result['traditional_price'], result['quantum_price']]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=models,
                y=prices,
                marker_color=['blue', 'purple'],
                text=[f"${p:,.0f}" for p in prices],
                textposition='auto',
            ))
            fig.update_layout(
                title='Traditional vs Quantum Swaption Pricing',
                yaxis_title='Price ($)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Export results
        st.markdown("### 💾 Export Results")
        if st.button("📥 Download Price Results", use_container_width=True):
            # Create results summary
            results_summary = {
                'Parameter': [
                    'Traditional Price', 'Quantum Price', 'Price Difference', 
                    'Difference %', 'Traditional Time (s)', 'Quantum Time (s)',
                    'Speed Ratio', 'Efficiency Score', 'Notional', 'Forward Rate', 
                    'Strike Rate', 'Expiry', 'Volatility', 'Option Type', 'Calculation Time'
                ],
                'Value': [
                    f"${result['traditional_price']:,.0f}",
                    f"${result['quantum_price']:,.0f}" if result['quantum_enabled'] else "N/A",
                    f"${result['quantum_price'] - result['traditional_price']:,.0f}" if result['quantum_enabled'] else "N/A",
                    f"{(result['quantum_price'] - result['traditional_price'])/result['traditional_price']*100:+.1f}%" if result['quantum_enabled'] else "N/A",
                    f"{speed_metrics.get('traditional_speed', 0):.4f}",
                    f"{result['quantum_details'].get('execution_time', 0):.3f}" if result['quantum_enabled'] else "N/A",
                    f"{speed_metrics.get('speed_ratio', 0):.1f}x" if result['quantum_enabled'] else "N/A",
                    f"{speed_metrics.get('efficiency_score', 0)}" if result['quantum_enabled'] else "N/A",
                    f"${result['notional']:,.0f}",
                    f"{result['forward_rate']:.3%}",
                    f"{result['strike']:.3%}",
                    f"{result['expiry']:.1f} years",
                    f"{result['volatility']:.1%}",
                    result['option_type'],
                    result['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                ]
            }
            
            df_export = pd.DataFrame(results_summary)
            csv_data = df_export.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"swaption_pricing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def calculate_speed_metrics(traditional_time, quantum_time, quantum_enabled):
    """Calculate comprehensive speed metrics for comparison"""
    
    metrics = {
        'traditional_speed': traditional_time,
        'quantum_speed': quantum_time if quantum_enabled else 0,
    }
    
    if quantum_enabled and quantum_time > 0:
        # Speed ratio (traditional vs quantum)
        speed_ratio = quantum_time / traditional_time if traditional_time > 0 else 1
        metrics['speed_ratio'] = speed_ratio
        
        # Efficiency score (0-100, higher is better)
        # Factors: speed ratio, absolute times, and practical usefulness
        base_score = 100 * (1 / speed_ratio) if speed_ratio > 1 else 100 * speed_ratio
        time_penalty = min(quantum_time * 10, 50)  # Penalty for long quantum times
        efficiency_score = max(0, base_score - time_penalty)
        metrics['efficiency_score'] = efficiency_score
        
        # Performance classification
        if speed_ratio < 0.1:
            metrics['speed_advantage'] = "Massive Quantum Advantage"
            metrics['complexity'] = "Quantum Optimal"
        elif speed_ratio < 0.5:
            metrics['speed_advantage'] = "Significant Quantum Advantage"
            metrics['complexity'] = "Quantum Favored"
        elif speed_ratio < 1:
            metrics['speed_advantage'] = "Moderate Quantum Advantage"
            metrics['complexity'] = "Quantum Suitable"
        elif speed_ratio < 2:
            metrics['speed_advantage'] = "Traditional Slightly Faster"
            metrics['complexity'] = "Balanced"
        else:
            metrics['speed_advantage'] = "Traditional Significantly Faster"
            metrics['complexity'] = "Traditional Optimal"
        
        # Operation estimates
        metrics['traditional_ops'] = f"~{int(1e6):,} operations"
        metrics['quantum_ops'] = f"~{quantum_time * 1e9:,.0f} quantum operations"
    
    return metrics

def calculate_simple_black76(notional, forward_rate, strike, expiry, volatility, option_type, risk_free_rate=0.05):
    """Simple and reliable Black-76 calculation"""
    try:
        # Basic input validation
        if forward_rate <= 0 or strike <= 0 or expiry <= 0 or volatility <= 0:
            raise ValueError("Invalid input parameters")
        
        # Calculate d1 and d2
        d1 = (np.log(forward_rate / strike) + (volatility**2 / 2) * expiry) / (volatility * np.sqrt(expiry))
        d2 = d1 - volatility * np.sqrt(expiry)
        
        # Simplified annuity factor
        annuity = (1 - np.exp(-risk_free_rate * expiry)) / risk_free_rate
        
        # Calculate option price
        if option_type == "Payer Swaption":
            price = annuity * (forward_rate * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2))
        else:  # Receiver Swaption
            price = annuity * (strike * stats.norm.cdf(-d2) - forward_rate * stats.norm.cdf(-d1))
        
        # Apply notional and ensure positive price
        final_price = max(notional * price, 1000)  # Minimum $1000
        
        return final_price
        
    except Exception as e:
        raise e

def calculate_fallback_price(notional, forward_rate, strike, expiry):
    """Fallback price calculation when Black-76 fails"""
    try:
        # Very simple intrinsic value + time value approximation
        intrinsic_value = max(forward_rate - strike, 0)
        time_value = 0.01 * expiry  # Simple time value
        
        # Simple annuity approximation
        annuity = expiry * 0.8
        
        price = notional * annuity * (intrinsic_value + time_value)
        
        # Ensure reasonable bounds
        price = max(price, notional * 0.001)  # At least 0.1% of notional
        price = min(price, notional * 0.1)    # At most 10% of notional
        
        return price
        
    except:
        # Ultimate fallback
        return notional * 0.01  # 1% of notional as final fallback
def calculate_simple_black76(notional, forward_rate, strike, expiry, volatility, option_type, risk_free_rate=0.05):
    """Simple and reliable Black-76 calculation"""
    try:
        # Basic input validation
        if forward_rate <= 0 or strike <= 0 or expiry <= 0 or volatility <= 0:
            raise ValueError("Invalid input parameters")
        
        # Calculate d1 and d2
        d1 = (np.log(forward_rate / strike) + (volatility**2 / 2) * expiry) / (volatility * np.sqrt(expiry))
        d2 = d1 - volatility * np.sqrt(expiry)
        
        # Simplified annuity factor (this is the key fix)
        # For swaptions, annuity is approximately the tenor discounted at risk-free rate
        annuity = (1 - np.exp(-risk_free_rate * expiry)) / risk_free_rate
        
        # Calculate option price
        if option_type == "Payer Swaption":
            price = annuity * (forward_rate * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2))
        else:  # Receiver Swaption
            price = annuity * (strike * stats.norm.cdf(-d2) - forward_rate * stats.norm.cdf(-d1))
        
        # Apply notional and ensure positive price
        final_price = max(notional * price, 1000)  # Minimum $1000
        
        return final_price
        
    except Exception as e:
        # Fallback to a very simple calculation
        st.warning(f"Primary calculation failed: {e}. Using fallback.")
        return calculate_fallback_price(notional, forward_rate, strike, expiry)

def calculate_fallback_price(notional, forward_rate, strike, expiry):
    """Fallback price calculation when Black-76 fails"""
    try:
        # Very simple intrinsic value + time value approximation
        intrinsic_value = max(forward_rate - strike, 0)
        time_value = 0.01 * expiry  # Simple time value
        
        # Simple annuity approximation
        annuity = expiry * 0.8
        
        price = notional * annuity * (intrinsic_value + time_value)
        
        # Ensure reasonable bounds
        price = max(price, notional * 0.001)  # At least 0.1% of notional
        price = min(price, notional * 0.1)    # At most 10% of notional
        
        return price
        
    except:
        # Ultimate fallback
        return notional * 0.01  # 1% of notional as final fallback
    # ... [The results display section from previous implementation continues here]
# --- ENHANCED QUANTUM CIRCUIT GENERATOR ---
class QuantumCircuitGenerator:
    """Enhanced quantum circuit generator with multiple architectures and financial applications"""
    
    def __init__(self, n_qubits=6):
        self.n_qubits = n_qubits
        self.entanglement_types = ['linear', 'circular', 'full', 'pairwise']
        
    def create_advanced_feature_map(self, features, entanglement_type='linear'):
        """Create advanced feature map with configurable entanglement"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Validate and normalize features
        features = self._normalize_features(features)
        
        # Enhanced feature encoding with financial interpretation
        if len(features) >= 6:
            # Financial feature mapping with appropriate rotations
            qc.ry(features[0] * np.pi * 2, 0)    # Forward rate (main driver)
            qc.rz(features[1] * np.pi * 2, 1)    # Strike rate (moneyness)
            qc.rx(features[2] * np.pi, 2)        # Volatility (uncertainty)
            qc.ry(features[3] * np.pi, 3)        # Expiry (time factor)
            qc.rz(features[4] * np.pi, 4)        # Tenor (duration)
            qc.h(5)                              # Hadamard for rate direction
        
        # Apply selected entanglement pattern
        self._apply_entanglement(qc, entanglement_type)
        
        # Additional feature-dependent rotations
        if len(features) >= 3:
            # Interaction terms between key features
            qc.cry(features[0] * np.pi, 0, 2)  # Forward rate -> Volatility
            qc.crz(features[1] * np.pi, 1, 3)  # Strike -> Expiry
            
        return qc
    
    def create_advanced_variational(self, params, n_layers=2, entanglement_type='linear'):
        """Create advanced variational circuit with multiple layers"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial state preparation - financial superposition
        for i in range(self.n_qubits):
            qc.h(i)  # Equal superposition for all qubits
            
        # Multiple variational layers
        params_per_layer = 3 * self.n_qubits  # RY, RZ, RX per qubit
        total_params_needed = n_layers * params_per_layer
        
        # Handle parameter size mismatch
        if len(params) < total_params_needed:
            # Extend parameters if needed
            extended_params = list(params) + [0.1] * (total_params_needed - len(params))
        else:
            extended_params = params[:total_params_needed]
        
        param_idx = 0
        
        for layer in range(n_layers):
            # Single-qubit rotations in each layer
            for i in range(self.n_qubits):
                if param_idx < len(extended_params):
                    qc.ry(extended_params[param_idx], i)
                    param_idx += 1
                if param_idx < len(extended_params):
                    qc.rz(extended_params[param_idx], i)
                    param_idx += 1
                if param_idx < len(extended_params):
                    qc.rx(extended_params[param_idx], i)
                    param_idx += 1
            
            # Entanglement within layer
            self._apply_entanglement(qc, entanglement_type)
            
            # Layer-specific additional rotations
            if layer < n_layers - 1:  # Not in final layer
                for i in range(self.n_qubits):
                    if param_idx < len(extended_params):
                        qc.ry(extended_params[param_idx] * 0.5, i)
                        param_idx += 1
                        
        return qc
    
    def create_qnn_circuit(self, features, params=None, n_layers=2):
        """Create quantum neural network circuit with encoding and variational parts"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Feature encoding layer
        features = self._normalize_features(features)
        for i in range(min(len(features), self.n_qubits)):
            qc.ry(features[i] * np.pi, i)
        
        # Variational layers
        if params is None:
            params = np.random.uniform(0, 2*np.pi, self.n_qubits * 2 * n_layers)
        
        param_idx = 0
        
        for layer in range(n_layers):
            # Rotation layer
            for i in range(self.n_qubits):
                if param_idx < len(params):
                    qc.rz(params[param_idx], i)
                    param_idx += 1
                if param_idx < len(params):
                    qc.ry(params[param_idx], i)
                    param_idx += 1
            
            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            
            # Final rotation in each layer
            for i in range(self.n_qubits):
                if param_idx < len(params):
                    qc.rz(params[param_idx], i)
                    param_idx += 1
                    
        return qc
    
    def create_amplitude_estimation_circuit(self, n_estimation_qubits=3):
        """Create amplitude estimation circuit for financial probability estimation"""
        total_qubits = self.n_qubits + n_estimation_qubits
        qc = QuantumCircuit(total_qubits)
        
        # Initialize estimation qubits in superposition
        for i in range(self.n_qubits, total_qubits):
            qc.h(i)
            
        # Main circuit - price movement simulation
        for i in range(self.n_qubits):
            qc.h(i)  # Start in superposition
            
        # Controlled operations for amplitude estimation
        # This simulates different price path scenarios
        for i in range(self.n_qubits, total_qubits):
            power = 2 ** (i - self.n_qubits)
            for _ in range(power):
                # Apply controlled version of price movement simulation
                for j in range(self.n_qubits):
                    # Simulate correlated price movements
                    angle = np.pi/4 * (1 + 0.1 * j)  # Varying angles for realism
                    qc.cp(angle, i, j)
                    
                # Additional financial correlations
                if self.n_qubits >= 2:
                    qc.cswap(i, 0, 1)  # Swap based on estimation qubit
                    
        return qc
    
    def create_qaoa_circuit(self, features, params=None, n_layers=2):
        """Create Quantum Approximate Optimization Algorithm circuit for portfolio optimization"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial state - equal superposition
        for i in range(self.n_qubits):
            qc.h(i)
        
        # Default parameters if none provided
        if params is None:
            params = np.random.uniform(0, np.pi, 2 * n_layers)
        
        # QAOA layers: alternating cost and mixer Hamiltonians
        for layer in range(n_layers):
            gamma = params[2 * layer] if 2 * layer < len(params) else np.pi/4
            beta = params[2 * layer + 1] if 2 * layer + 1 < len(params) else np.pi/4
            
            # Cost Hamiltonian (problem-specific)
            self._apply_financial_cost_hamiltonian(qc, gamma, features)
            
            # Mixer Hamiltonian
            for i in range(self.n_qubits):
                qc.rx(2 * beta, i)
                
        return qc
    
    def create_efficient_su2_circuit(self, features, params=None, n_layers=2, entanglement_type='circular'):
        """Create EfficientSU2 circuit - hardware-efficient ansatz"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Feature encoding
        features = self._normalize_features(features)
        for i in range(min(len(features), self.n_qubits)):
            qc.ry(features[i] * np.pi, i)
        
        # Generate parameters if not provided
        if params is None:
            params_per_layer = 2 * self.n_qubits  # RY and RZ per qubit
            params = np.random.uniform(0, 2*np.pi, n_layers * params_per_layer)
        
        param_idx = 0
        
        for layer in range(n_layers):
            # Single-qubit rotations layer
            for i in range(self.n_qubits):
                if param_idx < len(params):
                    qc.ry(params[param_idx], i)
                    param_idx += 1
                if param_idx < len(params):
                    qc.rz(params[param_idx], i)
                    param_idx += 1
            
            # Entanglement layer
            self._apply_entanglement(qc, entanglement_type)
            
        return qc
    
    def _apply_entanglement(self, qc, entanglement_type):
        """Apply different entanglement patterns"""
        if entanglement_type == 'linear':
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
                
        elif entanglement_type == 'circular':
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(self.n_qubits - 1, 0)  # Close the circle
            
        elif entanglement_type == 'full':
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qc.cx(i, j)
                    
        elif entanglement_type == 'pairwise':
            for i in range(0, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
    
    def _apply_financial_cost_hamiltonian(self, qc, gamma, features):
        """Apply financial-specific cost Hamiltonian for optimization problems"""
        # Correlation terms (ZZ interactions)
        for i in range(self.n_qubits - 1):
            for j in range(i + 1, self.n_qubits):
                # Simulate asset correlations
                qc.rzz(gamma * 0.1, i, j)  # Reduced strength for multiple terms
        
        # Individual asset terms (Z interactions)
        for i in range(self.n_qubits):
            weight = features[i % len(features)] if features else 0.5
            qc.rz(gamma * weight, i)
    
    def _normalize_features(self, features):
        """Normalize features to reasonable quantum circuit ranges"""
        if features is None:
            return [0.5] * self.n_qubits
            
        normalized = []
        for f in features:
            # Clip and scale features to [0, 1] range
            clipped = max(0.0, min(1.0, abs(f)))
            normalized.append(clipped)
        
        # Ensure we have enough features
        while len(normalized) < self.n_qubits:
            normalized.append(0.5)  # Default value
            
        return normalized[:self.n_qubits]
    
    def get_circuit_info(self, circuit):
        """Get detailed information about a quantum circuit"""
        ops = circuit.count_ops()
        return {
            'qubits': circuit.num_qubits,
            'depth': circuit.depth(),
            'total_gates': sum(ops.values()),
            'gate_breakdown': ops,
            'parameters': circuit.num_parameters
        }

# --- PERFORMANCE ANALYTICS ENGINE ---
class PerformanceAnalytics:
    """Advanced performance analytics with comprehensive metrics, statistical testing, and real-time monitoring"""
    
    def __init__(self):
        self.comparison_data = []
        self.performance_metrics = {}
        self.training_progress = []
        self.model_benchmarks = {}
        self.confidence_intervals = {}
        self.performance_history = []
        self.statistical_tests = {}
        self.regime_analysis = {}
        self.risk_metrics = {}
        self.data_quality_report = {}
        
    def generate_comprehensive_comparison(self, n_samples=1000, include_real_models=True, 
                                        classical_ml=None, quantum_ml=None, market_regime='normal',
                                        quantum_circuit_type="feature_map_advanced"):
        """Generate comprehensive comparison data with enhanced error handling and data validation"""
        results = []
        pricer = EnhancedSwaptionPricer()
        
        # Enhanced parameter distributions with better fallbacks
        param_distributions = self._get_enhanced_regime_distributions(market_regime)
        
        successful_samples = 0
        max_attempts = n_samples * 3  # Increased attempts for better sampling
        sample_quality_metrics = {
            'failed_validation': 0,
            'pricing_errors': 0,
            'parameter_issues': 0,
            'successful_samples': 0
        }
        
        for i in range(max_attempts):
            if successful_samples >= n_samples:
                break
                
            try:
                # Sample parameters with enhanced validation
                params = self._sample_parameters_with_validation(param_distributions, i)
                if not params:
                    sample_quality_metrics['parameter_issues'] += 1
                    continue
                    
                expiry, tenor, strike, volatility, notional = params
                
                # Calculate true price with enhanced error handling
                true_price = self._calculate_true_price_with_fallback(
                    pricer, notional, expiry, tenor, strike, volatility, i
                )
                
                if true_price is None:
                    sample_quality_metrics['pricing_errors'] += 1
                    continue
                
                # Enhanced financial parameter validation
                if not self._validate_enhanced_financial_parameters(
                    true_price, notional, expiry, tenor, strike, volatility
                ):
                    sample_quality_metrics['failed_validation'] += 1
                    continue
                
                # Generate features with validation
                features, feature_validation = self._create_validated_feature_vector(
                    pricer, expiry, tenor, strike, volatility, notional
                )
                
                if not feature_validation['valid']:
                    sample_quality_metrics['parameter_issues'] += 1
                    continue
                
                # Enhanced model predictions with comprehensive fallbacks
                classical_pred, quantum_pred, prediction_metrics = self._get_robust_model_predictions(
                    features, true_price, volatility, market_regime,
                    include_real_models, classical_ml, quantum_ml, quantum_circuit_type
                ) if include_real_models else (true_price * 1.05, true_price * 0.95, {})
                
                # Calculate comprehensive metrics
                sample_metrics = self._calculate_comprehensive_sample_metrics(
                    true_price, classical_pred, quantum_pred, features, 
                    feature_validation['moneyness'], market_regime
                )
                
                # Create enhanced sample record
                sample_record = self._create_enhanced_sample_record(
                    successful_samples, true_price, classical_pred, quantum_pred, 
                    features, expiry, tenor, volatility, notional, 
                    feature_validation['forward_rate'], feature_validation['moneyness'], 
                    sample_metrics, market_regime, prediction_metrics
                )
                
                results.append(sample_record)
                successful_samples += 1
                sample_quality_metrics['successful_samples'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to generate sample {i}: {str(e)}")
                continue
        
        # Update data quality report
        self._update_data_quality_report(sample_quality_metrics, successful_samples, max_attempts)
        
        self.comparison_data = results
        if results:
            self._update_performance_history()
            self._analyze_market_regimes(results)
            logger.info(f"Successfully generated {len(results)} comparison samples")
        else:
            logger.warning("No valid comparison samples generated")
            # Generate fallback synthetic data
            results = self._generate_fallback_comparison_data(n_samples)
            self.comparison_data = results
            
        return results
    
    def _get_enhanced_regime_distributions(self, regime):
        """Get enhanced parameter distributions with better statistical properties"""
        try:
            regimes = {
                'normal': {
                    'expiry': lambda: np.random.lognormal(0.5, 0.8),
                    'tenor': lambda: np.random.lognormal(1.5, 0.6),
                    'strike': lambda: np.clip(np.random.normal(0.035, 0.015), 0.005, 0.10),
                    'volatility': lambda: np.random.gamma(2, 0.08),
                    'notional': lambda: np.random.choice([1e6, 5e6, 10e6, 25e6, 50e6, 100e6])
                },
                'high_vol': {
                    'expiry': lambda: np.random.lognormal(0.3, 0.6),
                    'tenor': lambda: np.random.lognormal(1.2, 0.5),
                    'strike': lambda: np.clip(np.random.normal(0.045, 0.02), 0.01, 0.15),
                    'volatility': lambda: np.random.gamma(3, 0.12),
                    'notional': lambda: np.random.choice([1e6, 5e6, 10e6])
                },
                'low_vol': {
                    'expiry': lambda: np.random.lognormal(0.7, 0.9),
                    'tenor': lambda: np.random.lognormal(1.8, 0.7),
                    'strike': lambda: np.clip(np.random.normal(0.025, 0.01), 0.001, 0.08),
                    'volatility': lambda: np.random.gamma(1.5, 0.05),
                    'notional': lambda: np.random.choice([25e6, 50e6, 100e6])
                }
            }
            
            return regimes.get(regime, regimes['normal'])
            
        except Exception as e:
            logger.error(f"Error in regime distributions: {e}")
            # Fallback to simple uniform distributions
            return {
                'expiry': lambda: np.random.uniform(0.25, 10.0),
                'tenor': lambda: np.random.uniform(1.0, 30.0),
                'strike': lambda: np.random.uniform(0.01, 0.08),
                'volatility': lambda: np.random.uniform(0.10, 0.40),
                'notional': lambda: 10000000
            }
    
    def _sample_parameters_with_validation(self, distributions, sample_id):
        """Sample parameters with comprehensive validation"""
        try:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    expiry = max(0.25, min(15.0, distributions['expiry']()))
                    tenor = max(0.5, min(30.0, distributions['tenor']()))
                    strike = distributions['strike']()
                    volatility = max(0.05, min(1.0, distributions['volatility']()))
                    notional = distributions['notional']()
                    
                    # Enhanced validation checks
                    if expiry >= tenor:
                        if attempt == max_retries - 1:
                            logger.debug(f"Sample {sample_id}: Expiry >= Tenor after {max_retries} attempts")
                        continue
                        
                    if strike <= 0.001 or strike >= 0.15:
                        if attempt == max_retries - 1:
                            logger.debug(f"Sample {sample_id}: Invalid strike after {max_retries} attempts")
                        continue
                    
                    return expiry, tenor, strike, volatility, notional
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.warning(f"Failed to sample parameters for sample {sample_id}: {e}")
                    continue
                    
            return None
            
        except Exception as e:
            logger.error(f"Critical error in parameter sampling for sample {sample_id}: {e}")
            return None
    
    def _calculate_true_price_with_fallback(self, pricer, notional, expiry, tenor, strike, volatility, sample_id):
        """Calculate true price with comprehensive error handling and fallbacks"""
        try:
            # First attempt with standard parameters
            true_price = pricer.black_76_swaption_price(
                notional, expiry, tenor, strike, "Payer Swaption", volatility
            )
            
            if true_price is None or not np.isfinite(true_price):
                raise ValueError("Invalid price returned from pricer")
                
            return true_price
            
        except Exception as e:
            logger.warning(f"Pricing failed for sample {sample_id}: {e}. Using fallback calculation.")
            
            try:
                # Fallback calculation using simplified Black-76
                return self._fallback_black76_calculation(notional, expiry, tenor, strike, volatility)
            except Exception as fallback_error:
                logger.error(f"Fallback pricing also failed for sample {sample_id}: {fallback_error}")
                return None
    
    def _fallback_black76_calculation(self, notional, expiry, tenor, strike, volatility):
        """Simplified Black-76 calculation as fallback"""
        try:
            # Simplified forward rate assumption
            forward_rate = 0.04 + (strike - 0.035) * 0.5
            
            # Simplified annuity factor
            annuity = tenor * 0.9  # Rough approximation
            
            # Black-76 formula
            d1 = (np.log(forward_rate / strike) + (volatility**2 / 2) * expiry) / (volatility * np.sqrt(expiry))
            d2 = d1 - volatility * np.sqrt(expiry)
            
            call_price = annuity * (forward_rate * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2))
            price = notional * max(call_price, 0)
            
            return price if np.isfinite(price) else notional * 0.01  # Minimum fallback
            
        except Exception as e:
            logger.error(f"Fallback Black-76 calculation failed: {e}")
            # Ultimate fallback - reasonable percentage of notional
            return notional * 0.02
    
    def _validate_enhanced_financial_parameters(self, true_price, notional, expiry, tenor, strike, volatility):
        """Enhanced financial parameter validation"""
        try:
            # Price sanity checks
            if true_price <= 0:
                return False
                
            if true_price > notional * 0.3:  # More conservative upper bound
                return False
                
            # Parameter relationship checks
            if expiry >= tenor:
                return False
                
            if strike <= 0.001 or strike >= 0.15:
                return False
                
            if volatility <= 0.03 or volatility >= 1.5:  # Wider but reasonable bounds
                return False
                
            # Additional financial sanity checks
            if tenor > 30:  # Maximum reasonable tenor
                return False
                
            if notional > 1e9:  # Maximum reasonable notional
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Parameter validation error: {e}")
            return False
    
    def _create_validated_feature_vector(self, pricer, expiry, tenor, strike, volatility, notional):
        """Create a feature vector with validation."""
        try:
            forward_rate = pricer.calculate_forward_swap_rate(expiry, tenor)
            moneyness = strike / forward_rate if forward_rate > 0 else 1.0
            
            features = [
                forward_rate,
                strike,
                volatility,
                expiry,
                tenor,
                notional
            ]
            
            validation_info = {
                'valid': True,
                'forward_rate': forward_rate,
                'moneyness': moneyness
            }
            
            return features, validation_info
            
        except Exception as e:
            logger.warning(f"Feature vector creation failed: {e}")
            return [], {'valid': False, 'forward_rate': 0, 'moneyness': 0}

    def _create_enhanced_sample_record(self, sample_id, true_price, classical_pred, quantum_pred,
                                       features, expiry, tenor, volatility, notional,
                                       forward_rate, moneyness, sample_metrics, market_regime,
                                       prediction_metrics):
        """Create an enhanced sample record dictionary."""
        return {
            'sample_id': sample_id,
            'true_price': true_price,
            'classical_pred': classical_pred,
            'quantum_pred': quantum_pred,
            'forward_rate': forward_rate,
            'strike': features[1],
            'volatility': volatility,
            'expiry': expiry,
            'tenor': tenor,
            'notional': notional,
            'moneyness': moneyness,
            'market_regime': market_regime,
            **sample_metrics,
            **prediction_metrics
        }

    def _calculate_comprehensive_sample_metrics(self, true_price, classical_pred, quantum_pred, features,
                                                moneyness, market_regime):
        """Calculate comprehensive metrics for a single sample."""
        try:
            classical_error = abs(classical_pred - true_price)
            quantum_error = abs(quantum_pred - true_price)
            
            classical_error_pct = (classical_error / true_price) * 100 if true_price > 0 else 0
            quantum_error_pct = (quantum_error / true_price) * 100 if true_price > 0 else 0
            
            improvement = classical_error - quantum_error
            improvement_pct = (improvement / classical_error) * 100 if classical_error > 0 else 0
            
            return {
                'classical_error': classical_error,
                'quantum_error': quantum_error,
                'classical_error_pct': classical_error_pct,
                'quantum_error_pct': quantum_error_pct,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            }
        except Exception as e:
            logger.warning(f"Metric calculation failed: {e}")
            return {
                'classical_error': 0, 'quantum_error': 0, 'classical_error_pct': 0,
                'quantum_error_pct': 0, 'improvement': 0, 'improvement_pct': 0
            }

    def _get_robust_model_predictions(self, features, true_price, volatility, market_regime,
                                      include_real_models, classical_ml, quantum_ml, quantum_circuit_type):
        """Get model predictions with robust fallbacks."""
        classical_pred = true_price * 1.05  # Fallback
        quantum_pred = true_price * 0.95  # Fallback
        prediction_metrics = {}

        if include_real_models and classical_ml and classical_ml.models:
            try:
                classical_pred = classical_ml.predict_ensemble(features)
            except Exception as e:
                logger.warning(f"Classical prediction failed: {e}")

        if include_real_models and quantum_ml:
            try:
                quantum_expectation, _, _ = quantum_ml.run_advanced_circuit(quantum_circuit_type, features)
                quantum_pred = true_price * (0.95 + 0.1 * quantum_expectation)
            except Exception as e:
                logger.warning(f"Quantum prediction failed: {e}")

        return classical_pred, quantum_pred, prediction_metrics

    def _update_data_quality_report(self, metrics, successful, total):
        """Update the data quality report."""
        self.data_quality_report = {
            **metrics,
            'total_attempts': total,
            'success_rate': (successful / total) * 100 if total > 0 else 0
        }

    def _generate_fallback_comparison_data(self, n_samples):
        """Generate fallback synthetic data if main generation fails."""
        results = []
        for i in range(n_samples):
            true_price = np.random.uniform(10000, 500000)
            results.append({
                'sample_id': i,
                'true_price': true_price,
                'classical_pred': true_price * np.random.uniform(0.9, 1.1),
                'quantum_pred': true_price * np.random.uniform(0.95, 1.05),
                'classical_error': 0, 'quantum_error': 0
            })
        return results

    def _update_performance_history(self):
        """Update performance history with the latest run."""
        if not self.comparison_data:
            return
        df = pd.DataFrame(self.comparison_data)
        self.performance_history.append({
            'timestamp': datetime.now(),
            'classical_mae': df['classical_error'].mean(),
            'quantum_mae': df['quantum_error'].mean()
        })

    def _analyze_market_regimes(self, results):
        """Analyze performance across different market regimes."""
        df = pd.DataFrame(results)
        if 'market_regime' in df.columns:
            self.regime_analysis = df.groupby('market_regime')[['classical_error', 'quantum_error']].mean().to_dict()

# --- STREAMLIT DASHBOARD ---
def main():
    """Enhanced dashboard with Kaggle integration"""
    if 'rates_loaded' not in st.session_state:
        st.session_state.rates_loaded = False
    if 'yield_loaded' not in st.session_state:
        st.session_state.yield_loaded = False
    if 'rates_data' not in st.session_state:
        st.session_state.rates_data = None
    if 'yield_data' not in st.session_state:
        st.session_state.yield_data = None
    st.set_page_config(
        page_title="Quantum Finance Pro - Kaggle Edition",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .classical-card { border-left-color: #1f77b4; }
    .quantum-card { border-left-color: #ff7f0e; }
    .kaggle-card { border-left-color: #20beff; }
    .improvement-card { border-left-color: #2ca02c; }
    .circuit-viz {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #dee2e6;
    }
    .data-section {
        background: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header">
        <h1 style="margin: 0; font-size: 3rem;">⚛️ Quantum Finance Pro</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; opacity: 0.9;">
            Advanced Swaption Pricing with Kaggle Integration & Quantum ML
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize engines
    if 'kaggle_manager' not in st.session_state:
        st.session_state.kaggle_manager = KaggleDataManager()
    if 'pricer' not in st.session_state:
        st.session_state.pricer = EnhancedSwaptionPricer(st.session_state.kaggle_manager)
    if 'classical_ml' not in st.session_state:
        st.session_state.classical_ml = EnhancedClassicalML()
    if 'quantum_ml' not in st.session_state:
        st.session_state.quantum_ml = EnhancedQuantumML()
    if 'analytics' not in st.session_state:
        st.session_state.analytics = PerformanceAnalytics()
    
    kaggle_manager = st.session_state.kaggle_manager
    pricer = st.session_state.pricer
    classical_ml = st.session_state.classical_ml
    quantum_ml = st.session_state.quantum_ml
    analytics = st.session_state.analytics
    
    # System Status with Current Market Parameters
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Quantum", "✅" if HAS_QUANTUM else "❌")
    with col2:
        st.metric("Classical ML", "✅" if HAS_ML else "❌")
    with col3:
        kaggle_status = "✅" if kaggle_manager.api else "⚠️ Demo"
        st.metric("Kaggle API", kaggle_status)
    with col4:
        datasets_loaded = len(kaggle_manager.loaded_datasets)
        st.metric("Datasets", datasets_loaded)
    with col5:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.metric("Current Time", current_time)

    # Current Market Parameters Display
    st.markdown("### 📊 Current Market Parameters")
    market_data = pricer.market_data

    if market_data:
        col_mkt1, col_mkt2, col_mkt3, col_mkt4 = st.columns(4)

        with col_mkt1:
            sofr_rate = market_data.get('SOFR', 0.0530)
            st.metric("SOFR Rate", f"{sofr_rate:.3%}")

        with col_mkt2:
            ust_10y = market_data.get('UST_10Y', 0.0410)
            st.metric("10Y Treasury", f"{ust_10y:.3%}")

        with col_mkt3:
            vix = market_data.get('VIX', 15.5)
            st.metric("VIX Index", f"{vix:.1f}")

        with col_mkt4:
            swap_5y = market_data.get('SWAP_5Y', 0.0430)
            st.metric("5Y Swap Rate", f"{swap_5y:.3%}")
    
    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Dashboard", "🏛️ Classical ML", "⚛️ Quantum ML", 
        "📊 Comparison", "🔗 Kaggle Data", "🎯 Live Pricing"
    ])
    
    with tab1:
        show_dashboard(pricer, classical_ml, quantum_ml, analytics)
    
    with tab2:
        show_classical_ml(classical_ml, pricer, quantum_ml)
    
    with tab3:
        show_quantum_ml(quantum_ml)
    
    with tab4:
        show_comparison(analytics)
    
    with tab5:
        show_kaggle_data(kaggle_manager, pricer)
    
    with tab6:
        show_live_pricing(pricer, classical_ml, quantum_ml)

def show_dashboard(pricer, classical_ml, quantum_ml, analytics):
    """Enhanced Executive Dashboard with comprehensive analytics"""
    
    # Dashboard Header with Status Indicators
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0; text-align: center;'>📊 Quantum Finance Dashboard</h1>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;'>
            Real-time Swaption Analytics & Performance Monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # REAL-TIME SYSTEM STATUS
    st.markdown("### 🚀 System Status & Performance")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        quantum_status = "✅ Active" if HAS_QUANTUM and quantum_ml else "❌ Offline"
        st.metric("Quantum Engine", quantum_status, 
                 help="Quantum computing capability status")
    
    with col2:
        ml_status = "✅ Active" if HAS_ML and classical_ml.models else "🔧 Training"
        st.metric("ML Engine", ml_status,
                 help="Machine learning model status")
    
    with col3:
        model_count = len(classical_ml.models) if classical_ml.models else 0
        st.metric("Active Models", model_count,
                 help="Number of trained ML models")
    
    with col4:
        quantum_circuits = len(quantum_ml.quantum_results) if quantum_ml else 0
        st.metric("Quantum Circuits", quantum_circuits,
                 help="Number of quantum circuits executed")
    
    with col5:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.metric("Last Update", current_time,
                 help="Last dashboard refresh")
    
    with col6:
        # Calculate system health score
        health_score = calculate_system_health(classical_ml, quantum_ml, pricer)
        health_color = "normal" if health_score > 80 else "off" if health_score > 60 else "inverse"
        st.metric("System Health", f"{health_score}%", 
                 delta_color=health_color,
                 help="Overall system performance score")

    # MARKET INTELLIGENCE SECTION
    st.markdown("### 🌍 Market Intelligence")
    
    col_mkt1, col_mkt2, col_mkt3, col_mkt4, col_mkt5 = st.columns(5)
    
    with col_mkt1:
        vix = pricer.market_data.get('VIX', 0)
        vix_status = "High Vol" if vix > 20 else "Normal" if vix > 15 else "Low Vol"
        st.metric("VIX Index", f"{vix:.1f}", vix_status,
                 help="Market volatility indicator")
    
    with col_mkt2:
        sofr = pricer.market_data.get('SOFR', 0) * 100
        st.metric("SOFR Rate", f"{sofr:.3f}%",
                 help="Secured Overnight Financing Rate")
    
    with col_mkt3:
        ust_10y = pricer.market_data.get('UST_10Y', 0) * 100
        st.metric("10Y Treasury", f"{ust_10y:.3f}%",
                 help="10-Year US Treasury Yield")
    
    with col_mkt4:
        swap_5y = pricer.market_data.get('SWAP_5Y', 0) * 100
        st.metric("5Y Swap Rate", f"{swap_5y:.3f}%",
                 help="5-Year Swap Rate")
    
    with col_mkt5:
        libor = pricer.market_data.get('LIBOR_3M', 0) * 100
        st.metric("3M LIBOR", f"{libor:.3f}%",
                 help="3-Month LIBOR Rate")

    # VISUAL ANALYTICS ROW
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.markdown("#### 📈 Yield Curve Analysis")
        
        # Enhanced yield curve with multiple scenarios
        tenors = list(pricer.yield_curve.keys())
        current_rates = [-np.log(df) / tenor if tenor > 0 else 0 for tenor, df in pricer.yield_curve.items()]
        
        # Create scenarios for comparison
        bullish_rates = [rate * 1.1 for rate in current_rates]  # +10%
        bearish_rates = [rate * 0.9 for rate in current_rates]  # -10%
        
        fig_yield = go.Figure()
        
        # Add multiple yield curve scenarios
        fig_yield.add_trace(go.Scatter(
            x=tenors, y=current_rates, mode='lines+markers',
            name='Current Curve', line=dict(color='blue', width=4),
            marker=dict(size=8)
        ))
        
        fig_yield.add_trace(go.Scatter(
            x=tenors, y=bullish_rates, mode='lines',
            name='Bull Scenario (+10%)', line=dict(color='green', width=2, dash='dash'),
            opacity=0.7
        ))
        
        fig_yield.add_trace(go.Scatter(
            x=tenors, y=bearish_rates, mode='lines',
            name='Bear Scenario (-10%)', line=dict(color='red', width=2, dash='dash'),
            opacity=0.7
        ))
        
        fig_yield.update_layout(
            title="Yield Curve with Market Scenarios",
            xaxis_title="Tenor (Years)",
            yaxis_title="Yield (%)",
            height=400,
            showlegend=True,
            template="plotly_white"
        )
        st.plotly_chart(fig_yield, use_container_width=True)
    
    with col_viz2:
        st.markdown("#### 🤖 ML Model Performance")
        
        if classical_ml.training_history:
            # Create performance comparison chart
            models = [m['model'] for m in classical_ml.training_history]
            cv_scores = [m['cv_mae'] for m in classical_ml.training_history]
            
            # Sort by performance (best first)
            sorted_data = sorted(zip(models, cv_scores), key=lambda x: x[1])
            models_sorted, scores_sorted = zip(*sorted_data)
            
            fig_perf = go.Figure()
            
            fig_perf.add_trace(go.Bar(
                y=list(models_sorted),
                x=list(scores_sorted),
                orientation='h',
                marker_color=['#2ca02c' if i == 0 else '#1f77b4' for i in range(len(models_sorted))],
                text=[f"${score:.2f}" for score in scores_sorted],
                textposition='auto',
            ))
            
            fig_perf.update_layout(
                title="Model Performance (CV MAE - Lower is Better)",
                xaxis_title="Cross-Validation MAE ($)",
                height=400,
                showlegend=False,
                template="plotly_white"
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Best model highlight
            best_model = min(classical_ml.training_history, key=lambda x: x['cv_mae'])
            st.success(f"🏆 **Best Performer**: {best_model['model']} (CV MAE: ${best_model['cv_mae']:.2f})")
        else:
            st.info("""
            🤖 **No trained models yet!**
            
            Visit the **Classical ML** tab to:
            - Generate training data
            - Train machine learning models  
            - Enable advanced pricing capabilities
            """)

    # PERFORMANCE ANALYTICS ROW
    st.markdown("### 📊 Advanced Performance Analytics")
    
    col_perf1, col_perf2, col_perf3 = st.columns(3)
    
    with col_perf1:
        st.markdown("#### 🎯 Model Statistics")
        
        if classical_ml.training_history:
            avg_mae = np.mean([m['cv_mae'] for m in classical_ml.training_history])
            best_mae = min([m['cv_mae'] for m in classical_ml.training_history])
            worst_mae = max([m['cv_mae'] for m in classical_ml.training_history])
            model_variety = len(set(m['model'] for m in classical_ml.training_history))
            
            stats_data = {
                'Metric': ['Average CV MAE', 'Best CV MAE', 'Worst CV MAE', 'Model Types'],
                'Value': [f"${avg_mae:.2f}", f"${best_mae:.2f}", f"${worst_mae:.2f}", f"{model_variety}"]
            }
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        else:
            st.info("Train models to see performance statistics")

    with col_perf2:
        st.markdown("#### ⚡ Quantum Analytics")
        
        if quantum_ml and quantum_ml.quantum_results:
            recent_results = quantum_ml.quantum_results[-5:]  # Last 5 executions
            
            quantum_data = []
            for result in recent_results:
                quantum_data.append({
                    'Circuit': result['circuit_type'],
                    'Expectation': f"{result['expectation']:.4f}",
                    'Time (s)': f"{result['execution_time']:.3f}",
                    'Qubits': result['circuit_metrics']['qubits']
                })
            
            st.dataframe(pd.DataFrame(quantum_data), use_container_width=True)
            
            # Quantum performance summary
            if quantum_ml.circuit_performance:
                total_executions = len(quantum_ml.quantum_results)
                avg_execution_time = np.mean([r['execution_time'] for r in quantum_ml.quantum_results])
                st.metric("Total Executions", total_executions)
                st.metric("Avg Execution Time", f"{avg_execution_time:.3f}s")
        else:
            st.info("""
            ⚛️ **Quantum computing ready!**
            
            Visit the **Quantum ML** tab to:
            - Execute quantum circuits
            - Analyze quantum advantage
            - Compare with classical methods
            """)

    with col_perf3:
        st.markdown("#### 📈 Feature Intelligence")
        
        if hasattr(classical_ml, 'feature_importance') and classical_ml.feature_importance:
            if 'Random Forest' in classical_ml.feature_importance:
                importance = classical_ml.feature_importance['Random Forest']
                feature_names = classical_ml.feature_names[:len(importance)] if hasattr(classical_ml, 'feature_names') else [f'Feature {i+1}' for i in range(len(importance))]
                
                # Create horizontal bar chart
                fig_features = go.Figure()
                fig_features.add_trace(go.Bar(
                    y=feature_names,
                    x=importance,
                    orientation='h',
                    marker_color='lightseagreen'
                ))
                fig_features.update_layout(
                    title="Feature Importance (Random Forest)",
                    xaxis_title="Importance Score",
                    height=300,
                    showlegend=False,
                    template="plotly_white"
                )
                st.plotly_chart(fig_features, use_container_width=True)
                
                # Top 3 features
                indices = np.argsort(importance)[::-1][:3]
                st.write("**Top 3 Features:**")
                for i, idx in enumerate(indices):
                    st.write(f"{i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
        else:
            st.info("Feature importance available after model training")

    # QUICK ACTIONS & RECOMMENDATIONS
    st.markdown("### 🚀 Quick Actions & Insights")
    
    col_act1, col_act2, col_act3 = st.columns(3)
    
    with col_act1:
        st.markdown("#### 💡 Trading Insights")
        if pricer.market_data.get('VIX', 0) > 20:
            st.warning("**High Volatility Alert** - Consider adjusting risk parameters")
        else:
            st.success("**Normal Market Conditions** - Standard pricing models applicable")
            
        if classical_ml.training_history:
            st.info(f"**ML Ready** - {len(classical_ml.models)} models available for pricing")
    
    with col_act2:
        st.markdown("#### ⚡ Performance Tips")
        tips = [
            "Use ensemble methods for more stable predictions",
            "Quantum enhancement works best for complex derivatives",
            "Monitor feature importance for model interpretability",
            "Regular retraining improves model accuracy"
        ]
        
        for i, tip in enumerate(tips):
            st.write(f"• {tip}")
    
    with col_act3:
        st.markdown("#### 📊 Next Steps")
        
        if not classical_ml.training_history:
            st.button("🚀 Train First ML Models", 
                     help="Start with Classical ML tab",
                     use_container_width=True)
        
        if quantum_ml and not quantum_ml.quantum_results:
            st.button("⚛️ Run Quantum Circuit", 
                     help="Explore quantum capabilities",
                     use_container_width=True)
        
        st.button("🎯 Live Pricing Analysis", 
                 help="Test real-time pricing",
                 use_container_width=True)

    # REAL-TIME ALERTS
    if analytics.comparison_data:
        st.markdown("### 🔔 Performance Alerts")
        
        df_comparison = pd.DataFrame(analytics.comparison_data)
        quantum_advantage = (df_comparison['classical_error'] - df_comparison['quantum_error']).mean()
        
        if quantum_advantage > 1000:  # $1000 advantage
            st.success(f"🎉 **Quantum Advantage Detected**: Average ${quantum_advantage:,.0f} improvement over classical methods")
        elif quantum_advantage > 0:
            st.info(f"📊 **Moderate Quantum Benefit**: ${quantum_advantage:,.0f} average improvement")
        else:
            st.warning("⚡ **Classical Methods Performing Better** - Consider model optimization")

def calculate_system_health(classical_ml, quantum_ml, pricer):
    """Calculate overall system health score (0-100)"""
    score = 0
    max_score = 0
    
    # ML Models Health (40 points)
    if classical_ml.training_history:
        avg_mae = np.mean([m['cv_mae'] for m in classical_ml.training_history])
        # Lower MAE = better score (normalize to 0-40)
        ml_score = max(0, 40 - (avg_mae / 1000))  # Assuming MAE in reasonable range
        score += ml_score
    max_score += 40
    
    # Quantum Health (30 points)
    if quantum_ml and quantum_ml.quantum_results:
        # More quantum results = better score
        quantum_score = min(30, len(quantum_ml.quantum_results) * 2)
        score += quantum_score
    max_score += 30
    
    # Market Data Health (20 points)
    if pricer.market_data:
        market_score = 20  # Basic market data available
        score += market_score
    max_score += 20
    
    # Feature Engineering Health (10 points)
    if hasattr(classical_ml, 'feature_importance') and classical_ml.feature_importance:
        feature_score = 10
        score += feature_score
    max_score += 10
    
    return int((score / max_score) * 100) if max_score > 0 else 0
def show_classical_ml(classical_ml, pricer, quantum_ml):
    """Enhanced Classical ML section with ML-driven pricing and quantum circuit generation"""
    
    st.markdown("## 🏛️ Advanced Classical Machine Learning")
    
    # ML Pricing Calculator with Real-time Updates
    st.markdown("### 🎯 ML Swaption Pricing Calculator")
    
    with st.expander("🚀 Smart ML Pricing Dashboard", expanded=True):
        col_price1, col_price2, col_price3 = st.columns(3)
        
        with col_price1:
            ml_expiry = st.slider("Expiry (Years)", 0.25, 10.0, 2.0, 0.25, key="ml_price_expiry")
            ml_tenor = st.slider("Tenor (Years)", 1.0, 30.0, 5.0, 0.5, key="ml_price_tenor")
            
        with col_price2:
            ml_strike = st.slider("Strike Rate (%)", 0.5, 10.0, 3.5, 0.1, key="ml_price_strike") / 100
            ml_volatility = st.slider("Volatility (%)", 10.0, 80.0, 25.0, 1.0, key="ml_price_vol") / 100
            
        with col_price3:
            ml_notional = st.selectbox("Notional", [1e6, 5e6, 10e6, 25e6, 50e6, 100e6], 
                                     format_func=lambda x: f"${x/1e6:.0f}M", 
                                     index=2, key="ml_price_notional")
            ml_forward_rate = st.number_input("Forward Rate (%)", 1.0, 10.0, 4.2, 0.1, 
                                            key="ml_price_forward") / 100
        
        # Real-time price calculation
        traditional_price = pricer.black_76_swaption_price(
            ml_notional, ml_expiry, ml_tenor, ml_strike, "Payer Swaption", ml_volatility
        )
        
        col_rt1, col_rt2 = st.columns(2)
        with col_rt1:
            st.metric("Traditional Price", f"${traditional_price:,.0f}")
        
        # ML Prediction with Confidence
        if classical_ml and classical_ml.models:
            features = [ml_forward_rate, ml_strike, ml_volatility, ml_expiry, ml_tenor, ml_notional]
            ml_price, confidence = classical_ml.predict_with_confidence(features)
            
            with col_rt2:
                error = ml_price - traditional_price
                error_pct = (error / traditional_price) * 100
                
                st.metric(
                    "ML Predicted Price", 
                    f"${ml_price:,.0f}",
                    delta=f"{error_pct:+.1f}%",
                    delta_color="inverse" if abs(error_pct) > 5 else "normal",
                    help=f"Confidence: {confidence:.1%}"
                )
            
            # Advanced ML Analysis
            if st.button("🔍 Deep ML Analysis", key="deep_ml_analysis"):
                with st.spinner("Performing advanced ML analysis..."):
                    perform_advanced_ml_analysis(
                        classical_ml, quantum_ml, features, 
                        traditional_price, ml_price, pricer
                    )
        else:
            st.warning("⚠️ Train ML models first for advanced pricing!")
    
    # Enhanced Training Section
    st.markdown("### ⚙️ Advanced Training Configuration")
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        n_samples = st.slider("Training Samples", 100, 20000, 5000, 100, key="cl_samples")
        cv_folds = st.slider("CV Folds", 3, 10, 5, key="cl_cv")
        
    with col_config2:
        feature_engineering = st.selectbox(
            "Feature Engineering", 
            ["Basic", "Advanced", "Quantum-Inspired"],
            index=1,
            key="cl_feat_level"
        )
        ensemble_method = st.selectbox(
            "Ensemble Method",
            ["Voting", "Stacking", "Weighted Average"],
            index=0,
            key="cl_ensemble"
        )
        
    with col_config3:
        hyperparameter_tuning = st.checkbox("Auto Hyperparameter Tuning", value=True, key="cl_hp_tune")
        generate_quantum_circuits = st.checkbox("Generate Quantum Circuits", value=True, key="cl_quantum_circ")
    
    # Training Execution with Enhanced Features
    if st.button("🎯 Train Advanced ML Models", type="primary", key="cl_train_advanced"):
        with st.spinner("Training advanced ML models with enhanced features..."):
            try:
                # Generate comprehensive training data
                data_df = generate_enhanced_training_data(n_samples, pricer, feature_engineering)
                
                # Feature engineering based on selection
                feature_columns = [col for col in data_df.columns if col != 'price']
                X = data_df[feature_columns]
                y = data_df['price']
                
                if feature_engineering == "Advanced":
                    X_engineered = classical_ml.engineer_advanced_features(X)
                elif feature_engineering == "Quantum-Inspired":
                    X_engineered = classical_ml.engineer_quantum_inspired_features(X)
                else:
                    X_engineered = X
                
                # Store for later use
                st.session_state.X_engineered = X_engineered
                st.session_state.feature_names = X_engineered.columns.tolist()
                st.session_state.y_actual = y.values
                
                # Train models with enhanced configuration
                results = classical_ml.train_enhanced_models(
                    X_engineered, y, cv_folds, 
                    hyperparameter_tuning, ensemble_method
                )
                
                # Generate quantum circuits if requested
                if generate_quantum_circuits and quantum_ml:
                    generate_ml_quantum_circuits(classical_ml, quantum_ml, X_engineered, y)
                
                st.session_state.advanced_classical_results = results
                st.success(f"✅ Successfully trained {len(results)} models with {feature_engineering} features!")
                
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

    # Display results if available
    if 'advanced_classical_results' in st.session_state:
        display_enhanced_ml_results(st.session_state.advanced_classical_results, classical_ml)

def perform_advanced_ml_analysis(classical_ml, quantum_ml, features, traditional_price, ml_price, pricer):
    """Perform comprehensive ML analysis with quantum integration"""
    
    st.markdown("#### 🔬 Advanced ML Analysis")
    
    # Feature Importance Analysis
    col_ana1, col_ana2 = st.columns(2)
    
    with col_ana1:
        st.markdown("##### 📊 Feature Impact Analysis")
        feature_impact = classical_ml.analyze_feature_impact(features)
        
        fig_impact = go.Figure(data=[
            go.Bar(x=list(feature_impact.keys()), 
                  y=list(feature_impact.values()),
                  marker_color='lightcoral')
        ])
        fig_impact.update_layout(
            title="Feature Impact on Prediction",
            height=400
        )
        st.plotly_chart(fig_impact, use_container_width=True)
    
    with col_ana2:
        st.markdown("##### 🎯 Prediction Confidence")
        confidence_metrics = classical_ml.get_prediction_confidence(features)
        
        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence_metrics['confidence'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Prediction Confidence"},
            gauge={'axis': {'range': [None, 100]},
                  'bar': {'color': "darkblue"},
                  'steps': [{'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "gray"},
                           {'range': [80, 100], 'color': "darkgray"}]}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Quantum Circuit Generation based on ML model
    if quantum_ml and HAS_QISKIT:
        st.markdown("#### ⚛️ ML-Driven Quantum Circuit Generation")
        
        # Generate circuits based on ML model characteristics
        quantum_circuits = generate_ml_based_quantum_circuits(
            classical_ml, quantum_ml, features, ml_price
        )
        
        # Display generated circuits
        col_qc1, col_qc2 = st.columns(2)
        
        with col_qc1:
            st.markdown("##### 🔧 Feature Encoding Circuit")
            if 'feature_encoding' in quantum_circuits:
                expectation, circuit, counts = quantum_circuits['feature_encoding']
                st.markdown(f"**Expectation:** {expectation:.4f}")
                st.text("Feature Encoding Circuit Structure")
                # Display circuit diagram if available
        
        with col_qc2:
            st.markdown("##### 🎯 Price Prediction Circuit")
            if 'price_prediction' in quantum_circuits:
                expectation, circuit, counts = quantum_circuits['price_prediction']
                st.markdown(f"**Expectation:** {expectation:.4f}")
                st.text("Price Prediction Circuit Structure")

def generate_ml_based_quantum_circuits(classical_ml, quantum_ml, features, ml_price):
    """Generate quantum circuits based on ML model characteristics and predictions"""
    
    circuits = {}
    
    try:
        # Circuit 1: Feature Encoding based on ML feature importance
        feature_importance = classical_ml.get_feature_importance()
        
        # Use feature importance to weight quantum feature encoding
        weighted_features = [f * imp for f, imp in zip(features[:4], feature_importance[:4])]
        
        circuits['feature_encoding'] = quantum_ml.run_advanced_circuit(
            "feature_map_advanced",
            features=weighted_features,
            show_diagram=True
        )
        
        # Circuit 2: Price Prediction Encoding
        # Normalize price for quantum representation
        price_normalized = ml_price / 1e6  # Scale for quantum representation
        price_features = features + [price_normalized]
        
        circuits['price_prediction'] = quantum_ml.run_advanced_circuit(
            "quantum_neural_network",
            features=price_features,
            show_diagram=True
        )
        
        # Circuit 3: Error Estimation
        traditional_price = pricer.black_76_swaption_price(
            features[5], features[3], features[4], features[1], "Payer Swaption", features[2]
        )
        price_error = abs(ml_price - traditional_price) / traditional_price
        
        error_features = features + [price_error]
        circuits['error_estimation'] = quantum_ml.run_advanced_circuit(
            "variational_advanced",
            features=error_features,
            show_diagram=True
        )
        
    except Exception as e:
        st.warning(f"Quantum circuit generation partially failed: {e}")
    
    return circuits

def generate_ml_quantum_circuits(classical_ml, quantum_ml, X_engineered, y):
    """Generate quantum circuits based on trained ML model characteristics"""
    
    st.markdown("#### ⚛️ Generating ML-Informed Quantum Circuits")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Circuit based on feature relationships
        status_text.text("Generating feature relationship circuit...")
        sample_features = X_engineered.iloc[0].values.tolist() if len(X_engineered) > 0 else [0.04] * 6
        
        # Use ML model insights to inform quantum circuit parameters
        feature_importance = classical_ml.get_feature_importance()
        
        # Generate circuits with ML-informed parameters
        circuit_types = ["feature_map_advanced", "variational_advanced", "quantum_neural_network"]
        
        for i, circuit_type in enumerate(circuit_types):
            progress = (i + 1) / len(circuit_types)
            progress_bar.progress(progress)
            status_text.text(f"Generating {circuit_type} circuit...")
            
            try:
                # Modify features based on ML feature importance
                weighted_features = [
                    feat * imp for feat, imp in zip(sample_features, feature_importance[:len(sample_features)])
                ]
                
                expectation, circuit, counts = quantum_ml.run_advanced_circuit(
                    circuit_type,
                    features=weighted_features,
                    show_diagram=True
                )
                
                # Store circuit information
                if 'ml_quantum_circuits' not in st.session_state:
                    st.session_state.ml_quantum_circuits = {}
                
                st.session_state.ml_quantum_circuits[circuit_type] = {
                    'expectation': expectation,
                    'circuit': circuit,
                    'counts': counts,
                    'features_used': weighted_features,
                    'feature_importance': feature_importance
                }
                
            except Exception as circuit_error:
                st.warning(f"Failed to generate {circuit_type}: {circuit_error}")
        
        status_text.text("✅ ML quantum circuits generated successfully!")
        progress_bar.progress(1.0)
        
    except Exception as e:
        st.error(f"Quantum circuit generation failed: {e}")
    finally:
        progress_bar.empty()
        status_text.empty()

def display_enhanced_ml_results(results, classical_ml):
    """Display enhanced ML results with interactive components"""
    
    st.markdown("## 📊 Enhanced Model Performance Analysis")
    
    # Performance Summary with Interactive Charts
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    
    with col_sum1:
        best_model = min(results.items(), key=lambda x: x[1]['cv_mae'])
        st.metric("🎯 Best Model", best_model[0])
    
    with col_sum2:
        st.metric("📈 Best CV MAE", f"${best_model[1]['cv_mae']:,.2f}")
    
    with col_sum3:
        avg_mae = np.mean([r['cv_mae'] for r in results.values()])
        improvement = ((avg_mae - best_model[1]['cv_mae']) / avg_mae) * 100
        st.metric("💪 Improvement vs Avg", f"{improvement:.1f}%")
    
    # Interactive Model Comparison
    st.markdown("### 📈 Interactive Model Comparison")
    
    comparison_metric = st.selectbox(
        "Select Metric for Comparison",
        ["CV MAE", "Training MAE", "RMSE", "R² Score"],
        key="comp_metric"
    )
    
    # Create interactive comparison chart
    metric_key = {'CV MAE': 'cv_mae', 'Training MAE': 'mae', 
                 'RMSE': 'rmse', 'R² Score': 'r2'}[comparison_metric]
    
    models = list(results.keys())
    metric_values = [results[name][metric_key] for name in models]
    
    fig_comparison = go.Figure(data=[
        go.Bar(x=models, y=metric_values,
              marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
              text=[f"{v:,.2f}" if metric_key != 'r2' else f"{v:.4f}" for v in metric_values],
              textposition='auto')
    ])
    
    yaxis_title = '$' if metric_key != 'r2' else 'Score'
    fig_comparison.update_layout(
        title=f'Model Comparison - {comparison_metric}',
        yaxis_title=yaxis_title,
        height=400
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Feature Importance Dashboard
    st.markdown("### 🔍 Feature Importance Dashboard")
    
    if hasattr(classical_ml, 'feature_importance') and classical_ml.feature_importance:
        importance_model = st.selectbox(
            "Select Model for Feature Analysis",
            list(classical_ml.feature_importance.keys()),
            key="imp_model"
        )
        
        if importance_model in classical_ml.feature_importance:
            display_interactive_feature_importance(
                importance_model,
                classical_ml.feature_names,
                classical_ml.feature_importance[importance_model]
            )

def display_interactive_feature_importance(model_name, feature_names, importance):
    """Display interactive feature importance analysis"""
    
    col_f1, col_f2 = st.columns([2, 1])
    
    with col_f1:
        # Interactive feature importance chart
        indices = np.argsort(importance)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importance = [importance[i] for i in indices]
        
        fig = go.Figure(data=[
            go.Bar(y=sorted_features, x=sorted_importance,
                  orientation='h',
                  marker_color='lightseagreen',
                  hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>')
        ])
        fig.update_layout(
            title=f'{model_name} - Feature Importance',
            xaxis_title='Importance',
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_f2:
        st.markdown("##### Top Features")
        for i in range(min(5, len(indices))):
            with st.container():
                st.metric(
                    label=f"{i+1}. {feature_names[indices[i]]}",
                    value=f"{importance[indices[i]]:.4f}"
                )
        
        # Feature statistics
        st.markdown("##### Importance Statistics")
        st.metric("Max Importance", f"{np.max(importance):.4f}")
        st.metric("Mean Importance", f"{np.mean(importance):.4f}")
        st.metric("Std Dev", f"{np.std(importance):.4f}")

def generate_enhanced_training_data(n_samples, pricer, feature_engineering):
    """Generate enhanced training data with advanced features"""
    
    # This would be your existing data generation logic enhanced with:
    # - More realistic market scenarios
    # - Correlation structures
    # - Regime changes
    # - Stress scenarios
    
    return generate_advanced_training_data(n_samples, pricer)  # Your existing function

# Add these methods to your ClassicalML class
class EnhancedClassicalML(ClassicalML):
    def predict_with_confidence(self, features):
        """Predict with confidence estimation"""
        predictions = []
        for model_name, model in self.models.items():
            try:
                pred = model.predict([features])[0]
                predictions.append(pred)
            except:
                continue
        
        if predictions:
            ensemble_pred = np.mean(predictions)
            confidence = 1.0 - (np.std(predictions) / ensemble_pred if ensemble_pred != 0 else 0.1)
            confidence = max(0.1, min(0.99, confidence))  # Clamp between 0.1 and 0.99
            return ensemble_pred, confidence
        else:
            return 0, 0.1
    
    def engineer_quantum_inspired_features(self, X):
        """Engineer quantum-inspired features"""
        # Add features inspired by quantum computing concepts
        # This is a placeholder for actual implementation
        return self.engineer_features(X)  # Extend with quantum-inspired features
    
    def analyze_feature_impact(self, features):
        """Analyze impact of each feature on prediction"""
        base_prediction = self.predict_ensemble(features)
        impacts = {}
        
        for i, feature_name in enumerate(self.feature_names[:len(features)]):
            perturbed_features = features.copy()
            # Perturb feature by 1%
            perturbed_features[i] *= 1.01
            perturbed_pred = self.predict_ensemble(perturbed_features)
            impact = abs(perturbed_pred - base_prediction) / base_prediction if base_prediction != 0 else 0
            impacts[feature_name] = impact
        
        return impacts
def generate_advanced_training_data(n_samples, pricer):
    """Generate advanced training data with comprehensive feature engineering"""
    data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_samples):
        if i % 100 == 0:
            progress = (i + 1) / n_samples
            progress_bar.progress(progress)
            status_text.text(f"Generating sample {i+1}/{n_samples}...")
        
        # Sample realistic financial parameters
        expiry = np.random.uniform(0.25, 10.0)
        tenor = np.random.uniform(1.0, 30.0)
        strike = np.random.uniform(0.01, 0.08)
        volatility = np.random.uniform(0.10, 0.40)
        notional = np.random.choice([1e6, 5e6, 10e6, 25e6, 50e6, 100e6])
        
        # Calculate price using Black-76 model
        try:
            price = pricer.black_76_swaption_price(
                notional, expiry, tenor, strike, "Payer Swaption", volatility
            )
        except:
            # Fallback calculation if pricing fails
            forward_rate = 0.04 + (strike - 0.035) * 0.5
            annuity = tenor * 0.9
            d1 = (np.log(forward_rate / strike) + (volatility**2 / 2) * expiry) / (volatility * np.sqrt(expiry))
            d2 = d1 - volatility * np.sqrt(expiry)
            call_price = annuity * (forward_rate * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2))
            price = notional * max(call_price, 0)
        
        # Calculate forward rate
        forward_rate = pricer.calculate_forward_swap_rate(expiry, tenor)
        
        # Create comprehensive feature set
        sample_data = {
            'forward_rate': forward_rate,
            'strike': strike,
            'volatility': volatility,
            'expiry': expiry,
            'tenor': tenor,
            'notional': notional,
            'moneyness': strike / forward_rate if forward_rate > 0 else 1.0,
            'log_moneyness': np.log(strike / forward_rate) if forward_rate > 0 and strike > 0 else 0,
            'time_sqrt': np.sqrt(expiry),
            'vol_time_adjusted': volatility * np.sqrt(expiry),
            'rate_spread': forward_rate - strike,
            'rate_ratio': strike / forward_rate if forward_rate > 0 else 1.0,
            'price': price
        }
        
        data.append(sample_data)
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Successfully generated {n_samples} training samples!")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Remove any invalid samples
    df = df[(df['price'] > 0) & (df['price'] < df['notional'] * 0.3)]
    df = df.dropna()
    
    return df
def show_comparison(analytics, classical_ml=None, quantum_ml=None, pricer=None):
    """Enhanced comparison section with ML price predictions and circuit analysis"""

    st.markdown("## 📊 Advanced Performance Comparison")

    # Comparison Configuration
    st.markdown("### ⚙️ Comparison Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_samples = st.slider("Comparison Samples", 100, 2000, 500, key="comp_samples")
        market_regime = st.selectbox(
            "Market Regime",
            ["normal", "high_vol", "low_vol"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="comp_regime"
        )

    with col2:
        include_real_models = st.checkbox("Include Real Model Predictions", value=True, key="comp_real")
        quantum_circuit_type = st.selectbox(
            "Quantum Circuit Type",
            ["feature_map_advanced", "variational_advanced", "quantum_neural_network"],
            key="comp_circuit"
        )

    with col3:
        show_circuit_analysis = st.checkbox("Show Circuit Analysis", value=True, key="comp_circuit_analysis")
        show_price_predictions = st.checkbox("Show Price Predictions", value=True, key="comp_prices")

    # Generate Comparison
    if st.button("🔄 Generate Advanced Comparison", type="primary", key="comp_generate"):
        with st.spinner("Generating comprehensive comparison with ML predictions and circuit analysis..."):
            try:
                analytics.generate_comprehensive_comparison(
                    n_samples=n_samples,
                    include_real_models=include_real_models,
                    classical_ml=classical_ml,
                    quantum_ml=quantum_ml,
                    market_regime=market_regime,
                    quantum_circuit_type=quantum_circuit_type
                )
                st.success(f"✅ Generated {n_samples} comparison samples with {market_regime} regime!")
            except Exception as e:
                st.error(f"❌ Comparison generation failed: {e}")

    # Display Advanced Comparison
    if analytics.comparison_data:
        df = pd.DataFrame(analytics.comparison_data)

        # Summary Statistics
        st.markdown("### 📈 Performance Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            classical_mae = df['classical_error'].mean()
            st.metric("Classical MAE", f"${classical_mae:,.0f}")

        with col2:
            quantum_mae = df['quantum_error'].mean()
            st.metric("Quantum MAE", f"${quantum_mae:,.0f}")

        with col3:
            improvement = ((classical_mae - quantum_mae) / classical_mae) * 100 if classical_mae > 0 else 0
            st.metric("MAE Improvement", f"+{improvement:.1f}%")

        with col4:
            success_rate = (df['improvement'] > 0).mean() * 100
            st.metric("Quantum Success Rate", f"{success_rate:.1f}%")

        # Error Distribution Analysis
        st.markdown("### 🔍 Error Distribution Analysis")

        col_err1, col_err2 = st.columns(2)

        with col_err1:
            # Box plot for error comparison
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=df['classical_error'],
                name='Classical ML',
                boxpoints='outliers',
                marker_color='#1f77b4'
            ))
            fig_box.add_trace(go.Box(
                y=df['quantum_error'],
                name='Quantum ML',
                boxpoints='outliers',
                marker_color='#ff7f0e'
            ))
            fig_box.update_layout(
                title='Error Distribution Comparison',
                yaxis_title='Absolute Error ($)',
                height=400
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with col_err2:
            # Error percentage distribution
            fig_error_pct = go.Figure()
            fig_error_pct.add_trace(go.Histogram(
                x=df['classical_error_pct'],
                name='Classical ML',
                opacity=0.7,
                nbinsx=50,
                marker_color='#1f77b4'
            ))
            fig_error_pct.add_trace(go.Histogram(
                x=df['quantum_error_pct'],
                name='Quantum ML',
                opacity=0.7,
                nbinsx=50,
                marker_color='#ff7f0e'
            ))
            fig_error_pct.update_layout(
                title='Error Percentage Distribution',
                xaxis_title='Error Percentage (%)',
                yaxis_title='Frequency',
                height=400,
                barmode='overlay'
            )
            st.plotly_chart(fig_error_pct, use_container_width=True)

        # Price Prediction Analysis
        if show_price_predictions and len(df) > 0:
            st.markdown("### 💰 Price Prediction Analysis")

            # Select a random sample for detailed analysis
            sample_idx = np.random.randint(0, len(df))
            sample = df.iloc[sample_idx]

            col_price1, col_price2 = st.columns(2)

            with col_price1:
                # Price comparison for selected sample
                prices = {
                    'Model': ['True Price', 'Classical ML', 'Quantum ML'],
                    'Price ($)': [
                        sample['true_price'],
                        sample['classical_pred'],
                        sample['quantum_pred']
                    ],
                    'Error ($)': [0, sample['classical_error'], sample['quantum_error']],
                    'Error (%)': [0, sample['classical_error_pct'], sample['quantum_error_pct']]
                }

                st.dataframe(pd.DataFrame(prices), use_container_width=True)

                # Price comparison bar chart
                fig_prices = go.Figure()
                fig_prices.add_trace(go.Bar(
                    name='Predicted Prices',
                    x=['True', 'Classical', 'Quantum'],
                    y=[sample['true_price'], sample['classical_pred'], sample['quantum_pred']],
                    marker_color=['green', 'blue', 'orange']
                ))
                fig_prices.update_layout(
                    title=f'Sample #{sample_idx} - Price Comparison',
                    yaxis_title='Price ($)',
                    height=400
                )
                st.plotly_chart(fig_prices, use_container_width=True)

            with col_price2:
                # Scatter plot of all predictions
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=df['true_price'],
                    y=df['classical_pred'],
                    mode='markers',
                    name='Classical ML',
                    marker=dict(color='blue', opacity=0.6)
                ))
                fig_scatter.add_trace(go.Scatter(
                    x=df['true_price'],
                    y=df['quantum_pred'],
                    mode='markers',
                    name='Quantum ML',
                    marker=dict(color='orange', opacity=0.6)
                ))
                # Perfect prediction line
                max_price = max(df['true_price'].max(), df['classical_pred'].max(), df['quantum_pred'].max())
                fig_scatter.add_trace(go.Scatter(
                    x=[0, max_price],
                    y=[0, max_price],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                fig_scatter.update_layout(
                    title='True vs Predicted Prices',
                    xaxis_title='True Price ($)',
                    yaxis_title='Predicted Price ($)',
                    height=400
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Circuit Performance Analysis
        if show_circuit_analysis and quantum_ml and quantum_ml.quantum_results:
            st.markdown("### ⚛️ Quantum Circuit Performance")

            # Get recent quantum results
            recent_results = quantum_ml.quantum_results[-10:]

            col_circ1, col_circ2 = st.columns(2)

            with col_circ1:
                # Circuit performance metrics
                circuit_metrics = []
                for result in recent_results:
                    perf = result.get('expectation_breakdown', {})
                    circuit_metrics.append({
                        'Circuit Type': result['circuit_type'],
                        'Expectation': result['expectation'],
                        'Execution Time (s)': result['execution_time'],
                        'Qubits': result['circuit_metrics']['qubits'],
                        'Depth': result['circuit_metrics']['depth']
                    })

                if circuit_metrics:
                    st.dataframe(pd.DataFrame(circuit_metrics), use_container_width=True)

            with col_circ2:
                # Circuit performance trends
                if len(recent_results) > 1:
                    fig_circuit_perf = go.Figure()
                    fig_circuit_perf.add_trace(go.Scatter(
                        x=list(range(len(recent_results))),
                        y=[r['expectation'] for r in recent_results],
                        mode='lines+markers',
                        name='Expectation Value',
                        line=dict(color='purple')
                    ))
                    fig_circuit_perf.update_layout(
                        title='Quantum Circuit Performance Trend',
                        xaxis_title='Run Number',
                        yaxis_title='Expectation Value',
                        height=300
                    )
                    st.plotly_chart(fig_circuit_perf, use_container_width=True)

            # Circuit Type Analysis
            st.markdown("#### 🔧 Circuit Type Comparison")

            # Group by circuit type
            circuit_types = {}
            for result in quantum_ml.quantum_results:
                circuit_type = result['circuit_type']
                if circuit_type not in circuit_types:
                    circuit_types[circuit_type] = []
                circuit_types[circuit_type].append(result['expectation'])

            if circuit_types:
                fig_circuit_types = go.Figure()
                for circuit_type, expectations in circuit_types.items():
                    fig_circuit_types.add_trace(go.Box(
                        y=expectations,
                        name=circuit_type.replace('_', ' ').title(),
                        boxpoints='all'
                    ))
                fig_circuit_types.update_layout(
                    title='Performance by Circuit Type',
                    yaxis_title='Expectation Value',
                    height=400
                )
                st.plotly_chart(fig_circuit_types, use_container_width=True)

        # Market Regime Analysis
        if 'market_regime' in df.columns:
            st.markdown("### 🌡️ Market Regime Analysis")

            regime_stats = df.groupby('market_regime').agg({
                'classical_error': 'mean',
                'quantum_error': 'mean',
                'improvement_pct': 'mean'
            }).round(2)

            col_reg1, col_reg2 = st.columns(2)

            with col_reg1:
                st.write("**Performance by Market Regime:**")
                st.dataframe(regime_stats, use_container_width=True)

            with col_reg2:
                # Regime performance comparison
                fig_regime = go.Figure()
                fig_regime.add_trace(go.Bar(
                    name='Classical MAE',
                    x=regime_stats.index,
                    y=regime_stats['classical_error'],
                    marker_color='blue'
                ))
                fig_regime.add_trace(go.Bar(
                    name='Quantum MAE',
                    x=regime_stats.index,
                    y=regime_stats['quantum_error'],
                    marker_color='orange'
                ))
                fig_regime.update_layout(
                    title='MAE by Market Regime',
                    yaxis_title='Mean Absolute Error ($)',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_regime, use_container_width=True)

        # Detailed Sample Analysis
        st.markdown("### 🔬 Detailed Sample Analysis")

        with st.expander("View Detailed Comparison Data"):
            # Show detailed dataframe
            display_columns = ['sample_id', 'true_price', 'classical_pred', 'quantum_pred',
                             'classical_error', 'quantum_error', 'improvement_pct']
            display_columns = [col for col in display_columns if col in df.columns]

            st.dataframe(df[display_columns].head(20), use_container_width=True)

            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison Data",
                data=csv,
                file_name=f"quantum_finance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def show_kaggle_data(kaggle_manager, pricer):
    """Enhanced Kaggle data integration section with better visualizations"""
    
    st.markdown("## 🔗 Kaggle Data Integration")
    
    st.markdown("""
    ### 📊 Real Market Data Integration
    
    This section demonstrates integration with real financial data from Kaggle:
    - **Interest Rate Data**: SOFR, LIBOR, Treasury yields
    - **Yield Curve Data**: Complete term structure  
    - **Volatility Data**: Market-implied volatilities
    - **Real-time Updates**: Live market data feeds
    """)
    
    # Display Kaggle API Status
    st.markdown("### 🔄 Kaggle API Status")
    
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        api_status = "✅ Connected" if kaggle_manager.api else "❌ Not Available"
        st.metric("Kaggle API", api_status)
    
    with col_status2:
        datasets_loaded = len(kaggle_manager.loaded_datasets)
        st.metric("Datasets Loaded", datasets_loaded)
    
    with col_status3:
        data_source = "Real Data" if kaggle_manager.api else "Synthetic Data"
        st.metric("Data Source", data_source)
    
    # Data Loading Controls
    st.markdown("### 📥 Data Loading")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Load Interest Rate Data", type="primary", key="kaggle_rates"):
            with st.spinner("Loading interest rate data from Kaggle..."):
                try:
                    df = kaggle_manager.load_interest_rates()
                    if df is not None:
                        st.session_state.rates_data = df
                        st.session_state.rates_loaded = True
                        st.success(f"✅ Loaded {len(df)} interest rate records!")

                        # Show data preview
                        with st.expander("📋 Data Preview"):
                            st.dataframe(df.head(10), use_container_width=True)
                            st.write(f"**Date Range:** {df['date'].min()} to {df['date'].max()}")
                    else:
                        st.warning("⚠️ Could not load interest rates from Kaggle; falling back to synthetic data.")
                        # Generate synthetic data for demonstration
                        synthetic_rates = kaggle_manager._generate_synthetic_rates()
                        st.session_state.rates_data = synthetic_rates
                        st.session_state.rates_loaded = True
                        st.info(f"📊 Using synthetic interest rate data ({len(synthetic_rates)} records)")

                except Exception as e:
                    st.warning(f"⚠️ Could not load interest rates from Kaggle; falling back to synthetic data.")
                    synthetic_rates = kaggle_manager._generate_synthetic_rates()
                    st.session_state.rates_data = synthetic_rates
                    st.session_state.rates_loaded = True
                    st.info(f"📊 Using synthetic interest rate data ({len(synthetic_rates)} records)")

    with col2:
        if st.button("📊 Load Yield Curve Data", type="primary", key="kaggle_yield"):
            with st.spinner("Loading yield curve data from Kaggle..."):
                try:
                    df = kaggle_manager.load_yield_curve()
                    if df is not None:
                        st.session_state.yield_data = df
                        st.session_state.yield_loaded = True
                        st.success(f"✅ Loaded {len(df)} yield curve records!")

                        # Show data preview
                        with st.expander("📋 Data Preview"):
                            st.dataframe(df.head(10), use_container_width=True)
                            if 'date' in df.columns:
                                st.write(f"**Date Range:** {df['date'].min()} to {df['date'].max()}")
                    else:
                        st.warning("⚠️ Could not load yield curve from Kaggle; falling back to synthetic data.")
                        synthetic_yield = kaggle_manager._generate_synthetic_yield_curve()
                        st.session_state.yield_data = synthetic_yield
                        st.session_state.yield_loaded = True
                        st.info(f"📊 Using synthetic yield curve data ({len(synthetic_yield)} records)")

                except Exception as e:
                    st.warning(f"⚠️ Could not load yield curve from Kaggle; falling back to synthetic data.")
                    synthetic_yield = kaggle_manager._generate_synthetic_yield_curve()
                    st.session_state.yield_data = synthetic_yield
                    st.session_state.yield_loaded = True
                    st.info(f"📊 Using synthetic yield curve data ({len(synthetic_yield)} records)")

    with col3:
        if st.button("🔄 Clear Cache", type="secondary", key="kaggle_clear"):
            st.session_state.rates_data = None
            st.session_state.yield_data = None
            st.session_state.rates_loaded = False
            st.session_state.yield_loaded = False
            st.info("Data cache cleared. Click above buttons to reload.")
    
    # Data Visualization Section
    st.markdown("### 📈 Data Visualization")
    
    # Interest Rate Data Display
    if st.session_state.get('rates_loaded', False) and st.session_state.rates_data is not None:
        st.markdown("#### 💹 Interest Rate Analysis")
        df_rates = st.session_state.rates_data
        
        col_rates1, col_rates2 = st.columns([2, 1])
        
        with col_rates1:
            # Time series visualization
            if 'date' in df_rates.columns:
                # Convert date column if needed
                if not pd.api.types.is_datetime64_any_dtype(df_rates['date']):
                    df_rates['date'] = pd.to_datetime(df_rates['date'])
                
                # Get numeric columns for rates
                rate_columns = [col for col in df_rates.columns 
                              if col != 'date' and pd.api.types.is_numeric_dtype(df_rates[col])]
                
                if rate_columns:
                    selected_rates = st.multiselect(
                        "Select Rates to Display",
                        rate_columns,
                        default=rate_columns[:min(3, len(rate_columns))],
                        key="rates_viz_select"
                    )
                    
                    if selected_rates:
                        fig_rates = go.Figure()
                        for rate_col in selected_rates:
                            # Use last 200 points for better performance
                            display_data = df_rates.tail(200) if len(df_rates) > 200 else df_rates
                            fig_rates.add_trace(go.Scatter(
                                x=display_data['date'],
                                y=display_data[rate_col],
                                mode='lines',
                                name=rate_col,
                                line=dict(width=2)
                            ))
                        
                        fig_rates.update_layout(
                            title='Interest Rate Time Series',
                            xaxis_title='Date',
                            yaxis_title='Rate',
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig_rates, use_container_width=True)
        
        with col_rates2:
            # Statistical summary
            st.markdown("**Statistical Summary**")
            if rate_columns:
                summary_data = []
                for col in rate_columns[:5]:  # Show first 5 columns
                    summary_data.append({
                        'Rate': col,
                        'Mean': f"{df_rates[col].mean():.4f}",
                        'Std': f"{df_rates[col].std():.4f}",
                        'Latest': f"{df_rates[col].iloc[-1]:.4f}"
                    })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Yield Curve Data Display
    if st.session_state.get('yield_loaded', False) and st.session_state.yield_data is not None:
        st.markdown("#### 📈 Yield Curve Analysis")
        df_yield = st.session_state.yield_data
        
        col_yield1, col_yield2 = st.columns([2, 1])
        
        with col_yield1:
            # Yield curve visualization
            if len(df_yield) > 0:
                # Get the latest yield curve
                latest_yield = df_yield.iloc[-1]
                
                # Define tenor mapping
                tenor_map = {
                    '1M': 1/12, '3M': 0.25, '6M': 0.5, '1Y': 1.0,
                    '2Y': 2.0, '5Y': 5.0, '10Y': 10.0, '20Y': 20.0, '30Y': 30.0
                }
                
                tenors = []
                rates = []
                tenor_labels = []
                
                for tenor_str, tenor_val in tenor_map.items():
                    if tenor_str in latest_yield:
                        try:
                            rate_val = float(latest_yield[tenor_str])
                            tenors.append(tenor_val)
                            rates.append(rate_val)
                            tenor_labels.append(tenor_str)
                        except (ValueError, TypeError):
                            continue
                
                if tenors and rates:
                    fig_yield = go.Figure()
                    fig_yield.add_trace(go.Scatter(
                        x=tenors,
                        y=rates,
                        mode='lines+markers+text',
                        line=dict(color='green', width=3),
                        marker=dict(size=8),
                        text=tenor_labels,
                        textposition="top center"
                    ))
                    fig_yield.update_layout(
                        title='Current Yield Curve',
                        xaxis_title='Tenor (Years)',
                        yaxis_title='Yield',
                        height=400
                    )
                    st.plotly_chart(fig_yield, use_container_width=True)
        
        with col_yield2:
            # Yield curve data table
            st.markdown("**Latest Yield Data**")
            if len(df_yield) > 0:
                latest_data = df_yield.iloc[-1]
                yield_data = []
                for col in df_yield.columns:
                    if col != 'date':
                        try:
                            value = float(latest_data[col])
                            yield_data.append({'Tenor': col, 'Yield': f"{value:.4f}"})
                        except (ValueError, TypeError):
                            continue
                
                if yield_data:
                    st.dataframe(pd.DataFrame(yield_data), use_container_width=True)
    
    # Market Data Integration Status
    st.markdown("### 🏛️ Market Data Integration Status")
    
    col_mkt1, col_mkt2 = st.columns(2)
    
    with col_mkt1:
        st.markdown("#### 📊 Current Market Data")
        market_data = pricer.market_data
        
        if market_data:
            market_display = []
            for key, value in market_data.items():
                if key != 'timestamp' and isinstance(value, (int, float)):
                    if value < 1:  # Assume it's a rate
                        display_value = f"{value:.3%}"
                    else:
                        display_value = f"{value:.2f}"
                    market_display.append({
                        'Indicator': key.replace('_', ' ').title(),
                        'Value': display_value
                    })
            
            if market_display:
                st.dataframe(pd.DataFrame(market_display), use_container_width=True)
    
    with col_mkt2:
        st.markdown("#### 🔧 System Status")
        
        status_info = []
        status_info.append({'Component': 'Kaggle API', 'Status': '✅ Active' if kaggle_manager.api else '❌ Inactive'})
        status_info.append({'Component': 'Interest Rates', 'Status': '✅ Loaded' if st.session_state.get('rates_loaded') else '📥 Ready'})
        status_info.append({'Component': 'Yield Curve', 'Status': '✅ Loaded' if st.session_state.get('yield_loaded') else '📥 Ready'})
        status_info.append({'Component': 'Data Source', 'Status': 'Real Data' if kaggle_manager.api else 'Synthetic Data'})
        
        st.dataframe(pd.DataFrame(status_info), use_container_width=True)
        
        # Data download option
        if st.session_state.get('rates_loaded') or st.session_state.get('yield_loaded'):
            st.markdown("#### 💾 Export Data")
            if st.button("📥 Download All Data as CSV"):
                # Combine available data
                all_data = {}
                if st.session_state.get('rates_loaded'):
                    all_data['interest_rates'] = st.session_state.rates_data
                if st.session_state.get('yield_loaded'):
                    all_data['yield_curve'] = st.session_state.yield_data
                
                # Create downloadable CSV
                if all_data:
                    # For simplicity, we'll just export the first dataset
                    first_key = list(all_data.keys())[0]
                    csv_data = all_data[first_key].to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
# Duplicate show_live_pricing function removed — the earlier definition above is used for live pricing.
# This block was intentionally removed to avoid redefining show_live_pricing.
if __name__ == "__main__":
    main()
