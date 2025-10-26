"""
Pydantic models for ML service API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

class PricingModel(str, Enum):
    BLACK_SCHOLES = "black_scholes"
    MONTE_CARLO = "monte_carlo"
    ML = "ml"

class InstrumentType(str, Enum):
    OPTION = "option"
    SWAPTION = "swaption"
    BOND = "bond"

# Base pricing request
class BasePricingRequest(BaseModel):
    model: PricingModel = Field(PricingModel.BLACK_SCHOLES, description="Pricing model to use")

# Option pricing request
class OptionPricingRequest(BasePricingRequest):
    spot_price: float = Field(..., gt=0, description="Current spot price of underlying asset")
    strike_price: float = Field(..., gt=0, description="Option strike price")
    time_to_expiry: float = Field(..., ge=0, le=10, description="Time to expiry in years")
    risk_free_rate: float = Field(..., ge=-0.1, le=0.2, description="Risk-free interest rate (annual)")
    volatility: float = Field(..., gt=0, le=5, description="Implied volatility (annual)")
    option_type: OptionType = Field(..., description="Type of option")
    dividend_yield: float = Field(0.0, ge=0, le=1, description="Dividend yield (annual)")

    @validator('strike_price')
    def strike_must_be_reasonable(cls, v, values):
        if 'spot_price' in values and abs(v - values['spot_price']) / values['spot_price'] > 5:
            raise ValueError('Strike price seems unreasonable compared to spot price')
        return v

# Swaption pricing request
class SwaptionPricingRequest(BasePricingRequest):
    swap_rate: float = Field(..., ge=0, le=0.2, description="Current swap rate")
    strike_rate: float = Field(..., ge=0, le=0.2, description="Strike swap rate")
    option_tenor: float = Field(..., gt=0, le=10, description="Time to option expiry in years")
    swap_tenor: float = Field(..., ge=0.5, le=50, description="Underlying swap tenor in years")
    volatility: float = Field(..., gt=0, le=2, description="Swaption volatility")

# Bond pricing request
class BondPricingRequest(BaseModel):
    face_value: float = Field(..., gt=0, description="Bond face value")
    coupon_rate: float = Field(..., ge=0, le=1, description="Annual coupon rate")
    maturity: float = Field(..., gt=0, le=100, description="Time to maturity in years")
    yield_to_maturity: float = Field(..., ge=-0.1, le=0.2, description="Yield to maturity")
    frequency: int = Field(2, ge=1, le=12, description="Coupon payment frequency per year")

# Risk metrics request
class RiskMetricsRequest(BaseModel):
    returns: List[float] = Field(..., min_items=30, description="Historical returns series")
    confidence_level: float = Field(0.95, gt=0, lt=1, description="Confidence level for VaR")
    time_horizon: int = Field(1, ge=1, le=252, description="Time horizon in days")
    method: str = Field("parametric", regex="^(historical|parametric|monte_carlo)$")

# Batch pricing request
class BatchPricingRequest(BaseModel):
    instruments: List[Dict[str, Any]] = Field(..., min_items=1, max_items=50, description="List of instruments to price")

    @validator('instruments')
    def validate_instruments(cls, v):
        for i, instrument in enumerate(v):
            if 'type' not in instrument:
                raise ValueError(f"Instrument {i}: missing 'type' field")
            if 'parameters' not in instrument:
                raise ValueError(f"Instrument {i}: missing 'parameters' field")
            if instrument['type'] not in [e.value for e in InstrumentType]:
                raise ValueError(f"Instrument {i}: invalid type '{instrument['type']}'")
        return v

# Greeks response
class Greeks(BaseModel):
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

# Confidence interval
class ConfidenceInterval(BaseModel):
    lower: float
    upper: float

# Base pricing response
class BasePricingResponse(BaseModel):
    price: float = Field(..., description="Calculated price")
    greeks: Optional[Greeks] = None
    confidence_interval: Optional[ConfidenceInterval] = None
    model: str = Field(..., description="Model used for pricing")
    timestamp: str = Field(..., description="Calculation timestamp")
    calculation_time: float = Field(..., description="Calculation time in seconds")

# Option pricing response
class OptionPricingResponse(BasePricingResponse):
    option_type: OptionType
    moneyness: float = Field(..., description="Moneyness (S/K)")
    intrinsic_value: float = Field(..., description="Intrinsic value")
    time_value: float = Field(..., description="Time value")

# Swaption pricing response
class SwaptionPricingResponse(BasePricingResponse):
    annuity_factor: Optional[float] = None
    forward_rate: float = Field(..., description="Forward swap rate")

# Bond pricing response
class BondPricingResponse(BasePricingResponse):
    duration: float = Field(..., description="Macaulay duration")
    convexity: float = Field(..., description="Bond convexity")
    current_yield: float = Field(..., description="Current yield")

# Risk metrics response
class RiskMetricsResponse(BaseModel):
    VaR: float = Field(..., description="Value at Risk")
    CVaR: float = Field(..., description="Conditional VaR")
    confidence_level: float
    time_horizon: int
    method: str
    timestamp: str

# Batch pricing response
class BatchPricingResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Individual pricing results")
    summary: Dict[str, int] = Field(..., description="Summary statistics")
    timestamp: str
    total_calculation_time: float

# Health check response
class HealthResponse(BaseModel):
    status: str = Field(..., regex="^(healthy|unhealthy)$")
    timestamp: str
    version: str
    models_loaded: bool
    uptime: Optional[float] = None
    memory_usage: Optional[Dict[str, float]] = None

# Error response
class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Market data models
class MarketDataPoint(BaseModel):
    symbol: str
    price: float
    timestamp: str
    volume: Optional[float] = None

class YieldCurveData(BaseModel):
    tenors: List[float]
    rates: List[float]
    timestamp: str
    interpolation_method: str = "linear"

class VolatilitySurfaceData(BaseModel):
    expiries: List[float]
    strikes: List[float]
    surface: List[List[float]]
    timestamp: str

# Model training request
class ModelTrainingRequest(BaseModel):
    model_type: str = Field(..., regex="^(option|swaption|ensemble)$")
    hyperparameters: Optional[Dict[str, Any]] = None
    training_data_path: Optional[str] = None
    validation_split: float = Field(0.2, gt=0, lt=1)
    epochs: int = Field(100, gt=0)

# Model training response
class ModelTrainingResponse(BaseModel):
    model_id: str
    model_type: str
    training_started: str
    estimated_completion: Optional[str] = None
    status: str = "running"