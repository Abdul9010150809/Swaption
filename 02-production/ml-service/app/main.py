from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quantum computing configuration
QUANTUM_CONFIG = {
    'enabled': os.getenv('QUANTUM_ENABLED', 'true').lower() == 'true',
    'api_key': os.getenv('QUANTUM_API_KEY', 'wPQOh--o2TjczKSr8xYZXZPudXBm4Ia6m__gdphs-5IR'),
    'provider': os.getenv('QUANTUM_PROVIDER', 'ibm'),
    'backend': os.getenv('QUANTUM_BACKEND', 'simulator'),
    'shots': int(os.getenv('QUANTUM_SHOTS', '1024')),
    'max_circuits': int(os.getenv('QUANTUM_MAX_CIRCUITS', '100')),
    'optimization_level': int(os.getenv('QUANTUM_OPTIMIZATION_LEVEL', '1')),
    'job_timeout': int(os.getenv('QUANTUM_JOB_TIMEOUT', '300')),
    'max_retries': int(os.getenv('QUANTUM_MAX_RETRIES', '3'))
}

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ML Service...")
    await load_models()
    logger.info("ML Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down ML Service...")

app = FastAPI(
    title="PriceMatrix ML Service",
    description="Machine Learning service for financial derivative pricing",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
option_model = None
swaption_model = None
feature_scaler = None
target_scaler = None

# Pydantic models for request/response
class OptionPricingRequest(BaseModel):
    spot_price: float = Field(..., gt=0, description="Current spot price")
    strike_price: float = Field(..., gt=0, description="Option strike price")
    time_to_expiry: float = Field(..., ge=0, description="Time to expiry in years")
    risk_free_rate: float = Field(..., ge=-0.1, le=0.2, description="Risk-free interest rate")
    volatility: float = Field(..., gt=0, le=5, description="Implied volatility")
    option_type: str = Field(..., pattern=r"^(call|put)$", description="Option type")
    dividend_yield: float = Field(0.0, ge=0, le=1, description="Dividend yield")

class SwaptionPricingRequest(BaseModel):
    swap_rate: float = Field(..., ge=0, le=0.2, description="Current swap rate")
    strike_rate: float = Field(..., ge=0, le=0.2, description="Strike swap rate")
    option_tenor: float = Field(..., gt=0, le=10, description="Time to option expiry")
    swap_tenor: float = Field(..., ge=0.5, le=50, description="Underlying swap tenor")
    volatility: float = Field(..., gt=0, le=2, description="Swaption volatility")

class BatchPricingRequest(BaseModel):
    instruments: List[Dict[str, Any]] = Field(..., min_items=1, max_items=50)

class PricingResponse(BaseModel):
    price: float
    confidence_interval: Optional[Dict[str, float]] = None
    model: str
    timestamp: str
    calculation_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    models_loaded: bool
    quantum_enabled: bool
    quantum_provider: str

class QuantumConfigResponse(BaseModel):
    enabled: bool
    provider: str
    backend: str
    shots: int
    max_circuits: int
    optimization_level: int
    api_key_configured: bool

# Model loading functions
async def load_models():
    """Load ML models and scalers on startup."""
    global option_model, swaption_model, feature_scaler, target_scaler

    try:
        # Load models (these would be trained and saved during development)
        # For now, we'll create placeholder models
        logger.info("Loading ML models...")

        # Option pricing model
        option_model = create_option_model()

        # Swaption pricing model
        swaption_model = create_swaption_model()

        # Feature and target scalers
        feature_scaler = joblib.load('models/feature_scaler.pkl') if os.path.exists('models/feature_scaler.pkl') else None
        target_scaler = joblib.load('models/target_scaler.pkl') if os.path.exists('models/target_scaler.pkl') else None

        logger.info("Models loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Continue without models - will fall back to analytical pricing

def create_option_model():
    """Create neural network model for option pricing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(8,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_swaption_model():
    """Create neural network model for swaption pricing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Utility functions
def preprocess_option_features(request: OptionPricingRequest) -> np.ndarray:
    """Preprocess option pricing features."""
    features = np.array([
        request.spot_price,
        request.strike_price,
        request.time_to_expiry,
        request.risk_free_rate,
        request.volatility,
        request.dividend_yield,
        request.spot_price / request.strike_price,  # moneyness
        request.volatility * np.sqrt(request.time_to_expiry)  # vol-time
    ]).reshape(1, -1)

    if feature_scaler:
        features = feature_scaler.transform(features)

    return features

def preprocess_swaption_features(request: SwaptionPricingRequest) -> np.ndarray:
    """Preprocess swaption pricing features."""
    features = np.array([
        request.swap_rate,
        request.strike_rate,
        request.option_tenor,
        request.swap_tenor,
        request.volatility
    ]).reshape(1, -1)

    if feature_scaler:
        features = feature_scaler.transform(features)

    return features

def postprocess_price(price: float) -> float:
    """Postprocess predicted price."""
    if target_scaler:
        price = target_scaler.inverse_transform([[price]])[0][0]
    return max(0, price)  # Ensure non-negative price

# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "PriceMatrix ML Service", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0",
        models_loaded=option_model is not None and swaption_model is not None,
        quantum_enabled=QUANTUM_CONFIG['enabled'],
        quantum_provider=QUANTUM_CONFIG['provider']
    )

@app.get("/quantum/config", response_model=QuantumConfigResponse)
async def get_quantum_config():
    """Get quantum computing configuration."""
    return QuantumConfigResponse(
        enabled=QUANTUM_CONFIG['enabled'],
        provider=QUANTUM_CONFIG['provider'],
        backend=QUANTUM_CONFIG['backend'],
        shots=QUANTUM_CONFIG['shots'],
        max_circuits=QUANTUM_CONFIG['max_circuits'],
        optimization_level=QUANTUM_CONFIG['optimization_level'],
        api_key_configured=bool(QUANTUM_CONFIG['api_key'] and len(QUANTUM_CONFIG['api_key']) > 10)
    )

@app.get("/quantum/status")
async def get_quantum_status():
    """Get quantum service status and connectivity."""
    if not QUANTUM_CONFIG['enabled']:
        return {
            "status": "disabled",
            "message": "Quantum computing is disabled",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    # In a real implementation, this would test connectivity to the quantum service
    # For now, we'll just check if the API key is configured
    if not QUANTUM_CONFIG['api_key'] or len(QUANTUM_CONFIG['api_key']) < 10:
        return {
            "status": "error",
            "message": "Quantum API key not properly configured",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    return {
        "status": "ready",
        "message": f"Quantum service ready with {QUANTUM_CONFIG['provider']} provider",
        "provider": QUANTUM_CONFIG['provider'],
        "backend": QUANTUM_CONFIG['backend'],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/pricing/options", response_model=PricingResponse)
async def price_option(request: OptionPricingRequest, background_tasks: BackgroundTasks):
    """Price European option using ML model."""
    start_time = datetime.utcnow()

    try:
        if option_model is None:
            raise HTTPException(status_code=503, detail="ML model not available")

        # Preprocess features
        features = preprocess_option_features(request)

        # Make prediction
        prediction = option_model.predict(features, verbose=0)[0][0]

        # Postprocess price
        price = postprocess_price(prediction)

        # Calculate confidence interval (simplified)
        confidence_interval = {
            "lower": max(0, price * 0.95),
            "upper": price * 1.05
        }

        calculation_time = (datetime.utcnow() - start_time).total_seconds()

        # Log pricing request
        logger.info(f"Option priced: {request.option_type}, price=${price:.4f}")

        return PricingResponse(
            price=round(price, 4),
            confidence_interval=confidence_interval,
            model="ml_neural_network",
            timestamp=datetime.utcnow().isoformat() + "Z",
            calculation_time=round(calculation_time, 4)
        )

    except Exception as e:
        logger.error(f"Option pricing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pricing calculation failed: {str(e)}")

@app.post("/pricing/swaptions", response_model=PricingResponse)
async def price_swaption(request: SwaptionPricingRequest, background_tasks: BackgroundTasks):
    """Price European swaption using ML model or quantum engines."""
    start_time = datetime.utcnow()

    try:
        # Check if quantum pricing is requested
        if hasattr(request, 'model') and request.model in ['quantum_monte_carlo', 'quantum_amplitude_estimation', 'quantum_hybrid']:
            # Import quantum engines
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '../../../01-research-development'))

            from src.pricing.quantum_pricing import QuantumMonteCarloEngine, QuantumAmplitudeEstimationEngine, HybridPricingEngine

            config = {
                'backend': 'simulator',
                'shots': 1024,
                'n_qubits': 8,
                'precision_qubits': 6
            }

            if request.model == 'quantum_monte_carlo':
                engine = QuantumMonteCarloEngine(config)
                price = engine.price_option(
                    spot=request.swap_rate * 100,  # Convert to price-like scale
                    strike=request.strike_rate * 100,
                    time_to_expiry=request.option_tenor,
                    risk_free_rate=0.03,  # Default risk-free rate
                    volatility=request.volatility,
                    option_type='call'
                )
                model_name = "quantum_monte_carlo"
            elif request.model == 'quantum_amplitude_estimation':
                engine = QuantumAmplitudeEstimationEngine(config)
                price = engine.price_option(
                    spot=request.swap_rate * 100,
                    strike=request.strike_rate * 100,
                    time_to_expiry=request.option_tenor,
                    risk_free_rate=0.03,
                    volatility=request.volatility,
                    option_type='call'
                )
                model_name = "quantum_amplitude_estimation"
            else:  # quantum_hybrid
                engine = HybridPricingEngine(config)
                price = engine.price_swaption_quantum(
                    notional=1000000,  # Default notional
                    strike_rate=request.strike_rate,
                    time_to_expiry=request.option_tenor,
                    risk_free_rate=0.03,
                    volatility=request.volatility,
                    swap_tenor=request.swap_tenor
                )
                model_name = "quantum_hybrid"

            # Calculate confidence interval (simplified)
            confidence_interval = {
                "lower": max(0, price * 0.95),
                "upper": price * 1.05
            }

            calculation_time = (datetime.utcnow() - start_time).total_seconds()

            logger.info(f"Quantum swaption priced: model={model_name}, price=${price:.6f}")

            return PricingResponse(
                price=round(price, 6),
                confidence_interval=confidence_interval,
                model=model_name,
                timestamp=datetime.utcnow().isoformat() + "Z",
                calculation_time=round(calculation_time, 4)
            )

        # Fallback to ML model
        if swaption_model is None:
            raise HTTPException(status_code=503, detail="ML model not available")

        # Preprocess features
        features = preprocess_swaption_features(request)

        # Make prediction
        prediction = swaption_model.predict(features, verbose=0)[0][0]

        # Postprocess price
        price = postprocess_price(prediction)

        # Calculate confidence interval (simplified)
        confidence_interval = {
            "lower": max(0, price * 0.9),
            "upper": price * 1.1
        }

        calculation_time = (datetime.utcnow() - start_time).total_seconds()

        # Log pricing request
        logger.info(f"ML swaption priced: tenor={request.swap_tenor}, price=${price:.6f}")

        return PricingResponse(
            price=round(price, 6),
            confidence_interval=confidence_interval,
            model="ml_neural_network",
            timestamp=datetime.utcnow().isoformat() + "Z",
            calculation_time=round(calculation_time, 4)
        )

    except Exception as e:
        logger.error(f"Swaption pricing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pricing calculation failed: {str(e)}")

@app.post("/pricing/batch")
async def price_batch(request: BatchPricingRequest, background_tasks: BackgroundTasks):
    """Batch pricing for multiple instruments."""
    start_time = datetime.utcnow()

    try:
        results = []

        for i, instrument in enumerate(request.instruments):
            try:
                if instrument.get("type") == "option":
                    # Convert dict to OptionPricingRequest
                    opt_request = OptionPricingRequest(**instrument.get("parameters", {}))
                    result = await price_option(opt_request, background_tasks)
                    results.append({
                        "instrument_id": instrument.get("id", f"instrument_{i}"),
                        "type": "option",
                        "result": result.dict(),
                        "success": True
                    })

                elif instrument.get("type") == "swaption":
                    # Convert dict to SwaptionPricingRequest
                    swap_request = SwaptionPricingRequest(**instrument.get("parameters", {}))
                    result = await price_swaption(swap_request, background_tasks)
                    results.append({
                        "instrument_id": instrument.get("id", f"instrument_{i}"),
                        "type": "swaption",
                        "result": result.dict(),
                        "success": True
                    })

                else:
                    results.append({
                        "instrument_id": instrument.get("id", f"instrument_{i}"),
                        "error": f"Unsupported instrument type: {instrument.get('type')}",
                        "success": False
                    })

            except Exception as e:
                results.append({
                    "instrument_id": instrument.get("id", f"instrument_{i}"),
                    "error": str(e),
                    "success": False
                })

        calculation_time = (datetime.utcnow() - start_time).total_seconds()

        logger.info(f"Batch pricing completed: {len(results)} instruments")

        return {
            "results": results,
            "summary": {
                "total": len(results),
                "successful": sum(1 for r in results if r["success"]),
                "failed": sum(1 for r in results if not r["success"])
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_calculation_time": round(calculation_time, 4)
        }

    except Exception as e:
        logger.error(f"Batch pricing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch pricing failed: {str(e)}")

@app.post("/models/train")
async def train_models(background_tasks: BackgroundTasks):
    """Trigger model training (admin endpoint)."""
    # This would typically be called by a training pipeline
    background_tasks.add_task(train_ml_models)
    return {"message": "Model training started", "timestamp": datetime.utcnow().isoformat() + "Z"}

async def train_ml_models():
    """Background task to train ML models."""
    try:
        logger.info("Starting model training...")

        # This would implement the actual training logic
        # For now, just log that training would happen
        logger.info("Model training completed (placeholder)")

    except Exception as e:
        logger.error(f"Model training failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
