"""
Database schemas and data models for ML service
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ModelStatus(str, Enum):
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class ModelType(str, Enum):
    OPTION_PRICING = "option_pricing"
    SWAPTION_PRICING = "swaption_pricing"
    ENSEMBLE = "ensemble"

class ModelMetadata(BaseModel):
    """Metadata for trained models."""
    id: str
    name: str
    type: ModelType
    version: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    accuracy_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_info: Dict[str, Any]
    feature_names: List[str]
    target_name: str
    framework: str  # e.g., "tensorflow", "sklearn"
    framework_version: str

class PricingRequestLog(BaseModel):
    """Log of pricing requests for analytics."""
    id: str
    timestamp: datetime
    instrument_type: str
    model_used: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    calculation_time: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None

class ModelPerformanceMetrics(BaseModel):
    """Performance metrics for models."""
    model_id: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    dataset_type: str  # "training", "validation", "test"
    additional_info: Optional[Dict[str, Any]] = None

class ABTestResult(BaseModel):
    """Results from A/B testing different models."""
    id: str
    experiment_name: str
    model_a: str
    model_b: str
    start_date: datetime
    end_date: datetime
    metric_name: str
    model_a_performance: float
    model_b_performance: float
    winner: str
    confidence_level: float
    sample_size: int

class DataQualityMetrics(BaseModel):
    """Data quality metrics for input validation."""
    timestamp: datetime
    feature_name: str
    metric_type: str  # "missing_rate", "outlier_rate", "distribution_shift"
    metric_value: float
    threshold: float
    is_alert: bool
    details: Optional[Dict[str, Any]] = None

class SystemHealthMetrics(BaseModel):
    """System health and performance metrics."""
    timestamp: datetime
    service_name: str
    metric_name: str
    metric_value: float
    unit: str
    tags: Optional[Dict[str, str]] = None

# Training pipeline schemas
class TrainingJob(BaseModel):
    """Training job configuration."""
    id: str
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    training_data_path: str
    validation_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    logs: List[str] = []
    error_message: Optional[str] = None

class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering pipeline."""
    id: str
    name: str
    description: str
    transformations: List[Dict[str, Any]]
    input_features: List[str]
    output_features: List[str]
    created_at: datetime
    updated_at: datetime

class ModelValidationResult(BaseModel):
    """Results from model validation."""
    model_id: str
    validation_type: str  # "cross_validation", "backtesting", "stress_testing"
    timestamp: datetime
    metrics: Dict[str, float]
    passed: bool
    details: Dict[str, Any]
    recommendations: List[str]

# API response schemas
class APIResponse(BaseModel):
    """Base API response."""
    success: bool
    message: str
    timestamp: str
    request_id: Optional[str] = None

class PricingAPIResponse(APIResponse):
    """Pricing API response."""
    data: Dict[str, Any]

class BatchPricingAPIResponse(APIResponse):
    """Batch pricing API response."""
    data: Dict[str, Any]
    summary: Dict[str, int]

class ModelManagementAPIResponse(APIResponse):
    """Model management API response."""
    data: Dict[str, Any]

# Configuration schemas
class ServiceConfig(BaseModel):
    """Service configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "INFO"
    model_cache_size: int = 10
    max_batch_size: int = 50
    timeout_seconds: int = 30

class ModelConfig(BaseModel):
    """Model configuration."""
    type: ModelType
    framework: str
    hyperparameters: Dict[str, Any]
    feature_engineering: FeatureEngineeringConfig
    validation_config: Dict[str, Any]

class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = True
    ttl_seconds: int = 3600
    max_size_mb: int = 100
    redis_url: Optional[str] = None

class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    enabled: bool = True
    metrics_interval_seconds: int = 60
    health_check_interval_seconds: int = 30
    alert_thresholds: Dict[str, float] = {
        "prediction_time_p95": 5.0,
        "error_rate": 0.05,
        "memory_usage": 0.8
    }

# Analytics schemas
class UsageAnalytics(BaseModel):
    """Usage analytics data."""
    date: str
    endpoint: str
    request_count: int
    avg_response_time: float
    error_count: int
    unique_users: int

class ModelAnalytics(BaseModel):
    """Model performance analytics."""
    model_id: str
    date: str
    predictions_count: int
    avg_prediction_time: float
    accuracy_metrics: Dict[str, float]
    drift_detected: bool

class BusinessMetrics(BaseModel):
    """Business-level metrics."""
    date: str
    total_pricing_requests: int
    unique_instruments_priced: int
    avg_deal_size: float
    pricing_model_distribution: Dict[str, int]
    user_satisfaction_score: Optional[float] = None