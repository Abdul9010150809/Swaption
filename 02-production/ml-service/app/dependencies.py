"""
Dependencies and utilities for ML service
"""

from fastapi import Request, HTTPException, Depends
from typing import Optional
import time
import logging
import uuid

logger = logging.getLogger(__name__)

# Request timing middleware
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Request ID middleware
async def add_request_id(request: Request, call_next):
    """Add request ID to request state and response headers."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Authentication dependency
async def get_current_user(request: Request) -> Optional[str]:
    """Extract user ID from request (JWT token or API key)."""
    # Check for JWT token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        # In production, validate JWT token here
        # For now, return a mock user ID
        return f"user_{hash(token) % 1000}"

    # Check for API key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        # In production, validate API key against database
        # For now, accept any non-empty key
        if len(api_key) >= 10:
            return f"apikey_{hash(api_key) % 1000}"

    return None

# Rate limiting dependency (simplified in-memory version)
class InMemoryRateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self):
        self.requests = {}

    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        window_start = now - window

        # Clean old requests
        if key in self.requests:
            self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]

        # Check limit
        if key not in self.requests:
            self.requests[key] = []

        if len(self.requests[key]) < limit:
            self.requests[key].append(now)
            return True

        return False

# Global rate limiter instance
rate_limiter = InMemoryRateLimiter()

async def check_rate_limit(
    request: Request,
    user: Optional[str] = Depends(get_current_user)
) -> None:
    """Check rate limit for the request."""
    # Use user ID or IP address as key
    key = user or request.client.host

    # Different limits for authenticated vs anonymous users
    if user:
        limit = 100  # requests per minute for authenticated users
    else:
        limit = 10   # requests per minute for anonymous users

    window = 60  # 1 minute window

    if not rate_limiter.is_allowed(key, limit, window):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(window)}
        )

# Request logging dependency
async def log_request(
    request: Request,
    user: Optional[str] = Depends(get_current_user)
) -> None:
    """Log incoming requests."""
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "user_id": user,
            "ip": request.client.host,
            "user_agent": request.headers.get("User-Agent"),
            "request_id": getattr(request.state, 'request_id', None)
        }
    )

# Model availability dependency
def require_model(model_name: str):
    """Dependency that ensures a specific model is available."""
    def dependency():
        # In production, check if model is loaded and ready
        # For now, just return True
        return True

        # Example production implementation:
        # if model_name == "option_model" and option_model is None:
        #     raise HTTPException(status_code=503, detail="Option pricing model not available")
        # if model_name == "swaption_model" and swaption_model is None:
        #     raise HTTPException(status_code=503, detail="Swaption pricing model not available")
        # return True

    return dependency

# Input validation helpers
def validate_option_parameters(spot: float, strike: float, time: float, rate: float, vol: float):
    """Validate option pricing parameters."""
    if spot <= 0:
        raise HTTPException(status_code=400, detail="Spot price must be positive")
    if strike <= 0:
        raise HTTPException(status_code=400, detail="Strike price must be positive")
    if time < 0:
        raise HTTPException(status_code=400, detail="Time to expiry cannot be negative")
    if rate < -0.1 or rate > 0.2:
        raise HTTPException(status_code=400, detail="Risk-free rate out of reasonable range")
    if vol <= 0 or vol > 5:
        raise HTTPException(status_code=400, detail="Volatility out of reasonable range")

def validate_swaption_parameters(swap_rate: float, strike_rate: float, option_tenor: float, swap_tenor: float, vol: float):
    """Validate swaption pricing parameters."""
    if swap_rate < 0 or swap_rate > 0.2:
        raise HTTPException(status_code=400, detail="Swap rate out of reasonable range")
    if strike_rate < 0 or strike_rate > 0.2:
        raise HTTPException(status_code=400, detail="Strike rate out of reasonable range")
    if option_tenor <= 0 or option_tenor > 10:
        raise HTTPException(status_code=400, detail="Option tenor out of reasonable range")
    if swap_tenor < 0.5 or swap_tenor > 50:
        raise HTTPException(status_code=400, detail="Swap tenor out of reasonable range")
    if vol <= 0 or vol > 2:
        raise HTTPException(status_code=400, detail="Volatility out of reasonable range")

# Error handling
class PricingError(Exception):
    """Custom exception for pricing errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

# Global exception handler would be defined in main.py
def handle_pricing_error(error: PricingError):
    """Handle pricing-specific errors."""
    logger.error(f"Pricing error: {error.message}")
    raise HTTPException(status_code=error.status_code, detail=error.message)

# Caching utilities
class SimpleCache:
    """Simple in-memory cache."""

    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, key: str):
        """Get value from cache."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value, ttl: int = None):
        """Set value in cache."""
        ttl = ttl or self.ttl
        self.cache[key] = (value, time.time())

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()

# Global cache instance
cache = SimpleCache()

# Performance monitoring
class PerformanceMonitor:
    """Monitor API performance."""

    def __init__(self):
        self.metrics = {}

    def record_timing(self, endpoint: str, duration: float):
        """Record endpoint timing."""
        if endpoint not in self.metrics:
            self.metrics[endpoint] = []
        self.metrics[endpoint].append(duration)

        # Keep only last 100 measurements
        if len(self.metrics[endpoint]) > 100:
            self.metrics[endpoint] = self.metrics[endpoint][-100:]

    def get_average_timing(self, endpoint: str) -> Optional[float]:
        """Get average timing for endpoint."""
        if endpoint in self.metrics and self.metrics[endpoint]:
            return sum(self.metrics[endpoint]) / len(self.metrics[endpoint])
        return None

# Global performance monitor
performance_monitor = PerformanceMonitor()

async def monitor_performance(request: Request, call_next):
    """Middleware to monitor API performance."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    performance_monitor.record_timing(request.url.path, duration)

    # Add performance header
    response.headers["X-Response-Time"] = str(duration)
    return response