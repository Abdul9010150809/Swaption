#!/bin/bash

# Price Matrix ML Service Startup Script
# This script handles the startup process for the ML service including
# model loading, health checks, and graceful shutdown

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="ml-service"
LOG_FILE="${SCRIPT_DIR}/logs/${SERVICE_NAME}.log"
PID_FILE="${SCRIPT_DIR}/logs/${SERVICE_NAME}.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            log_info "Stopping service (PID: $PID)"
            kill "$PID"
            wait "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
    exit 0
}

# Trap signals
trap cleanup SIGTERM SIGINT

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:${PORT:-8000}/health" > /dev/null 2>&1; then
            log_info "Health check passed"
            return 0
        fi

        log_info "Health check failed, attempt $attempt/$max_attempts"
        sleep 2
        ((attempt++))
    done

    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Model validation function
validate_models() {
    log_info "Validating models..."

    # Check if model files exist
    if [ ! -f "models/swaption_pricer.h5" ]; then
        log_error "Swaption pricer model not found"
        return 1
    fi

    if [ ! -f "models/feature_scaler.pkl" ]; then
        log_error "Feature scaler not found"
        return 1
    fi

    if [ ! -f "models/target_scaler.pkl" ]; then
        log_error "Target scaler not found"
        return 1
    fi

    log_info "All model files found"
    return 0
}

# Environment setup
setup_environment() {
    log_info "Setting up environment..."

    # Create log directory
    mkdir -p "${SCRIPT_DIR}/logs"

    # Set Python path
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

    # Set environment variables
    export SERVICE_NAME="$SERVICE_NAME"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"

    log_info "Environment setup complete"
}

# Main startup function
main() {
    log_info "Starting $SERVICE_NAME..."

    # Setup environment
    setup_environment

    # Validate models
    if ! validate_models; then
        log_error "Model validation failed"
        exit 1
    fi

    # Change to script directory
    cd "$SCRIPT_DIR"

    # Start the service
    log_info "Starting FastAPI server..."

    # Use gunicorn for production
    if [ "$ENVIRONMENT" = "production" ]; then
        log_info "Starting in production mode with gunicorn"

        gunicorn \
            --bind 0.0.0.0:${PORT:-8000} \
            --workers ${WORKERS:-4} \
            --worker-class uvicorn.workers.UvicornWorker \
            --pid "$PID_FILE" \
            --log-file "$LOG_FILE" \
            --log-level "${LOG_LEVEL:-INFO}" \
            --access-logfile "${SCRIPT_DIR}/logs/access.log" \
            --error-logfile "${SCRIPT_DIR}/logs/error.log" \
            app.main:app &

    else
        log_info "Starting in development mode with uvicorn"

        python -m uvicorn \
            app.main:app \
            --host 0.0.0.0 \
            --port ${PORT:-8000} \
            --reload \
            --log-level "${LOG_LEVEL:-INFO}" &
    fi

    # Get the PID
    SERVICE_PID=$!
    echo $SERVICE_PID > "$PID_FILE"
    log_info "Service started with PID: $SERVICE_PID"

    # Wait for service to be ready
    if health_check; then
        log_info "$SERVICE_NAME started successfully"

        # Keep the script running
        wait $SERVICE_PID
    else
        log_error "Service failed to start properly"
        cleanup
        exit 1
    fi
}

# Run main function
main "$@"