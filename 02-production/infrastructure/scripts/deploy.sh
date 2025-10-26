#!/bin/bash

# Deployment Script for Price Matrix
# This script handles the deployment of the Price Matrix system to various environments

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENVIRONMENT="${ENVIRONMENT:-staging}"
DOCKER_COMPOSE_FILE="${SCRIPT_DIR}/../docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

log_step() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] STEP: $1${NC}"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_step "Running pre-deployment checks"

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi

    # Check if docker-compose.yml exists
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
        exit 1
    fi

    # Check available disk space (minimum 5GB)
    local available_space=$(df / | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 5242880 ]; then  # 5GB in KB
        log_error "Insufficient disk space. Need at least 5GB available."
        exit 1
    fi

    log_info "Pre-deployment checks passed"
}

# Build Docker images
build_images() {
    log_step "Building Docker images for $ENVIRONMENT"

    cd "$PROJECT_ROOT"

    # Set build arguments based on environment
    local build_args=""
    if [ "$ENVIRONMENT" = "production" ]; then
        build_args="--build-arg NODE_ENV=production"
    fi

    # Build images
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" build $build_args
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" build $build_args
    fi

    log_info "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log_step "Deploying services to $ENVIRONMENT"

    cd "$PROJECT_ROOT"

    # Set environment variables
    export ENVIRONMENT="$ENVIRONMENT"

    # Start services
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" up -d
    fi

    log_info "Services deployed successfully"
}

# Health checks
health_checks() {
    log_step "Running health checks"

    local max_attempts=30
    local attempt=1

    # Services to check
    local services=("api-gateway" "ml-service" "frontend")

    for service in "${services[@]}"; do
        log_info "Checking health of $service"

        local attempt=1
        while [ $attempt -le $max_attempts ]; do
            if [ "$service" = "frontend" ]; then
                # Frontend health check
                if curl -f -s "http://localhost:3001/health" > /dev/null 2>&1; then
                    log_info "$service is healthy"
                    break
                fi
            elif [ "$service" = "api-gateway" ]; then
                # API Gateway health check
                if curl -f -s "http://localhost:3000/health" > /dev/null 2>&1; then
                    log_info "$service is healthy"
                    break
                fi
            elif [ "$service" = "ml-service" ]; then
                # ML Service health check
                if curl -f -s "http://localhost:8000/health" > /dev/null 2>&1; then
                    log_info "$service is healthy"
                    break
                fi
            fi

            if [ $attempt -eq $max_attempts ]; then
                log_error "$service failed health check after $max_attempts attempts"
                return 1
            fi

            log_info "Health check failed for $service, attempt $attempt/$max_attempts"
            sleep 5
            ((attempt++))
        done
    done

    log_info "All health checks passed"
}

# Run database migrations (if applicable)
run_migrations() {
    log_step "Running database migrations"

    # This would be specific to your database setup
    # For now, just log that migrations would run here
    log_info "Database migrations completed (placeholder)"
}

# Backup current deployment
backup_current() {
    log_step "Creating backup of current deployment"

    # This would create backups of current models, configurations, etc.
    log_info "Backup completed (placeholder)"
}

# Rollback function
rollback() {
    log_error "Deployment failed, initiating rollback"

    cd "$PROJECT_ROOT"

    # Stop services
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" down
    fi

    # Restore from backup if available
    log_info "Rollback completed"
}

# Post-deployment tasks
post_deployment() {
    log_step "Running post-deployment tasks"

    # Run any smoke tests
    log_info "Running smoke tests"

    # Test basic functionality
    if curl -f -s "http://localhost:3000/health" > /dev/null 2>&1; then
        log_info "API Gateway smoke test passed"
    else
        log_error "API Gateway smoke test failed"
        return 1
    fi

    # Log deployment information
    cat > "${SCRIPT_DIR}/../logs/deployment_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).log" << EOF
Deployment completed successfully
Environment: $ENVIRONMENT
Timestamp: $(date)
Services deployed: api-gateway, ml-service, frontend
Health checks: PASSED
EOF

    log_info "Post-deployment tasks completed"
}

# Main deployment function
main() {
    log_info "Starting deployment to $ENVIRONMENT environment"

    # Error handling
    trap rollback ERR

    # Run deployment steps
    pre_deployment_checks
    backup_current
    build_images
    run_migrations
    deploy_services
    health_checks
    post_deployment

    log_info "Deployment to $ENVIRONMENT completed successfully!"

    # Print deployment summary
    echo
    echo "========================================"
    echo "DEPLOYMENT SUMMARY"
    echo "========================================"
    echo "Environment: $ENVIRONMENT"
    echo "Completed: $(date)"
    echo "Services:"
    echo "  - API Gateway: http://localhost:3000"
    echo "  - Frontend: http://localhost:3001"
    echo "  - ML Service: http://localhost:8000"
    echo "========================================"
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Deploy Price Matrix to specified environment"
    echo
    echo "Options:"
    echo "  -e, --environment ENV    Deployment environment (staging|production) [default: staging]"
    echo "  -h, --help              Show this help message"
    echo
    echo "Examples:"
    echo "  $0 -e production    # Deploy to production"
    echo "  $0                  # Deploy to staging (default)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
    exit 1
fi

# Run main function
main "$@"