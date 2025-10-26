#!/bin/bash

# Health Check Script for Price Matrix
# This script performs comprehensive health checks on all system components

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/../logs/health_check_$(date +%Y%m%d).log"
TIMEOUT=10
RETRIES=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Health check results
declare -A RESULTS
OVERALL_STATUS="HEALTHY"

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

log_step() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] STEP: $1${NC}"
}

# Generic HTTP health check
check_http_service() {
    local name="$1"
    local url="$2"
    local expected_status="${3:-200}"

    log_step "Checking $name at $url"

    local attempt=1
    while [ $attempt -le $RETRIES ]; do
        if curl -f -s --max-time $TIMEOUT -o /dev/null -w "%{http_code}" "$url" | grep -q "^$expected_status$"; then
            RESULTS["$name"]="HEALTHY"
            log_info "$name is healthy"
            return 0
        fi

        log_warn "$name health check failed (attempt $attempt/$RETRIES)"
        sleep 2
        ((attempt++))
    done

    RESULTS["$name"]="UNHEALTHY"
    OVERALL_STATUS="UNHEALTHY"
    log_error "$name is unhealthy"
    return 1
}

# Check Docker containers
check_docker_containers() {
    log_step "Checking Docker containers"

    if ! command -v docker &> /dev/null; then
        log_warn "Docker not available, skipping container checks"
        return 0
    fi

    local containers=("price-matrix-api-gateway" "price-matrix-ml-service" "price-matrix-frontend")

    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^${container}$"; then
            # Check container health
            local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "unknown")

            if [ "$health_status" = "healthy" ]; then
                RESULTS["container_$container"]="HEALTHY"
                log_info "Container $container is healthy"
            else
                RESULTS["container_$container"]="UNHEALTHY"
                OVERALL_STATUS="UNHEALTHY"
                log_error "Container $container is unhealthy (status: $health_status)"
            fi
        else
            RESULTS["container_$container"]="NOT_RUNNING"
            OVERALL_STATUS="UNHEALTHY"
            log_error "Container $container is not running"
        fi
    done
}

# Check system resources
check_system_resources() {
    log_step "Checking system resources"

    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    if (( $(echo "$cpu_usage < 90" | bc -l) )); then
        RESULTS["cpu_usage"]="HEALTHY"
        log_info "CPU usage: ${cpu_usage}%"
    else
        RESULTS["cpu_usage"]="WARNING"
        log_warn "High CPU usage: ${cpu_usage}%"
    fi

    # Memory usage
    local mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ "$mem_usage" -lt 90 ]; then
        RESULTS["memory_usage"]="HEALTHY"
        log_info "Memory usage: ${mem_usage}%"
    else
        RESULTS["memory_usage"]="WARNING"
        OVERALL_STATUS="WARNING"
        log_warn "High memory usage: ${mem_usage}%"
    fi

    # Disk usage
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 90 ]; then
        RESULTS["disk_usage"]="HEALTHY"
        log_info "Disk usage: ${disk_usage}%"
    else
        RESULTS["disk_usage"]="WARNING"
        OVERALL_STATUS="WARNING"
        log_warn "High disk usage: ${disk_usage}%"
    fi
}

# Check database connectivity (Redis)
check_database() {
    log_step "Checking database connectivity"

    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            RESULTS["redis"]="HEALTHY"
            log_info "Redis is healthy"
        else
            RESULTS["redis"]="UNHEALTHY"
            OVERALL_STATUS="UNHEALTHY"
            log_error "Redis is unhealthy"
        fi
    else
        log_warn "redis-cli not available, skipping Redis check"
        RESULTS["redis"]="UNKNOWN"
    fi
}

# Check model files
check_models() {
    log_step "Checking model files"

    local model_dir="${SCRIPT_DIR}/../../ml-service/models"

    if [ -f "$model_dir/swaption_pricer.h5" ]; then
        RESULTS["swaption_model"]="HEALTHY"
        log_info "Swaption pricing model exists"
    else
        RESULTS["swaption_model"]="UNHEALTHY"
        OVERALL_STATUS="UNHEALTHY"
        log_error "Swaption pricing model not found"
    fi

    if [ -f "$model_dir/feature_scaler.pkl" ]; then
        RESULTS["feature_scaler"]="HEALTHY"
        log_info "Feature scaler exists"
    else
        RESULTS["feature_scaler"]="WARNING"
        log_warn "Feature scaler not found"
    fi

    if [ -f "$model_dir/target_scaler.pkl" ]; then
        RESULTS["target_scaler"]="HEALTHY"
        log_info "Target scaler exists"
    else
        RESULTS["target_scaler"]="WARNING"
        log_warn "Target scaler not found"
    fi
}

# Check API endpoints functionality
check_api_endpoints() {
    log_step "Checking API endpoints"

    local base_url="http://localhost:3000"

    # Basic health check
    check_http_service "api_health" "$base_url/health"

    # Test pricing endpoint with sample data
    local pricing_response=$(curl -s -X POST "$base_url/api/pricing/options" \
        -H "Content-Type: application/json" \
        -d '{
            "spot_price": 100.0,
            "strike_price": 105.0,
            "time_to_expiry": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.20,
            "option_type": "call"
        }' 2>/dev/null)

    if echo "$pricing_response" | grep -q "price"; then
        RESULTS["api_pricing"]="HEALTHY"
        log_info "API pricing endpoint is functional"
    else
        RESULTS["api_pricing"]="UNHEALTHY"
        OVERALL_STATUS="UNHEALTHY"
        log_error "API pricing endpoint is not functional"
    fi
}

# Generate report
generate_report() {
    log_step "Generating health check report"

    local report_file="${SCRIPT_DIR}/../logs/health_report_$(date +%Y%m%d_%H%M%S).json"

    # Create JSON report
    cat > "$report_file" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "overall_status": "$OVERALL_STATUS",
  "checks": {
EOF

    local first=true
    for check in "${!RESULTS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$report_file"
        fi
        echo "    \"$check\": \"${RESULTS[$check]}\"" >> "$report_file"
    done

    cat >> "$report_file" << EOF

  },
  "system_info": {
    "hostname": "$(hostname)",
    "uptime": "$(uptime -p)",
    "load_average": "$(uptime | awk -F'load average:' '{ print $2 }')"
  }
}
EOF

    log_info "Health check report saved to: $report_file"
}

# Main function
main() {
    log_info "Starting comprehensive health check"

    # Initialize log file
    mkdir -p "${SCRIPT_DIR}/../logs"
    echo "=== Health Check Report $(date) ===" > "$LOG_FILE"

    # Run all checks
    check_docker_containers
    check_system_resources
    check_database
    check_models
    check_api_endpoints

    # Generate report
    generate_report

    # Print summary
    echo
    echo "========================================"
    echo "HEALTH CHECK SUMMARY"
    echo "========================================"
    echo "Overall Status: $OVERALL_STATUS"
    echo "Timestamp: $(date)"
    echo "========================================"

    for check in "${!RESULTS[@]}"; do
        local status="${RESULTS[$check]}"
        case $status in
            "HEALTHY")
                echo -e "${GREEN}✓ $check: $status${NC}"
                ;;
            "WARNING")
                echo -e "${YELLOW}⚠ $check: $status${NC}"
                ;;
            "UNHEALTHY")
                echo -e "${RED}✗ $check: $status${NC}"
                ;;
            *)
                echo -e "$check: $status"
                ;;
        esac
    done

    echo "========================================"

    # Exit with appropriate code
    if [ "$OVERALL_STATUS" = "HEALTHY" ]; then
        log_info "All health checks passed"
        exit 0
    elif [ "$OVERALL_STATUS" = "WARNING" ]; then
        log_warn "Health checks completed with warnings"
        exit 1
    else
        log_error "Health checks failed"
        exit 2
    fi
}

# Run main function
main "$@"