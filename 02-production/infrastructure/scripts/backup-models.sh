#!/bin/bash

# Model Backup Script for Price Matrix
# This script creates backups of trained models and related artifacts

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_ROOT="${SCRIPT_DIR}/../backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="models_backup_${TIMESTAMP}"
BACKUP_DIR="${BACKUP_ROOT}/${BACKUP_NAME}"

# Source directories
MODEL_DIR="${SCRIPT_DIR}/../../ml-service/models"
EXPERIMENT_DIR="${SCRIPT_DIR}/../../01-research-development/experiments"
LOG_DIR="${SCRIPT_DIR}/../../01-research-development/logs"

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

# Create backup directory
create_backup_dir() {
    log_step "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"

    # Create subdirectories
    mkdir -p "$BACKUP_DIR/models"
    mkdir -p "$BACKUP_DIR/experiments"
    mkdir -p "$BACKUP_DIR/logs"
    mkdir -p "$BACKUP_DIR/metadata"
}

# Backup models
backup_models() {
    log_step "Backing up models from: $MODEL_DIR"

    if [ ! -d "$MODEL_DIR" ]; then
        log_warn "Model directory not found: $MODEL_DIR"
        return 1
    fi

    # Copy model files
    if ls "$MODEL_DIR"/*.h5 1> /dev/null 2>&1; then
        cp "$MODEL_DIR"/*.h5 "$BACKUP_DIR/models/" 2>/dev/null || true
        log_info "Backed up $(ls "$MODEL_DIR"/*.h5 2>/dev/null | wc -l) H5 model files"
    fi

    if ls "$MODEL_DIR"/*.pkl 1> /dev/null 2>&1; then
        cp "$MODEL_DIR"/*.pkl "$BACKUP_DIR/models/" 2>/dev/null || true
        log_info "Backed up $(ls "$MODEL_DIR"/*.pkl 2>/dev/null | wc -l) pickle model files"
    fi

    if ls "$MODEL_DIR"/*.joblib 1> /dev/null 2>&1; then
        cp "$MODEL_DIR"/*.joblib "$BACKUP_DIR/models/" 2>/dev/null || true
        log_info "Backed up $(ls "$MODEL_DIR"/*.joblib 2>/dev/null | wc -l) joblib model files"
    fi

    # Copy model metadata
    if [ -f "$MODEL_DIR/model_metadata.json" ]; then
        cp "$MODEL_DIR/model_metadata.json" "$BACKUP_DIR/models/"
        log_info "Backed up model metadata"
    fi
}

# Backup experiments
backup_experiments() {
    log_step "Backing up experiments from: $EXPERIMENT_DIR"

    if [ ! -d "$EXPERIMENT_DIR" ]; then
        log_warn "Experiment directory not found: $EXPERIMENT_DIR"
        return 1
    fi

    # Copy experiment results
    if [ -d "$EXPERIMENT_DIR/model_checkpoints" ]; then
        cp -r "$EXPERIMENT_DIR/model_checkpoints" "$BACKUP_DIR/experiments/" 2>/dev/null || true
        log_info "Backed up model checkpoints"
    fi

    if [ -d "$EXPERIMENT_DIR/results" ]; then
        cp -r "$EXPERIMENT_DIR/results" "$BACKUP_DIR/experiments/" 2>/dev/null || true
        log_info "Backed up experiment results"
    fi

    # Copy experiment configurations
    if ls "$EXPERIMENT_DIR"/*.yaml 1> /dev/null 2>&1; then
        cp "$EXPERIMENT_DIR"/*.yaml "$BACKUP_DIR/experiments/" 2>/dev/null || true
        log_info "Backed up experiment configurations"
    fi
}

# Backup logs
backup_logs() {
    log_step "Backing up logs from: $LOG_DIR"

    if [ ! -d "$LOG_DIR" ]; then
        log_warn "Log directory not found: $LOG_DIR"
        return 1
    fi

    # Copy recent logs (last 30 days)
    find "$LOG_DIR" -name "*.log" -mtime -30 -exec cp {} "$BACKUP_DIR/logs/" \; 2>/dev/null || true

    if [ "$(ls -A "$BACKUP_DIR/logs" 2>/dev/null)" ]; then
        log_info "Backed up recent log files"
    else
        log_info "No recent log files to backup"
    fi
}

# Create backup metadata
create_metadata() {
    log_step "Creating backup metadata"

    cat > "$BACKUP_DIR/metadata/backup_info.json" << EOF
{
  "backup_name": "$BACKUP_NAME",
  "timestamp": "$TIMESTAMP",
  "created_by": "$(whoami)",
  "hostname": "$(hostname)",
  "backup_type": "models",
  "version": "1.0.0",
  "directories": {
    "source_model_dir": "$MODEL_DIR",
    "source_experiment_dir": "$EXPERIMENT_DIR",
    "source_log_dir": "$LOG_DIR",
    "backup_dir": "$BACKUP_DIR"
  },
  "contents": {
    "models": $(ls "$BACKUP_DIR/models/" 2>/dev/null | wc -l),
    "experiments": $(find "$BACKUP_DIR/experiments/" -type f 2>/dev/null | wc -l),
    "logs": $(ls "$BACKUP_DIR/logs/" 2>/dev/null | wc -l)
  }
}
EOF

    # Create file manifest
    find "$BACKUP_DIR" -type f -exec ls -lh {} \; > "$BACKUP_DIR/metadata/file_manifest.txt"

    log_info "Backup metadata created"
}

# Compress backup
compress_backup() {
    log_step "Compressing backup"

    cd "$BACKUP_ROOT"
    tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"

    # Calculate sizes
    BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
    COMPRESSED_SIZE=$(du -sh "${BACKUP_NAME}.tar.gz" | cut -f1)

    log_info "Backup compressed: $BACKUP_SIZE -> $COMPRESSED_SIZE"

    # Clean up uncompressed directory
    rm -rf "$BACKUP_DIR"
}

# Cleanup old backups
cleanup_old_backups() {
    log_step "Cleaning up old backups"

    # Keep only last 10 backups
    cd "$BACKUP_ROOT"
    ls -t *.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm -f

    log_info "Old backups cleaned up"
}

# Main function
main() {
    log_info "Starting model backup process"

    # Check if running as appropriate user
    if [ "$EUID" -eq 0 ]; then
        log_warn "Running as root - this may cause permission issues"
    fi

    # Create backup directory
    create_backup_dir

    # Perform backups
    backup_models
    backup_experiments
    backup_logs

    # Create metadata
    create_metadata

    # Compress
    compress_backup

    # Cleanup
    cleanup_old_backups

    log_info "Model backup completed successfully: ${BACKUP_NAME}.tar.gz"
    log_info "Backup location: ${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz"

    # Print summary
    echo
    echo "========================================"
    echo "BACKUP SUMMARY"
    echo "========================================"
    echo "Name: $BACKUP_NAME"
    echo "Location: ${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz"
    echo "Created: $(date)"
    echo "========================================"
}

# Error handling
error_handler() {
    log_error "Backup failed with error code $?"
    log_error "Backup directory: $BACKUP_DIR"

    # Cleanup partial backup
    if [ -d "$BACKUP_DIR" ]; then
        log_info "Cleaning up partial backup"
        rm -rf "$BACKUP_DIR"
    fi

    exit 1
}

# Set error handler
trap error_handler ERR

# Run main function
main "$@"