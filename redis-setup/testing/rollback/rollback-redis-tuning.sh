#!/bin/bash
# Redis HFT Tuning Rollback Procedure
# Quick recovery script for Redis configuration issues
# Created: September 27, 2025

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${SCRIPT_DIR}/backups/20250927_124118_pre_tuning"
ORIGINAL_CONFIG="/opt/redis-hft/config/redis-hft.conf"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[EMERGENCY]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Emergency rollback - restore original configuration
emergency_rollback() {
    log "🚨 EMERGENCY ROLLBACK INITIATED"
    
    if [[ ! -f "$BACKUP_DIR/redis-hft.conf" ]]; then
        error "CRITICAL: Backup configuration not found!"
        error "Location: $BACKUP_DIR/redis-hft.conf"
        exit 1
    fi
    
    log "Stopping Redis service..."
    sudo systemctl stop redis-hft || warning "Failed to stop Redis gracefully"
    
    log "Restoring original configuration..."
    sudo cp "$BACKUP_DIR/redis-hft.conf" "$ORIGINAL_CONFIG"
    
    log "Starting Redis with original configuration..."
    sudo systemctl start redis-hft
    
    # Wait and verify
    sleep 5
    if redis-cli ping >/dev/null 2>&1; then
        success "✅ EMERGENCY ROLLBACK SUCCESSFUL"
        log "Redis is responding normally"
        
        # Quick performance check
        log "Running quick performance verification..."
        cd "$SCRIPT_DIR"
        ../monitoring/redis-hft-monitor_to_json.sh > /tmp/rollback_verify.json
        
        if command -v jq >/dev/null 2>&1; then
            local set_p99=$(jq -r '.set.p99' /tmp/rollback_verify.json)
            local rtt_p99=$(jq -r '.rtt.p99' /tmp/rollback_verify.json)
            log "Performance check - SET P99: ${set_p99}μs, RTT P99: ${rtt_p99}μs"
        fi
        
    else
        error "❌ EMERGENCY ROLLBACK FAILED"
        error "Redis is not responding after rollback"
        error "Manual intervention required"
        exit 1
    fi
}

# Verify system state
verify_system() {
    log "Verifying system state..."
    
    # Check Redis service
    if ! systemctl is-active redis-hft >/dev/null 2>&1; then
        warning "Redis service is not active"
        return 1
    fi
    
    # Check Redis connectivity
    if ! redis-cli ping >/dev/null 2>&1; then
        warning "Redis is not responding to ping"
        return 1
    fi
    
    # Check performance gate
    if [[ -f "$SCRIPT_DIR/perf-gate.sh" ]]; then
        cd "$SCRIPT_DIR"
        if ./perf-gate.sh >/dev/null 2>&1; then
            success "Performance gate: PASS"
        else
            warning "Performance gate: FAIL"
            return 1
        fi
    fi
    
    success "System state verification: PASS"
    return 0
}

# Show current vs baseline performance
show_performance_comparison() {
    log "Performance comparison:"
    
    if [[ -f "$BACKUP_DIR/performance_baseline.json" ]]; then
        echo ""
        echo "=== BASELINE (Pre-tuning) ==="
        if command -v jq >/dev/null 2>&1; then
            echo "SET P99: $(jq -r '.set.p99' "$BACKUP_DIR/performance_baseline.json")μs"
            echo "RTT P99: $(jq -r '.rtt.p99' "$BACKUP_DIR/performance_baseline.json")μs"
            echo "Jitter: $(jq -r '.set.jitter' "$BACKUP_DIR/performance_baseline.json")μs"
        else
            cat "$BACKUP_DIR/performance_baseline.json"
        fi
    fi
    
    echo ""
    echo "=== CURRENT ==="
    cd "$SCRIPT_DIR"
    ../monitoring/redis-hft-monitor_to_json.sh > /tmp/current_perf.json
    
    if command -v jq >/dev/null 2>&1; then
        echo "SET P99: $(jq -r '.set.p99' /tmp/current_perf.json)μs"
        echo "RTT P99: $(jq -r '.rtt.p99' /tmp/current_perf.json)μs"  
        echo "Jitter: $(jq -r '.set.jitter' /tmp/current_perf.json)μs"
    else
        cat /tmp/current_perf.json
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "🔧 Redis HFT Tuning Rollback Options"
    echo "====================================="
    echo "1) Emergency Rollback (restore original config)"
    echo "2) Verify System State"
    echo "3) Show Performance Comparison"
    echo "4) Check Backup Status"
    echo "5) Exit"
    echo ""
    read -p "Select option [1-5]: " choice
    
    case $choice in
        1) emergency_rollback ;;
        2) verify_system ;;
        3) show_performance_comparison ;;
        4) 
            log "Backup status:"
            ls -la "$BACKUP_DIR/"
            if [[ -f "$BACKUP_DIR/checksums.txt" ]]; then
                echo ""
                echo "Backup checksums:"
                cat "$BACKUP_DIR/checksums.txt"
            fi
            ;;
        5) log "Exiting"; exit 0 ;;
        *) warning "Invalid option"; show_menu ;;
    esac
}

# Check if this is an emergency call
if [[ $# -gt 0 && "$1" == "--emergency" ]]; then
    emergency_rollback
    exit $?
fi

# Safety checks
if [[ $EUID -eq 0 ]]; then
    error "Do not run this script as root"
    exit 1
fi

if [[ ! -f "$BACKUP_DIR/redis-hft.conf" ]]; then
    error "Backup configuration not found: $BACKUP_DIR/redis-hft.conf"
    error "Cannot perform rollback without backup"
    exit 1
fi

log "Redis HFT Tuning Rollback Utility"
log "Backup location: $BACKUP_DIR"

# Interactive mode
while true; do
    show_menu
    echo ""
    read -p "Continue? [y/N]: " continue_choice
    if [[ "$continue_choice" != "y" && "$continue_choice" != "Y" ]]; then
        break
    fi
done

log "Rollback utility finished"