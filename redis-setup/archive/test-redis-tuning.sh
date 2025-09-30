#!/bin/bash
# Redis HFT Incremental Tuning Test Plan
# Tests each configuration change individually to isolate impact
# Created: September 27, 2025

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${SCRIPT_DIR}/backups/20250927_124118_pre_tuning"
RESULTS_DIR="${SCRIPT_DIR}/tuning_results"
ORIGINAL_CONFIG="/opt/redis-hft/config/redis-hft.conf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Initialize results directory
init_testing() {
    mkdir -p "$RESULTS_DIR"
    log "Initialized testing environment"
    log "Backup location: $BACKUP_DIR"
    log "Results location: $RESULTS_DIR"
}

# Run performance test and save results
run_performance_test() {
    local test_name="$1"
    local result_file="${RESULTS_DIR}/${test_name}_performance.json"
    
    log "Running performance test: $test_name"
    
    # Wait for Redis to stabilize
    sleep 2
    
    # Run performance monitoring
    cd "$SCRIPT_DIR"
    ./redis-hft-monitor_to_json.sh > "$result_file"
    
    # Extract key metrics for quick comparison
    if command -v jq >/dev/null 2>&1; then
        local set_p99=$(jq -r '.set.p99' "$result_file")
        local rtt_p99=$(jq -r '.rtt.p99' "$result_file") 
        local jitter=$(jq -r '.set.jitter' "$result_file")
        log "Results - SET P99: ${set_p99}μs, RTT P99: ${rtt_p99}μs, Jitter: ${jitter}μs"
    else
        log "Results saved to: $result_file"
    fi
    
    return 0
}

# Apply single configuration change  
apply_config_change() {
    local change_name="$1"
    local config_line="$2"
    local temp_config="/tmp/redis-hft-test.conf"
    
    log "Applying change: $change_name"
    log "Configuration: $config_line"
    
    # Copy original config
    sudo cp "$ORIGINAL_CONFIG" "$temp_config"
    
    # Apply the specific change
    case "$change_name" in
        "databases")
            sudo sed -i '/^databases/d' "$temp_config"
            echo "$config_line" | sudo tee -a "$temp_config" >/dev/null
            ;;
        "tcp-keepalive")
            sudo sed -i '/^tcp-keepalive/d' "$temp_config"
            echo "$config_line" | sudo tee -a "$temp_config" >/dev/null
            ;;
        "tcp-nodelay")
            echo "$config_line" | sudo tee -a "$temp_config" >/dev/null
            ;;
        "hz")
            sudo sed -i '/^hz/d' "$temp_config"
            echo "$config_line" | sudo tee -a "$temp_config" >/dev/null
            ;;
        "maxmemory")
            sudo sed -i '/^maxmemory /d' "$temp_config"
            echo "$config_line" | sudo tee -a "$temp_config" >/dev/null
            ;;
    esac
    
    # Deploy temporary config
    sudo cp "$temp_config" "$ORIGINAL_CONFIG"
    
    # Restart Redis
    log "Restarting Redis with new configuration..."
    sudo systemctl restart redis-hft
    sleep 3
    
    # Verify Redis is running
    if ! redis-cli ping >/dev/null 2>&1; then
        error "Redis failed to start with change: $change_name"
        return 1
    fi
    
    success "Change applied successfully: $change_name"
    return 0
}

# Restore original configuration
restore_original() {
    log "Restoring original configuration"
    sudo cp "${BACKUP_DIR}/redis-hft.conf" "$ORIGINAL_CONFIG"
    sudo systemctl restart redis-hft
    sleep 3
    
    if redis-cli ping >/dev/null 2>&1; then
        success "Original configuration restored"
    else
        error "Failed to restore original configuration"
        return 1
    fi
}

# Test individual changes
test_individual_changes() {
    log "Starting individual change testing"
    
    # Baseline measurement (should already exist, but taking fresh one)
    run_performance_test "00_baseline_fresh"
    
    # Test 1: databases 16 → 1
    if apply_config_change "databases" "databases 1"; then
        run_performance_test "01_databases_1"
        restore_original
    fi
    
    # Test 2: tcp-keepalive 300 → 30  
    if apply_config_change "tcp-keepalive" "tcp-keepalive 30"; then
        run_performance_test "02_tcp_keepalive_30"
        restore_original
    fi
    
    # Test 3: explicit tcp-nodelay
    if apply_config_change "tcp-nodelay" "tcp-nodelay yes"; then
        run_performance_test "03_tcp_nodelay_yes"
        restore_original
    fi
    
    # Test 4: hz 10 → 8
    if apply_config_change "hz" "hz 8"; then
        run_performance_test "04_hz_8"
        restore_original
    fi
    
    # Test 5: maxmemory 4gb → 1gb
    if apply_config_change "maxmemory" "maxmemory 1gb"; then
        run_performance_test "05_maxmemory_1gb"
        restore_original
    fi
    
    success "Individual testing completed"
}

# Generate test report
generate_report() {
    local report_file="${RESULTS_DIR}/tuning_test_report.txt"
    
    log "Generating test report"
    
    cat > "$report_file" << EOF
Redis HFT Tuning Test Report
Generated: $(date)
Backup: $BACKUP_DIR

Test Results:
EOF
    
    for result_file in "${RESULTS_DIR}"/*.json; do
        if [[ -f "$result_file" ]]; then
            local test_name=$(basename "$result_file" .json)
            echo "" >> "$report_file"
            echo "=== $test_name ===" >> "$report_file"
            
            if command -v jq >/dev/null 2>&1; then
                echo "SET P99: $(jq -r '.set.p99' "$result_file")μs" >> "$report_file"
                echo "XADD P99: $(jq -r '.xadd.p99' "$result_file")μs" >> "$report_file" 
                echo "RTT P99: $(jq -r '.rtt.p99' "$result_file")μs" >> "$report_file"
                echo "SET Jitter: $(jq -r '.set.jitter' "$result_file")μs" >> "$report_file"
                echo "Throughput: $(jq -r '.health.ops_per_sec' "$result_file") ops/sec" >> "$report_file"
            else
                echo "Raw data: $result_file" >> "$report_file"
            fi
        fi
    done
    
    success "Report generated: $report_file"
}

# Main execution
main() {
    log "Redis HFT Incremental Tuning Test"
    log "=================================="
    
    init_testing
    test_individual_changes
    generate_report
    
    log "Testing completed. Review results in: $RESULTS_DIR"
    warning "Remember to analyze results before applying any changes permanently"
}

# Safety checks
if [[ $EUID -eq 0 ]]; then
    error "Do not run this script as root"
    exit 1
fi

if [[ ! -f "$BACKUP_DIR/redis-hft.conf" ]]; then
    error "Backup configuration not found: $BACKUP_DIR/redis-hft.conf"
    exit 1
fi

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi