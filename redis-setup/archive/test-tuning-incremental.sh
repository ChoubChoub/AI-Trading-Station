#!/bin/bash
# Redis HFT Incremental Tuning Tester
# Created: September 28, 2025
# Purpose: Test each Redis configuration change individually to isolate impact

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REDIS_CONFIG="/opt/redis-hft/config/redis-hft.conf"
BACKUP_CONFIG="${SCRIPT_DIR}/backups/original_redis_hft_conf_backup.txt"
STAGING_CONFIG="${SCRIPT_DIR}/redis-hft.conf.tuning-proposal"
RESULTS_FILE="${SCRIPT_DIR}/TUNING_RESULTS.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Test configuration changes incrementally
test_incremental_changes() {
    local changes=(
        "databases:1:databases 1"
        "tcp-keepalive:30:tcp-keepalive 30"
        "hz:8:hz 8"
        "maxmemory:1gb:maxmemory 1gb"
    )
    
    log "Starting incremental Redis tuning tests..."
    echo
    
    # Get baseline performance
    log "Measuring baseline performance..."
    baseline_perf=$(./redis-hft-monitor_to_json.sh 2>/dev/null || echo "FAILED")
    if [[ "$baseline_perf" == "FAILED" ]]; then
        error "Failed to get baseline performance"
        return 1
    fi
    
    baseline_set_p99=$(echo "$baseline_perf" | grep -o '"set":{"p50":[0-9]*,"p95":[0-9]*,"p99":[0-9]*' | grep -o 'p99":[0-9]*' | cut -d':' -f2)
    baseline_rtt_p99=$(echo "$baseline_perf" | grep -o '"rtt":{"p50":[^}]*,"p99":[0-9.]*' | grep -o 'p99":[0-9.]*' | cut -d':' -f2)
    
    success "Baseline: SET P99=${baseline_set_p99}μs, RTT P99=${baseline_rtt_p99}μs"
    echo
    
    # Create working copy of original config
    local working_config="${SCRIPT_DIR}/backups/redis-hft-working.conf"
    cp "$BACKUP_CONFIG" "$working_config"
    
    # Test each change individually
    for change_info in "${changes[@]}"; do
        IFS=':' read -r setting value config_line <<< "$change_info"
        
        log "Testing: $setting = $value"
        
        # Apply this change to working config
        echo "$config_line" >> "$working_config"
        
        # Deploy working config
        sudo cp "$working_config" "$REDIS_CONFIG"
        
        # Restart Redis
        log "Restarting Redis with $setting change..."
        if ! sudo systemctl restart redis-hft; then
            error "Failed to restart Redis with $setting change"
            rollback_and_exit
        fi
        
        # Wait for Redis to stabilize
        sleep 3
        
        # Test connectivity
        if ! redis-cli ping >/dev/null 2>&1; then
            error "Redis not responding after $setting change"
            rollback_and_exit
        fi
        
        # Measure performance
        log "Measuring performance with $setting change..."
        test_perf=$(./redis-hft-monitor_to_json.sh 2>/dev/null || echo "FAILED")
        if [[ "$test_perf" == "FAILED" ]]; then
            error "Performance test failed with $setting change"
            rollback_and_exit
        fi
        
        test_set_p99=$(echo "$test_perf" | grep -o '"set":{"p50":[0-9]*,"p95":[0-9]*,"p99":[0-9]*' | grep -o 'p99":[0-9]*' | cut -d':' -f2)
        test_rtt_p99=$(echo "$test_perf" | grep -o '"rtt":{"p50":[^}]*,"p99":[0-9.]*' | grep -o 'p99":[0-9.]*' | cut -d':' -f2)
        
        # Compare performance
        set_change=$(echo "scale=1; (($test_set_p99 - $baseline_set_p99) / $baseline_set_p99) * 100" | bc -l)
        rtt_change=$(echo "scale=1; (($test_rtt_p99 - $baseline_rtt_p99) / $baseline_rtt_p99) * 100" | bc -l)
        
        echo "Results: SET P99=${test_set_p99}μs (${set_change}%), RTT P99=${test_rtt_p99}μs (${rtt_change}%)"
        
        # Check for regression
        if (( $(echo "$set_change > 10" | bc -l) )) || (( $(echo "$rtt_change > 10" | bc -l) )); then
            error "Performance regression detected: SET ${set_change}%, RTT ${rtt_change}%"
            warning "Rolling back $setting change..."
            
            # Remove the problematic line from working config
            head -n -1 "$working_config" > "${working_config}.tmp" && mv "${working_config}.tmp" "$working_config"
            continue
        else
            success "$setting change: SET ${set_change}%, RTT ${rtt_change}% - KEEPING"
        fi
        
        echo
    done
    
    # Apply final configuration
    log "Applying final optimized configuration..."
    sudo cp "$working_config" "$REDIS_CONFIG"
    sudo systemctl restart redis-hft
    sleep 3
    
    # Final performance test
    log "Final performance validation..."
    final_perf=$(./redis-hft-monitor_to_json.sh 2>/dev/null || echo "FAILED")
    if [[ "$final_perf" != "FAILED" ]]; then
        final_set_p99=$(echo "$final_perf" | grep -o '"set":{"p50":[0-9]*,"p95":[0-9]*,"p99":[0-9]*' | grep -o 'p99":[0-9]*' | cut -d':' -f2)
        final_rtt_p99=$(echo "$final_perf" | grep -o '"rtt":{"p50":[^}]*,"p99":[0-9.]*' | grep -o 'p99":[0-9.]*' | cut -d':' -f2)
        
        total_set_change=$(echo "scale=1; (($final_set_p99 - $baseline_set_p99) / $baseline_set_p99) * 100" | bc -l)
        total_rtt_change=$(echo "scale=1; (($final_rtt_p99 - $baseline_rtt_p99) / $baseline_rtt_p99) * 100" | bc -l)
        
        success "FINAL RESULTS:"
        success "  SET P99: ${baseline_set_p99}μs → ${final_set_p99}μs (${total_set_change}%)"
        success "  RTT P99: ${baseline_rtt_p99}μs → ${final_rtt_p99}μs (${total_rtt_change}%)"
    fi
    
    # Run performance gate
    log "Running performance gate validation..."
    if ./perf-gate.sh >/dev/null 2>&1; then
        success "Performance gate: PASS ✅"
    else
        warning "Performance gate: FAIL - Consider rollback"
    fi
    
    # Cleanup
    rm -f "$working_config" "${working_config}.tmp"
    
    success "Incremental tuning test complete!"
}

rollback_and_exit() {
    error "Rolling back to original configuration..."
    ./rollback-tuning.sh
    exit 1
}

# Main execution
main() {
    echo "🚀 Redis HFT Incremental Tuning Tester"
    echo "======================================"
    echo
    
    # Verify prerequisites
    if [[ ! -f "$BACKUP_CONFIG" ]]; then
        error "Original backup not found: $BACKUP_CONFIG"
        error "Run backup first before testing"
        exit 1
    fi
    
    if [[ ! -f "$STAGING_CONFIG" ]]; then
        error "Staging config not found: $STAGING_CONFIG"
        exit 1
    fi
    
    # Check if bc is available for calculations
    if ! command -v bc >/dev/null 2>&1; then
        error "bc calculator not found. Installing..."
        sudo apt-get update && sudo apt-get install -y bc
    fi
    
    # Start testing
    test_incremental_changes
}

# Run if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi