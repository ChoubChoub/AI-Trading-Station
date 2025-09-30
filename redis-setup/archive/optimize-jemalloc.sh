#!/bin/bash
# jemalloc Configuration Optimization Script
# Phase 3A - HFT Redis Memory Allocator Tuning

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REDIS_SERVICE="redis-server"

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
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%H:%M:%S')]${NC} $1"
}

# Test configuration function
test_jemalloc_config() {
    local config_name="$1"
    local malloc_conf="$2"
    local output_file="$3"
    
    log "Testing jemalloc configuration: $config_name"
    log "MALLOC_CONF: $malloc_conf"
    
    # Create systemd override directory
    sudo mkdir -p /etc/systemd/system/redis-server.service.d
    
    # Create override file with MALLOC_CONF
    sudo tee /etc/systemd/system/redis-server.service.d/jemalloc.conf > /dev/null <<EOF
[Service]
Environment="MALLOC_CONF=$malloc_conf"
EOF
    
    # Reload systemd and restart Redis
    sudo systemctl daemon-reload
    sudo systemctl restart redis-server
    
    # Wait for Redis to be ready
    sleep 2
    
    # Verify Redis is running
    if ! redis-cli ping > /dev/null 2>&1; then
        error "Redis not responding after configuration change!"
        return 1
    fi
    
    # Run performance test
    log "Running performance test..."
    RTT_COUNT=2000 taskset -c 4 "${SCRIPT_DIR}/redis-hft-monitor_to_json.sh" > "$output_file" 2>/dev/null
    
    # Extract key metrics
    local rtt_p99=$(jq -r '.rtt.p99' "$output_file")
    local rtt_jitter=$(jq -r '.rtt.jitter' "$output_file")
    
    success "Configuration: $config_name - RTT P99: ${rtt_p99}μs, Jitter: ${rtt_jitter}μs"
    
    # Get memory stats
    redis-cli MEMORY STATS | grep -E "(fragmentation|allocator)" > "${output_file%.json}_memory.txt"
    
    return 0
}

# Rollback function
rollback_config() {
    log "Rolling back to default jemalloc configuration..."
    
    # Remove override file
    sudo rm -f /etc/systemd/system/redis-server.service.d/jemalloc.conf
    
    # Reload and restart
    sudo systemctl daemon-reload
    sudo systemctl restart redis-server
    
    # Wait for Redis
    sleep 2
    
    if redis-cli ping > /dev/null 2>&1; then
        success "Rollback successful - Redis running with default configuration"
    else
        error "Rollback failed - Redis not responding!"
        return 1
    fi
}

# Main optimization testing
main() {
    log "=== Phase 3A: jemalloc Configuration Optimization ==="
    
    # Create diagnostics directory
    mkdir -p "${SCRIPT_DIR}/diagnostics_phase3"
    cd "${SCRIPT_DIR}/diagnostics_phase3"
    
    # Backup current performance
    log "Taking baseline measurement..."
    RTT_COUNT=2000 taskset -c 4 "${SCRIPT_DIR}/redis-hft-monitor_to_json.sh" > baseline_default.json 2>/dev/null
    local baseline_p99=$(jq -r '.rtt.p99' baseline_default.json)
    success "Baseline RTT P99: ${baseline_p99}μs"
    
    # Test configurations
    local configs=(
        "config1_arenas:narenas:2"
        "config2_decay:narenas:2,dirty_decay_ms:1000"
        "config3_thp:narenas:2,dirty_decay_ms:1000,metadata_thp:auto"
        "config4_optimal:narenas:2,dirty_decay_ms:1000,metadata_thp:auto,tcache_max:65536"
    )
    
    log "Testing ${#configs[@]} jemalloc configurations..."
    
    local best_config=""
    local best_p99="$baseline_p99"
    local best_file=""
    
    for config in "${configs[@]}"; do
        local name="${config%%:*}"
        local malloc_conf="${config#*:}"
        local output_file="${name}_result.json"
        
        if test_jemalloc_config "$name" "$malloc_conf" "$output_file"; then
            local current_p99=$(jq -r '.rtt.p99' "$output_file")
            
            # Compare performance (lower is better)
            if (( $(echo "$current_p99 < $best_p99" | bc -l) )); then
                best_config="$malloc_conf"
                best_p99="$current_p99"
                best_file="$output_file"
                success "New best configuration: $name (${current_p99}μs)"
            fi
        else
            warning "Configuration $name failed - skipping"
        fi
        
        sleep 1
    done
    
    # Summary
    echo
    log "=== OPTIMIZATION RESULTS ==="
    log "Baseline P99: ${baseline_p99}μs"
    log "Best P99: ${best_p99}μs"
    
    if [[ -n "$best_config" ]]; then
        local improvement=$(echo "scale=2; ($baseline_p99 - $best_p99) / $baseline_p99 * 100" | bc -l)
        success "Best configuration: $best_config"
        success "Improvement: ${improvement}% (${best_p99}μs vs ${baseline_p99}μs)"
        
        # Ask user to apply best configuration
        echo
        read -p "Apply best configuration permanently? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log "Applying best configuration permanently..."
            test_jemalloc_config "optimal" "$best_config" "final_optimized.json"
            success "Optimization applied! New RTT P99: $(jq -r '.rtt.p99' final_optimized.json)μs"
        else
            log "Keeping current configuration for now"
        fi
    else
        warning "No configuration showed improvement - keeping baseline"
    fi
    
    # Final rollback to default for safety
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        rollback_config
    fi
    
    log "jemalloc optimization testing complete!"
}

# Trap for cleanup
trap 'rollback_config' EXIT

# Run main function
main "$@"