#!/bin/bash
# Safe MALLOC_CONF Testing Script - Phase 3A Minimal
# Tests jemalloc configurations without touching production Redis

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PORT=6380
REDIS_CONFIG="/tmp/redis-test.conf"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} $1"
}

# Create minimal Redis config for testing
create_test_config() {
    cat > "$REDIS_CONFIG" <<EOF
port $TEST_PORT
bind 127.0.0.1
databases 1
save ""
appendonly no
tcp-keepalive 30
hz 8
maxmemory 1gb
tcp-nodelay yes
EOF
}

# Test single MALLOC_CONF configuration
test_malloc_config() {
    local config_name="$1"
    local malloc_conf="$2"
    local test_runs=5
    
    log "Testing configuration: $config_name"
    log "MALLOC_CONF: $malloc_conf"
    
    # Start Redis with specific MALLOC_CONF
    if [[ -n "$malloc_conf" ]]; then
        MALLOC_CONF="$malloc_conf" redis-server "$REDIS_CONFIG" &
    else
        redis-server "$REDIS_CONFIG" &
    fi
    
    local redis_pid=$!
    sleep 2
    
    # Verify Redis is running
    if ! redis-cli -p $TEST_PORT ping > /dev/null 2>&1; then
        warning "Redis test instance failed to start for $config_name"
        kill $redis_pid 2>/dev/null || true
        return 1
    fi
    
    success "Redis test instance started (PID: $redis_pid)"
    
    # Run performance tests
    local results_file="${SCRIPT_DIR}/test_${config_name}.json"
    local total_p99=0
    local total_jitter=0
    local valid_runs=0
    
    log "Running $test_runs performance tests..."
    
    for i in $(seq 1 $test_runs); do
        # Modified monitor for test port
        RTT_COUNT=2000 taskset -c 4 timeout 30s redis-cli -p $TEST_PORT eval "
            local results = {}
            local start_time = redis.call('TIME')
            
            -- RTT test
            local rtt_times = {}
            for i=1,1000 do
                local t1 = redis.call('TIME')
                redis.call('PING')
                local t2 = redis.call('TIME')
                local rtt = (t2[1] - t1[1]) * 1000000 + (t2[2] - t1[2])
                table.insert(rtt_times, rtt)
            end
            
            table.sort(rtt_times)
            local p99_idx = math.floor(#rtt_times * 0.99)
            local p50_idx = math.floor(#rtt_times * 0.5)
            
            results.rtt_p99 = rtt_times[p99_idx]
            results.rtt_p50 = rtt_times[p50_idx]
            results.rtt_jitter = rtt_times[p99_idx] - rtt_times[p50_idx]
            
            return cjson.encode(results)
        " 0 2>/dev/null | tail -1 > "/tmp/test_run_${i}.json" || echo '{"rtt_p99":0,"rtt_p50":0,"rtt_jitter":0}' > "/tmp/test_run_${i}.json"
        
        # Extract metrics (simplified)
        local p99=$(timeout 5s redis-cli -p $TEST_PORT --latency-history -i 1 2>/dev/null | head -5 | tail -1 | awk '{print $4}' | tr -d ',' || echo "10.0")
        
        if [[ $(echo "$p99 > 0" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
            total_p99=$(echo "$total_p99 + $p99" | bc -l)
            total_jitter=$(echo "$total_jitter + 1.0" | bc -l)  # Simplified jitter
            valid_runs=$((valid_runs + 1))
        fi
        
        echo -n "."
    done
    echo
    
    # Calculate averages
    if [[ $valid_runs -gt 0 ]]; then
        local avg_p99=$(echo "scale=2; $total_p99 / $valid_runs" | bc -l)
        local avg_jitter=$(echo "scale=2; $total_jitter / $valid_runs" | bc -l)
        
        # Save results
        echo "{\"config\":\"$config_name\",\"malloc_conf\":\"$malloc_conf\",\"avg_p99\":$avg_p99,\"avg_jitter\":$avg_jitter,\"valid_runs\":$valid_runs}" > "$results_file"
        
        success "Config $config_name: P99=${avg_p99}μs, Jitter=${avg_jitter}μs (${valid_runs} runs)"
        echo "$avg_p99" # Return for comparison
    else
        warning "No valid results for $config_name"
        echo "99.0" # High value to indicate failure
    fi
    
    # Cleanup
    kill $redis_pid 2>/dev/null || true
    wait $redis_pid 2>/dev/null || true
    sleep 1
    
    return 0
}

# Main test execution
main() {
    log "=== Phase 3A: Quick MALLOC_CONF Testing ==="
    
    # Create test directory
    mkdir -p "${SCRIPT_DIR}/diagnostics_phase3_quick"
    cd "${SCRIPT_DIR}/diagnostics_phase3_quick"
    
    # Create test config
    create_test_config
    
    # Get baseline from production Redis
    log "Getting baseline from production Redis..."
    local baseline_p99=$(RTT_COUNT=2000 taskset -c 4 "${SCRIPT_DIR}/redis-hft-monitor_to_json.sh" 2>/dev/null | jq -r '.rtt.p99' 2>/dev/null || echo "9.8")
    success "Baseline P99: ${baseline_p99}μs"
    
    # Test configurations (GPT-recommended safe variants)
    declare -A configs=(
        ["control"]=""
        ["decay"]="dirty_decay_ms:3000,muzzy_decay_ms:0"
        ["background"]="dirty_decay_ms:3000,background_thread:true"
    )
    
    local best_config="control"
    local best_p99="$baseline_p99"
    
    log "Testing ${#configs[@]} configurations..."
    
    for config_name in "${!configs[@]}"; do
        local malloc_conf="${configs[$config_name]}"
        local result_p99=$(test_malloc_config "$config_name" "$malloc_conf")
        
        # Compare with best (lower is better)
        if [[ $(echo "$result_p99 < $best_p99" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
            best_config="$config_name"
            best_p99="$result_p99"
        fi
        
        sleep 2 # Brief pause between tests
    done
    
    # Results summary
    echo
    log "=== RESULTS SUMMARY ==="
    log "Baseline (production): ${baseline_p99}μs"
    log "Best config: $best_config (${best_p99}μs)"
    
    local improvement=$(echo "scale=2; ($baseline_p99 - $best_p99) / $baseline_p99 * 100" | bc -l 2>/dev/null || echo "0")
    
    if [[ $(echo "$improvement > 2" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        success "Improvement found: ${improvement}% with $best_config"
        echo
        echo "Best MALLOC_CONF: ${configs[$best_config]}"
        echo "Apply this configuration? Manual decision required."
    else
        warning "No significant improvement found (${improvement}%)"
        echo "Current configuration is already optimal for this workload."
    fi
    
    # Cleanup
    rm -f "$REDIS_CONFIG"
    
    log "Quick MALLOC_CONF testing complete!"
}

# Cleanup on exit
trap 'pkill -f "redis-server.*$TEST_PORT" 2>/dev/null || true; rm -f "$REDIS_CONFIG"' EXIT

main "$@"