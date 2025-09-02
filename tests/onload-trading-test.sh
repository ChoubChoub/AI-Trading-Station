#!/bin/bash

# ============================================================================
# ONLOAD TRADING TEST WRAPPER
# ============================================================================
# PURPOSE: Shell wrapper for OnLoad performance validation
# TARGET: Validate sub-microsecond latency in trading scenarios
# INTEGRATION: Works with comprehensive_onload_test.py
# ============================================================================

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/onload-trading-test.log"
readonly RESULTS_DIR="/tmp/onload-test-results"
readonly TEST_DURATION=60

# Test configuration
ONLOAD_ENABLED=true
TRADING_CORES="2,3"
SYSTEM_CORES="0,1"
MESSAGE_SIZE=64
ITERATIONS=10000
VERBOSE=false

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

check_dependencies() {
    log "Checking test dependencies..."
    
    local missing_deps=()
    
    # Check for required Python modules
    if ! python3 -c "import numpy, socket, threading, json" 2>/dev/null; then
        missing_deps+=("python3-numpy")
    fi
    
    # Check for network tools
    for cmd in ss netstat iperf3; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            log "Optional tool not found: $cmd"
        fi
    done
    
    # Install missing dependencies
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "Installing missing dependencies: ${missing_deps[*]}"
        apt-get update && apt-get install -y "${missing_deps[@]}"
    fi
    
    log "Dependencies verified"
}

setup_test_environment() {
    log "Setting up OnLoad test environment..."
    
    mkdir -p "$RESULTS_DIR"
    
    # Check OnLoad availability
    if [[ "$ONLOAD_ENABLED" == "true" ]]; then
        if ! command -v onload >/dev/null 2>&1; then
            log "WARNING: OnLoad not found, testing with regular sockets"
            ONLOAD_ENABLED=false
        else
            log "OnLoad available: $(onload --version 2>&1 | head -1)"
        fi
    fi
    
    # Configure CPU affinity for test process
    if [[ -n "$TRADING_CORES" ]]; then
        log "Setting CPU affinity for test to cores: $TRADING_CORES"
        # This will be applied when launching the test
    fi
    
    # Check network interface status
    local interfaces
    interfaces=$(ip link show | grep -E "^[0-9]+:" | awk -F: '{print $2}' | tr -d ' ' | grep -v lo || echo "")
    
    for interface in $interfaces; do
        if ip link show "$interface" | grep -q "state UP"; then
            log "Network interface active: $interface"
        fi
    done
    
    # Configure firewall if needed
    if command -v ufw >/dev/null 2>&1; then
        ufw allow 12345:12347/tcp >/dev/null 2>&1 || true
        ufw allow 12345:12347/udp >/dev/null 2>&1 || true
    fi
    
    log "Test environment setup completed"
}

run_baseline_network_test() {
    log "Running baseline network performance test..."
    
    local baseline_file="${RESULTS_DIR}/baseline_network.txt"
    
    # Test local loopback latency
    {
        echo "=== BASELINE NETWORK PERFORMANCE ==="
        echo "Timestamp: $(date)"
        echo "Hostname: $(hostname)"
        echo "Kernel: $(uname -r)"
        echo
        
        # Ping localhost for basic latency
        echo "Ping localhost (10 packets):"
        ping -c 10 127.0.0.1 2>/dev/null | tail -1 || echo "Ping failed"
        echo
        
        # Check network buffer sizes
        echo "Network buffer sizes:"
        echo "  rmem_default: $(cat /proc/sys/net/core/rmem_default 2>/dev/null || echo 'unknown')"
        echo "  wmem_default: $(cat /proc/sys/net/core/wmem_default 2>/dev/null || echo 'unknown')"
        echo "  rmem_max: $(cat /proc/sys/net/core/rmem_max 2>/dev/null || echo 'unknown')"
        echo "  wmem_max: $(cat /proc/sys/net/core/wmem_max 2>/dev/null || echo 'unknown')"
        echo
        
        # Check TCP settings
        echo "TCP settings:"
        echo "  tcp_rmem: $(cat /proc/sys/net/ipv4/tcp_rmem 2>/dev/null || echo 'unknown')"
        echo "  tcp_wmem: $(cat /proc/sys/net/ipv4/tcp_wmem 2>/dev/null || echo 'unknown')"
        echo "  tcp_congestion_control: $(cat /proc/sys/net/ipv4/tcp_congestion_control 2>/dev/null || echo 'unknown')"
        echo
        
        # Active connections
        echo "Active network connections:"
        ss -tuln | head -10 || echo "ss command failed"
        
    } > "$baseline_file"
    
    log "Baseline network test completed: $baseline_file"
}

run_python_latency_test() {
    log "Running comprehensive Python latency test..."
    
    local test_script="comprehensive_onload_test.py"
    local script_dir="$(dirname "$0")"
    local full_script_path="${script_dir}/${test_script}"
    
    # Check if test script exists
    if [[ ! -f "$full_script_path" ]]; then
        error "Test script not found: $full_script_path"
    fi
    
    # Prepare test command
    local test_cmd="python3 $full_script_path --iterations $ITERATIONS"
    
    if [[ "$ONLOAD_ENABLED" != "true" ]]; then
        test_cmd="$test_cmd --no-onload"
    fi
    
    # Run with CPU affinity if specified
    if [[ -n "$TRADING_CORES" ]]; then
        if [[ "$ONLOAD_ENABLED" == "true" ]]; then
            # Run with OnLoad and CPU affinity
            test_cmd="taskset -c $TRADING_CORES onload-trading $test_cmd"
        else
            # Run with CPU affinity only
            test_cmd="taskset -c $TRADING_CORES $test_cmd"
        fi
    elif [[ "$ONLOAD_ENABLED" == "true" ]]; then
        # Run with OnLoad only
        test_cmd="onload $test_cmd"
    fi
    
    log "Executing test command: $test_cmd"
    
    # Execute the test
    local test_output="${RESULTS_DIR}/python_test_output.log"
    local test_exit_code=0
    
    eval "$test_cmd" 2>&1 | tee "$test_output" || test_exit_code=$?
    
    if [[ $test_exit_code -eq 0 ]]; then
        log "Python latency test completed successfully"
    else
        log "Python latency test completed with warnings (exit code: $test_exit_code)"
    fi
    
    # Copy results file if generated
    local results_pattern="${script_dir}/onload_test_results_*.json"
    for results_file in $results_pattern; do
        if [[ -f "$results_file" ]]; then
            cp "$results_file" "$RESULTS_DIR/"
            log "Copied results file: $(basename "$results_file")"
        fi
    done
    
    return $test_exit_code
}

run_socket_micro_benchmark() {
    log "Running socket micro-benchmark..."
    
    local benchmark_file="${RESULTS_DIR}/socket_microbenchmark.log"
    
    # Create simple socket benchmark
    cat > "${RESULTS_DIR}/socket_bench.py" << 'EOF'
#!/usr/bin/env python3
import socket
import time
import sys

def socket_creation_benchmark(iterations=1000):
    """Benchmark socket creation/destruction."""
    times = []
    
    for _ in range(iterations):
        start = time.time_ns()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.close()
        end = time.time_ns()
        times.append((end - start) / 1000.0)  # microseconds
    
    return times

def socket_connect_benchmark(host='127.0.0.1', port=12345, iterations=100):
    """Benchmark socket connection time."""
    # Start a simple server
    import threading
    
    def simple_server():
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(10)
        
        try:
            while True:
                conn, addr = server.accept()
                conn.close()
        except:
            pass
        finally:
            server.close()
    
    server_thread = threading.Thread(target=simple_server)
    server_thread.daemon = True
    server_thread.start()
    
    time.sleep(0.1)  # Allow server to start
    
    times = []
    for _ in range(iterations):
        start = time.time_ns()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            sock.close()
            end = time.time_ns()
            times.append((end - start) / 1000.0)  # microseconds
        except:
            pass
    
    return times

if __name__ == "__main__":
    print("Socket Micro-Benchmark Results")
    print("=" * 40)
    
    # Socket creation benchmark
    creation_times = socket_creation_benchmark()
    if creation_times:
        avg_creation = sum(creation_times) / len(creation_times)
        print(f"Socket creation: {avg_creation:.2f}μs average")
    
    # Socket connection benchmark
    connection_times = socket_connect_benchmark()
    if connection_times:
        avg_connection = sum(connection_times) / len(connection_times)
        print(f"Socket connection: {avg_connection:.2f}μs average")
    
    print(f"Completed {len(creation_times)} creation and {len(connection_times)} connection tests")
EOF
    
    # Run the micro-benchmark
    python3 "${RESULTS_DIR}/socket_bench.py" > "$benchmark_file" 2>&1 || true
    
    log "Socket micro-benchmark completed: $benchmark_file"
}

analyze_test_results() {
    log "Analyzing test results..."
    
    local analysis_file="${RESULTS_DIR}/test_analysis.txt"
    
    {
        echo "=== ONLOAD TRADING TEST ANALYSIS ==="
        echo "Timestamp: $(date)"
        echo "OnLoad Enabled: $ONLOAD_ENABLED"
        echo "Trading Cores: $TRADING_CORES"
        echo "Message Size: $MESSAGE_SIZE bytes"
        echo "Iterations: $ITERATIONS"
        echo
        
        # Analyze JSON results if available
        local json_results
        json_results=($(find "$RESULTS_DIR" -name "onload_test_results_*.json" | head -1))
        
        if [[ ${#json_results[@]} -gt 0 ]] && [[ -f "${json_results[0]}" ]]; then
            echo "=== PERFORMANCE RESULTS ==="
            
            # Extract key metrics using python
            python3 -c "
import json
import sys

try:
    with open('${json_results[0]}', 'r') as f:
        data = json.load(f)
    
    # TCP ping-pong results
    tcp_results = data.get('test_results', {}).get('tcp_ping_pong', {})
    if tcp_results:
        stats = tcp_results.get('statistics', {})
        eval_result = tcp_results.get('evaluation', {})
        
        print('TCP Ping-Pong Test:')
        print(f'  Mean Latency: {stats.get(\"mean\", 0):.3f}μs [{eval_result.get(\"mean_target\", \"UNKNOWN\")}]')
        print(f'  P95 Latency:  {stats.get(\"p95\", 0):.3f}μs [{eval_result.get(\"p95_target\", \"UNKNOWN\")}]')
        print(f'  P99 Latency:  {stats.get(\"p99\", 0):.3f}μs [{eval_result.get(\"p99_target\", \"UNKNOWN\")}]')
        print(f'  Max Latency:  {stats.get(\"max_val\", 0):.3f}μs')
        print(f'  Sample Count: {stats.get(\"sample_count\", 0)}')
        print(f'  Overall:      [{eval_result.get(\"overall\", \"UNKNOWN\")}]')
        print()
    
    # UDP results
    udp_results = data.get('test_results', {}).get('udp_ping_pong', {})
    if udp_results:
        stats = udp_results.get('statistics', {})
        eval_result = udp_results.get('evaluation', {})
        
        print('UDP Ping-Pong Test:')
        print(f'  Mean Latency: {stats.get(\"mean\", 0):.3f}μs [{eval_result.get(\"mean_target\", \"UNKNOWN\")}]')
        print(f'  P95 Latency:  {stats.get(\"p95\", 0):.3f}μs [{eval_result.get(\"p95_target\", \"UNKNOWN\")}]')
        print(f'  P99 Latency:  {stats.get(\"p99\", 0):.3f}μs [{eval_result.get(\"p99_target\", \"UNKNOWN\")}]')
        print()
    
    # Overall assessment
    assessment = data.get('overall_assessment', {})
    if assessment:
        print('Overall Assessment:')
        print(f'  Performance Grade: {assessment.get(\"performance_grade\", \"UNKNOWN\")}')
        print(f'  Tests Passed: {assessment.get(\"tests_passed\", 0)}')
        print(f'  Tests Failed: {assessment.get(\"tests_failed\", 0)}')
        
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            print('  Recommendations:')
            for rec in recommendations:
                print(f'    • {rec}')

except Exception as e:
    print(f'Error analyzing results: {e}')
    sys.exit(1)
" 2>/dev/null || echo "Could not analyze JSON results"
        fi
        
        echo
        echo "=== SYSTEM INFORMATION ==="
        echo "CPU Info:"
        grep "model name" /proc/cpuinfo | head -1 || echo "CPU info not available"
        echo "Memory Info:"
        grep -E "MemTotal|MemAvailable" /proc/meminfo || echo "Memory info not available"
        echo "Network Interfaces:"
        ip addr show | grep -E "^[0-9]+:" | awk '{print $2}' | tr -d ':' || echo "Interface info not available"
        
        echo
        echo "=== LOG FILES GENERATED ==="
        ls -la "$RESULTS_DIR"/ 2>/dev/null || echo "No result files found"
        
    } > "$analysis_file"
    
    log "Test analysis completed: $analysis_file"
    
    # Display summary
    echo
    echo "=== TEST SUMMARY ==="
    cat "$analysis_file" | grep -A 20 "=== PERFORMANCE RESULTS ===" || echo "No performance summary available"
    echo
    echo "Full analysis available at: $analysis_file"
}

cleanup_test_environment() {
    log "Cleaning up test environment..."
    
    # Kill any remaining test processes
    pkill -f "socket_bench.py" 2>/dev/null || true
    pkill -f "comprehensive_onload_test.py" 2>/dev/null || true
    
    # Remove temporary files
    rm -f "${RESULTS_DIR}/socket_bench.py" 2>/dev/null || true
    
    log "Test environment cleanup completed"
}

usage() {
    cat << EOF
OnLoad Trading Test Wrapper

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --onload-enabled     Enable OnLoad acceleration (default: true)
    --onload-disabled    Disable OnLoad (baseline test)
    --cores=X,Y          Trading cores for CPU affinity (default: $TRADING_CORES)
    --iterations=N       Number of test iterations (default: $ITERATIONS)
    --message-size=N     Message size in bytes (default: $MESSAGE_SIZE)
    --verbose           Enable verbose output
    --help              Show this help

TESTS PERFORMED:
    1. Baseline network performance measurement
    2. Comprehensive Python latency testing
    3. Socket micro-benchmarks
    4. Results analysis and reporting

PERFORMANCE TARGETS:
    Mean Latency: ≤4.37μs
    P95 Latency:  ≤4.53μs
    P99 Latency:  ≤4.89μs

EXAMPLES:
    $0                                    # Run with OnLoad enabled
    $0 --onload-disabled                 # Baseline test without OnLoad
    $0 --cores=4,5 --iterations=20000    # Custom configuration
    $0 --verbose                         # Detailed output

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --onload-enabled)
            ONLOAD_ENABLED=true
            shift
            ;;
        --onload-disabled)
            ONLOAD_ENABLED=false
            shift
            ;;
        --cores=*)
            TRADING_CORES="${1#*=}"
            shift
            ;;
        --iterations=*)
            ITERATIONS="${1#*=}"
            shift
            ;;
        --message-size=*)
            MESSAGE_SIZE="${1#*=}"
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Main execution
log "Starting OnLoad trading test suite..."
log "Configuration: OnLoad=$ONLOAD_ENABLED, Cores=$TRADING_CORES, Iterations=$ITERATIONS"

# Trap cleanup on exit
trap cleanup_test_environment EXIT

# Execute test sequence
check_dependencies
setup_test_environment
run_baseline_network_test
run_socket_micro_benchmark

# Run main test and capture exit code
test_exit_code=0
run_python_latency_test || test_exit_code=$?

analyze_test_results

log "OnLoad trading test suite completed"
log "Results directory: $RESULTS_DIR"

if [[ $test_exit_code -eq 0 ]]; then
    log "All tests completed successfully"
else
    log "Tests completed with warnings (some targets may not have been met)"
fi

exit $test_exit_code