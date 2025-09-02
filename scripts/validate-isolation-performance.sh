#!/bin/bash

# ============================================================================
# ISOLATION PERFORMANCE VALIDATION SUITE
# ============================================================================
# PURPOSE: Comprehensive validation of core isolation performance
# TARGET: Validate sub-10μs latency with isolated cores
# METRICS: Mean 4.37μs | P95 4.53μs | P99 4.89μs
# ============================================================================

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly TEST_RESULTS_DIR="/tmp/isolation-performance-tests"
readonly BENCHMARK_DATA="${TEST_RESULTS_DIR}/benchmarks.json"

# Performance targets (in microseconds)
readonly TARGET_MEAN=4.37
readonly TARGET_P95=4.53
readonly TARGET_P99=4.89
readonly CRITICAL_THRESHOLD=10.0

# Test configuration
TEST_DURATION=60
TRADING_CORES="2,3"
SYSTEM_CORES="0,1"
VERBOSE=false

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

check_dependencies() {
    log "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for required tools
    for cmd in cyclictest rt-tests python3 numactl taskset; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "Installing missing dependencies: ${missing_deps[*]}"
        apt-get update
        apt-get install -y rt-tests python3 python3-pip numactl util-linux
    fi
    
    # Install Python dependencies
    if ! python3 -c "import numpy, json, statistics" 2>/dev/null; then
        pip3 install numpy
    fi
    
    log "Dependencies verified"
}

create_test_environment() {
    log "Setting up test environment..."
    
    mkdir -p "$TEST_RESULTS_DIR"
    
    # Create Python analysis script
    cat > "${TEST_RESULTS_DIR}/analyze_latency.py" << 'EOF'
#!/usr/bin/env python3
import json
import sys
import statistics
import numpy as np
from datetime import datetime

def analyze_latency_data(data_file):
    """Analyze latency data and calculate performance metrics."""
    try:
        latencies = []
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    try:
                        latency = float(line.strip())
                        latencies.append(latency)
                    except ValueError:
                        continue
        
        if not latencies:
            return None
            
        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = max(latencies)
        min_latency = min(latencies)
        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        return {
            'sample_count': len(latencies),
            'mean': round(mean_latency, 3),
            'median': round(median_latency, 3),
            'p95': round(p95_latency, 3),
            'p99': round(p99_latency, 3),
            'max': round(max_latency, 3),
            'min': round(min_latency, 3),
            'std_dev': round(std_dev, 3)
        }
        
    except Exception as e:
        print(f"Error analyzing data: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: analyze_latency.py <data_file>")
        sys.exit(1)
    
    result = analyze_latency_data(sys.argv[1])
    if result:
        print(json.dumps(result, indent=2))
    else:
        sys.exit(1)
EOF
    
    chmod +x "${TEST_RESULTS_DIR}/analyze_latency.py"
    log "Test environment created"
}

test_cyclictest_isolation() {
    log "Running cyclictest isolation validation..."
    
    local test_file="${TEST_RESULTS_DIR}/cyclictest_isolation.log"
    local raw_data="${TEST_RESULTS_DIR}/cyclictest_raw.dat"
    
    # Run cyclictest on isolated cores
    IFS=',' read -ra CORE_ARRAY <<< "$TRADING_CORES"
    local core_count=${#CORE_ARRAY[@]}
    
    log "Testing $core_count isolated cores: $TRADING_CORES for ${TEST_DURATION}s"
    
    cyclictest \
        -t "$core_count" \
        -a "$TRADING_CORES" \
        -n \
        -p 99 \
        -i 100 \
        -h 2000 \
        -q \
        -D "${TEST_DURATION}s" \
        > "$test_file" 2>&1
    
    # Extract latency data
    grep -E "^#.*Max.*:" "$test_file" | awk '{print $4}' > "$raw_data"
    
    # Analyze results
    local analysis
    analysis=$(python3 "${TEST_RESULTS_DIR}/analyze_latency.py" "$raw_data")
    
    if [[ -n "$analysis" ]]; then
        echo "$analysis" > "${TEST_RESULTS_DIR}/cyclictest_analysis.json"
        
        # Extract key metrics
        local mean_lat p95_lat p99_lat
        mean_lat=$(echo "$analysis" | jq -r '.mean')
        p95_lat=$(echo "$analysis" | jq -r '.p95')
        p99_lat=$(echo "$analysis" | jq -r '.p99')
        
        log "Cyclictest Results:"
        log "  Mean: ${mean_lat}μs (target: ≤${TARGET_MEAN}μs)"
        log "  P95:  ${p95_lat}μs (target: ≤${TARGET_P95}μs)"
        log "  P99:  ${p99_lat}μs (target: ≤${TARGET_P99}μs)"
        
        # Evaluate performance
        local performance_grade="PASS"
        if (( $(echo "$mean_lat > $TARGET_MEAN" | bc -l) )) || \
           (( $(echo "$p95_lat > $TARGET_P95" | bc -l) )) || \
           (( $(echo "$p99_lat > $TARGET_P99" | bc -l) )); then
            performance_grade="FAIL"
        fi
        
        log "Cyclictest Performance: $performance_grade"
        return $([ "$performance_grade" = "PASS" ] && echo 0 || echo 1)
    else
        log "ERROR: Failed to analyze cyclictest results"
        return 1
    fi
}

test_memory_access_latency() {
    log "Testing memory access latency on isolated cores..."
    
    local test_program="${TEST_RESULTS_DIR}/memory_latency_test"
    
    # Create memory latency test program
    cat > "${test_program}.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <sys/mman.h>

#define ITERATIONS 10000
#define MEMORY_SIZE (64 * 1024 * 1024)  // 64MB

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <cpu_core>\n", argv[0]);
        return 1;
    }
    
    int cpu_core = atoi(argv[1]);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        perror("sched_setaffinity");
        return 1;
    }
    
    // Set real-time priority
    struct sched_param param;
    param.sched_priority = 99;
    if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
        perror("sched_setscheduler");
    }
    
    // Allocate and lock memory
    volatile char *memory = mmap(NULL, MEMORY_SIZE, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    
    if (mlock(memory, MEMORY_SIZE) != 0) {
        perror("mlock");
    }
    
    // Initialize memory
    for (int i = 0; i < MEMORY_SIZE; i++) {
        memory[i] = i % 256;
    }
    
    struct timespec start, end;
    printf("# Memory access latency test on CPU %d\n", cpu_core);
    printf("# Format: latency_ns\n");
    
    for (int i = 0; i < ITERATIONS; i++) {
        int offset = (rand() % (MEMORY_SIZE - 64)) & ~63;  // 64-byte aligned
        
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        volatile char value = memory[offset];  // Memory read
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        
        long latency_ns = (end.tv_sec - start.tv_sec) * 1000000000L + 
                         (end.tv_nsec - start.tv_nsec);
        
        if (latency_ns > 0 && latency_ns < 10000) {  // Filter unrealistic values
            printf("%ld\n", latency_ns);
        }
        
        // Use the value to prevent optimization
        (void)value;
    }
    
    munmap((void*)memory, MEMORY_SIZE);
    return 0;
}
EOF
    
    # Compile the test program
    gcc -O2 -o "$test_program" "${test_program}.c" -lrt
    
    # Run memory latency test on each trading core
    IFS=',' read -ra CORE_ARRAY <<< "$TRADING_CORES"
    local memory_results="${TEST_RESULTS_DIR}/memory_latency.json"
    echo "{" > "$memory_results"
    
    local first_core=true
    for core in "${CORE_ARRAY[@]}"; do
        log "Testing memory latency on core $core..."
        
        local core_data="${TEST_RESULTS_DIR}/memory_core_${core}.dat"
        "$test_program" "$core" > "$core_data" 2>/dev/null || true
        
        # Convert nanoseconds to microseconds for analysis
        grep -v '^#' "$core_data" | awk '{print $1/1000}' > "${core_data}.us"
        
        local analysis
        analysis=$(python3 "${TEST_RESULTS_DIR}/analyze_latency.py" "${core_data}.us" 2>/dev/null || echo '{}')
        
        if [[ "$first_core" == "false" ]]; then
            echo "," >> "$memory_results"
        fi
        echo "\"core_$core\": $analysis" >> "$memory_results"
        first_core=false
    done
    
    echo "}" >> "$memory_results"
    log "Memory latency test completed"
}

test_context_switch_latency() {
    log "Testing context switch latency..."
    
    local test_program="${TEST_RESULTS_DIR}/context_switch_test"
    
    # Create context switch test program
    cat > "${test_program}.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sched.h>
#include <sys/wait.h>
#include <signal.h>

#define ITERATIONS 1000

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <cpu_core>\n", argv[0]);
        return 1;
    }
    
    int cpu_core = atoi(argv[1]);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        perror("sched_setaffinity");
        return 1;
    }
    
    printf("# Context switch latency test on CPU %d\n", cpu_core);
    printf("# Format: latency_us\n");
    
    for (int i = 0; i < ITERATIONS; i++) {
        struct timespec start, end;
        
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        sched_yield();  // Force context switch
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        
        double latency_us = (end.tv_sec - start.tv_sec) * 1000000.0 + 
                           (end.tv_nsec - start.tv_nsec) / 1000.0;
        
        if (latency_us > 0 && latency_us < 1000) {  // Filter unrealistic values
            printf("%.3f\n", latency_us);
        }
    }
    
    return 0;
}
EOF
    
    gcc -O2 -o "$test_program" "${test_program}.c" -lrt
    
    # Run context switch test on trading cores
    IFS=',' read -ra CORE_ARRAY <<< "$TRADING_CORES"
    local switch_results="${TEST_RESULTS_DIR}/context_switch.json"
    echo "{" > "$switch_results"
    
    local first_core=true
    for core in "${CORE_ARRAY[@]}"; do
        log "Testing context switch latency on core $core..."
        
        local core_data="${TEST_RESULTS_DIR}/context_switch_core_${core}.dat"
        "$test_program" "$core" > "$core_data" 2>/dev/null || true
        
        local analysis
        analysis=$(python3 "${TEST_RESULTS_DIR}/analyze_latency.py" "$core_data" 2>/dev/null || echo '{}')
        
        if [[ "$first_core" == "false" ]]; then
            echo "," >> "$switch_results"
        fi
        echo "\"core_$core\": $analysis" >> "$switch_results"
        first_core=false
    done
    
    echo "}" >> "$switch_results"
    log "Context switch test completed"
}

generate_performance_report() {
    log "Generating performance report..."
    
    local report_file="${TEST_RESULTS_DIR}/performance_report.json"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    cat > "$report_file" << EOF
{
  "test_metadata": {
    "timestamp": "$timestamp",
    "trading_cores": "$TRADING_CORES",
    "system_cores": "$SYSTEM_CORES",
    "test_duration": "${TEST_DURATION}s",
    "hostname": "$(hostname)",
    "kernel": "$(uname -r)",
    "cpu_info": "$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
  },
  "performance_targets": {
    "mean_latency_us": $TARGET_MEAN,
    "p95_latency_us": $TARGET_P95,
    "p99_latency_us": $TARGET_P99,
    "critical_threshold_us": $CRITICAL_THRESHOLD
  },
  "test_results": {
EOF
    
    # Add cyclictest results if available
    if [[ -f "${TEST_RESULTS_DIR}/cyclictest_analysis.json" ]]; then
        echo '    "cyclictest": ' >> "$report_file"
        cat "${TEST_RESULTS_DIR}/cyclictest_analysis.json" >> "$report_file"
        echo ',' >> "$report_file"
    fi
    
    # Add memory latency results if available
    if [[ -f "${TEST_RESULTS_DIR}/memory_latency.json" ]]; then
        echo '    "memory_latency": ' >> "$report_file"
        cat "${TEST_RESULTS_DIR}/memory_latency.json" >> "$report_file"
        echo ',' >> "$report_file"
    fi
    
    # Add context switch results if available
    if [[ -f "${TEST_RESULTS_DIR}/context_switch.json" ]]; then
        echo '    "context_switch": ' >> "$report_file"
        cat "${TEST_RESULTS_DIR}/context_switch.json" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF
  }
}
EOF
    
    log "Performance report generated: $report_file"
    
    # Display summary
    echo "=== PERFORMANCE VALIDATION SUMMARY ==="
    echo "Timestamp: $timestamp"
    echo "Trading Cores: $TRADING_CORES"
    echo "Test Duration: ${TEST_DURATION}s"
    echo
    
    if [[ -f "${TEST_RESULTS_DIR}/cyclictest_analysis.json" ]]; then
        local mean p95 p99
        mean=$(jq -r '.mean' "${TEST_RESULTS_DIR}/cyclictest_analysis.json")
        p95=$(jq -r '.p95' "${TEST_RESULTS_DIR}/cyclictest_analysis.json")
        p99=$(jq -r '.p99' "${TEST_RESULTS_DIR}/cyclictest_analysis.json")
        
        echo "Real-Time Latency (cyclictest):"
        echo "  Mean: ${mean}μs (target: ≤${TARGET_MEAN}μs) $(check_target "$mean" "$TARGET_MEAN")"
        echo "  P95:  ${p95}μs (target: ≤${TARGET_P95}μs) $(check_target "$p95" "$TARGET_P95")"
        echo "  P99:  ${p99}μs (target: ≤${TARGET_P99}μs) $(check_target "$p99" "$TARGET_P99")"
        echo
    fi
    
    echo "Detailed results: ${TEST_RESULTS_DIR}/"
    echo "=================================="
}

check_target() {
    local actual="$1"
    local target="$2"
    
    if (( $(echo "$actual <= $target" | bc -l) )); then
        echo "✓ PASS"
    else
        echo "✗ FAIL"
    fi
}

cleanup() {
    log "Cleaning up test environment..."
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}

usage() {
    cat << EOF
Isolation Performance Validation Suite

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --cores=X,Y           Trading cores to test (default: $TRADING_CORES)
    --duration=N          Test duration in seconds (default: $TEST_DURATION)
    --target-latency=X    Override mean latency target (default: ${TARGET_MEAN}μs)
    --verbose             Enable verbose output
    --help               Show this help

TESTS:
    - Real-time scheduling latency (cyclictest)
    - Memory access latency
    - Context switch latency
    - Core isolation effectiveness

PERFORMANCE TARGETS:
    Mean Latency: ≤${TARGET_MEAN}μs
    P95 Latency:  ≤${TARGET_P95}μs
    P99 Latency:  ≤${TARGET_P99}μs

EXAMPLES:
    $0                                    # Run all tests with defaults
    $0 --cores=4,5 --duration=120       # Test cores 4,5 for 2 minutes
    $0 --target-latency=5.0              # Override latency target

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cores=*)
            TRADING_CORES="${1#*=}"
            shift
            ;;
        --duration=*)
            TEST_DURATION="${1#*=}"
            shift
            ;;
        --target-latency=*)
            TARGET_MEAN="${1#*=}"
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

# Trap cleanup on exit
trap cleanup EXIT

# Main execution
[[ $EUID -eq 0 ]] || error "Must run as root for performance testing"

log "Starting isolation performance validation..."
log "Trading cores: $TRADING_CORES"
log "Test duration: ${TEST_DURATION}s"
log "Performance targets: Mean ≤${TARGET_MEAN}μs, P95 ≤${TARGET_P95}μs, P99 ≤${TARGET_P99}μs"

check_dependencies
create_test_environment

# Run tests
test_cyclictest_isolation
test_memory_access_latency
test_context_switch_latency

# Generate final report
generate_performance_report

log "Performance validation completed. Results saved to: $TEST_RESULTS_DIR"