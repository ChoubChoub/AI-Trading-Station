#!/bin/bash

echo "=== CORE ISOLATION PERFORMANCE VALIDATION ==="
echo "Timestamp: $(date)"
echo ""

# Test 1: Basic network connectivity
echo "=== NETWORK CONNECTIVITY TEST ==="
if command -v ping >/dev/null 2>&1; then
    echo "Testing connectivity to 8.8.8.8:"
    ping -c 3 -W 2 8.8.8.8 2>/dev/null | tail -2 || echo "Ping test failed or unavailable"
else
    echo "Ping command not available"
fi

echo ""

# Test 2: CPU performance on non-isolated cores
echo "=== CPU PERFORMANCE TEST (Non-isolated cores 0,1,4-7) ==="
if command -v sysbench >/dev/null 2>&1; then
    echo "Running CPU benchmark on available cores..."
    AVAILABLE_CORES="0,1,4,5,6,7"
    # Test if cores exist
    ACTUAL_CORES=""
    for core in 0 1 4 5 6 7; do
        if [ -d "/sys/devices/system/cpu/cpu$core" ]; then
            if [ -z "$ACTUAL_CORES" ]; then
                ACTUAL_CORES="$core"
            else
                ACTUAL_CORES="$ACTUAL_CORES,$core"
            fi
        fi
    done
    
    if [ -n "$ACTUAL_CORES" ]; then
        echo "Testing cores: $ACTUAL_CORES"
        taskset -c "$ACTUAL_CORES" sysbench cpu --threads=4 --time=5 run 2>/dev/null | grep -E "events per second|total time" || echo "CPU benchmark failed"
    else
        echo "No suitable cores found for testing"
    fi
else
    echo "sysbench not available for CPU testing"
fi

echo ""

# Test 3: Memory performance
echo "=== MEMORY PERFORMANCE TEST ==="
if command -v sysbench >/dev/null 2>&1; then
    echo "Testing memory bandwidth..."
    sysbench memory --memory-total-size=1G --time=3 run 2>/dev/null | grep -E "transferred|per second" || echo "Memory test failed"
else
    echo "sysbench not available for memory testing"
fi

echo ""

# Test 4: Core isolation effectiveness
echo "=== CORE ISOLATION EFFECTIVENESS ==="
echo "Checking if isolated cores (2,3) are idle..."

# Sample CPU usage for isolated cores
if [ -f /proc/stat ]; then
    echo "CPU statistics for cores 2 and 3:"
    grep -E "cpu[23] " /proc/stat || echo "CPU stats not available for cores 2,3"
fi

# Check load distribution
echo ""
echo "Current system load distribution:"
if command -v mpstat >/dev/null 2>&1; then
    mpstat -P 0,1,2,3 1 1 2>/dev/null | grep Average || echo "Load distribution data not available"
fi

echo ""

# Test 5: Network performance indicators
echo "=== NETWORK PERFORMANCE INDICATORS ==="
PRIMARY_NIC=$(ip route | grep default | awk '{print $5}' | head -1)
if [ -n "$PRIMARY_NIC" ]; then
    echo "Network statistics for $PRIMARY_NIC:"
    cat /proc/net/dev | grep "$PRIMARY_NIC" || echo "Network stats not available"
    
    echo ""
    echo "Network errors and drops:"
    ethtool -S "$PRIMARY_NIC" 2>/dev/null | grep -E "error|drop|discard" | head -5 || echo "Network error stats not available"
fi

echo ""
echo "=== VALIDATION COMPLETE ==="
echo "Summary:"
echo "  - Network connectivity: $(ping -c 1 -W 1 8.8.8.8 >/dev/null 2>&1 && echo "OK" || echo "Failed")"
echo "  - CPU performance test: $(command -v sysbench >/dev/null && echo "Available" || echo "Unavailable")"
echo "  - Core isolation active: $(grep -q "isolcpus" /proc/cmdline && echo "Yes" || echo "No")"
echo "  - Primary NIC: ${PRIMARY_NIC:-"Not detected"}"
