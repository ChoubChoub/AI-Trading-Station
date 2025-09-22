#!/bin/bash

echo "=== CORE ISOLATION MONITORING REPORT ==="
echo "Date: $(date)"
echo "Uptime: $(uptime -p)"
echo ""

# Check isolated cores parameter from kernel command line
ISOLATED_CORES=$(cat /proc/cmdline | grep -o 'isolcpus=[0-9,]*' | cut -d= -f2 2>/dev/null || echo "Not set")
echo "Isolated cores from cmdline: $ISOLATED_CORES"

# Check if isolation is active
if [ -f /sys/devices/system/cpu/isolated ]; then
    ACTIVE_ISOLATED=$(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo "Not available")
    echo "Currently isolated cores: $ACTIVE_ISOLATED"
else
    echo "Isolated cores status: Not available (check cmdline parameter)"
fi

echo ""
echo "=== CPU UTILIZATION PER CORE ==="
if command -v mpstat >/dev/null 2>&1; then
    mpstat -P ALL 1 1 2>/dev/null | grep -E "Average|CPU" || echo "mpstat data not available"
else
    echo "mpstat not available"
fi

echo ""
echo "=== CPU GOVERNORS ==="
if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    echo "Current governors per core:"
    for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [ -f "$i" ]; then
            core=$(echo "$i" | grep -o 'cpu[0-9]*')
            governor=$(cat "$i")
            echo "  $core: $governor"
        fi
    done | head -10
else
    echo "CPU frequency scaling not available"
fi

echo ""
echo "=== IRQ DISTRIBUTION ==="
echo "IRQ distribution across cores (top 15):"
grep -E "CPU|eth|enp|TxRx|network" /proc/interrupts | head -15

echo ""
echo "=== NETWORK INTERFACE STATUS ==="
PRIMARY_NIC=$(ip route | grep default | awk '{print $5}' | head -1)
if [ -n "$PRIMARY_NIC" ]; then
    echo "Primary interface: $PRIMARY_NIC"
    ip link show "$PRIMARY_NIC" 2>/dev/null || echo "Interface details not available"
    
    if command -v ethtool >/dev/null 2>&1; then
        echo "Link status:"
        ethtool "$PRIMARY_NIC" 2>/dev/null | grep -E "Speed|Duplex|Link detected" || echo "ethtool info not available"
    fi
else
    echo "Primary interface not detected"
fi

echo ""
echo "=== PROCESSES ON ISOLATED CORES ==="
if command -v ps >/dev/null 2>&1; then
    echo "Processes running on cores 2 and 3:"
    ps -eLo pid,tid,psr,comm | grep -E "^\s*[0-9]*\s*[0-9]*\s*[23]\s" | head -10 || echo "No processes found on isolated cores"
else
    echo "Process monitoring not available"
fi

echo ""
echo "=== SYSTEM LOAD AND MEMORY ==="
echo "Load average: $(cat /proc/loadavg)"
echo "Memory usage:"
free -h

echo ""
echo "=== IRQ AFFINITY LOG ==="
if [ -f /var/log/nic-irq-affinity.log ]; then
    echo "IRQ affinity configuration:"
    tail -10 /var/log/nic-irq-affinity.log
else
    echo "IRQ affinity log not available"
fi

echo ""
echo "=== MONITORING COMPLETE ==="

