#!/bin/bash
# Verify CPU Affinity for All Services (Enhanced with Opus Feedback)
# See: Documentation/CPU_AFFINITY_AUDIT_AND_PLAN.md
#
# Enhancements:
# - Service dependency check
# - CPU frequency governor display
# - Context switch monitoring suggestions

set -euo pipefail

echo "🔍 CPU Affinity Verification Report"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Generated: $(date)"
echo ""

# ============================================================================
# SYSTEMD CONFIGURATION
# ============================================================================

echo "📋 Systemd CPUAffinity Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for service in prometheus redis-hft binance-trades binance-bookticker questdb batch-writer; do
    affinity=$(systemctl show -p CPUAffinity "$service.service" 2>/dev/null | cut -d= -f2)
    if [ -n "$affinity" ] && [ "$affinity" != "" ]; then
        echo "  ✅ $service: $affinity"
    else
        echo "  ❌ $service: (not set)"
    fi
done
echo ""

# ============================================================================
# ACTUAL PROCESS AFFINITY
# ============================================================================

check_service() {
    local service=$1
    local expected=$2
    local pid=$(systemctl show -p MainPID "$service" 2>/dev/null | cut -d= -f2)
    
    if [ "$pid" = "0" ] || [ -z "$pid" ]; then
        echo "  ❌ $service: Not running"
        return 1
    fi
    
    local actual=$(taskset -cp "$pid" 2>/dev/null | awk '{print $NF}')
    
    if [ "$actual" = "$expected" ]; then
        echo "  ✅ $service (PID $pid): CPU $actual"
    else
        echo "  ⚠️  $service (PID $pid): Expected CPU $expected, Got CPU $actual"
    fi
}

echo "🎯 Actual Process Affinity:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_service "prometheus.service" "2"
check_service "redis-hft.service" "4"
check_service "binance-trades.service" "3"
check_service "binance-bookticker.service" "3"
check_service "questdb.service" "5"
check_service "batch-writer.service" "6,7"
echo ""

# ============================================================================
# SERVICE DEPENDENCIES (Opus Recommendation)
# ============================================================================

echo "🔗 Service Dependencies:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if systemctl list-dependencies market-data.target --all 2>/dev/null | grep -qE "(binance|batch|quest|redis)"; then
    systemctl list-dependencies market-data.target --all 2>/dev/null | grep -E "(binance|batch|quest|redis)" | sed 's/^/  /'
else
    echo "  market-data.target dependencies:"
    systemctl list-dependencies market-data.target 2>/dev/null | grep "●" | sed 's/^/  /'
fi
echo ""

# ============================================================================
# CPU FREQUENCY GOVERNORS (Opus Recommendation)
# ============================================================================

echo "⚡ CPU Frequency Governors:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    for cpu in /sys/devices/system/cpu/cpu[0-7]; do
        if [ -f "$cpu/cpufreq/scaling_governor" ]; then
            gov=$(cat "$cpu/cpufreq/scaling_governor")
            freq=$(cat "$cpu/cpufreq/scaling_cur_freq" 2>/dev/null || echo "N/A")
            if [ "$freq" != "N/A" ]; then
                freq_mhz=$((freq / 1000))
                echo "  $(basename $cpu): $gov (${freq_mhz} MHz)"
            else
                echo "  $(basename $cpu): $gov"
            fi
        fi
    done
    echo ""
    
    # Check if performance governor is set
    non_perf=$(grep -L "performance" /sys/devices/system/cpu/cpu[0-7]/cpufreq/scaling_governor 2>/dev/null | wc -l)
    if [ "$non_perf" -gt 0 ]; then
        echo "  ⚠️  Warning: Some CPUs not using 'performance' governor"
        echo "     For HFT workloads, consider: sudo cpupower frequency-set -g performance"
    fi
else
    echo "  (No CPU frequency scaling available)"
fi
echo ""

# ============================================================================
# CPU UTILIZATION
# ============================================================================

echo "📊 Current CPU Utilization:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v mpstat &> /dev/null; then
    mpstat -P ALL 1 1 | grep -E "CPU|Average" | grep -v "Linux" | sed 's/^/  /'
else
    echo "  (mpstat not available - install sysstat package)"
    echo "  Current load: $(uptime | awk -F'load average:' '{print $2}')"
fi
echo ""

# ============================================================================
# MEMORY USAGE
# ============================================================================

echo "💾 Process Memory Usage:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for service in prometheus redis-hft binance-trades binance-bookticker questdb batch-writer; do
    pid=$(systemctl show -p MainPID "$service.service" 2>/dev/null | cut -d= -f2)
    if [ "$pid" != "0" ] && [ -n "$pid" ]; then
        mem=$(ps -p "$pid" -o rss= 2>/dev/null || echo "0")
        mem_mb=$((mem / 1024))
        echo "  $service: ${mem_mb} MB"
    fi
done
echo ""

# ============================================================================
# MONITORING SUGGESTIONS (Opus Recommendations)
# ============================================================================

echo "📈 Advanced Monitoring Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Monitor IRQ distribution (network interrupts):"
echo "   watch -n 5 'cat /proc/interrupts | grep -E \"CPU|enp130s0\"'"
echo ""
echo "2. Check process migration events (batch-writer):"
echo "   perf stat -e migrations -p \$(pgrep -f batch-writer) sleep 60"
echo ""
echo "3. Monitor cache efficiency (QuestDB):"
echo "   perf stat -e cache-misses,cache-references -p \$(pgrep -f questdb) sleep 60"
echo ""
echo "4. Watch context switches (WebSocket collectors):"
echo "   pidstat -w 5 -p \$(pgrep -f 'binance.*collector')"
echo ""
echo "5. Monitor per-CPU usage continuously:"
echo "   mpstat -P ALL 2"
echo ""
echo "6. Check CPU affinity for all market data processes:"
echo "   for pid in \$(pgrep -f 'binance|batch-writer|questdb'); do"
echo "     echo \"PID \$pid: \$(taskset -cp \$pid 2>/dev/null | awk '{print \$NF}')\""
echo "   done"
echo ""

# ============================================================================
# HEALTH CHECK
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏥 Overall Health Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

all_running=true
all_pinned=true

for service in prometheus redis-hft binance-trades binance-bookticker questdb batch-writer; do
    if ! systemctl is-active --quiet "$service.service" 2>/dev/null; then
        all_running=false
        echo "  ⚠️  $service is not running"
    fi
    
    affinity=$(systemctl show -p CPUAffinity "$service.service" 2>/dev/null | cut -d= -f2)
    if [ -z "$affinity" ] || [ "$affinity" = "" ]; then
        all_pinned=false
    fi
done

echo ""
if [ "$all_running" = true ] && [ "$all_pinned" = true ]; then
    echo "  ✅ All services running with CPU affinity configured"
    echo ""
    echo "  Next steps:"
    echo "  1. Monitor Grafana dashboard: http://localhost:3000"
    echo "  2. Verify capture rate remains >99%"
    echo "  3. Watch for QuestDB GC pauses (first 24 hours)"
    echo "  4. Monitor batch-writer CPU utilization"
    echo "  5. Check context switches for WebSocket collectors"
elif [ "$all_running" = false ]; then
    echo "  ⚠️  Some services are not running"
    echo ""
    echo "  Start services with:"
    echo "  sudo systemctl start market-data.target"
elif [ "$all_pinned" = false ]; then
    echo "  ℹ️  Some services don't have CPU affinity configured"
    echo ""
    echo "  Configure with:"
    echo "  ./configure_cpu_affinity.sh"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Report complete: $(date)"
