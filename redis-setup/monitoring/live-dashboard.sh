#!/bin/bash
# Redis HFT Live Monitoring Dashboard
# Usage: ./monitoring/live-dashboard.sh [interval_seconds]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

INTERVAL=${1:-10}  # Default 10 seconds

echo "Redis HFT Live Dashboard - Press Ctrl+C to exit"
echo "Update interval: ${INTERVAL} seconds"
echo

trap 'echo "Dashboard stopped."; exit 0' INT TERM

while true; do
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "  REDIS HFT LIVE DASHBOARD - $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"
    
    # Performance Gate Status
    echo -n "Performance Gate: "
    if ./production/perf-gate.sh --metrics-only >/dev/null 2>&1; then
        echo "🟢 PASS"
    else
        exit_code=$?
        case $exit_code in
            1) echo "🟡 SOFT FAIL" ;;
            2) echo "🔴 HARD FAIL" ;;
            *) echo "❓ ERROR" ;;
        esac
    fi
    
    # Current Metrics
    METRICS=$(./monitoring/redis-hft-monitor_to_json.sh 2>/dev/null || echo '{"error":"failed"}')
    echo
    echo "📊 CURRENT PERFORMANCE:"
    echo "─────────────────────────────────────────────────"
    
    if echo "$METRICS" | jq -e .error >/dev/null 2>&1; then
        echo "❌ Monitoring failed"
    else
        P50=$(echo "$METRICS" | jq -r '.rtt.p50')
        P95=$(echo "$METRICS" | jq -r '.rtt.p95') 
        P99=$(echo "$METRICS" | jq -r '.rtt.p99')
        JITTER=$(echo "$METRICS" | jq -r '.rtt.jitter')
        MEM=$(echo "$METRICS" | jq -r '.health.mem_used_human')
        OPS=$(echo "$METRICS" | jq -r '.health.ops_per_sec')
        
        # Color code based on thresholds
        P99_COLOR="🟢"
        if (( $(echo "$P99 > 15" | bc -l) )); then P99_COLOR="🟡"; fi
        if (( $(echo "$P99 > 20" | bc -l) )); then P99_COLOR="🔴"; fi
        
        JITTER_COLOR="🟢"
        if (( $(echo "$JITTER > 3" | bc -l) )); then JITTER_COLOR="🟡"; fi
        if (( $(echo "$JITTER > 5" | bc -l) )); then JITTER_COLOR="🔴"; fi
        
        OPS_COLOR="🟢"
        if (( $(echo "$OPS < 15000" | bc -l) )); then OPS_COLOR="🟡"; fi
        if (( $(echo "$OPS < 10000" | bc -l) )); then OPS_COLOR="🔴"; fi
        
        printf "P50: %8.2fμs   P95: %8.2fμs   P99: %s %8.2fμs\n" "$P50" "$P95" "$P99_COLOR" "$P99"
        printf "Jitter: %s %6.2fμs   Memory: %8s   Ops/sec: %s %8.0f\n" "$JITTER_COLOR" "$JITTER" "$MEM" "$OPS_COLOR" "$OPS"
    fi
    
    # System Status
    echo
    echo "🔍 SYSTEM STATUS:"
    echo "─────────────────────────────────────────────────"
    REDIS_STATUS=$(systemctl is-active redis-hft 2>/dev/null || echo "unknown")
    LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
    
    if [[ "$REDIS_STATUS" == "active" ]]; then
        echo -n "Redis: 🟢 Active"
    else
        echo -n "Redis: 🔴 $REDIS_STATUS"
    fi
    
    echo "      Load: $LOAD      Memory: $(free -h | grep Mem | awk '{print $3"/"$2}')"
    
    # Connection Test
    if redis-cli ping >/dev/null 2>&1; then
        echo "Connection: 🟢 PONG"
    else
        echo "Connection: 🔴 FAILED"
    fi
    
    echo
    echo "────────────────────────────────────────────────────────────"
    echo "Next update in ${INTERVAL}s... (Ctrl+C to exit)"
    
    sleep $INTERVAL
done