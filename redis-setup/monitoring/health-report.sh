#!/bin/bash
# Redis HFT Health Report - Complete System Status
# Usage: ./monitoring/health-report.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "════════════════════════════════════════════════════════════"
echo "           REDIS HFT HEALTH REPORT"
echo "════════════════════════════════════════════════════════════"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo

# Performance Gate Check
echo "🎯 PERFORMANCE GATE STATUS:"
echo "─────────────────────────────────────────────────────────"
if ./production/perf-gate.sh --metrics-only 2>/dev/null; then
    echo "✅ Performance Gate: PASS"
else
    exit_code=$?
    case $exit_code in
        1) echo "⚠️  Performance Gate: SOFT FAIL (degraded performance)" ;;
        2) echo "❌ Performance Gate: HARD FAIL (environment drift)" ;;
        3) echo "🔧 Performance Gate: CONFIG ERROR" ;;
        *) echo "❓ Performance Gate: UNKNOWN ERROR ($exit_code)" ;;
    esac
fi
echo

# Current Performance Metrics
echo "📊 CURRENT PERFORMANCE METRICS:"
echo "─────────────────────────────────────────────────────────"
METRICS=$(./monitoring/redis-hft-monitor_to_json.sh 2>/dev/null || echo '{"error":"monitoring failed"}')
echo "$METRICS" | jq -r '
if .error then
  "❌ Monitoring Error: " + .error
else
  "RTT Performance:",
  "  P50: " + (.rtt.p50|tostring) + "μs",
  "  P95: " + (.rtt.p95|tostring) + "μs", 
  "  P99: " + (.rtt.p99|tostring) + "μs",
  "  Jitter: " + (.rtt.jitter|tostring) + "μs",
  "",
  "System Health:",
  "  Memory: " + .health.mem_used_human,
  "  Clients: " + (.health.clients|tostring),
  "  Ops/sec: " + (.health.ops_per_sec|tostring)
end' 2>/dev/null || echo "❌ Failed to parse performance metrics"
echo

# System Status
echo "🔍 SYSTEM STATUS:"
echo "─────────────────────────────────────────────────────────"
echo "Redis Service: $(systemctl is-active redis-hft 2>/dev/null || echo 'UNKNOWN')"
echo "Redis Connection: $(redis-cli ping 2>/dev/null || echo 'FAILED')"
echo "Memory Usage: $(free -h | grep Mem | awk '{print $3"/"$2}' 2>/dev/null || echo 'UNKNOWN')"
echo "CPU Load: $(uptime | awk -F'load average:' '{print $2}' | xargs 2>/dev/null || echo 'UNKNOWN')"
echo "CPU Isolation: $(cat /proc/cmdline 2>/dev/null | grep -o 'isolcpus=[^ ]*' || echo 'NOT SET')"
echo

# Recent Performance History
echo "⚡ RECENT PERFORMANCE DECISIONS:"
echo "─────────────────────────────────────────────────────────"
if python3 production/gate_decision_ledger.py --show-recent 5 2>/dev/null; then
    echo "✅ Decision history available"
else
    echo "⚠️  No recent decisions found (system may be new)"
fi
echo

# Tail Analysis Status
echo "📈 TAIL MONITORING STATUS:"
echo "─────────────────────────────────────────────────────────"
if [[ -f "state/tail-run.json" ]]; then
    TAIL_WINDOWS=$(cat state/tail-run.json | jq '.recent_windows | length' 2>/dev/null || echo "0")
    if [[ "$TAIL_WINDOWS" -gt 0 ]]; then
        echo "Tail history available: $TAIL_WINDOWS windows"
        echo "Latest tail metrics:"
        cat state/tail-run.json | jq -r '.recent_windows[-1] | 
          "  P99.9: " + (.p99_9|tostring) + "μs",
          "  Tail Span: " + (.tail_span|tostring) + "μs", 
          "  Classification: " + .classification,
          "  Sample Count: " + (.samples|tostring)' 2>/dev/null || echo "  ⚠️  Failed to parse tail data"
    else
        echo "⚠️  No tail monitoring data available"
    fi
else
    echo "⚠️  Tail monitoring state file not found"
fi
echo

# Summary Assessment
echo "🎯 SYSTEM ASSESSMENT:"
echo "─────────────────────────────────────────────────────────"

# Determine overall status
REDIS_OK=$(systemctl is-active redis-hft 2>/dev/null | grep -q "active" && echo "true" || echo "false")
PERF_OK=$(./production/perf-gate.sh --metrics-only >/dev/null 2>&1 && echo "true" || echo "false")
CONN_OK=$(redis-cli ping 2>/dev/null | grep -q "PONG" && echo "true" || echo "false")

if [[ "$REDIS_OK" == "true" && "$PERF_OK" == "true" && "$CONN_OK" == "true" ]]; then
    echo "🟢 OVERALL STATUS: HEALTHY - System ready for trading"
    echo "✅ All critical systems operational"
elif [[ "$REDIS_OK" == "true" && "$CONN_OK" == "true" ]]; then
    echo "🟡 OVERALL STATUS: DEGRADED - System functional but performance issues"
    echo "⚠️  Performance gate failing - investigate immediately"
else
    echo "🔴 OVERALL STATUS: CRITICAL - System issues detected"
    echo "❌ Redis server or connection problems - immediate attention required"
fi

echo "════════════════════════════════════════════════════════════"
echo "Report generated: $(date)"
echo "Next check recommended: $(date -d '+5 minutes')"
echo "════════════════════════════════════════════════════════════"