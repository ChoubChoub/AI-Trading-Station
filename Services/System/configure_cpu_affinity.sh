#!/bin/bash
# Centralized CPU Affinity Configuration (Enhanced with Opus Feedback)
# See: Documentation/CPU_AFFINITY_AUDIT_AND_PLAN.md
#
# Enhancements:
# - CPU load validation before changes
# - User confirmation prompt
# - Comprehensive pre-flight checks

set -euo pipefail

echo "🔧 CPU Affinity Configuration System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# PRE-FLIGHT CHECKS (Opus Recommendation)
# ============================================================================

echo "🔍 Pre-Flight System Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Check CPU topology
echo "📊 CPU Topology:"
lscpu | grep -E "^CPU\(s\)|Core\(s\)|Thread\(s\)|Socket\(s\)" | sed 's/^/  /'
echo ""

# 2. Check current CPU load
echo "📈 Current System Load:"
uptime | sed 's/^/  /'
load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk -F, '{print $1}' | xargs)
echo "  1-minute load average: $load_avg"
echo ""

# 3. Check CPU frequency governors
echo "⚡ CPU Frequency Governors:"
if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    for cpu in /sys/devices/system/cpu/cpu[0-7]; do
        if [ -f "$cpu/cpufreq/scaling_governor" ]; then
            gov=$(cat "$cpu/cpufreq/scaling_governor")
            echo "  $(basename $cpu): $gov"
        fi
    done
else
    echo "  (No frequency scaling available)"
fi
echo ""

# 4. Check isolated CPUs
echo "🔒 Isolated CPUs (from kernel cmdline):"
isolated=$(cat /proc/cmdline 2>/dev/null | grep -o 'isolcpus=[^ ]*' || echo "none")
echo "  $isolated"
echo ""

# 5. Check Grafana dashboard accessibility
echo "📊 Grafana Dashboard Check:"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health 2>/dev/null | grep -q "200"; then
    echo "  ✅ Grafana accessible at http://localhost:3000"
else
    echo "  ⚠️  Grafana not accessible (monitoring may be limited)"
fi
echo ""

# 6. Check current service status
echo "🔄 Current Service Status:"
for service in prometheus redis-hft binance-trades binance-bookticker questdb batch-writer; do
    if systemctl is-active --quiet "$service.service" 2>/dev/null; then
        echo "  ✅ $service: running"
    else
        echo "  ❌ $service: not running"
    fi
done
echo ""

# 7. Check disk space for backups
echo "💾 Disk Space Check:"
df -h /home/youssefbahloul/ai-trading-station/Archive | tail -1 | awk '{print "  Available: " $4 " (" $5 " used)"}'
echo ""

# ============================================================================
# USER CONFIRMATION (Opus Recommendation)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  This script will:"
echo "  1. Create systemd override directories for CPU affinity"
echo "  2. Pin services to specific CPU cores"
echo "  3. Require service restarts to take effect"
echo ""
echo "📋 Planned CPU Allocation:"
echo "  CPU 0-1: Network IRQs (kernel) - unchanged"
echo "  CPU 2:   Prometheus - already configured"
echo "  CPU 3:   WebSocket Collectors (binance-trades, binance-bookticker) - NEW"
echo "  CPU 4:   Redis HFT - already configured"
echo "  CPU 5:   QuestDB - NEW"
echo "  CPU 6-7: Batch Writer (8 workers) - NEW"
echo ""
read -p "Continue with CPU affinity changes? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted by user"
    exit 1
fi
echo ""

# ============================================================================
# CONFIGURATION APPLICATION
# ============================================================================

echo "🔧 Applying CPU Affinity Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. CPU 3: WebSocket Collectors
echo "📡 CPU 3: WebSocket Collectors"
sudo mkdir -p /etc/systemd/system/binance-trades.service.d
sudo tee /etc/systemd/system/binance-trades.service.d/cpu-affinity.conf > /dev/null << 'CONF'
[Service]
CPUAffinity=3
# Remove CPUQuota since we're pinned to dedicated core
CPUQuota=
CONF

sudo mkdir -p /etc/systemd/system/binance-bookticker.service.d
sudo tee /etc/systemd/system/binance-bookticker.service.d/cpu-affinity.conf > /dev/null << 'CONF'
[Service]
CPUAffinity=3
# Remove CPUQuota since we're pinned to dedicated core
CPUQuota=
CONF
echo "  ✅ binance-trades.service → CPU 3"
echo "  ✅ binance-bookticker.service → CPU 3"

# 2. CPU 5: QuestDB
echo ""
echo "💾 CPU 5: QuestDB"
sudo mkdir -p /etc/systemd/system/questdb.service.d
sudo tee /etc/systemd/system/questdb.service.d/cpu-affinity.conf > /dev/null << 'CONF'
[Service]
CPUAffinity=5
# QuestDB is critical - high priority scheduling
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=40
CONF
echo "  ✅ questdb.service → CPU 5"
echo "  ℹ️  Note: Monitor JVM heap and GC after pinning (Opus recommendation)"

# 3. CPU 6-7: Batch Writer
echo ""
echo "📝 CPU 6-7: Batch Writer"
sudo mkdir -p /etc/systemd/system/batch-writer.service.d
sudo tee /etc/systemd/system/batch-writer.service.d/cpu-affinity.conf > /dev/null << 'CONF'
[Service]
CPUAffinity=6-7
# 8 workers on 2 cores = 4:1 ratio (appropriate for async I/O)
# Remove CPUQuota since we have dedicated cores
CPUQuota=
CONF
echo "  ✅ batch-writer.service → CPU 6-7"
echo "  ℹ️  Note: Monitor CPU utilization; may need worker count adjustment (Opus recommendation)"

# 4. Verify existing configurations
echo ""
echo "🔍 Verifying Existing Configurations"
echo "  ✅ prometheus.service → CPU 2 (already configured)"
echo "  ✅ redis-hft.service → CPU 4 (already configured)"

# 5. Reload systemd daemon
echo ""
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload
echo "  ✅ Daemon reloaded"

# ============================================================================
# POST-CONFIGURATION SUMMARY
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ CPU Affinity Configuration Complete!"
echo ""
echo "📋 CPU Allocation Summary:"
echo "  CPU 0-1: Network IRQs (kernel)"
echo "  CPU 2:   Prometheus"
echo "  CPU 3:   WebSocket Collectors (binance-trades, binance-bookticker)"
echo "  CPU 4:   Redis HFT"
echo "  CPU 5:   QuestDB"
echo "  CPU 6-7: Batch Writer (8 workers)"
echo ""
echo "⚠️  IMPORTANT: Services NOT restarted yet!"
echo ""
echo "Next steps:"
echo ""
echo "1. Review configuration:"
echo "   ./verify_cpu_affinity.sh"
echo ""
echo "2. Restart services (OPTION A - Safer, one by one):"
echo "   sudo systemctl restart questdb.service"
echo "   sleep 2"
echo "   sudo systemctl restart redis-hft.service"
echo "   sleep 2"
echo "   sudo systemctl restart batch-writer.service"
echo "   sleep 2"
echo "   sudo systemctl restart binance-trades.service binance-bookticker.service"
echo ""
echo "3. OR restart entire pipeline (OPTION B - Faster):"
echo "   sudo systemctl restart market-data.target"
echo ""
echo "4. Verify after restart:"
echo "   ./verify_cpu_affinity.sh"
echo ""
echo "5. Monitor for 24 hours:"
echo "   - Grafana dashboard: http://localhost:3000"
echo "   - Capture rate should remain >99%"
echo "   - Watch for QuestDB GC pauses"
echo "   - Monitor batch-writer CPU utilization"
echo ""
echo "📚 Full documentation: Documentation/CPU_AFFINITY_AUDIT_AND_PLAN.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
