#!/bin/bash
# Redis HFT Tuning - Quick Rollback Script
# Created: September 28, 2025
# Purpose: Instant rollback from any tuning configuration

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${SCRIPT_DIR}/backups/tuning_rollback_$(date +%Y%m%d_%H%M%S)"
REDIS_CONFIG="/opt/redis-hft/config/redis-hft.conf"
ORIGINAL_BACKUP="${SCRIPT_DIR}/backups/original_redis_hft_conf_backup.txt"

echo "🚨 REDIS HFT TUNING ROLLBACK INITIATED"
echo

# Check if original backup exists
if [[ ! -f "$ORIGINAL_BACKUP" ]]; then
    echo "❌ ERROR: Original backup not found at $ORIGINAL_BACKUP"
    echo "Cannot safely rollback without original configuration."
    exit 1
fi

# Create emergency backup of current state
echo "1. Creating emergency backup of current state..."
mkdir -p "$BACKUP_DIR"
sudo cp "$REDIS_CONFIG" "$BACKUP_DIR/redis-hft.conf.pre-rollback"
echo "   Backup created: $BACKUP_DIR/"

# Restore original configuration
echo "2. Restoring original Redis configuration..."
sudo cp "$ORIGINAL_BACKUP" "$REDIS_CONFIG"
echo "   Original configuration restored"

# Restart Redis service
echo "3. Restarting Redis service..."
sudo systemctl restart redis-hft
sleep 3

# Verify service is running
if sudo systemctl is-active redis-hft >/dev/null; then
    echo "✅ Redis service restarted successfully"
else
    echo "❌ ERROR: Redis service failed to start after rollback"
    echo "Manual intervention required!"
    exit 1
fi

# Test connectivity
echo "4. Testing Redis connectivity..."
if redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis responding to ping"
else
    echo "❌ ERROR: Redis not responding after rollback"
    exit 1
fi

# Quick performance test
echo "5. Quick performance validation..."
cd "$SCRIPT_DIR"
PERF_RESULT=$(../monitoring/redis-hft-monitor_to_json.sh 2>/dev/null || echo "FAILED")

if [[ "$PERF_RESULT" == "FAILED" ]]; then
    echo "⚠️  WARNING: Performance test failed, but basic connectivity OK"
else
    echo "${PERF_RESULT}" | grep -E '"p99":[0-9]+' | head -1 || echo "Performance data collected"
    echo "✅ Basic performance test completed"
fi

echo
echo "🎯 ROLLBACK COMPLETE"
echo "   - Original configuration restored"
echo "   - Redis service operational"
echo "   - Emergency backup: $BACKUP_DIR/"
echo
echo "Next steps:"
echo "   - Run full performance gate: ./perf-gate.sh"
echo "   - Review what went wrong with tuning changes"
echo "   - Consider more conservative approach"