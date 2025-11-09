#!/bin/bash
# Rollback CPU Affinity Changes
# See: Documentation/CPU_AFFINITY_AUDIT_AND_PLAN.md
#
# This script removes all CPU affinity override configurations
# and restarts services to return to their original state

set -euo pipefail

echo "⚠️  CPU Affinity Rollback Procedure"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will:"
echo "  1. Remove CPU affinity overrides for:"
echo "     - binance-trades.service"
echo "     - binance-bookticker.service"
echo "     - questdb.service"
echo "     - batch-writer.service"
echo "  2. Reload systemd daemon"
echo "  3. Restart affected services"
echo ""
echo "Note: Prometheus and Redis-HFT configurations will NOT be touched"
echo "      (they were pre-existing and should remain)"
echo ""

read -p "Continue with rollback? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Rollback aborted by user"
    exit 1
fi
echo ""

echo "🔄 Removing CPU affinity override files..."
echo ""

# Remove override files (but keep directories for future use)
if [ -f /etc/systemd/system/binance-trades.service.d/cpu-affinity.conf ]; then
    sudo rm /etc/systemd/system/binance-trades.service.d/cpu-affinity.conf
    echo "  ✅ Removed binance-trades CPU affinity override"
else
    echo "  ℹ️  binance-trades: no override file found"
fi

if [ -f /etc/systemd/system/binance-bookticker.service.d/cpu-affinity.conf ]; then
    sudo rm /etc/systemd/system/binance-bookticker.service.d/cpu-affinity.conf
    echo "  ✅ Removed binance-bookticker CPU affinity override"
else
    echo "  ℹ️  binance-bookticker: no override file found"
fi

if [ -f /etc/systemd/system/questdb.service.d/cpu-affinity.conf ]; then
    sudo rm /etc/systemd/system/questdb.service.d/cpu-affinity.conf
    echo "  ✅ Removed questdb CPU affinity override"
else
    echo "  ℹ️  questdb: no override file found"
fi

if [ -f /etc/systemd/system/batch-writer.service.d/cpu-affinity.conf ]; then
    sudo rm /etc/systemd/system/batch-writer.service.d/cpu-affinity.conf
    echo "  ✅ Removed batch-writer CPU affinity override"
else
    echo "  ℹ️  batch-writer: no override file found"
fi

echo ""
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload
echo "  ✅ Daemon reloaded"

echo ""
echo "🔄 Restarting services..."
echo ""

# Restart in dependency order
echo "  Restarting questdb.service..."
sudo systemctl restart questdb.service
sleep 2

echo "  Restarting batch-writer.service..."
sudo systemctl restart batch-writer.service
sleep 2

echo "  Restarting binance-trades.service..."
sudo systemctl restart binance-trades.service

echo "  Restarting binance-bookticker.service..."
sudo systemctl restart binance-bookticker.service

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Rollback complete!"
echo ""
echo "Services restored to original configuration."
echo "Processes will now float across all available CPUs."
echo ""
echo "Verify rollback:"
echo "  ./verify_cpu_affinity.sh"
echo ""
echo "Monitor system:"
echo "  - Grafana dashboard: http://localhost:3000"
echo "  - Service status: systemctl status market-data.target"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
