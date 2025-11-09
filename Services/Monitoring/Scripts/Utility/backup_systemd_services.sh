#!/bin/bash
# Backup all market data and monitoring service files
# Part of CPU Affinity Management System
# See: Documentation/CPU_AFFINITY_AUDIT_AND_PLAN.md

set -euo pipefail

BACKUP_DIR="/home/youssefbahloul/ai-trading-station/Archive/systemd_services_backup_$(date +%Y%m%d_%H%M%S)"

echo "🗂️  Creating systemd services backup..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup service files and their override directories
echo "📦 Backing up service files..."
sudo cp -r /etc/systemd/system/binance-trades.service* "$BACKUP_DIR/" 2>/dev/null || true
sudo cp -r /etc/systemd/system/binance-bookticker.service* "$BACKUP_DIR/" 2>/dev/null || true
sudo cp -r /etc/systemd/system/batch-writer.service* "$BACKUP_DIR/" 2>/dev/null || true
sudo cp -r /etc/systemd/system/questdb.service* "$BACKUP_DIR/" 2>/dev/null || true
sudo cp -r /etc/systemd/system/prometheus.service* "$BACKUP_DIR/" 2>/dev/null || true
sudo cp -r /etc/systemd/system/redis-hft.service* "$BACKUP_DIR/" 2>/dev/null || true
sudo cp -r /etc/systemd/system/market-data.target "$BACKUP_DIR/" 2>/dev/null || true

# Change ownership to user
sudo chown -R youssefbahloul:youssefbahloul "$BACKUP_DIR"

# Create backup manifest
cat > "$BACKUP_DIR/BACKUP_MANIFEST.txt" << EOF
Systemd Services Backup
=======================
Date: $(date)
Host: $(hostname)
User: $(whoami)

Files Backed Up:
EOF

ls -lh "$BACKUP_DIR" | grep -v "total" | grep -v "BACKUP_MANIFEST" >> "$BACKUP_DIR/BACKUP_MANIFEST.txt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Backup complete: $BACKUP_DIR"
echo ""
echo "📋 Backed up files:"
ls -lh "$BACKUP_DIR" | grep -v "total"
echo ""
echo "💾 Backup size: $(du -sh "$BACKUP_DIR" | cut -f1)"
