#!/bin/bash
set -e

echo "=== CORE ISOLATION ROLLBACK SYSTEM ==="
echo "Timestamp: $(date)"

# Find most recent backup
LATEST_BACKUP=$(ls -t /etc/default/grub.backup.* 2>/dev/null | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "ERROR: No GRUB backup found"
    echo "Available backups:"
    ls -la /etc/default/grub.backup.* 2>/dev/null || echo "None found"
    exit 1
fi

echo "Found backup: $LATEST_BACKUP"

# Create rollback timestamp
ROLLBACK_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup current configuration before rollback
sudo cp /etc/default/grub "/etc/default/grub.pre-rollback.$ROLLBACK_TIMESTAMP"

# Restore GRUB configuration
if sudo cp "$LATEST_BACKUP" /etc/default/grub; then
    echo "✓ GRUB configuration restored from: $LATEST_BACKUP"
else
    echo "✗ ERROR: Failed to restore GRUB configuration"
    exit 1
fi

# Update GRUB
if sudo update-grub; then
    echo "✓ GRUB updated successfully"
else
    echo "✗ ERROR: GRUB update failed"
    # Restore the working configuration
    sudo cp "/etc/default/grub.pre-rollback.$ROLLBACK_TIMESTAMP" /etc/default/grub
    sudo update-grub
    exit 1
fi

# Reset CPU governor to default (if exists)
if [ -f /etc/default/cpupower ]; then
    sudo cp /etc/default/cpupower "/etc/default/cpupower.backup.$ROLLBACK_TIMESTAMP"
    echo 'GOVERNOR="ondemand"' | sudo tee /etc/default/cpupower
    sudo systemctl restart cpupower 2>/dev/null || true
    echo "✓ CPU governor reset to ondemand"
fi

# Disable IRQ affinity service
sudo systemctl disable nic-irq-affinity.service 2>/dev/null || true
echo "✓ IRQ affinity service disabled"

# Remove network optimizations (optional)
read -p "Remove network optimizations? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f /etc/sysctl.d/99-trading-network-performance.conf ]; then
        sudo mv /etc/sysctl.d/99-trading-network-performance.conf "/etc/sysctl.d/99-trading-network-performance.conf.disabled.$ROLLBACK_TIMESTAMP"
        echo "✓ Network optimizations disabled"
    fi
fi

echo ""
echo "=== ROLLBACK COMPLETE ==="
echo "Changes made:"
echo "  - GRUB configuration restored"
echo "  - CPU governor reset to ondemand"
echo "  - IRQ affinity service disabled"
echo ""
echo "REBOOT REQUIRED to take effect"
echo "To reboot now: sudo reboot"
echo ""
echo "Rollback log available at: /var/log/core-isolation-rollback.$ROLLBACK_TIMESTAMP.log"

# Create rollback log
{
    echo "Core Isolation Rollback Log"
    echo "Timestamp: $(date)"
    echo "User: $(whoami)"
    echo "Restored from: $LATEST_BACKUP"
    echo "Pre-rollback backup: /etc/default/grub.pre-rollback.$ROLLBACK_TIMESTAMP"
} | sudo tee "/var/log/core-isolation-rollback.$ROLLBACK_TIMESTAMP.log" > /dev/null

