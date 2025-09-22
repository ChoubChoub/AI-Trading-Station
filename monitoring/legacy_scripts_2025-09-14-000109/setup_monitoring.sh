#!/bin/bash
#
# Setup script for AI Trading Station monitoring
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
echo "=== AI Trading Station Monitoring Setup ==="
echo ""
# Check Python version
echo "Checking Python version..."
if ! python3 --version | grep -E "3\\.(8|9|10|11|12)" > /dev/null; then
    echo "Error: Python 3.8+ required"
    exit 1
fi
# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install psutil
# Check for nvidia-smi (for GPU monitoring)
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU tools detected"
else
    echo "⚠ nvidia-smi not found - GPU monitoring disabled"
fi
# Check for lm-sensors (for CPU temperature)
if command -v sensors &> /dev/null; then
    echo "✓ lm-sensors detected"
else
    echo "⚠ lm-sensors not found - installing..."
    sudo apt-get update && sudo apt-get install -y lm-sensors
    sudo sensors-detect --auto
fi
# Make scripts executable
echo "Setting script permissions..."
chmod +x "$SCRIPT_DIR"/*.sh
chmod +x "$SCRIPT_DIR"/*.py
# Create log directory
LOG_DIR="/var/log/trading_monitor"
if [[ ! -d "$LOG_DIR" ]]; then
    echo "Creating log directory..."
    sudo mkdir -p "$LOG_DIR"
    sudo chown $USER:$USER "$LOG_DIR"
fi
# Create systemd service (optional)
echo ""
read -p "Install as systemd service? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SERVICE_FILE="/etc/systemd/system/trading-monitor.service"
    sudo tee "$SERVICE_FILE" > /dev/null << 'INNER_EOF'
[Unit]
Description=AI Trading Station Monitor
After=network.target
[Service]
Type=simple
User=$USER
WorkingDirectory=$BASE_DIR
ExecStart=/usr/bin/python3 $SCRIPT_DIR/monitor_trading_system.py
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
INNER_EOF
    sudo systemctl daemon-reload
    sudo systemctl enable trading-monitor.service
    echo "✓ Systemd service installed"
    echo "  Start with: sudo systemctl start trading-monitor"
    echo "  View logs: sudo journalctl -u trading-monitor -f"
fi
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Available monitoring tools:"
echo "  1. Real-time dashboard: ./monitor_dashboard.py"
echo "  2. One-time check: ./monitor_trading_system.py --once"
echo "  3. Continuous monitoring: ./monitor_trading_system.py"
echo "  4. Check IRQ affinity: ./check_irq_affinity.sh"
echo "  5. Fix IRQ affinity: sudo ./fix_irq_affinity.sh"
echo ""
echo "Configuration file: $BASE_DIR/config/monitor_config.json"
