#!/bin/bash
# ============================================================================
# AI Trading Station - Monitoring Setup and Aliases
# ============================================================================
# PURPOSE: Setup monitoring aliases and shortcuts for traders
# USAGE: source setup_monitoring.sh
# ============================================================================

# Get the absolute path to monitoring directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Make monitoring scripts executable
chmod +x "$SCRIPT_DIR/monitor_dashboard_complete.py"
chmod +x "$SCRIPT_DIR/monitor_trading_system_optimized.py"

# Define monitoring aliases
alias monitordash="python3 $SCRIPT_DIR/monitor_dashboard_complete.py"
alias monitorcore="python3 $SCRIPT_DIR/monitor_trading_system_optimized.py"
alias monitorstatus="python3 $SCRIPT_DIR/monitor_dashboard_complete.py --export /tmp/monitor_status.json && cat /tmp/monitor_status.json"

# Define convenience functions for traders
monitor_quick() {
    echo "=== AI Trading Station - Quick Status ==="
    echo "OnLoad Wrapper: $REPO_ROOT/scripts/onload-trading"
    echo "Dashboard: monitordash"
    echo "Core Monitor: monitorcore"
    echo "Configuration: $SCRIPT_DIR/monitoring_config.json"
    echo ""
    
    # Check if onload is available
    if command -v onload &> /dev/null; then
        echo "✓ OnLoad: Available"
    else
        echo "✗ OnLoad: Not installed"
    fi
    
    # Check wrapper
    if [ -x "$REPO_ROOT/scripts/onload-trading" ]; then
        echo "✓ OnLoad Wrapper: Ready"
    else
        echo "✗ OnLoad Wrapper: Missing or not executable"
    fi
    
    # Check Python dependencies
    if python3 -c "import psutil" 2>/dev/null; then
        echo "✓ Python psutil: Available"
    else
        echo "⚠ Python psutil: Missing (pip install psutil)"
    fi
    
    echo ""
    echo "Usage:"
    echo "  monitordash      # Launch real-time dashboard"
    echo "  monitorcore      # Launch optimized core monitor"
    echo "  monitor_quick    # Show this status"
}

# Export functions so they're available in shell
export -f monitor_quick

echo "=== AI Trading Station Monitoring Setup ==="
echo "Monitoring aliases installed:"
echo "  monitordash      - Launch monitoring dashboard"
echo "  monitorcore      - Launch optimized monitor"
echo "  monitorstatus    - Export and show current status"
echo "  monitor_quick    - Quick status check"
echo ""
echo "Configuration: $SCRIPT_DIR/monitoring_config.json"
echo "OnLoad Wrapper: $REPO_ROOT/scripts/onload-trading"
echo ""
echo "Usage examples:"
echo "  monitordash                    # Interactive dashboard"
echo "  monitorcore --export data.json # Export 60s of metrics"
echo "  monitor_quick                  # Quick system check"
echo ""
echo "To make aliases permanent, add this to your ~/.bashrc:"
echo "  source $SCRIPT_DIR/setup_monitoring.sh"