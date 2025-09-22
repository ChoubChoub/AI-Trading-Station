#!/bin/bash
# Monitoring launcher using the onload-trading wrapper

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 AI Trading Station - Latency Monitor"
echo "========================================"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check for onload-trading wrapper
if command -v onload-trading &> /dev/null; then
    echo "✅ Using onload-trading wrapper"
    echo ""
    onload-trading python3 "$SCRIPT_DIR/monitor_trading_latency_wrapped.py" "$@"
else
    echo "⚠️  onload-trading wrapper not found"
    echo "   Running without acceleration (results will be inaccurate)"
    echo ""
    python3 "$SCRIPT_DIR/monitor_trading_latency_wrapped.py" "$@"
fi
