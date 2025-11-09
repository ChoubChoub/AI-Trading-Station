#!/bin/bash
# Toggle monitoring mode between trading (active) and not trading
#
# Usage:
#   ./set_monitoring_mode.sh trading       # Enable trading mode (no GPU blocking)
#   ./set_monitoring_mode.sh notrading     # Enable monitoring mode (with GPU benchmarks)
#   ./set_monitoring_mode.sh status        # Show current mode

MODE="${1:-status}"
CONFIG_FILE="$HOME/ai-trading-station/Services/Monitoring/Config/monitor_config.json"

# Function to update production_mode in JSON config
update_config_mode() {
    local new_value=$1
    
    # Use jq if available, otherwise use sed
    if command -v jq &> /dev/null; then
        # Create temp file with updated config
        jq ".monitoring.production_mode = $new_value" "$CONFIG_FILE" > "$CONFIG_FILE.tmp"
        mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
    else
        # Fallback to sed (less robust but works)
        sed -i "s/\"production_mode\": [^,]*/\"production_mode\": $new_value/" "$CONFIG_FILE"
    fi
}

# Function to read current mode from config
read_config_mode() {
    if command -v jq &> /dev/null; then
        jq -r '.monitoring.production_mode' "$CONFIG_FILE"
    else
        # Fallback to grep/sed
        grep -o '"production_mode": [^,]*' "$CONFIG_FILE" | sed 's/.*: //'
    fi
}

case "$MODE" in
    trading)
        echo "🔒 ENABLING TRADING MODE - TRADING ACTIVE"
        echo "   ├─ GPU benchmarks: DISABLED (no blocking)"
        echo "   ├─ Latency: Estimated from typical values"
        echo "   ├─ TFLOPS: Estimated from locked clocks"
        echo "   └─ Memory/Clocks: REAL-TIME (lightweight)"
        update_config_mode "true"
        echo ""
        echo "✅ Trading mode enabled in $CONFIG_FILE"
        echo "   Dashboard will use estimates on next collection cycle (~30s)."
        echo "   No dashboard restart needed - config reloaded automatically."
        ;;
        
    notrading)
        echo "📊 ENABLING MONITORING MODE - TRADING NOT ACTIVE"
        echo "   ├─ GPU benchmarks: ENABLED (2-3s blocking every 30s)"
        echo "   ├─ Latency: Live measurement (50 iterations)"
        echo "   ├─ TFLOPS: Live measurement (every 30s)"
        echo "   └─ Memory/Clocks: REAL-TIME"
        update_config_mode "false"
        echo ""
        echo "✅ Monitoring mode enabled in $CONFIG_FILE"
        echo "   Dashboard will use live benchmarks on next collection cycle (~30s)."
        echo "   No dashboard restart needed - config reloaded automatically."
        ;;
        
    status)
        CURRENT_MODE=$(read_config_mode)
        if [ "$CURRENT_MODE" = "true" ]; then
            echo "📊 Current Mode: 🔒 TRADING ACTIVE"
            echo "   GPU blocking: DISABLED"
            echo "   Monitoring: Estimates + lightweight metrics"
        else
            echo "📊 Current Mode: 📊 TRADING NOT ACTIVE"
            echo "   GPU blocking: ~2s every 30s"
            echo "   Monitoring: Live benchmarks + all metrics"
        fi
        echo ""
        echo "Config file: $CONFIG_FILE"
        echo ""
        echo "To change mode:"
        echo "  trading-mode    # Enable trading mode (estimates only)"
        echo "  monitor-mode    # Enable monitoring mode (live benchmarks)"
        ;;
        
    *)
        echo "❌ Invalid mode: $MODE"
        echo "Usage: $0 {trading|notrading|status}"
        exit 1
        ;;
esac

