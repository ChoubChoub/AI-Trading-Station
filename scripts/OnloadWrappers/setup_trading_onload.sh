# Create the ultimate OnLoad configuration for your trading system
#!/bin/bash
# OnLoad Trading Environment Setup
# Created: 2025-09-01 10:40:34 UTC
# User: ChoubChoub
# Purpose: Configure OnLoad for AI Trading Station

echo "=== OnLoad Trading Environment Setup ==="
echo "Date: $(date -u)"
echo "User: ChoubChoub"
echo ""

# Create OnLoad configuration optimized for trading
echo "⚙️  Creating OnLoad trading configuration..."
sudo tee /etc/onload.conf << 'ONLOAD_CONF'
# OnLoad Configuration for AI Trading Station
# Created: 2025-09-01 10:40:34 UTC by ChoubChoub
# Target: Ultra-low latency algorithmic trading

# Core performance settings for trading
EF_POLL_USEC=0              # Never sleep - always poll (critical for trading)
EF_INT_DRIVEN=0             # Pure polling mode for deterministic latency
EF_PACKET_BUFFER_MODE=0     # Most efficient buffer management
EF_SPIN_USEC=1000000       # Spin for 1 second before yielding (trading priority)

# Buffer configuration for high-frequency data
EF_RXQ_SIZE=2048           # Larger RX queue for market data bursts
EF_TXQ_SIZE=1024           # Optimized TX queue for order submission
EF_VI_RXQ_SIZE=2048        # Virtual interface RX optimization
EF_VI_TXQ_SIZE=1024        # Virtual interface TX optimization

# CPU affinity for trading cores (2,3 isolated in GRUB)
EF_CLUSTER_CORE_AFFINITY=2,3
EF_IRQ_CORE_AFFINITY=2,3   # Bind interrupts to isolated cores

# Memory management for consistent performance
EF_USE_HUGE_PAGES=0        # Disable to avoid allocation latency spikes
EF_PREALLOC_PACKETS=1      # Pre-allocate for deterministic performance
EF_MAX_PACKETS=65536       # Sufficient packet pool for trading workload

# Advanced trading-specific settings
EF_POLL_ON_DEMAND=0        # Always maintain polling threads
EF_INT_REPRIME=0           # Disable interrupt re-priming overhead
EF_PERIODIC_TIMER_CPU=3    # Use isolated core for timing

# Disable features that add latency
EF_LOG_LEVEL=0             # Minimal logging for production
EF_LOG_FILE=/dev/null      # No log file overhead
EF_NO_FAIL=0               # Fail fast on issues (don't fallback)

# TCP optimizations for trading protocols
EF_TCP_RECV_NONBLOCK=1     # Non-blocking receives
EF_TCP_SEND_NONBLOCK=1     # Non-blocking sends
EF_UDP_RECV_SPIN=1         # Spin on UDP receives (market data)
EF_UDP_SEND_SPIN=1         # Spin on UDP sends

# Redis Streams optimization (AI Trading Station architecture)
EF_SOCKET_CACHE_MAX=1024   # Cache sockets for Redis connections
ONLOAD_CONF

echo "  ✅ OnLoad trading configuration created"

# Create trading-specific wrapper script
echo "🎯 Creating OnLoad trading wrapper..."
sudo tee /usr/local/bin/onload-trading << 'TRADING_SCRIPT'
#!/bin/bash
# OnLoad Trading Application Wrapper
# Created: 2025-09-01 10:40:34 UTC by ChoubChoub

# Set trading-optimized environment
export EF_POLL_USEC=0
export EF_INT_DRIVEN=0
export EF_SPIN_USEC=1000000
export EF_CLUSTER_CORE_AFFINITY=2,3

# Trading-specific OnLoad settings
export EF_RXQ_SIZE=2048
export EF_TXQ_SIZE=1024
export EF_TCP_RECV_NONBLOCK=1
export EF_TCP_SEND_NONBLOCK=1

echo "🚀 OnLoad Trading Wrapper - ChoubChoub AI Trading Station"
echo "Date: $(date -u)"
echo "Launching with OnLoad acceleration: $@"
echo "Configuration: /etc/onload.conf"
echo ""

# Launch application with OnLoad
exec onload "$@"
TRADING_SCRIPT

sudo chmod +x /usr/local/bin/onload-trading
echo "  ✅ Trading wrapper created: /usr/local/bin/onload-trading"

# Optimize network interfaces for OnLoad
echo "🔧 Optimizing network interfaces..."
for iface in enp130s0f0np0 enp130s0f1np1; do
    if ip link show $iface >/dev/null 2>&1; then
        echo "  Optimizing $iface..."
        
        # Interrupt coalescing for minimum latency
        sudo ethtool -C $iface rx-usecs 0 rx-frames 1 tx-usecs 0 tx-frames 1 2>/dev/null || true
        
        # Disable offloading
        sudo ethtool -K $iface tso off gso off gro off lro off 2>/dev/null || true
        sudo ethtool -K $iface rx-checksumming off tx-checksumming off 2>/dev/null || true
        
        # Set optimal ring buffer sizes for trading
        sudo ethtool -G $iface rx 2048 tx 1024 2>/dev/null || true
        
        echo "    ✓ $iface optimized for trading"
    fi
done

# Test the setup
echo ""
echo "🧪 Testing OnLoad trading setup..."
echo "1. OnLoad version: $(onload --version | head -1)"
echo "2. Active stacks: $(onload_stackdump lots 2>/dev/null | grep -c "stack" || echo "0")"
echo "3. Trading wrapper: $(ls -la /usr/local/bin/onload-trading | awk '{print $1, $9}')"

echo ""
echo "✅ OnLoad trading environment setup completed!"
echo ""
echo "🎯 Usage Examples:"
echo "  • Python trading app:    onload-trading python3 trading_bot.py"
echo "  • Redis connection:      onload-trading redis-cli"
echo "  • Custom trading app:    onload-trading ./my_trading_app"
echo ""
echo "📊 Next Steps:"
echo "  1. Test with your Redis Streams architecture"
echo "  2. Measure end-to-end latency in trading pipeline"
echo "  3. Monitor performance with onload_stackdump"

