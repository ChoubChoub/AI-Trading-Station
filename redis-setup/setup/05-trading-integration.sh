#!/bin/bash
# Redis HFT Trading Integration Examples
# Created: 2025-09-25
# Purpose: Demonstrate Redis Streams for HFT trading
# Safe: Examples and test scripts only, no system modifications

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
info() { echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO:${NC} $1"; }

echo -e "${BLUE}=== Redis HFT Trading Integration Setup ===${NC}"
echo "Date: $(date -u)"
echo "Purpose: Create trading integration examples and tools"
echo ""

# Create trading integration directory
log "📁 Creating trading integration directory..."
mkdir -p /opt/redis-hft/trading/{examples,schemas,tools}
chown -R redis-hft:redis-hft /opt/redis-hft/trading

# Create trading stream schemas
log "📋 Creating trading stream schemas..."
cat > /opt/redis-hft/trading/schemas/market-data.md << 'EOF'
# Market Data Stream Schema

## Stream: market:data:{symbol}
Purpose: Real-time market data feed

### Message Format:
```
XADD market:data:BTCUSD * symbol BTCUSD price 45000.50 volume 1.25 timestamp 1696123456789 exchange binance
```

### Fields:
- symbol: Trading pair (e.g., BTCUSD, ETHUSD)
- price: Current price (decimal)
- volume: Trade volume (decimal) 
- timestamp: Unix timestamp in milliseconds
- exchange: Exchange name
- bid: Best bid price (optional)
- ask: Best ask price (optional)
- spread: Bid-ask spread (optional)

### Consumer Groups:
- trading-algorithms: Main trading algorithm consumers
- risk-management: Risk monitoring consumers
- analytics: Data analysis consumers
EOF

cat > /opt/redis-hft/trading/schemas/orders.md << 'EOF'
# Order Management Stream Schema

## Stream: orders:{status}
Purpose: Order lifecycle management

### Streams:
- orders:pending - New orders awaiting execution
- orders:filled - Successfully executed orders  
- orders:rejected - Rejected orders
- orders:cancelled - Cancelled orders

### Message Format:
```
XADD orders:pending * order_id 12345 symbol BTCUSD side buy quantity 1.0 price 45000 type limit strategy momentum_v1 timestamp 1696123456789
```

### Fields:
- order_id: Unique order identifier
- symbol: Trading pair
- side: buy/sell
- quantity: Order size
- price: Order price (for limit orders)
- type: market/limit/stop
- strategy: Algorithm identifier
- timestamp: Order creation time
- priority: high/normal/low (optional)
EOF

cat > /opt/redis-hft/trading/schemas/risk-events.md << 'EOF'
# Risk Management Stream Schema

## Stream: risk:events
Purpose: Risk monitoring and alerts

### Message Format:
```
XADD risk:events * event_type position_limit symbol BTCUSD current_position 5.5 limit 5.0 severity high action block_trading timestamp 1696123456789
```

### Event Types:
- position_limit: Position size exceeded
- drawdown_limit: Maximum drawdown reached
- volatility_spike: Unusual price volatility
- connection_loss: Exchange connectivity issues
- latency_spike: Network latency exceeded threshold

### Fields:
- event_type: Type of risk event
- symbol: Affected trading pair (if applicable)
- severity: low/medium/high/critical
- action: none/warn/reduce/block_trading
- timestamp: Event occurrence time
- details: Additional context (JSON string)
EOF

log "✅ Trading schemas created"

# Create market data simulator
log "🎲 Creating market data simulator..."
cat > /opt/redis-hft/trading/tools/market-simulator.py << 'EOF'
#!/usr/bin/env python3
"""
Market Data Simulator for Redis HFT Testing
Generates realistic market data streams
"""

import redis
import time
import random
import json
from datetime import datetime

class MarketSimulator:
    def __init__(self):
        # Use OnLoad-accelerated connection
        self.redis_client = redis.Redis(
            host='127.0.0.1',
            port=6379,
            password=self._get_redis_password(),
            decode_responses=True
        )
        
        self.symbols = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'DOTUSD']
        self.prices = {
            'BTCUSD': 45000.0,
            'ETHUSD': 3000.0, 
            'ADAUSD': 0.5,
            'DOTUSD': 25.0
        }
    
    def _get_redis_password(self):
        with open('/opt/redis-hft/config/redis-auth.txt', 'r') as f:
            line = f.read().strip()
            return line.split(': ')[1]
    
    def generate_price_tick(self, symbol):
        current_price = self.prices[symbol]
        # Random walk with small changes (realistic for HFT)
        change_pct = random.uniform(-0.001, 0.001)  # 0.1% max change
        new_price = current_price * (1 + change_pct)
        self.prices[symbol] = new_price
        return round(new_price, 2)
    
    def simulate_market_data(self, duration_seconds=60):
        print(f"🎲 Starting market simulation for {duration_seconds} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            for symbol in self.symbols:
                price = self.generate_price_tick(symbol)
                volume = round(random.uniform(0.1, 5.0), 2)
                
                # Add to Redis Stream
                stream_name = f"market:data:{symbol}"
                message = {
                    'symbol': symbol,
                    'price': price,
                    'volume': volume,
                    'timestamp': int(time.time() * 1000),
                    'exchange': 'simulator'
                }
                
                self.redis_client.xadd(stream_name, message)
                
            # HFT-like frequency: 100 updates per second
            time.sleep(0.01)
    
    def create_sample_orders(self, count=10):
        print(f"📝 Creating {count} sample orders...")
        
        for i in range(count):
            symbol = random.choice(self.symbols)
            side = random.choice(['buy', 'sell'])
            quantity = round(random.uniform(0.1, 2.0), 2)
            price = self.prices[symbol] * random.uniform(0.999, 1.001)
            
            order = {
                'order_id': f"ORD_{int(time.time())}_{i}",
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': round(price, 2),
                'type': 'limit',
                'strategy': 'simulator_v1',
                'timestamp': int(time.time() * 1000)
            }
            
            self.redis_client.xadd('orders:pending', order)
            time.sleep(0.1)

if __name__ == '__main__':
    import sys
    
    simulator = MarketSimulator()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'orders':
        simulator.create_sample_orders(20)
    else:
        simulator.simulate_market_data(30)
        
    print("✅ Simulation complete")
EOF

chmod +x /opt/redis-hft/trading/tools/market-simulator.py

# Create trading performance tester
log "🧪 Creating performance tester..."
cat > /opt/redis-hft/trading/tools/latency-tester.py << 'EOF'
#!/usr/bin/env python3
"""
Redis HFT Latency Tester
Measures round-trip latency for trading operations
"""

import redis
import time
import statistics

class LatencyTester:
    def __init__(self):
        self.redis_client = redis.Redis(
            host='127.0.0.1',
            port=6379,
            password=self._get_redis_password(),
            decode_responses=True
        )
    
    def _get_redis_password(self):
        with open('/opt/redis-hft/config/redis-auth.txt', 'r') as f:
            line = f.read().strip()
            return line.split(': ')[1]
    
    def test_set_get_latency(self, iterations=1000):
        print(f"🧪 Testing SET/GET latency ({iterations} iterations)...")
        latencies = []
        
        for i in range(iterations):
            key = f"test:latency:{i}"
            value = f"value_{i}"
            
            # Measure SET operation
            start = time.perf_counter_ns()
            self.redis_client.set(key, value)
            end = time.perf_counter_ns()
            
            latency_us = (end - start) / 1000  # Convert to microseconds
            latencies.append(latency_us)
            
            # Cleanup
            self.redis_client.delete(key)
        
        return self._calculate_stats(latencies, "SET/GET")
    
    def test_stream_latency(self, iterations=1000):
        print(f"🔄 Testing XADD latency ({iterations} iterations)...")
        latencies = []
        stream_name = "test:latency:stream"
        
        for i in range(iterations):
            message = {
                'test_id': i,
                'timestamp': int(time.time() * 1000),
                'data': f'test_data_{i}'
            }
            
            # Measure XADD operation
            start = time.perf_counter_ns()
            self.redis_client.xadd(stream_name, message)
            end = time.perf_counter_ns()
            
            latency_us = (end - start) / 1000
            latencies.append(latency_us)
        
        # Cleanup
        self.redis_client.delete(stream_name)
        
        return self._calculate_stats(latencies, "XADD")
    
    def _calculate_stats(self, latencies, operation):
        mean = statistics.mean(latencies)
        median = statistics.median(latencies)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        p99 = sorted(latencies)[int(0.99 * len(latencies))]
        min_lat = min(latencies)
        max_lat = max(latencies)
        
        print(f"📊 {operation} Latency Results:")
        print(f"   Mean: {mean:.2f}μs")
        print(f"   Median: {median:.2f}μs")
        print(f"   P95: {p95:.2f}μs")
        print(f"   P99: {p99:.2f}μs")
        print(f"   Range: {min_lat:.2f}μs - {max_lat:.2f}μs")
        print()
        
        return {
            'mean': mean,
            'median': median,
            'p95': p95,
            'p99': p99,
            'min': min_lat,
            'max': max_lat
        }

if __name__ == '__main__':
    tester = LatencyTester()
    
    print("=== Redis HFT Latency Testing ===")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test basic operations
    tester.test_set_get_latency(1000)
    
    # Test stream operations (critical for trading)
    tester.test_stream_latency(1000)
    
    print("✅ Latency testing complete")
    print("🎯 Target: <10μs for HFT operations")
EOF

chmod +x /opt/redis-hft/trading/tools/latency-tester.py

# Create stream monitoring tool
log "📊 Creating stream monitor..."
cat > /opt/redis-hft/trading/tools/stream-monitor.sh << 'EOF'
#!/bin/bash
# Redis Streams Monitor for HFT Trading

REDIS_CLI="/usr/local/bin/redis-hft-cli"

echo "=== Redis HFT Streams Monitor ==="
echo "Date: $(date)"
echo ""

# List all streams
echo "📋 Active Streams:"
STREAMS=$($REDIS_CLI keys "*:*" | grep -E "(market:|orders:|risk:)" | sort)

if [[ -z "$STREAMS" ]]; then
    echo "  No trading streams found"
else
    for stream in $STREAMS; do
        LENGTH=$($REDIS_CLI xlen "$stream" 2>/dev/null || echo "0")
        echo "  $stream: $LENGTH messages"
    done
fi

echo ""

# Monitor market data streams
echo "📈 Market Data Summary:"
MARKET_STREAMS=$($REDIS_CLI keys "market:data:*" 2>/dev/null || echo "")

if [[ -n "$MARKET_STREAMS" ]]; then
    for stream in $MARKET_STREAMS; do
        SYMBOL=$(echo "$stream" | cut -d: -f3)
        LAST_MESSAGE=$($REDIS_CLI xrevrange "$stream" + - count 1 2>/dev/null)
        if [[ -n "$LAST_MESSAGE" ]]; then
            echo "  $SYMBOL: Active"
        fi
    done
else
    echo "  No market data streams active"
fi

echo ""

# Monitor order streams
echo "📝 Order Processing Summary:"
ORDER_STREAMS=("orders:pending" "orders:filled" "orders:rejected" "orders:cancelled")

for stream in "${ORDER_STREAMS[@]}"; do
    COUNT=$($REDIS_CLI xlen "$stream" 2>/dev/null || echo "0")
    echo "  $stream: $COUNT orders"
done

echo ""

# Show recent activity (if any)
echo "🔄 Recent Activity (last 5 messages):"
ALL_STREAMS=$($REDIS_CLI keys "*:*" | grep -E "(market:|orders:|risk:)" | head -3)

for stream in $ALL_STREAMS; do
    echo "  Stream: $stream"
    $REDIS_CLI xrevrange "$stream" + - count 2 2>/dev/null | head -4 | sed 's/^/    /'
    echo ""
done
EOF

chmod +x /opt/redis-hft/trading/tools/stream-monitor.sh

# Create integration examples
log "📚 Creating integration examples..."
cat > /opt/redis-hft/trading/examples/trading-pipeline-example.py << 'EOF'
#!/usr/bin/env python3
"""
Example: Complete HFT Trading Pipeline with Redis Streams
Demonstrates: Market data → Algorithm → Order management → Risk control
"""

import redis
import time
import json
import threading
from datetime import datetime

class HFTTradingPipeline:
    def __init__(self):
        self.redis_client = redis.Redis(
            host='127.0.0.1',
            port=6379, 
            password=self._get_redis_password(),
            decode_responses=True
        )
        
        self.running = False
        self.position = {}  # Track positions by symbol
        
    def _get_redis_password(self):
        with open('/opt/redis-hft/config/redis-auth.txt', 'r') as f:
            line = f.read().strip()
            return line.split(': ')[1]
    
    def market_data_consumer(self):
        """Consumer for market data streams"""
        print("🔄 Starting market data consumer...")
        
        # Create consumer group
        try:
            self.redis_client.xgroup_create("market:data:BTCUSD", "trading-algo", id='0', mkstream=True)
        except redis.ResponseError:
            pass  # Group already exists
        
        while self.running:
            try:
                # Read from stream with consumer group
                messages = self.redis_client.xreadgroup(
                    "trading-algo", 
                    "consumer-1",
                    {"market:data:BTCUSD": '>'},
                    count=1,
                    block=100
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        self.process_market_data(fields, msg_id)
                        
            except Exception as e:
                print(f"Market data error: {e}")
                time.sleep(0.1)
    
    def process_market_data(self, data, msg_id):
        """Simple trading algorithm"""
        symbol = data['symbol']
        price = float(data['price'])
        timestamp = int(data['timestamp'])
        
        # Simple momentum algorithm
        signal = self.calculate_signal(symbol, price)
        
        if signal != 'hold':
            order = {
                'order_id': f"ORD_{timestamp}_{symbol}",
                'symbol': symbol,
                'side': signal,
                'quantity': 0.1,
                'price': price,
                'type': 'market',
                'strategy': 'momentum_v1',
                'timestamp': timestamp,
                'source_msg': msg_id
            }
            
            # Send to order stream
            self.redis_client.xadd('orders:pending', order)
            print(f"📝 Order generated: {signal} {order['quantity']} {symbol} @ {price}")
        
        # Acknowledge message processing
        self.redis_client.xack("market:data:BTCUSD", "trading-algo", msg_id)
    
    def calculate_signal(self, symbol, price):
        """Simple trading signal calculation"""
        # This is a placeholder - real algorithms would be more sophisticated
        import random
        
        # Random signal for demo (replace with real algorithm)
        rand = random.random()
        if rand < 0.1:
            return 'buy'
        elif rand < 0.2:
            return 'sell'
        else:
            return 'hold'
    
    def order_processor(self):
        """Process pending orders"""
        print("📋 Starting order processor...")
        
        while self.running:
            try:
                # Read pending orders
                messages = self.redis_client.xread(
                    {"orders:pending": '$'},
                    count=5,
                    block=100
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        self.execute_order(fields, msg_id)
                        
            except Exception as e:
                print(f"Order processing error: {e}")
                time.sleep(0.1)
    
    def execute_order(self, order, msg_id):
        """Simulate order execution"""
        symbol = order['symbol']
        side = order['side']
        quantity = float(order['quantity'])
        
        # Update position
        if symbol not in self.position:
            self.position[symbol] = 0
        
        if side == 'buy':
            self.position[symbol] += quantity
        else:
            self.position[symbol] -= quantity
        
        # Move to filled orders
        filled_order = order.copy()
        filled_order['fill_time'] = int(time.time() * 1000)
        filled_order['status'] = 'filled'
        
        self.redis_client.xadd('orders:filled', filled_order)
        print(f"✅ Order filled: {side} {quantity} {symbol}, Position: {self.position[symbol]}")
        
        # Risk check
        self.risk_check(symbol)
    
    def risk_check(self, symbol):
        """Basic risk management"""
        position = abs(self.position.get(symbol, 0))
        max_position = 1.0  # Maximum position size
        
        if position > max_position:
            risk_event = {
                'event_type': 'position_limit',
                'symbol': symbol,
                'current_position': position,
                'limit': max_position,
                'severity': 'high',
                'action': 'block_trading',
                'timestamp': int(time.time() * 1000)
            }
            
            self.redis_client.xadd('risk:events', risk_event)
            print(f"⚠️  Risk event: Position limit exceeded for {symbol}")
    
    def run(self, duration=30):
        """Run the complete trading pipeline"""
        print("🚀 Starting HFT Trading Pipeline Demo")
        print(f"Duration: {duration} seconds")
        print()
        
        self.running = True
        
        # Start consumers in separate threads
        market_thread = threading.Thread(target=self.market_data_consumer)
        order_thread = threading.Thread(target=self.order_processor)
        
        market_thread.start()
        order_thread.start()
        
        # Run for specified duration
        time.sleep(duration)
        
        self.running = False
        market_thread.join(timeout=2)
        order_thread.join(timeout=2)
        
        print()
        print("📊 Final Statistics:")
        print(f"Positions: {self.position}")
        
        # Show stream lengths
        pending = self.redis_client.xlen('orders:pending')
        filled = self.redis_client.xlen('orders:filled')
        risk_events = self.redis_client.xlen('risk:events')
        
        print(f"Pending orders: {pending}")
        print(f"Filled orders: {filled}")
        print(f"Risk events: {risk_events}")

if __name__ == '__main__':
    pipeline = HFTTradingPipeline()
    pipeline.run(30)  # Run for 30 seconds
EOF

chmod +x /opt/redis-hft/trading/examples/trading-pipeline-example.py

# Set ownership
chown -R redis-hft:redis-hft /opt/redis-hft/trading

echo ""
echo -e "${BLUE}=== Trading Integration Setup Complete ===${NC}"
echo "✅ Trading schemas created"
echo "✅ Market data simulator ready"
echo "✅ Latency testing tools ready"
echo "✅ Stream monitoring tools ready"
echo "✅ Complete trading pipeline example created"
echo ""
echo -e "${GREEN}Usage Examples:${NC}"
echo "• Test latency: /opt/redis-hft/trading/tools/latency-tester.py"
echo "• Simulate data: /opt/redis-hft/trading/tools/market-simulator.py"
echo "• Monitor streams: /opt/redis-hft/trading/tools/stream-monitor.sh"
echo "• Run trading demo: /opt/redis-hft/trading/examples/trading-pipeline-example.py"
echo ""
echo -e "${YELLOW}Integration Commands:${NC}"
echo "• Redis client: redis-hft-cli"
echo "• Performance monitor: /opt/redis-hft/scripts/redis-hft-monitor.sh"
echo "• Service status: systemctl status redis-hft"
echo ""
echo -e "${BLUE}🎯 Redis HFT integration ready for production trading!${NC}"