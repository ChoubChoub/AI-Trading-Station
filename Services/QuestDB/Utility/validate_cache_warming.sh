#!/bin/bash
# Post-Reboot Validation Script for Smart Cache Warming
# Tests all critical functionality after system reboot

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     🔄 POST-REBOOT VALIDATION - SMART CACHE WARMING          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

VALIDATION_PASSED=0
VALIDATION_FAILED=0

# Test 1: Check if batch-writer service started automatically
echo "Test 1: Batch-Writer Service Auto-Start"
echo "────────────────────────────────────────────────────────────────"
if systemctl is-active --quiet batch-writer; then
    echo "✅ batch-writer service is running"
    VALIDATION_PASSED=$((VALIDATION_PASSED + 1))
    
    # Get PID and uptime
    PID=$(systemctl show batch-writer -p MainPID --value)
    UPTIME=$(systemctl show batch-writer -p ActiveEnterTimestamp --value)
    echo "   PID: $PID"
    echo "   Started: $UPTIME"
else
    echo "❌ batch-writer service is NOT running"
    VALIDATION_FAILED=$((VALIDATION_FAILED + 1))
fi
echo ""

# Test 2: Check smart warming executed
echo "Test 2: Smart Cache Warming Execution"
echo "────────────────────────────────────────────────────────────────"
WARMING_LOG=$(journalctl -u batch-writer --since "5 minutes ago" | grep "Smart warming complete")
if [ -n "$WARMING_LOG" ]; then
    echo "✅ Smart warming executed on startup"
    VALIDATION_PASSED=$((VALIDATION_PASSED + 1))
    
    # Extract warming stats
    SYMBOLS=$(echo "$WARMING_LOG" | grep -oP '\d+ symbols' | head -1)
    TICKS=$(echo "$WARMING_LOG" | grep -oP '\d+ ticks' | head -1)
    TIME=$(echo "$WARMING_LOG" | grep -oP '\d+ms' | head -1)
    
    echo "   Warmed: $SYMBOLS, $TICKS in $TIME"
    
    # Show detected symbols
    DETECTED=$(journalctl -u batch-writer --since "5 minutes ago" | grep "Most active symbols")
    if [ -n "$DETECTED" ]; then
        echo "   $DETECTED" | sed 's/.*batch-writer\[.*\]: //'
    fi
else
    echo "⚠️  No smart warming found in recent logs"
    echo "   (May still be starting up - check again in 30 seconds)"
    VALIDATION_FAILED=$((VALIDATION_FAILED + 1))
fi
echo ""

# Test 3: Check Prometheus metrics available
echo "Test 3: Prometheus Metrics Server"
echo "────────────────────────────────────────────────────────────────"
if curl -s http://localhost:9092/metrics > /dev/null 2>&1; then
    echo "✅ Prometheus metrics server responding on port 9092"
    VALIDATION_PASSED=$((VALIDATION_PASSED + 1))
    
    # Check for strategic metrics
    METRICS=$(curl -s http://localhost:9092/metrics | grep -E "cache_hot_miss_penalty|cache_warm_latency_impact|cache_query_patterns" | wc -l)
    echo "   Strategic metrics found: $METRICS types"
else
    echo "❌ Prometheus metrics server not responding"
    VALIDATION_FAILED=$((VALIDATION_FAILED + 1))
fi
echo ""

# Test 4: Check QuestDB connection
echo "Test 4: QuestDB Database Connection"
echo "────────────────────────────────────────────────────────────────"
if nc -z localhost 8812 2>/dev/null; then
    echo "✅ QuestDB PostgreSQL wire protocol accessible (port 8812)"
    VALIDATION_PASSED=$((VALIDATION_PASSED + 1))
else
    echo "❌ QuestDB port 8812 not accessible"
    VALIDATION_FAILED=$((VALIDATION_FAILED + 1))
fi
echo ""

# Test 5: Check Redis connection
echo "Test 5: Redis Cache Connection"
echo "────────────────────────────────────────────────────────────────"
if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
    echo "✅ Redis server responding on port 6379"
    VALIDATION_PASSED=$((VALIDATION_PASSED + 1))
    
    # Check for cached data
    KEYS=$(redis-cli -h localhost -p 6379 --scan --pattern "cache:binance_spot:*" | wc -l)
    echo "   Cache keys found: $KEYS"
else
    echo "❌ Redis server not responding"
    VALIDATION_FAILED=$((VALIDATION_FAILED + 1))
fi
echo ""

# Test 6: Check worker activity
echo "Test 6: Batch Writer Workers Processing Data"
echo "────────────────────────────────────────────────────────────────"
WORKER_ACTIVITY=$(journalctl -u batch-writer --since "1 minute ago" | grep -E "Worker [0-7] added" | wc -l)
if [ "$WORKER_ACTIVITY" -gt 0 ]; then
    echo "✅ Workers processing batches (${WORKER_ACTIVITY} batches in last minute)"
    VALIDATION_PASSED=$((VALIDATION_PASSED + 1))
else
    echo "⚠️  No worker activity detected in last minute"
    echo "   (May be low market activity - check again during trading hours)"
    VALIDATION_FAILED=$((VALIDATION_FAILED + 1))
fi
echo ""

# Test 7: Run functional cache test
echo "Test 7: Functional Cache Query Test"
echo "────────────────────────────────────────────────────────────────"
cd /home/youssefbahloul/ai-trading-station/Services/QuestDB/Config

CACHE_TEST=$(timeout 15 python3 << 'PYEOF'
import asyncio
import sys
sys.path.insert(0, '.')
from tiered_market_data import TieredMarketData

async def quick_test():
    tiered = TieredMarketData()
    await tiered.start()
    try:
        ticks, tier, latency = await tiered.get_recent_ticks(
            symbol='BTCUSDT',
            exchange='binance_spot',
            data_type='TRADE',
            seconds=60
        )
        print(f"SUCCESS:{len(ticks)}:{tier}:{latency:.2f}")
        return True
    except Exception as e:
        print(f"FAILED:{str(e)}")
        return False
    finally:
        await tiered.stop()

asyncio.run(quick_test())
PYEOF
)

if echo "$CACHE_TEST" | grep -q "SUCCESS:"; then
    TICKS=$(echo "$CACHE_TEST" | grep "SUCCESS:" | cut -d: -f2)
    TIER=$(echo "$CACHE_TEST" | grep "SUCCESS:" | cut -d: -f3)
    LATENCY=$(echo "$CACHE_TEST" | grep "SUCCESS:" | cut -d: -f4)
    echo "✅ Cache query successful: $TICKS ticks from $TIER tier in ${LATENCY}ms"
    VALIDATION_PASSED=$((VALIDATION_PASSED + 1))
else
    echo "❌ Cache query failed"
    echo "   Error: $CACHE_TEST"
    VALIDATION_FAILED=$((VALIDATION_FAILED + 1))
fi
echo ""

# Test 8: CRITICAL - Redis Connection Leak Check (Opus Recommendation)
echo "Test 8: Redis Connection Leak Detection"
echo "────────────────────────────────────────────────────────────────"
REDIS_CONNS=$(netstat -tn 2>/dev/null | grep ':6379' | wc -l)
if [ "$REDIS_CONNS" -gt 50 ]; then
    echo "❌ FAILED: $REDIS_CONNS connections to Redis (limit: 50)"
    echo "   ⚠️  CONNECTION LEAK DETECTED - Check WarmCache connection pooling"
    echo "   Expected: <20 connections (BlockingConnectionPool with max_connections=20)"
    echo "   Action required: Investigate tiered_market_data.py WarmCache.connect()"
    VALIDATION_FAILED=$((VALIDATION_FAILED + 1))
elif [ "$REDIS_CONNS" -le 20 ]; then
    echo "✅ EXCELLENT: $REDIS_CONNS connections (healthy pool usage)"
    VALIDATION_PASSED=$((VALIDATION_PASSED + 1))
else
    echo "⚠️  WARNING: $REDIS_CONNS connections (acceptable but elevated)"
    echo "   Expected: <20 connections, monitor for growth"
    VALIDATION_PASSED=$((VALIDATION_PASSED + 1))
fi
echo ""

# Final Summary
echo "═══════════════════════════════════════════════════════════════"
echo "VALIDATION SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Tests Passed: $VALIDATION_PASSED / 8"
echo "Tests Failed: $VALIDATION_FAILED / 8"
echo ""

if [ $VALIDATION_FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - System fully operational after reboot"
    echo ""
    echo "Smart Cache Warming Status:"
    echo "  • Auto-start: ✅ Working"
    echo "  • Symbol detection: ✅ Working"
    echo "  • Cache warming: ✅ Working"
    echo "  • Metrics: ✅ Working"
    echo "  • Connection pooling: ✅ Healthy"
    echo "  • Production ready: ✅ YES"
    exit 0
elif [ $VALIDATION_FAILED -le 2 ]; then
    echo "⚠️  PARTIAL SUCCESS - Minor issues detected"
    echo "   Review failed tests above"
    exit 1
else
    echo "❌ VALIDATION FAILED - Critical issues detected"
    echo "   System requires attention"
    exit 2
fi
