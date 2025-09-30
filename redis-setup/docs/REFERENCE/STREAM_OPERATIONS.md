# Redis Streams Trimming Guide for HFT Trading

## 🎯 **Critical Implementation Note**

With `noeviction` policy configured for deterministic performance, **your trading applications MUST implement stream trimming** to prevent memory exhaustion.

## 📊 **When You'll Need This**

When implementing market data feeds, order streams, or any time-series data in Redis:

```python
import redis

r = redis.Redis(host='127.0.0.1', port=6379, password='your_redis_password')

# ✅ CORRECT: Auto-trim to keep only last 10,000 entries
r.xadd("market_data:EURUSD", {
    "price": 1.0847,
    "volume": 1000000,
    "timestamp": time.time()
}, maxlen=10000, approximate=True)

# ❌ WRONG: No trimming - will fill memory and hit noeviction limit
r.xadd("market_data:EURUSD", {
    "price": 1.0847,
    "volume": 1000000,  
    "timestamp": time.time()
})
```

## 🔧 **Implementation Examples**

### **Market Data Feed:**
```python
# Keep last 1 hour of tick data (assuming 1 tick/second)
MARKET_DATA_MAXLEN = 3600

def publish_market_tick(symbol, price, volume):
    r.xadd(f"market_data:{symbol}", {
        "price": price,
        "volume": volume,
        "timestamp": time.time()
    }, maxlen=MARKET_DATA_MAXLEN, approximate=True)
```

### **Order Events:**
```python
# Keep last 1000 order events per symbol
ORDER_EVENTS_MAXLEN = 1000

def publish_order_event(symbol, order_id, event_type, details):
    r.xadd(f"orders:{symbol}", {
        "order_id": order_id,
        "event": event_type,
        "details": json.dumps(details),
        "timestamp": time.time()
    }, maxlen=ORDER_EVENTS_MAXLEN, approximate=True)
```

### **Performance Metrics:**
```python
# Keep last 10 minutes of latency measurements (1/second)
PERF_METRICS_MAXLEN = 600

def log_latency_metric(latency_us):
    r.xadd("performance:latency", {
        "latency_us": latency_us,
        "timestamp": time.time()
    }, maxlen=PERF_METRICS_MAXLEN, approximate=True)
```

## ⚡ **Why `approximate=True`?**

- **Faster**: Doesn't guarantee exact count, saves CPU cycles
- **HFT-Friendly**: Reduces latency spikes from exact counting
- **Good Enough**: For trading, ~10,000 vs exactly 10,000 entries doesn't matter

## 🚨 **Memory Planning**

```python
# Estimate memory usage:
# Stream entry ≈ 100-500 bytes (depends on data)
# 10,000 entries ≈ 1-5 MB per stream
# With 100 symbols ≈ 100-500 MB total

# Adjust MAXLEN based on:
# - Available memory (4GB configured)
# - Number of symbols/streams
# - Data retention needs
```

## 🎯 **Quick Reference**

| Data Type | Suggested MAXLEN | Memory/Stream | Use Case |
|-----------|------------------|---------------|----------|
| High-freq ticks | 3600 (1hr) | ~1-2 MB | Real-time pricing |
| Order events | 1000 | ~100-500 KB | Order lifecycle |
| Performance logs | 600 (10min) | ~50-100 KB | Monitoring |
| OHLCV bars | 1440 (24hr) | ~150-300 KB | Daily analysis |

## 🔍 **Monitoring Stream Memory**

Add to your monitoring:
```bash
# Check stream memory usage
redis-hft-cli MEMORY USAGE market_data:EURUSD

# List all streams and their lengths  
redis-hft-cli SCAN 0 MATCH "*:*" TYPE stream
```

## 📝 **Implementation Checklist**

- [ ] All `XADD` commands include `maxlen` parameter
- [ ] `approximate=True` set for performance
- [ ] MAXLEN values calculated based on memory budget
- [ ] Stream memory usage monitored
- [ ] Retention periods align with business needs

**Remember**: This is required due to the `noeviction` policy chosen for deterministic HFT performance. Without trimming, Redis will reject new writes when memory limit is reached.