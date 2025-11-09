# Datafeed User Guide
**Market Data Collection System - Quick Reference**

---

## 🚀 Quick Start

```bash
# Start the entire data collection system
datafeed start

# Check everything is running
datafeed status

# Stop the system cleanly
datafeed stop
```

---

## 📋 Command Reference

### **Basic Operations**

| Command | Description | Example |
|---------|-------------|---------|
| `datafeed start` | Start all collectors (trades + orderbook) and batch writer | `datafeed start` |
| `datafeed stop` | Gracefully stop entire system with data flush | `datafeed stop` |
| `datafeed restart` | Restart all services | `datafeed restart` |
| `datafeed status` | Show comprehensive system status | `datafeed status` |

### **Health Monitoring**

| Command | Description | Use Case |
|---------|-------------|----------|
| `datafeed test` | Test all connectivity (Redis, QuestDB, Binance, endpoints) | Pre-flight checks |
| `datafeed health` | Quick health check with exit code (0=healthy, 1=unhealthy) | Monitoring scripts |
| `datafeed metrics` | View detailed metrics (rates, errors, latency) | Performance analysis |

### **Logging**

| Command | Description | Example |
|---------|-------------|---------|
| `datafeed logs` | View recent logs from all services | `datafeed logs` |
| `datafeed logs -f` | Follow logs in real-time | `datafeed logs -f` |
| `datafeed logs -n 100` | Show last 100 log lines | `datafeed logs -n 100` |

### **Service Management**

| Command | Description | When to Use |
|---------|-------------|-------------|
| `datafeed enable` | Enable auto-start at boot | Production deployment |
| `datafeed disable` | Disable auto-start (manual control) | Development/testing |
| `datafeed help` | Show help and usage examples | First time usage |

---

## 📊 Understanding Status Output

```bash
$ datafeed status

Prerequisites:
  ✓ Redis (PID 2402, CPU 4)          # Redis HFT is running
  ✓ QuestDB (PID 1575)                # QuestDB is running

Data Collectors:
  Trades:    ✓ Running (PID 2409, up 0d 0h 19m)
             Memory: 27M               # Trade collector health
  Orderbook: ✓ Running (PID 3748, up 0d 0h 5m)
             Memory: 19M               # Orderbook collector health
  BatchWriter: ✓ Running (PID 2410, up 0d 0h 19m)
             Memory: 39M               # Redis→QuestDB writer health

Data Flow (Last 60 seconds):
  Redis Streams:
    Trades:    100039 messages        # Messages in Redis buffer
    Orderbook: 100025 messages
  QuestDB Tables:
    Trades       1151 events (19.2/sec)   # Events written to database
    Orderbook   33793 events (563.2/sec)

Prometheus Metrics:
  ✓ Available at http://localhost:9091/metrics
```

### **What to Look For:**
- ✅ **All services show ✓** - System is healthy
- ⚠️ **Redis/QuestDB missing** - Prerequisites not running
- ❌ **Service stopped** - Collector or writer has crashed
- 📊 **Data flow >0 events/sec** - Pipeline is processing data

---

## 🔍 Health Check Details

```bash
$ datafeed health
✅ HEALTHY
Exit code: 0
```

**Exit Codes:**
- `0` = All services running, prerequisites OK, data flowing
- `1` = One or more services down or prerequisites missing

**Use in Scripts:**
```bash
if datafeed health; then
    echo "System healthy, proceeding..."
else
    echo "System unhealthy, sending alert!"
    # Send notification
fi
```

---

## 📈 Metrics Breakdown

```bash
$ datafeed metrics
```

### **Batch Writer (Prometheus)**
- `total_ticks_inserted` - Total events written to QuestDB
- `errors` - Write errors (should be 0)
- `per_symbol` - Events per trading pair

### **Trades Collector (Health Endpoint)**
- `status` - healthy/unhealthy
- `messages_written` - Total messages processed
- `reconnections` - WebSocket reconnection count
- `errors` - Processing errors
- `uptime_seconds` - Time since service start

### **QuestDB Event Rates**
- Shows events/second over last 60 seconds
- Separate rates for trades and orderbook

---

## 🔧 Common Tasks

### **Daily Startup**
```bash
# 1. Verify prerequisites
datafeed test

# 2. Start system
datafeed start

# 3. Confirm health
datafeed status
```

### **Investigate Issues**
```bash
# 1. Check connectivity
datafeed test

# 2. View real-time logs
datafeed logs -f

# 3. Check metrics
datafeed metrics
```

### **Clean Shutdown**
```bash
# Graceful stop (flushes data, closes connections)
datafeed stop

# Verify everything stopped
datafeed status
```

### **Performance Check**
```bash
# Get current rates
datafeed metrics | grep "events/sec"

# Check memory usage
datafeed status | grep "Memory"

# View detailed stats
curl http://localhost:8001/health | jq
```

---

## 🎯 What Data is Being Collected?

### **Trades Stream** (`binance_spot_trades`)
- Price, quantity, timestamp (microsecond precision)
- Trade ID, buyer/seller indicator
- ~20-50 events/second per symbol

### **Orderbook Stream** (`binance_spot_orderbook`)
- Best bid/ask prices + quantities
- Updated on every price change
- ~500-1000 events/second per symbol

### **Default Symbols**
- BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT

---

## 🔗 Integration Endpoints

### **Health Endpoint** (Trades Collector)
```bash
curl http://localhost:8001/health | jq
```

**Returns:**
```json
{
  "status": "healthy",
  "last_message_age_seconds": 0.02,
  "messages_written": 9612,
  "reconnections": 0,
  "errors": 0,
  "uptime_seconds": 16.28
}
```

### **Prometheus Metrics** (Batch Writer)
```bash
curl http://localhost:8000/metrics
```

**Key Metrics:**
- `ticks_inserted_total` - Total events written
- `write_errors_total` - Write error count
- `processing_time_seconds` - Write latency

### **QuestDB Console**
- **URL**: http://localhost:19000
- **Query trades**: `SELECT * FROM binance_spot_trades LIMIT 100;`
- **Query orderbook**: `SELECT * FROM binance_spot_orderbook LIMIT 100;`

---

## ⚠️ Troubleshooting

### **Service Won't Start**
```bash
# Check what's wrong
datafeed test

# View error logs
datafeed logs -f

# Check systemd status
sudo systemctl status market-data.target
```

**Common Issues:**
- Redis not running → `sudo systemctl start redis-hft`
- QuestDB not running → Start QuestDB instance
- Port already in use → Check for zombie processes

### **No Data in QuestDB**
```bash
# Verify Redis has data
redis-cli XLEN market:binance_spot:trades:BTCUSDT

# Check batch writer is processing
curl http://localhost:8000/metrics | grep ticks_inserted

# Verify QuestDB connection
nc -zv localhost 9009
```

### **High Memory Usage**
```bash
# Check current usage
datafeed status | grep Memory

# Reduce Redis buffer size (requires restart)
# Edit: /etc/systemd/system/binance-*.service
# Add: Environment="REDIS_MAXLEN=50000"

sudo systemctl daemon-reload
datafeed restart
```

### **Connection Keeps Dropping**
```bash
# Check WebSocket connectivity
curl -s https://api.binance.com/api/v3/ping

# View reconnection logs
datafeed logs | grep "reconnection"

# Check network stability
ping -c 10 stream.binance.com
```

---

## 📁 File Locations

### **Scripts**
- `/home/youssefbahloul/ai-trading-station/QuestDB/scripts/binance_trades_collector.py`
- `/home/youssefbahloul/ai-trading-station/QuestDB/scripts/binance_bookticker_collector.py`
- `/home/youssefbahloul/ai-trading-station/QuestDB/scripts/redis_to_questdb_v2.py`

### **Systemd Services**
- `/etc/systemd/system/market-data.target`
- `/etc/systemd/system/binance-trades.service`
- `/etc/systemd/system/binance-bookticker.service`
- `/etc/systemd/system/batch-writer.service`

### **Control Script**
- `/usr/local/bin/datafeed` → `/home/youssefbahloul/ai-trading-station/QuestDB/scripts/datafeed_v2.sh`

### **Logs**
- View via: `datafeed logs`
- Direct access: `journalctl -u market-data.target`

---

## 🚦 Production Checklist

### **Before Going Live**
- [ ] Run `datafeed test` - all checks pass
- [ ] Enable auto-start: `datafeed enable`
- [ ] Verify 24/7 monitoring setup
- [ ] Document alert response procedures
- [ ] Test graceful restart: `datafeed restart`

### **Daily Monitoring**
- [ ] Check `datafeed status` morning and evening
- [ ] Review `datafeed metrics` for anomalies
- [ ] Monitor memory usage trends
- [ ] Verify QuestDB disk space

### **Weekly Maintenance**
- [ ] Review error logs: `datafeed logs | grep ERROR`
- [ ] Check reconnection frequency
- [ ] Validate data completeness in QuestDB
- [ ] Archive old QuestDB partitions if needed

---

## 📚 Additional Resources

- **Full Documentation**: `/home/youssefbahloul/ai-trading-station/QuestDB/docs/DAY3_PHASE2_COMPLETION_REPORT.md`
- **Architecture Details**: See completion report for diagrams and technical specs
- **Performance Metrics**: Baseline 99.60% capture rate, <1ms latency
- **Support**: Check logs first, then review completion report troubleshooting section

---

## 💡 Tips & Best Practices

1. **Always use `datafeed stop`** (not `kill`) for clean shutdown
2. **Check `datafeed health`** before making system changes
3. **Monitor `datafeed metrics`** for performance trends
4. **Use `datafeed logs -f`** during development for real-time feedback
5. **Run `datafeed test`** after any configuration changes
6. **Enable auto-start** in production: `datafeed enable`

---

**Quick Help**: `datafeed help`  
**System Status**: `datafeed status`  
**Current Version**: Phase 2 Complete (October 2025)
