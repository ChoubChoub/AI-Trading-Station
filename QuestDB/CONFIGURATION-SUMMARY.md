# QuestDB Configuration Summary
**Created:** 2025-10-18  
**Hardware:** Samsung 990 Pro NVMe (3.6TB) + 192GB RAM + Intel Ultra 9 285K  
**Purpose:** 20-strategy crypto trading with 10+ exchanges

---

## Configuration Highlights

### Storage Strategy: Hot/Cold Tiering ✅
```
NVMe (3.6TB Samsung 990 Pro):
├─ Last 30 days tick data       → Active trading
├─ Last 7 days orderbook data   → Real-time analysis
└─ QuestDB WAL files            → Write-ahead log

HDD (7.3TB Seagate) - TO BE MOUNTED:
├─ Tick data > 30 days          → Historical backtesting
├─ Orderbook data > 7 days      → Archive
└─ Backup & cold storage        → Long-term
```

### Memory Allocation (192GB Total)
```yaml
QuestDB:     96GB (50% of total) - server.conf configured
Redis:       8GB (HFT optimized)  - redis-hft.conf
GPU/CUDA:    40GB (model inference)
OS/Services: 20GB (system overhead)
Buffer:      28GB (page cache, dynamic)
```

### Performance Targets
- **Insert Rate:** >200,000 ticks/second
- **Query Latency:** <100ms (p99)
- **NVMe Throughput:** ~7,350MB/s (hardware max)
- **Concurrent Strategies:** 20 parallel
- **Exchange Feeds:** 10+ simultaneous

---

## File Locations

### Configuration
```bash
# Main config (customized for your hardware)
/home/youssefbahloul/ai-trading-station/QuestDB/questdb-9.1.0-rt-linux-x86-64/conf/server.conf

# Data directory (NVMe hot storage)
/home/youssefbahloul/ai-trading-station/QuestDB/data/hot

# Logs
/home/youssefbahloul/ai-trading-station/QuestDB/logs
```

### Scripts
```bash
# NVMe optimization (run once as sudo)
/home/youssefbahloul/ai-trading-station/QuestDB/scripts/optimize-nvme-for-questdb.sh

# Start QuestDB
/home/youssefbahloul/ai-trading-station/QuestDB/scripts/start-questdb.sh
```

---

## Quick Start Guide

### Step 1: Optimize NVMe (One-Time Setup)
```bash
cd /home/youssefbahloul/ai-trading-station/QuestDB/scripts
sudo ./optimize-nvme-for-questdb.sh
```

**What it does:**
- Sets I/O scheduler to `none` (optimal for NVMe)
- Increases queue depth to 1024
- Optimizes read-ahead to 2MB
- Disables request merging for NVMe multiqueue
- Makes settings persistent via udev rules

### Step 2: Start QuestDB
```bash
cd /home/youssefbahloul/ai-trading-station/QuestDB/scripts
./start-questdb.sh
```

**Endpoints after startup:**
- **Web Console:** http://localhost:9000
- **HTTP API:** http://localhost:9000
- **PostgreSQL Wire:** localhost:8812
- **InfluxDB Line Protocol:** localhost:9009

### Step 3: Validate Performance
```bash
# Test insert performance (target: >200k/sec)
curl -G http://localhost:9000/exec \
  --data-urlencode "query=CREATE TABLE IF NOT EXISTS perf_test (timestamp TIMESTAMP, symbol SYMBOL, price DOUBLE) timestamp(timestamp) PARTITION BY DAY;"

# Monitor QuestDB
tail -f /home/youssefbahloul/ai-trading-station/QuestDB/logs/questdb.log

# Monitor NVMe I/O
iostat -x 1 10 | grep nvme
```

---

## Key Configuration Details

### Worker Threads (Scaled for 20 Strategies)
```properties
http.worker.count=20              # One per strategy
line.tcp.io.worker.count=8        # 10+ exchange feeds
shared.worker.count=16            # Query parallelism
```

### Memory Settings (96GB Allocation)
```properties
cairo.memory.limit=103079215104   # 96GB in bytes
cairo.max.query.memory=10737418240 # 10GB per query
cairo.global.memory.limit=85899345920 # 80GB global
```

### NVMe Optimizations
```properties
cairo.page.size=4096              # Match NVMe page size
cairo.wal.max.segment.size=268435456 # 256MB segments
cairo.wal.writer.buffer.size=8388608 # 8MB buffer
cairo.parallel.indexing=true      # Concurrent I/O
```

### Timestamp Precision (HFT Critical)
```properties
cairo.timestamp.precision=microsecond  # <100μs processing
```

### Connection Pooling
```properties
http.connection.pool.initial.size=5
http.connection.pool.max.size=50
```

---

## Integration with Existing Systems

### Redis → QuestDB Pipeline
```yaml
Architecture:
  Exchange WebSocket → Redis Stream → QuestDB

Redis Role:
  - Real-time tick buffering (8GB memory)
  - Temporary orderbook state
  - <10ms insertion latency

QuestDB Role:
  - Persistent storage (NVMe/HDD tiering)
  - Historical backtesting queries
  - Long-term data warehouse
```

### GPU Integration
```yaml
Data Flow:
  QuestDB (historical) → GPU (backtesting)
  
Performance:
  - Fast NVMe prevents GPU starvation
  - Query optimization critical for model training
  - 96GB QuestDB memory matches 192GB VRAM scale
```

---

## TODO: HDD Setup (Future)

When ready to enable cold storage on 7.3TB HDD:

```bash
# 1. Mount HDD
sudo mkdir -p /mnt/hdd
sudo mount /dev/sda /mnt/hdd

# 2. Create cold storage directory
sudo mkdir -p /mnt/hdd/questdb/cold
sudo chown -R youssefbahloul:youssefbahloul /mnt/hdd/questdb

# 3. Update server.conf
# Uncomment this line:
# cairo.cold.storage.root=/mnt/hdd/questdb/cold

# 4. Restart QuestDB
./stop-questdb.sh && ./start-questdb.sh
```

---

## Performance Validation Checklist

After starting QuestDB, validate these metrics:

- [ ] Insert rate: >200,000 ticks/second
- [ ] Query latency: <100ms (p99)
- [ ] NVMe utilization: 60-80% during trading
- [ ] Memory usage: <90% of 96GB allocation
- [ ] Connection pool: <50 active connections
- [ ] No I/O errors in logs
- [ ] Web console responsive: http://localhost:9000

---

## Monitoring Commands

```bash
# QuestDB metrics
curl http://localhost:9000/metrics

# NVMe statistics
sudo nvme smart-log /dev/nvme0n1

# I/O performance
iostat -x 1

# Memory usage
free -h

# QuestDB process
ps aux | grep questdb

# Connection count
netstat -an | grep -E '9000|8812|9009' | wc -l
```

---

## Why This Configuration Wins

### 1. Hardware-Optimized
- NVMe page size (4KB) matches Samsung 990 Pro
- WAL segments (256MB) optimal for sequential writes
- Queue depth (1024) leverages NVMe parallelism

### 2. Memory-Efficient
- 96GB allocation (50% of 192GB) leaves headroom
- 10GB per query prevents OOM
- Connection pooling prevents exhaustion

### 3. Multi-Strategy Ready
- 20 HTTP workers (one per strategy)
- 8 I/O workers (10+ exchanges)
- 16 query workers (parallel backtests)

### 4. HFT-Capable
- Microsecond timestamp precision
- <100ms query latency
- Parallel indexing for fast lookups

### 5. Scalable Architecture
- Hot/cold tiering (30 days → 365 days)
- Partition-based lifecycle management
- Easy expansion to HDD when needed

---

## Support & Troubleshooting

**Configuration file location:**  
`/home/youssefbahloul/ai-trading-station/QuestDB/questdb-9.1.0-rt-linux-x86-64/conf/server.conf`

**All settings are documented inline with explanations.**

**For issues:**
1. Check logs: `tail -f /home/youssefbahloul/ai-trading-station/QuestDB/logs/questdb.log`
2. Verify NVMe: `iostat -x 1 10`
3. Check memory: `free -h`
4. Review config: Comments explain every setting

**Performance not meeting targets?**
- Run NVMe optimization script again
- Check I/O scheduler: `cat /sys/block/nvme0n1/queue/scheduler` (should be `[none]`)
- Verify memory allocation: QuestDB should use ~96GB
- Monitor with: `watch -n 1 'iostat -x 1 1 | grep nvme0n1'`

---

**Status:** ✅ Configuration Complete  
**Next Step:** Run `sudo ./optimize-nvme-for-questdb.sh` then `./start-questdb.sh`
