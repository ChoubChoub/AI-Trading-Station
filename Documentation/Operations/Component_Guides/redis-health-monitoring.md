# Redis Connection Health - Quick Reference

**Last Updated:** October 27, 2025  
**Status:** ✅ Production Ready

---

## Quick Health Check (30 seconds)

```bash
# 1. Connection count (should be ~24)
netstat -tn | grep ':6379' | grep ESTABLISHED | wc -l

# 2. Health monitor status
systemctl status redis-health-monitor --no-pager | head -5

# 3. Recent health monitor logs
journalctl -u redis-health-monitor --since "5 minutes ago" --no-pager | tail -3

# 4. Batch writer status
systemctl status batch-writer --no-pager | head -5
```

**Expected Output:**
- Connections: 20-30 (baseline: 25)
- Health monitor: ✅ HEALTHY
- Batch writer: Active (running)

---

## Performance Benchmarks

### Target Metrics
| Window | Average | P99 | Status |
|--------|---------|-----|--------|
| 5s | <1ms | <1ms | ✅ 0.26ms / 0.74ms |
| 30s | <2ms | <4ms | ✅ 1.69ms / 3.01ms |
| 60s | <3ms | <5ms | ✅ 2.86ms / 3.73ms |

### Quick Latency Test
```bash
cd /home/youssefbahloul/ai-trading-station/QuestDB/scripts
python3 -c "
import asyncio
from tiered_market_data import TieredMarketData

async def test():
    t = TieredMarketData()
    await t.start()
    await asyncio.sleep(1)
    
    for window in [5, 30, 60]:
        _, _, lat = await t.get_recent_ticks('BTCUSDT', 'binance_spot', 'TRADE', window)
        print(f'{window}s: {lat:.2f}ms')
    
    await t.stop()

asyncio.run(test())
"
```

---

## Alert Thresholds

| Level | Connections | Action |
|-------|-------------|--------|
| ✅ **HEALTHY** | 20-25 | Normal operation |
| ⚠️ **WARNING** | 26-49 | Monitor closely |
| 🚨 **LEAK** | 50-99 | Manual investigation |
| 🔥 **CRITICAL** | 100+ | Auto-restart triggered |

---

## Common Operations

### Restart Services
```bash
# Restart batch writer
sudo systemctl restart batch-writer.service

# Restart health monitor
sudo systemctl restart redis-health-monitor.service

# Restart both
sudo systemctl restart batch-writer redis-health-monitor

# Check status
systemctl status batch-writer redis-health-monitor
```

### Monitor Connections in Real-Time
```bash
# Watch connection count (updates every 2s)
watch -n 2 'netstat -tn | grep ":6379" | grep ESTABLISHED | wc -l'

# Watch health monitor logs
journalctl -u redis-health-monitor -f

# Watch batch writer logs
journalctl -u batch-writer -f
```

### Check for Errors
```bash
# Batch writer errors (last hour)
journalctl -u batch-writer --since "1 hour ago" | grep -i error

# Health monitor alerts (last hour)
journalctl -u redis-health-monitor --since "1 hour ago" | grep -E "WARNING|ERROR|CRITICAL"

# Redis errors
redis-cli INFO stats | grep -E "rejected|error"
```

---

## Troubleshooting

### Issue: Connections Growing Above Baseline

**Symptoms:** Connection count steadily increasing above 30

**Diagnosis:**
```bash
# Check connection growth over 5 minutes
for i in {1..10}; do
    echo "$(date +%T): $(netstat -tn | grep ':6379' | grep ESTABLISHED | wc -l) connections"
    sleep 30
done
```

**Solution:**
```bash
# If growth detected, restart batch-writer
sudo systemctl restart batch-writer.service

# Verify connections stabilize
sleep 10
netstat -tn | grep ':6379' | grep ESTABLISHED | wc -l
```

---

### Issue: High Latency

**Symptoms:** Cache queries taking >5ms consistently

**Diagnosis:**
```bash
# Test Redis latency
redis-cli --latency

# Check system load
uptime

# Check Redis memory
redis-cli INFO memory | grep used_memory_human
```

**Solution:**
```bash
# If Redis is slow, check for other processes
ps aux | grep redis

# Restart Redis if needed (CAUTION: will clear cache)
sudo systemctl restart redis-hft.service

# Restart batch-writer to reconnect
sudo systemctl restart batch-writer.service
```

---

### Issue: Health Monitor Not Reporting

**Symptoms:** No logs from redis-health-monitor in last 2 minutes

**Diagnosis:**
```bash
# Check service status
systemctl status redis-health-monitor

# Check for crashes
journalctl -u redis-health-monitor --since "10 minutes ago" | grep -i "error\|exception"
```

**Solution:**
```bash
# Restart health monitor
sudo systemctl restart redis-health-monitor

# Verify it's running
systemctl status redis-health-monitor

# Watch for new logs
journalctl -u redis-health-monitor -f
```

---

## Rollback Procedure

**If issues arise, rollback to pool version (stable but slower):**

```bash
# 1. Stop batch writer
sudo systemctl stop batch-writer.service

# 2. Restore pool version
cd /home/youssefbahloul/ai-trading-station/QuestDB/scripts
cp ../Archive/redis_pool_backup_20251027_104148/tiered_market_data.py.pool_version \
   tiered_market_data.py

# 3. Restart service
sudo systemctl start batch-writer.service

# 4. Verify connections stabilize at ~31
sleep 10
netstat -tn | grep ':6379' | grep ESTABLISHED | wc -l

# 5. Check latency (will be ~3-4ms avg)
python3 -c "
import asyncio
from tiered_market_data import TieredMarketData
async def test():
    t = TieredMarketData()
    await t.start()
    _, _, lat = await t.get_recent_ticks('BTCUSDT', 'binance_spot', 'TRADE', 30)
    print(f'30s latency: {lat:.2f}ms (expect ~3-4ms with pool)')
    await t.stop()
asyncio.run(test())
"
```

---

## Service Files

### Health Monitor
- **Service:** `/etc/systemd/system/redis-health-monitor.service`
- **Script:** `/home/youssefbahloul/ai-trading-station/QuestDB/scripts/redis_health_monitor.py`
- **Logs:** `journalctl -u redis-health-monitor -f`

### Batch Writer
- **Service:** `/etc/systemd/system/batch-writer.service`
- **Script:** `/home/youssefbahloul/ai-trading-station/QuestDB/scripts/redis_to_questdb_v2.py`
- **Logs:** `journalctl -u batch-writer -f`

### Redis HFT
- **Service:** `/etc/systemd/system/redis-hft.service`
- **Config:** `/opt/redis-hft/config/redis-hft.conf`
- **Logs:** `journalctl -u redis-hft -f`

---

## Contact & Escalation

### Automated Recovery
- Health monitor auto-restarts batch-writer if connections exceed 100
- Cooldown: 5 minutes between restarts to prevent restart loops

### Manual Intervention Required
- Connections growing despite restarts
- Latency consistently >10ms
- Health monitor repeatedly crashing
- Redis errors in logs

### Documentation
- **Full Analysis:** `Documentation/REDIS_CONNECTION_LEAK_FINAL_RESOLUTION.md`
- **Regression Report:** `Documentation/CONNECTION_POOL_LATENCY_REGRESSION.md`
- **CPU Affinity:** `Documentation/CPU_AFFINITY_STRATEGY_FINAL.md`

---

## Success Criteria

**System is healthy when ALL of these are true:**

- ✅ Connections: 20-30 (stable, no growth)
- ✅ 5s latency: <1ms average
- ✅ 30s latency: <2ms average
- ✅ Health monitor: ✅ HEALTHY status
- ✅ No errors in logs (last hour)
- ✅ Batch writer: Active (running)
- ✅ Redis: Active (running)

**Run this one-liner to check all:**
```bash
echo "=== System Health Check ===" && \
echo "Connections: $(netstat -tn | grep ':6379' | grep ESTABLISHED | wc -l)" && \
echo "Health Monitor: $(systemctl is-active redis-health-monitor)" && \
echo "Batch Writer: $(systemctl is-active batch-writer)" && \
echo "Redis HFT: $(systemctl is-active redis-hft)" && \
journalctl -u redis-health-monitor --since "2 minutes ago" --no-pager | tail -1
```

---

**Quick Reference Version:** 1.0  
**Generated:** October 27, 2025  
**Status:** Production Ready ✅
