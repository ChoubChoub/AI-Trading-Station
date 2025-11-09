# System Reboot Procedure - CPU Affinity Verification

**Date:** October 23, 2025  
**Purpose:** Verify CPU affinity configuration persists after reboot and all services operate correctly

---

## Pre-Reboot Status

### Current Configuration
- **QuestDB**: CPUs 5-6 (2 cores for Java GC + database)
- **Batch-writer**: CPU 7 (1 core for I/O-bound work)
- **WebSocket collectors**: CPU 3 (binance-trades, binance-bookticker)
- **Redis-HFT**: CPU 4
- **Prometheus**: CPU 2
- **IRQ handlers**: CPUs 0-1 (isolated)

### Pre-Reboot Metrics
- Temperature: 67°C (stable)
- QuestDB CPU usage: ~95% across CPUs 5-6
- Batch-writer CPU usage: ~14% on CPU 7
- All services: Active

---

## Reboot Commands

### Safe Reboot Sequence
```bash
# 1. Verify current state is saved
systemctl status prometheus redis-hft questdb batch-writer binance-trades binance-bookticker

# 2. Ensure systemd override files are in place
ls -la /etc/systemd/system/*.service.d/

# 3. Reboot
sudo reboot
```

**Estimated downtime:** 2-3 minutes

---

## Post-Reboot Diagnostic

### Automatic Diagnostic Script
After reboot, run the comprehensive diagnostic:

```bash
cd /home/youssefbahloul/ai-trading-station/Tests/System
./post_reboot_diagnostic.sh
```

This script will verify:
1. ✅ System health (CPU topology, governor, load, memory, temperature)
2. ✅ All 6 services running (prometheus, redis, questdb, batch-writer, websocket collectors)
3. ✅ CPU affinity configuration (systemd settings + actual process affinity)
4. ✅ CPU utilization per service and per core
5. ✅ Data ingestion pipeline (Redis streams, QuestDB tables, recent data)
6. ✅ Network connectivity and latency
7. ✅ Grafana monitoring accessibility

**Output:** 
- Console output with color-coded status
- Log file: `/home/youssefbahloul/ai-trading-station/Services/Monitoring/logs/post_reboot_diagnostic_YYYYMMDD_HHMMSS.log`

---

## Manual Verification (Quick Check)

If you need a quick manual check:

```bash
# 1. Check all services are active
systemctl is-active prometheus redis-hft questdb batch-writer binance-trades binance-bookticker

# 2. Verify CPU affinity
taskset -cp $(pgrep prometheus)           # Should show: 2
taskset -cp $(pgrep redis-server)         # Should show: 4
taskset -cp $(pgrep -f io.questdb)        # Should show: 5,6
taskset -cp $(pgrep -f redis_to_questdb)  # Should show: 7
taskset -cp $(pgrep -f binance_trades)    # Should show: 3
taskset -cp $(pgrep -f binance_bookticker) # Should show: 3

# 3. Check temperature
sensors | grep "Package id 0"              # Should be <75°C

# 4. Check CPU usage
mpstat -P 5,6,7 1 3                       # QuestDB on 5-6, batch-writer on 7

# 5. Verify data flow
redis-cli DBSIZE                          # Should have keys
curl -s http://localhost:9000/exec?query=SELECT%201  # QuestDB responding
curl -s http://localhost:9091/metrics | grep records_processed  # Batch writer active
```

---

## Expected Results

### Services
All services should show `active (running)`:
```
prometheus        ✓ active
redis-hft         ✓ active
questdb           ✓ active
batch-writer      ✓ active
binance-trades    ✓ active
binance-bookticker ✓ active
```

### CPU Affinity
```
prometheus         → CPU 2      ✓
redis-hft          → CPU 4      ✓
questdb            → CPUs 5,6   ✓
batch-writer       → CPU 7      ✓
binance-trades     → CPU 3      ✓
binance-bookticker → CPU 3      ✓
```

### Temperature & Load
```
Package temperature: <75°C      ✓
System load:         <5.0       ✓
CPU 5-6 usage:       80-95%     ✓ (QuestDB - expected)
CPU 7 usage:         10-20%     ✓ (Batch-writer - I/O bound)
CPU 3 usage:         1-5%       ✓ (WebSocket - network I/O)
```

### Data Pipeline
```
Redis keys:          >0         ✓
QuestDB tables:      Exist      ✓
Recent ingestion:    <60s ago   ✓
Batch writer metrics: Available ✓
```

---

## Troubleshooting

### If Services Don't Start
```bash
# Check service status and logs
sudo systemctl status <service-name>
sudo journalctl -u <service-name> -n 50 --no-pager

# Restart if needed
sudo systemctl restart <service-name>
```

### If CPU Affinity Not Applied
```bash
# Verify systemd overrides exist
ls -la /etc/systemd/system/questdb.service.d/
ls -la /etc/systemd/system/batch-writer.service.d/

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart questdb batch-writer
```

### If Temperature High
```bash
# Check which CPU is hot
sensors

# Check CPU utilization
mpstat -P ALL 1 5

# If QuestDB saturating single core again
taskset -cp $(pgrep -f io.questdb)  # Should be 5,6 not just 5
```

### If Data Not Flowing
```bash
# Check WebSocket collectors
sudo systemctl status binance-trades binance-bookticker
sudo journalctl -u binance-trades -n 20 --no-pager

# Check Redis
redis-cli PING
redis-cli KEYS "market:binance_spot:*" | head -5

# Check QuestDB
curl http://localhost:9000/
curl "http://localhost:9000/exec?query=SELECT%20*%20FROM%20tables()"

# Check batch writer
sudo systemctl status batch-writer
curl http://localhost:9091/metrics | grep -E "processed|failed|latency"
```

---

## Rollback Procedure (Emergency)

If CPU affinity causes issues after reboot:

```bash
cd /home/youssefbahloul/ai-trading-station/Monitoring/Scripts
./rollback_cpu_affinity.sh
```

This will:
1. Remove CPU affinity overrides for binance services and batch-writer
2. Keep Prometheus and Redis affinity (pre-existing)
3. Restart affected services
4. Verify changes

---

## Documentation Updates

After successful reboot verification, update:

1. **CPU_AFFINITY_IMPLEMENTATION_RECORD.md**
   - Add "Post-Reboot Verification" section
   - Record reboot timestamp and diagnostic results

2. **PHASE3_FINAL_REPORT.md**
   - Mark CPU affinity optimization as "Production Verified"
   - Include temperature stability data

3. **CPU_AFFINITY_THERMAL_FIX.md**
   - Confirm thermal fix persists across reboots

---

## Success Criteria

✅ All services start automatically  
✅ CPU affinity correctly applied to all processes  
✅ Temperature remains <75°C under load  
✅ QuestDB utilizes CPUs 5-6 efficiently  
✅ Batch-writer runs on CPU 7 with low usage  
✅ Data flows: WebSocket → Redis → Batch-writer → QuestDB  
✅ Grafana displays real-time metrics  
✅ No service failures in logs  

**Status upon completion:** PRODUCTION READY ✅

---

## Timeline

- **22:20 EDT:** Initial CPU affinity implementation
- **23:00 EDT:** Thermal fix applied (QuestDB: 5→5-6, Batch-writer: 6-7→7)
- **23:05 EDT:** Post-reboot diagnostic script created
- **23:XX EDT:** System reboot initiated
- **23:XX EDT:** Post-reboot diagnostic executed
- **23:XX EDT:** Production verification complete

---

**Next:** Execute `sudo reboot` when ready
