# VPS to Brain Machine Architecture

**Date:** October 26, 2025  
**Status:** Planning Phase  
**Target:** Distributed data ingestion with disconnection resilience

---

## Executive Summary

This document outlines the proposed distributed architecture where a dedicated **VPS (Vultr)** handles exchange data ingestion and buffering, feeding a **Brain Machine** (current trading setup) via Redis Streams. This design provides **24-hour buffer capacity** for network disconnections while preserving all existing optimizations (Day 3/4 tiered cache, CPU affinity, batch writer).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ VPS - Data Ingestion Server (Vultr)                                 │
│ Configuration: 2 CPU | 4GB RAM | 50GB NVMe                          │
│                                                                      │
│  ┌──────────────┐     ┌─────────────┐     ┌──────────────────┐    │
│  │  Exchange    │────>│   Redis     │────>│  Redis Stream    │    │
│  │  Collectors  │     │ (HOT+WARM)  │     │   Forwarder      │    │
│  │  (Binance)   │     │  Cache      │     │                  │    │
│  └──────────────┘     └─────────────┘     └────────┬─────────┘    │
│   50 pairs/WS               │                       │               │
│   ~1,200 ticks/sec          │ (overflow)            │               │
│                             ↓                       │               │
│                     ┌─────────────┐                 │               │
│                     │  QuestDB    │                 │               │
│                     │  (Buffer)   │                 │               │
│                     │  50GB NVMe  │                 │               │
│                     └─────────────┘                 │               │
│                      ~22h capacity                  │               │
└─────────────────────────────────────────────────────┼───────────────┘
                                                      │
                                    Redis Streams     │
                                    (~2-5ms latency)  │
                                    Guaranteed        │
                                    Delivery          │
                                                      │
┌─────────────────────────────────────────────────────┼───────────────┐
│ BRAIN - Trading Machine (Existing Setup)            ↓               │
│                                                                      │
│  ┌──────────────┐     ┌─────────────┐     ┌──────────────────┐    │
│  │   Stream     │────>│   Redis     │────>│  Batch Writer    │    │
│  │  Consumer    │     │  (Local)    │     │  (EXISTING)      │    │
│  │  (50 lines)  │     │             │     │  redis_to_       │    │
│  └──────────────┘     └──────┬──────┘     │  questdb_v2.py   │    │
│   New component              │             └────────┬─────────┘    │
│                              │                      │               │
│                              ↓                      ↓               │
│                     ┌──────────────┐      ┌─────────────────┐     │
│                     │ Tiered Cache │      │    QuestDB      │     │
│                     │ (HOT+WARM)   │      │  Long-term      │     │
│                     │ Day 3/4      │      │  SSD/HDD        │     │
│                     └──────────────┘      └─────────────────┘     │
│                      UNCHANGED              UNCHANGED              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### VPS (Vultr) - Data Ingestion Server

**Hardware:**
- **CPU:** 2 dedicated cores
- **RAM:** 4 GB
- **Storage:** 50 GB NVMe SSD
- **Network:** 1 Gbps+ (near exchange locations)
- **Estimated Cost:** ~$12-18/month (Vultr mid-tier)

**Software Stack:**
- Ubuntu 22.04 LTS
- Redis 7.x (in-memory cache)
- QuestDB 7.x (buffer storage)
- Python 3.11+ (exchange collectors)

**Components:**

1. **Exchange Collectors (Python WebSocket)**
   - Binance WebSocket clients (50 trading pairs)
   - Handles: trades, orderbook updates
   - Rate: ~1,200 ticks/sec average
   - CPU Affinity: Core 0

2. **Redis (HOT + WARM Cache)**
   - HOT tier: Last 100 ticks/symbol (Python deque in RAM)
   - WARM tier: Last 5 minutes/symbol (Redis sorted sets)
   - Memory allocation: ~2 GB
   - Buffer capacity: ~1.7 hours
   - Persistence: RDB snapshots every 5 minutes

3. **QuestDB (COLD Buffer)**
   - Stores last 24-48 hours on NVMe
   - Acts as overflow buffer during disconnections
   - Auto-cleanup after successful transmission to Brain
   - Storage allocation: ~40 GB usable
   - Buffer capacity: ~22 hours

4. **Data Forwarder (NEW - 100 lines Python)**
   - Reads from Redis WARM cache
   - Sends to Brain via Redis Streams (XADD)
   - Implements retry logic with exponential backoff
   - Falls back to QuestDB on sustained failures
   - CPU Affinity: Core 1
   - Monitoring: Prometheus metrics

**Data Flow (Normal Operation):**
```
Exchange → WebSocket → Redis WARM → Redis Stream → Brain
         ~10-20ms      ~0.3ms        ~2-5ms
```

**Data Flow (Disconnection):**
```
Exchange → WebSocket → Redis WARM → QuestDB Buffer
                                      ↓ (accumulates)
                                   ~22 hours capacity
                                      ↓ (reconnection)
                                   Replay to Brain
```

---

### Brain Machine (Existing Setup) - UNCHANGED

**Hardware:**
- Existing CPU configuration (8 cores with affinity)
- Existing RAM allocation
- Existing SSD/HDD storage strategy

**Software Stack:**
- All existing services preserved
- No changes to batch writer
- No changes to QuestDB schema
- No changes to CPU affinity

**Components:**

1. **Stream Consumer (NEW - 50 lines Python)**
   - Subscribes to Redis Stream from VPS
   - Consumer group: "brain-consumer-group"
   - Acknowledges messages (XACK)
   - Writes to local Redis (market:* keys)
   - Monitoring: Connection health, lag metrics

2. **Redis (Local)**
   - Same structure as current setup
   - Receives data from Stream Consumer
   - Feeds Batch Writer (unchanged)

3. **Batch Writer (EXISTING - NO CHANGES)**
   - File: `redis_to_questdb_v2.py`
   - Reads from local Redis streams
   - Populates tiered cache (Day 3/4)
   - Writes to QuestDB with ILP
   - All CPU affinity preserved
   - All optimizations preserved

4. **Tiered Cache (Day 3/4 - UNCHANGED)**
   - HOT tier: deque (~0.01ms)
   - WARM tier: Redis (~0.1-0.3ms)
   - COLD tier: QuestDB (~50-200ms)
   - All code from Day 3/4 preserved

5. **QuestDB (Long-term Storage - UNCHANGED)**
   - SSD: Last 30-90 days (hot queries)
   - HDD: Historical data (months/years)
   - Existing partitioning strategy
   - All schemas unchanged

---

## Communication Protocol: Redis Streams

**Why Redis Streams?**
- ✅ **Throughput:** 100K+ messages/sec
- ✅ **Latency:** ~0.5-2ms over network (local LAN) or ~2-5ms (internet)
- ✅ **Guaranteed Delivery:** Built-in acknowledgment (XACK)
- ✅ **Automatic Retry:** Consumer groups handle failures
- ✅ **Scalability:** Multiple consumers possible (future: multiple Brain machines)
- ✅ **Native Protocol:** No extra dependencies, pure Redis
- ✅ **Persistence:** Survives Redis restarts with AOF/RDB

**Message Format:**
```python
{
    'symbol': 'BTCUSDT',
    'exchange': 'binance_spot',
    'type': 'TRADE',
    'price': 67234.56,
    'volume': 0.15,
    'side': 'BUY',
    'timestamp': 1729901234567890  # microseconds
}
```

**Stream Name:** `market:vps:stream`

**Consumer Group:** `brain-consumer-group`

**Acknowledgment:** After successful write to Brain Redis

---

## Buffer Capacity Analysis

### Data Volume Calculation

**Current Baseline (Measured):**
- 5 pairs × 1 exchange = 5 symbols
- Tick rate: 120 ticks/sec
- Data rate: ~3 MB/min
- Redis memory: 15 MB

**Target Scenario (50 pairs, Binance only):**
- 50 pairs × 1 exchange = 50 symbols
- Scaling factor: 10x
- Tick rate: ~1,200 ticks/sec
- Data rate: ~30 MB/min

### VPS Buffer Capacity

**RAM Buffer (4 GB):**
```
Total RAM:        4,000 MB
OS + Services:    -1,000 MB (Redis, QuestDB, OS overhead)
Available:         3,000 MB (safe limit, 75% utilization)
Data Rate:           30 MB/min
─────────────────────────────────
Buffer Time:       3,000 ÷ 30 = 100 minutes = 1.7 hours ✅
```

**NVMe Buffer (50 GB):**
```
Total NVMe:       50,000 MB
OS + Apps:       -10,000 MB (system, logs, apps)
Available:        40,000 MB (safe limit, 80% utilization)
Data Rate:           30 MB/min
─────────────────────────────────
Buffer Time:      40,000 ÷ 30 = 1,333 minutes = 22.2 hours ✅
```

**Total Buffer Capacity:**
```
RAM:     1.7 hours
NVMe:   22.2 hours
─────────────────
TOTAL:  ~24 hours ✅✅
```

**Conclusion:** VPS can survive **1 full day** of Brain disconnection without data loss.

---

## Performance Characteristics

### Latency Analysis

**Current Setup (Direct):**
```
Exchange → Brain Redis = ~10-20ms
```

**Distributed Setup:**
```
Exchange → VPS Redis → Redis Stream → Brain Redis
  ~10ms      ~0.3ms      ~2-5ms         ~0ms
────────────────────────────────────────────────
Total: ~12-25ms (+2-5ms overhead)
```

**Latency Optimization Strategies:**

1. **VPS Location:** Deploy near exchange servers
   - AWS Tokyo for Binance (reduces Exchange→VPS to ~1-3ms)
   - AWS Singapore for regional redundancy

2. **Redis Pipelining:** Batch 10-50 messages per network round-trip
   - Amortizes network latency
   - Increases throughput to 50K+ msgs/sec

3. **Network Optimization:** 
   - Dedicated VPN/VPC between VPS and Brain
   - QoS prioritization for Redis Streams traffic
   - TCP tuning (window size, congestion control)

4. **Future: Kernel Bypass (if critical):**
   - DPDK or Solarflare TCPDirect
   - Reduces network to ~0.5-1ms (complex setup)

### CPU Utilization

**VPS (2 CPUs):**
- Core 0: Exchange collectors + Redis server (~60% load)
- Core 1: Data forwarder + QuestDB writes (~40% load)
- **Total:** ~50% average utilization ✅ (comfortable margin)

**Brain (Unchanged):**
- Existing CPU affinity preserved
- No additional CPU load (Stream Consumer is lightweight)
- Batch Writer continues as-is

### Throughput Capacity

**VPS Maximum:**
- Redis Streams: 100K+ messages/sec (theoretical)
- Network: 1 Gbps = ~125 MB/sec (practical: ~80 MB/sec)
- Target load: 1,200 ticks/sec = ~30 MB/min = ~0.5 MB/sec
- **Headroom:** 160x capacity ✅

**Brain Maximum:**
- Existing batch writer handles 10K+ ticks/sec
- Target load: 1,200 ticks/sec
- **Headroom:** 8x capacity ✅

---

## Disconnection Scenarios & Resilience

### Scenario 1: Temporary Network Glitch (<1 minute)

**VPS Behavior:**
- Redis Stream accumulates in RAM
- No overflow to QuestDB
- Forwarder retries with exponential backoff

**Brain Behavior:**
- Stream Consumer detects lag
- Alerts sent to monitoring
- Catches up within seconds after reconnection

**Data Loss:** None ✅  
**Recovery Time:** <1 minute ✅

---

### Scenario 2: Short Disconnection (1-90 minutes)

**VPS Behavior:**
- Redis Stream fills RAM buffer (~1.7 hours capacity)
- Forwarder continues retrying
- No overflow to QuestDB yet

**Brain Behavior:**
- Stream Consumer in retry loop
- Monitoring alerts: "Brain disconnected"
- No data processing during outage

**Data Loss:** None ✅  
**Recovery Time:** 5-10 minutes (catch-up processing) ✅  
**Buffer Status:** RAM sufficient ✅

---

### Scenario 3: Extended Disconnection (2-24 hours)

**VPS Behavior:**
- Redis Stream fills RAM (~1.7 hours)
- Forwarder switches to QuestDB overflow mode
- Writes to QuestDB with timestamps
- QuestDB accumulates up to 22 hours of data
- Monitoring alerts: "QuestDB buffer in use"

**Brain Behavior:**
- Stream Consumer in retry loop
- Monitoring alerts: "Extended disconnection"
- System waits for reconnection

**Recovery Process:**
1. Brain reconnects
2. Forwarder detects reconnection
3. Reads oldest data from QuestDB
4. Replays to Redis Stream in order
5. Brain processes backlog (fast-forward mode)
6. QuestDB cleanup after confirmation
7. System returns to normal operation

**Data Loss:** None ✅  
**Recovery Time:** 30-60 minutes (depends on backlog size) ✅  
**Buffer Status:** NVMe sufficient for 24 hours ✅

---

### Scenario 4: Critical Failure (>24 hours)

**VPS Behavior:**
- QuestDB buffer fills completely
- Forwarder switches to "drop oldest" mode
- Keeps most recent 24 hours
- Monitoring alerts: "CRITICAL - Buffer full"

**Brain Behavior:**
- Prolonged disconnection
- Manual intervention required

**Data Loss:** Data older than 24 hours ✅ (acceptable trade-off)  
**Recovery Time:** 1-2 hours + manual intervention ⚠️  
**Likelihood:** Extremely rare (>24h outage is catastrophic) ⚠️

**Mitigation:**
- Dual Brain setup (primary + backup)
- VPS alerts at 80% buffer (19 hours)
- Automatic failover to backup Brain

---

## Integration with Existing Setup

### What DOES NOT Change

✅ **Batch Writer** (`redis_to_questdb_v2.py`)
- No modifications needed
- Reads from local Redis as before
- All CPU affinity preserved
- All optimizations preserved

✅ **Tiered Cache** (Day 3/4 Implementation)
- `tiered_market_data.py` unchanged
- `feature_engineering.py` unchanged
- HOT/WARM/COLD tiers work as-is
- All performance characteristics maintained

✅ **QuestDB Schema**
- No schema changes
- Same table structures
- Same partitioning strategy
- Same queries work

✅ **CPU Affinity Configuration**
- All existing affinity settings preserved
- No changes to systemd service files
- Network isolation maintained

✅ **Monitoring & Metrics**
- Existing Prometheus metrics work
- Grafana dashboards compatible
- No metric schema changes

### What DOES Change (Minimal)

**New Component 1: VPS Data Forwarder** (~100 lines Python)
- Location: VPS only
- Function: Redis WARM → Redis Stream
- Fallback: QuestDB on failure
- Monitoring: Connection health, buffer usage

**New Component 2: Brain Stream Consumer** (~50 lines Python)
- Location: Brain only
- Function: Redis Stream → local Redis
- Integration: Writes to `market:binance_spot:trades:*` keys
- Monitoring: Lag, throughput, errors

**Total New Code:** ~150 lines across 2 files

**Configuration Changes:**
- Brain Redis: Allow remote connections from VPS (firewall rule)
- VPS Redis: Configure persistence (RDB + AOF)
- Systemd: 2 new service files (forwarder, consumer)

---

## Deployment Plan

### Phase 1: VPS Setup (Week 1)

**Day 1-2: Infrastructure**
- [ ] Provision Vultr VPS (2 CPU, 4GB RAM, 50GB NVMe)
- [ ] Install Ubuntu 22.04 LTS
- [ ] Install Redis 7.x
- [ ] Install QuestDB 7.x
- [ ] Install Python 3.11+
- [ ] Configure firewall (allow Brain IP only)

**Day 3-4: Data Collection**
- [ ] Deploy exchange collectors
- [ ] Test WebSocket connections
- [ ] Verify data ingestion
- [ ] Monitor CPU/RAM usage
- [ ] Benchmark throughput

**Day 5-7: Forwarder Development**
- [ ] Implement Data Forwarder (100 lines)
- [ ] Test Redis Streams transmission
- [ ] Test QuestDB fallback
- [ ] Implement retry logic
- [ ] Add monitoring metrics

### Phase 2: Brain Integration (Week 2)

**Day 1-2: Consumer Development**
- [ ] Implement Stream Consumer (50 lines)
- [ ] Test local Redis integration
- [ ] Verify batch writer pickup
- [ ] Test end-to-end flow
- [ ] Monitor for errors

**Day 3-4: Validation**
- [ ] Compare data integrity (VPS vs Brain)
- [ ] Measure latency (end-to-end)
- [ ] Verify tiered cache population
- [ ] Test disconnection scenarios
- [ ] Load testing (1K, 5K, 10K ticks/sec)

**Day 5-7: Production Cutover**
- [ ] Parallel run (old + new paths)
- [ ] Monitor for 48 hours
- [ ] Cutover to VPS path
- [ ] Decommission old collectors
- [ ] Update documentation

### Phase 3: Monitoring & Optimization (Week 3+)

**Week 3:**
- [ ] Set up Grafana dashboards (VPS metrics)
- [ ] Configure alerting rules
- [ ] Tune buffer thresholds
- [ ] Optimize network settings
- [ ] Document runbooks

**Ongoing:**
- [ ] Monitor buffer usage trends
- [ ] Optimize forwarder batching
- [ ] Test failover procedures
- [ ] Plan for scaling (more exchanges)

---

## Monitoring & Observability

### VPS Metrics (Prometheus)

**System Metrics:**
- CPU utilization (per core)
- RAM usage (Redis, QuestDB, OS)
- Disk usage (NVMe)
- Network throughput (in/out)
- Network errors/retries

**Application Metrics:**
- Exchange connection status (per pair)
- Ticks received/sec (per symbol)
- Redis Stream depth (message count)
- Redis Stream lag (oldest message age)
- QuestDB buffer size (rows, GB)
- Forwarder send rate (msgs/sec)
- Forwarder retry count
- Forwarder errors

**Custom Alerts:**
- 🔴 CRITICAL: Exchange disconnected >1 min
- 🔴 CRITICAL: QuestDB buffer >80% (19 hours)
- 🟡 WARNING: Redis Stream lag >10 seconds
- 🟡 WARNING: Forwarder retry rate >10/min
- 🟢 INFO: Buffer switched to QuestDB mode

### Brain Metrics (Existing + New)

**Existing Metrics (Preserved):**
- Batch writer throughput
- QuestDB write latency
- Tiered cache hit rates
- All Day 3/4 metrics

**New Metrics:**
- Stream Consumer lag (messages behind)
- Stream Consumer throughput (msgs/sec)
- VPS connection status
- Data staleness (last received timestamp)

**Custom Alerts:**
- 🔴 CRITICAL: VPS disconnected >5 min
- 🟡 WARNING: Stream Consumer lag >1000 messages
- 🟡 WARNING: Data staleness >60 seconds

---

## Cost Analysis

### VPS Costs (Monthly)

**Vultr Pricing (2 CPU, 4GB RAM, 50GB NVMe):**
- Server cost: ~$12-18/month
- Bandwidth: Included (1-2 TB/month)
- Backups: ~$2/month (optional)
- **Total: ~$14-20/month**

**Alternative Providers:**
- DigitalOcean: ~$18/month (similar specs)
- Linode: ~$16/month (similar specs)
- Hetzner: ~$8-12/month (EU locations)

### Operational Costs

**Development Time:**
- Week 1 (VPS setup): ~20 hours
- Week 2 (Integration): ~20 hours
- Week 3 (Monitoring): ~10 hours
- **Total: ~50 hours one-time**

**Maintenance:**
- Monitoring: ~2 hours/month
- Updates: ~1 hour/month
- **Total: ~3 hours/month**

### ROI Analysis

**Benefits:**
- 24-hour buffer capacity (vs 0 currently)
- Reduced Brain network load
- Scalable to multiple exchanges
- Exchange-proximity deployment (lower latency)
- Independent scaling of ingestion vs trading

**Costs:**
- $14-20/month infrastructure
- ~50 hours initial development
- ~3 hours/month maintenance

**Payback Period:** <1 month (if disconnections cause missed trades)

---

## Risks & Mitigations

### Risk 1: VPS Downtime

**Impact:** Data ingestion stops, Brain has no new data

**Probability:** Low (~99.9% uptime = ~45 min/month)

**Mitigation:**
- Dual VPS setup (primary + backup)
- Automatic failover (Floating IP)
- Brain can revert to direct exchange connection (manual)

### Risk 2: Network Partition

**Impact:** VPS and Brain cannot communicate

**Probability:** Medium (ISP issues, routing problems)

**Mitigation:**
- 24-hour buffer on VPS (survives overnight)
- Automatic reconnection with replay
- Monitoring alerts at 19 hours (80% buffer)

### Risk 3: Redis Stream Corruption

**Impact:** Data loss or duplicate messages

**Probability:** Very Low (Redis Streams are durable)

**Mitigation:**
- Redis persistence (RDB + AOF)
- Consumer group acknowledgment (prevents duplicates)
- QuestDB buffer as backup (can replay)

### Risk 4: Brain Processing Backlog

**Impact:** Slow catch-up after extended disconnection

**Probability:** Low (only after >1 hour disconnection)

**Mitigation:**
- Batch writer designed for high throughput (10K+ ticks/sec)
- Fast-forward mode (prioritize recent data)
- Temporary rate limiting on trading during catch-up

### Risk 5: Message Ordering

**Impact:** Out-of-order tick processing

**Probability:** Very Low (Redis Streams preserve order)

**Mitigation:**
- Single stream per exchange (ordering guaranteed)
- Timestamp-based ordering in Brain (fallback)
- Consumer group ensures sequential consumption

---

## Future Enhancements

### Phase 4: Multi-Exchange Support

**Objective:** Add Coinbase and Kraken (150 total symbols)

**Changes:**
- VPS: Upgrade to 4 CPU, 8GB RAM, 100GB NVMe
- Multiple Redis Streams (one per exchange)
- Load balancing across forwarders
- **Timeline:** Q1 2026

### Phase 5: Multi-Region Redundancy

**Objective:** Deploy VPS in multiple regions (US, EU, Asia)

**Architecture:**
- Primary VPS: AWS Tokyo (Binance proximity)
- Backup VPS: AWS Frankfurt (EU markets)
- Failover: Automatic via health checks
- **Timeline:** Q2 2026

### Phase 6: Real-Time Analytics

**Objective:** Pre-process features on VPS before sending to Brain

**Components:**
- Feature engineering on VPS (Day 3 code reused)
- Send enriched data (not raw ticks)
- Reduce Brain processing load
- **Timeline:** Q3 2026

### Phase 7: WebSocket Tunneling

**Objective:** Replace Redis Streams with WebSocket for ultra-low latency

**Benefits:**
- ~0.5-1ms latency (vs 2-5ms)
- More flexible protocol
- Easier debugging

**Trade-offs:**
- No built-in acknowledgment (must implement)
- More complex error handling
- **Timeline:** Q4 2026 (if needed)

---

## Testing Strategy

### Unit Tests

**VPS Components:**
- Exchange collector WebSocket handling
- Redis WARM cache operations
- QuestDB buffer writes
- Forwarder retry logic
- Error handling paths

**Brain Components:**
- Stream Consumer message parsing
- Local Redis writes
- Connection failure handling
- Lag detection

### Integration Tests

**End-to-End:**
- VPS ingestion → Brain processing → QuestDB storage
- Data integrity (no loss, no duplicates)
- Timestamp preservation
- Order preservation

**Disconnection Scenarios:**
- 1-minute glitch (RAM buffer)
- 2-hour outage (QuestDB buffer)
- 12-hour outage (QuestDB buffer)
- 24-hour outage (buffer limit)

### Load Tests

**VPS Capacity:**
- 1,200 ticks/sec (normal load)
- 5,000 ticks/sec (peak load)
- 10,000 ticks/sec (stress test)

**Brain Capacity:**
- Backlog processing (10K messages)
- Fast-forward mode (100K messages)
- Concurrent trading + catch-up

### Chaos Engineering

**Scenarios:**
- Random VPS restarts
- Network packet loss (1%, 5%, 10%)
- Redis crashes
- QuestDB crashes
- Brain disconnections at random intervals

---

## Success Criteria

### Functional Requirements

✅ **Data Integrity:**
- Zero data loss during disconnections <24 hours
- No duplicate ticks
- Timestamp accuracy within ±1ms

✅ **Latency:**
- End-to-end latency <25ms (P95)
- Catch-up processing <60 minutes for 24h backlog

✅ **Reliability:**
- VPS uptime >99.9%
- Successful reconnection >99.9%
- Buffer capacity validated at 24 hours

✅ **Compatibility:**
- Existing batch writer works unchanged
- Existing cache works unchanged
- Existing monitoring works unchanged

### Performance Requirements

✅ **Throughput:**
- VPS handles 1,200 ticks/sec with <50% CPU
- Brain processes backlog at 5,000+ ticks/sec

✅ **Resource Usage:**
- VPS RAM <75% (3GB of 4GB)
- VPS Disk <80% (40GB of 50GB)
- Brain CPU unchanged

✅ **Scalability:**
- Ready to scale to 3 exchanges (450 symbols)
- Ready to scale to multiple Brain machines

---

## Documentation Deliverables

### Technical Documentation

- [x] This architecture document
- [ ] VPS setup guide (step-by-step)
- [ ] Forwarder code documentation
- [ ] Consumer code documentation
- [ ] Deployment runbook
- [ ] Troubleshooting guide
- [ ] Monitoring dashboard guide

### Operational Documentation

- [ ] VPS maintenance procedures
- [ ] Disaster recovery procedures
- [ ] Scaling procedures
- [ ] Cost optimization guide
- [ ] Performance tuning guide

---

## Appendix

### A. Redis Streams Commands

**Producer (VPS Forwarder):**
```bash
# Add message to stream
XADD market:vps:stream * symbol BTCUSDT price 67234.56 volume 0.15 ...

# Check stream length
XLEN market:vps:stream

# View recent messages
XRANGE market:vps:stream - + COUNT 10
```

**Consumer (Brain):**
```bash
# Create consumer group
XGROUP CREATE market:vps:stream brain-consumer-group 0 MKSTREAM

# Read messages
XREADGROUP GROUP brain-consumer-group consumer1 COUNT 100 STREAMS market:vps:stream >

# Acknowledge message
XACK market:vps:stream brain-consumer-group <message-id>

# Check pending messages
XPENDING market:vps:stream brain-consumer-group
```

### B. QuestDB Buffer Queries

**Write to buffer:**
```sql
INSERT INTO vps_buffer (timestamp, symbol, exchange, price, volume, side)
VALUES (to_timestamp('2025-10-26T12:34:56.789012Z'), 'BTCUSDT', 'binance_spot', 67234.56, 0.15, 'BUY');
```

**Read oldest data:**
```sql
SELECT * FROM vps_buffer
ORDER BY timestamp ASC
LIMIT 1000;
```

**Cleanup after transmission:**
```sql
DELETE FROM vps_buffer
WHERE timestamp < dateadd('h', -1, now());
```

### C. Monitoring Queries

**VPS buffer usage:**
```sql
SELECT
    count() as message_count,
    max(timestamp) as newest_message,
    min(timestamp) as oldest_message,
    datediff('h', min(timestamp), max(timestamp)) as buffer_hours
FROM vps_buffer;
```

**Stream lag (Redis CLI):**
```bash
redis-cli XINFO GROUPS market:vps:stream
# Look for "lag" field (messages not yet consumed)
```

---

## Summary

**Target Architecture:**
- VPS (Vultr): 2 CPU, 4GB RAM, 50GB NVMe
- Exchange: Binance only (50 pairs)
- Protocol: Redis Streams
- Buffer: 24 hours capacity
- Latency: +2-5ms overhead
- Cost: ~$14-20/month
- Development: ~50 hours one-time
- Compatibility: 100% with existing setup

**Key Benefits:**
- ✅ 24-hour disconnection resilience
- ✅ Scalable to multiple exchanges
- ✅ Preserves all Day 3/4 optimizations
- ✅ Minimal code changes (~150 lines)
- ✅ Low operational cost

**Next Steps:**
1. Provision Vultr VPS
2. Deploy exchange collectors
3. Implement Data Forwarder
4. Implement Stream Consumer
5. Test end-to-end
6. Production cutover

**Status:** Ready for implementation ✅

---

*Document Version: 1.0*  
*Last Updated: October 26, 2025*  
*Author: GitHub Copilot + User*  
*Related Documents: DAY4_COMPLETION_REPORT.md, DAY4_IMPLEMENTATION_SUMMARY.md*
