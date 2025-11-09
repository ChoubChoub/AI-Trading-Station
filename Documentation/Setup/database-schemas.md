# QuestDB Schema Design Assessment
**Date:** October 18, 2025  
**Purpose:** Optimize table schemas for 20-strategy crypto trading system

## Current Proposal Analysis

### ✅ Good Design Choices

1. **SYMBOL types for categorical data** - Excellent! (symbol, exchange, strategy_id)
2. **WAL enabled** - Critical for write durability
3. **PARTITION BY DAY/HOUR** - Good lifecycle management
4. **INDEX on strategy_id** - Enables fast per-strategy queries
5. **timestamp TIMESTAMP** - Designated timestamp column (required)

### ⚠️ Issues & Improvements Needed

#### 1. **crypto_ticks** Table

**Issues:**
- ❌ Missing `INDEX` on commonly queried symbols
- ❌ No deduplication strategy (duplicate ticks from multiple sources)
- ❌ PARTITION BY DAY might be too coarse for hot/cold tiering (30 days → 30 partitions minimum)
- ⚠️ Storing bid/ask separately is redundant with orderbook_snapshots

**Recommended Changes:**
```sql
CREATE TABLE IF NOT EXISTS crypto_ticks (
    symbol SYMBOL capacity 256 CACHE INDEX,  -- INDEX for fast symbol queries
    exchange SYMBOL capacity 32 CACHE INDEX, -- INDEX for per-exchange analysis
    price DOUBLE,
    volume DOUBLE,
    timestamp TIMESTAMP,
    DEDUP UPSERT KEYS(timestamp, symbol, exchange)  -- Prevent duplicates
) timestamp(timestamp) PARTITION BY DAY WAL;
```

**Why:**
- INDEX on symbol: Query specific coins (BTC, ETH) instantly
- INDEX on exchange: Compare prices across exchanges
- DEDUP: Prevent duplicate ticks if feed reconnects
- Removed bid/ask: Use orderbook_snapshots instead (cleaner separation)

#### 2. **strategy_signals** Table

**Issues:**
- ❌ PARTITION BY HOUR is too granular (20 strategies × 24 hours = 480 partitions/day)
- ❌ Missing INDEX on symbol (queries often filter by symbol + strategy)
- ⚠️ signal INT is inefficient (use SYMBOL for -1/0/1)

**Recommended Changes:**
```sql
CREATE TABLE IF NOT EXISTS strategy_signals (
    strategy_id SYMBOL capacity 64 CACHE INDEX,
    symbol SYMBOL capacity 256 CACHE INDEX,      -- Added INDEX
    signal SYMBOL capacity 4 CACHE,              -- 'SHORT', 'NEUTRAL', 'LONG'
    confidence DOUBLE,
    expected_return DOUBLE,
    position_size DOUBLE,                        -- Added: position sizing
    timestamp TIMESTAMP
) timestamp(timestamp) PARTITION BY DAY WAL;     -- Changed to DAY
```

**Why:**
- PARTITION BY DAY: Reduces partition count (20 strategies manageable)
- INDEX on symbol: Fast "show me all signals for BTC" queries
- SYMBOL for signal: Better compression, faster filtering
- position_size: Essential for risk management

#### 3. **strategy_performance** Table

**Issues:**
- ❌ Single row per strategy per day doesn't capture intraday performance
- ❌ Missing cumulative metrics (total PnL, total trades)
- ⚠️ No timestamp range for metrics (daily? inception-to-date?)

**Recommended Changes:**
```sql
CREATE TABLE IF NOT EXISTS strategy_performance (
    strategy_id SYMBOL capacity 64 CACHE INDEX,
    interval SYMBOL capacity 16 CACHE,           -- 'HOUR', 'DAY', 'WEEK', 'MONTH'
    pnl DOUBLE,
    pnl_cumulative DOUBLE,                       -- Running total
    sharpe_ratio DOUBLE,
    max_drawdown DOUBLE,
    win_rate DOUBLE,
    trades_count LONG,
    trades_cumulative LONG,                      -- Total trades since inception
    active_positions INT,                        -- Current open positions
    timestamp TIMESTAMP
) timestamp(timestamp) PARTITION BY DAY;         -- No WAL (analytics table)
```

**Why:**
- interval: Support multiple timeframes (hourly monitoring, daily reports)
- Cumulative metrics: Track total performance since strategy start
- active_positions: Monitor current exposure
- No WAL: Performance data can be recalculated if lost

#### 4. **orderbook_snapshots** Table

**Issues:**
- ❌ STRING for bids/asks is inefficient (can't query levels)
- ❌ PARTITION BY HOUR creates too many partitions (10 exchanges × 24 hours = 240/day)
- ⚠️ Missing depth metrics (total bid/ask volume)

**Recommended Changes:**
```sql
-- Option A: Flattened structure (recommended for analytics)
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    exchange SYMBOL capacity 32 CACHE INDEX,
    symbol SYMBOL capacity 256 CACHE INDEX,
    level INT,                                   -- 0=best, 1=second best, etc.
    bid_price DOUBLE,
    bid_volume DOUBLE,
    ask_price DOUBLE,
    ask_volume DOUBLE,
    mid_price DOUBLE,
    spread DOUBLE,
    depth_5_bps DOUBLE,                          -- Liquidity within 5bps
    timestamp TIMESTAMP
) timestamp(timestamp) PARTITION BY DAY WAL;

-- Option B: Keep JSON but add indexed columns
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    exchange SYMBOL capacity 32 CACHE INDEX,
    symbol SYMBOL capacity 256 CACHE INDEX,
    best_bid DOUBLE,                             -- Indexed for fast queries
    best_ask DOUBLE,
    mid_price DOUBLE,
    spread DOUBLE,
    bids_json STRING,                            -- Full depth as JSON
    asks_json STRING,
    depth_5_bps DOUBLE,                          -- Liquidity metric
    timestamp TIMESTAMP
) timestamp(timestamp) PARTITION BY DAY WAL;
```

**Why:**
- Flattened structure: Can query "show me best bid history" without JSON parsing
- PARTITION BY DAY: 7-day retention = 7 partitions (manageable)
- depth_5_bps: Critical for execution quality analysis

#### 5. **ml_predictions** Table

**Issues:**
- ✅ Actually quite good!
- ⚠️ STRING for features might be large (consider compression)

**Recommended Changes:**
```sql
CREATE TABLE IF NOT EXISTS ml_predictions (
    model_id SYMBOL capacity 64 CACHE INDEX,
    symbol SYMBOL capacity 256 CACHE INDEX,      -- Added INDEX
    prediction_type SYMBOL capacity 32 CACHE,
    prediction_value DOUBLE,
    confidence DOUBLE,
    actual_value DOUBLE,                         -- Added: for accuracy tracking
    error DOUBLE,                                -- Added: prediction - actual
    features STRING,
    timestamp TIMESTAMP
) timestamp(timestamp) PARTITION BY DAY WAL;     -- Changed to DAY
```

**Why:**
- INDEX on symbol: Fast per-coin prediction queries
- actual_value/error: Track model accuracy over time
- PARTITION BY DAY: Hourly partitioning unnecessary

### 🆕 Additional Recommended Tables

#### 6. **regime_states** (Critical for Multi-Strategy!)

```sql
-- Market regime classification for strategy selection
CREATE TABLE IF NOT EXISTS regime_states (
    symbol SYMBOL capacity 256 CACHE INDEX,
    regime SYMBOL capacity 32 CACHE,             -- 'TRENDING', 'RANGING', 'VOLATILE', 'CALM'
    volatility DOUBLE,
    trend_strength DOUBLE,
    correlation_btc DOUBLE,
    dominant_strategies STRING,                  -- JSON: ['momentum', 'mean_reversion']
    timestamp TIMESTAMP
) timestamp(timestamp) PARTITION BY DAY WAL;
```

**Why:** Your system has 20 strategies - knowing which regime you're in determines which strategies to weight heavily.

#### 7. **execution_quality** (For Multi-Exchange Optimization)

```sql
-- Track execution quality across exchanges
CREATE TABLE IF NOT EXISTS execution_quality (
    exchange SYMBOL capacity 32 CACHE INDEX,
    symbol SYMBOL capacity 256 CACHE INDEX,
    strategy_id SYMBOL capacity 64 CACHE,
    order_type SYMBOL capacity 16 CACHE,         -- 'MARKET', 'LIMIT', 'IOC'
    requested_price DOUBLE,
    executed_price DOUBLE,
    slippage DOUBLE,
    latency_ms INT,
    filled_volume DOUBLE,
    timestamp TIMESTAMP
) timestamp(timestamp) PARTITION BY DAY WAL;
```

**Why:** With 10+ exchanges, you need to know which exchange has best execution for each strategy.

#### 8. **risk_metrics** (Real-time Risk Monitoring)

```sql
-- Real-time portfolio risk metrics
CREATE TABLE IF NOT EXISTS risk_metrics (
    portfolio_id SYMBOL capacity 32 CACHE,       -- 'MAIN', 'STRATEGY_1', etc.
    total_exposure DOUBLE,
    var_95 DOUBLE,                               -- Value at Risk (95%)
    expected_shortfall DOUBLE,
    correlation_risk DOUBLE,                     -- Cross-asset correlation
    leverage DOUBLE,
    margin_usage_pct DOUBLE,
    timestamp TIMESTAMP
) timestamp(timestamp) PARTITION BY HOUR WAL;    -- Hourly risk snapshots
```

**Why:** JP Morgan risk management background - you need VAR and margin monitoring!

## Storage Estimates (20 Strategies, 10 Exchanges)

### Hot Storage (30 days on NVMe)

| Table | Records/Day | Size/Record | Daily Size | 30-Day Size |
|-------|-------------|-------------|------------|-------------|
| crypto_ticks | 86M | 48 bytes | 4.1 GB | 123 GB |
| strategy_signals | 2M | 64 bytes | 128 MB | 3.8 GB |
| orderbook_snapshots | 86M | 80 bytes | 6.9 GB | 207 GB |
| ml_predictions | 5M | 96 bytes | 480 MB | 14.4 GB |
| regime_states | 1M | 72 bytes | 72 MB | 2.2 GB |
| **TOTAL** | | | **11.7 GB/day** | **350 GB/30 days** ✅ |

**Fits comfortably in 3.6TB NVMe!**

### Cold Storage (365 days on HDD)

| Table | 365-Day Size |
|-------|--------------|
| crypto_ticks | 1.5 TB |
| strategy_signals | 47 GB |
| orderbook_snapshots | 2.5 TB |
| Others | 200 GB |
| **TOTAL** | **~4.2 TB** ✅ |

**Fits in 7.3TB HDD with room to grow!**

## Recommended Partitioning Strategy

```
crypto_ticks:           PARTITION BY DAY    (30 partitions hot, 365 cold)
strategy_signals:       PARTITION BY DAY    (30 partitions hot)
strategy_performance:   PARTITION BY DAY    (30 partitions hot)
orderbook_snapshots:    PARTITION BY DAY    (7 partitions hot, then DROP)
ml_predictions:         PARTITION BY DAY    (30 partitions hot)
regime_states:          PARTITION BY DAY    (30 partitions hot, 365 cold)
execution_quality:      PARTITION BY DAY    (90 partitions for analysis)
risk_metrics:           PARTITION BY HOUR   (720 partitions = 30 days × 24 hours)
```

## Index Strategy

**Add INDEX on:**
- All SYMBOL columns used in WHERE clauses
- strategy_id (most queries filter by strategy)
- symbol (most queries filter by coin)
- exchange (for exchange comparison)

**Skip INDEX on:**
- Timestamp columns (designated timestamp is auto-indexed)
- Numeric metrics (price, volume, pnl)
- Rarely queried columns

## DEDUP Strategy

**Enable DEDUP on:**
- crypto_ticks (prevent duplicate ticks)
- orderbook_snapshots (prevent duplicate snapshots)

**Skip DEDUP on:**
- strategy_signals (signals can change rapidly)
- ml_predictions (multiple models can predict at same time)
- performance metrics (always additive)

## Query Optimization Examples

### Fast Strategy Performance Query
```sql
-- Show top 3 performing strategies today
SELECT strategy_id, SUM(pnl) as daily_pnl
FROM strategy_performance
WHERE timestamp >= today()
GROUP BY strategy_id
ORDER BY daily_pnl DESC
LIMIT 3;
```

### Cross-Exchange Arbitrage Detection
```sql
-- Find arbitrage opportunities (price differences >0.5%)
SELECT a.symbol, a.exchange as buy_exchange, a.price as buy_price,
       b.exchange as sell_exchange, b.price as sell_price,
       ((b.price - a.price) / a.price * 100) as profit_pct
FROM crypto_ticks a
ASOF JOIN crypto_ticks b ON (symbol)
WHERE a.timestamp >= now() - INTERVAL '1' MINUTE
  AND ((b.price - a.price) / a.price * 100) > 0.5
ORDER BY profit_pct DESC;
```

### Strategy Correlation Analysis
```sql
-- Calculate strategy PNL correlation
SELECT s1.strategy_id as strategy_a,
       s2.strategy_id as strategy_b,
       corr(s1.pnl, s2.pnl) as correlation
FROM strategy_performance s1
JOIN strategy_performance s2 ON (s1.timestamp = s2.timestamp)
WHERE s1.timestamp >= dateadd('d', -30, now())
  AND s1.strategy_id != s2.strategy_id
GROUP BY s1.strategy_id, s2.strategy_id
ORDER BY correlation DESC;
```

## Recommendations Summary

### ✅ Keep from Original
- Overall table structure concept
- SYMBOL types for categorical data
- WAL on transactional tables
- Timestamp columns

### ✅ Add These Improvements
1. INDEX on symbol, exchange, strategy_id
2. DEDUP on tick and orderbook tables
3. PARTITION BY DAY instead of HOUR (reduces partition count)
4. Cumulative metrics in performance table
5. regime_states table (critical for multi-strategy!)
6. execution_quality table (multi-exchange optimization)
7. risk_metrics table (real-time risk monitoring)

### ❌ Remove/Change
- Bid/ask from crypto_ticks (use orderbook instead)
- JSON strings for orderbook (flatten or add indexed columns)
- PARTITION BY HOUR on high-volume tables
- INT for signals (use SYMBOL instead)

## Next Steps

1. Review this assessment
2. Approve final schema design
3. Create `/home/youssefbahloul/ai-trading-station/QuestDB/schemas/` directory
4. Generate optimized SQL files
5. Test schema with sample data
6. Integrate with monitoring dashboard

