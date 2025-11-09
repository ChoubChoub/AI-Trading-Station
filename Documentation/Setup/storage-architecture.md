# QuestDB Hot/Cold Storage - How It Actually Works
**Date:** October 18, 2025

## ⚠️ IMPORTANT: Do NOT Move Partition Files Manually!

Moving QuestDB partition files while the database is running **will corrupt your data**. QuestDB maintains internal metadata about partition locations.

## How QuestDB's Cold Storage Works

### 1. Configuration (Already Done ✅)
```properties
# Hot storage (NVMe)
cairo.root=/home/youssefbahloul/ai-trading-station/QuestDB/data/hot

# Cold storage (HDD) - ACTIVE
cairo.cold.storage.root=/mnt/hdd/questdb/cold
```

### 2. Partition Lifecycle (Manual SQL Commands)

QuestDB does **not** automatically move partitions. You must explicitly detach and attach partitions using SQL.

#### Drop Old Partitions (Delete Data)
```sql
-- Delete partitions older than 30 days for crypto_ticks table
ALTER TABLE crypto_ticks DROP PARTITION 
WHERE timestamp < dateadd('d', -30, now());
```

#### Detach Partitions (Preserve Data)
```sql
-- Detach specific partition (keeps data on disk but QuestDB won't query it)
ALTER TABLE crypto_ticks DETACH PARTITION LIST '2025-09-18';

-- Detach multiple partitions
ALTER TABLE crypto_ticks DETACH PARTITION 
WHERE timestamp < dateadd('d', -30, now());
```

### 3. Manual File Movement (Only After Detach!)

**ONLY after detaching partitions**, you can safely move files:

```bash
# 1. Detach partitions via SQL first (see above)

# 2. Wait for QuestDB to complete the detach operation

# 3. Now it's safe to move the .detached files
cd /home/youssefbahloul/ai-trading-station/QuestDB/data/hot/db/crypto_ticks
mv *.detached /mnt/hdd/questdb/cold/crypto_ticks/
```

### 4. Reattach from Cold Storage (If Needed)

```bash
# 1. Copy partition back to hot storage
cp -r /mnt/hdd/questdb/cold/crypto_ticks/2025-09-18.detached \
     /home/youssefbahloul/ai-trading-station/QuestDB/data/hot/db/crypto_ticks/

# 2. Attach via SQL
ALTER TABLE crypto_ticks ATTACH PARTITION LIST '2025-09-18';
```

## Recommended Approach: SQL-Based Retention Policies

### Option A: Simple DROP (Production Recommended)
```sql
-- Run daily via cron: delete data older than retention period
-- crypto_ticks: keep 365 days
ALTER TABLE crypto_ticks DROP PARTITION 
WHERE timestamp < dateadd('d', -365, now());

-- orderbook_snapshots: keep 90 days
ALTER TABLE orderbook_snapshots DROP PARTITION 
WHERE timestamp < dateadd('d', -90, now());
```

**Advantages:**
- ✅ Simple and reliable
- ✅ No file management needed
- ✅ QuestDB handles everything
- ✅ No corruption risk

**Disadvantages:**
- ❌ Data is permanently deleted (not archived)

### Option B: Detach + Archive (For Historical Analysis)
```sql
-- Detach old partitions
ALTER TABLE crypto_ticks DETACH PARTITION 
WHERE timestamp < dateadd('d', -30, now());
```

Then manually archive:
```bash
#!/bin/bash
# Archive detached partitions to cold storage
HOT="/home/youssefbahloul/ai-trading-station/QuestDB/data/hot/db"
COLD="/mnt/hdd/questdb/cold"

for table in crypto_ticks orderbook_snapshots regime_states; do
    if [ -d "$HOT/$table" ]; then
        find "$HOT/$table" -name "*.detached" -type d | while read partition; do
            echo "Archiving: $partition"
            mkdir -p "$COLD/$table"
            mv "$partition" "$COLD/$table/"
        done
    fi
done
```

**Advantages:**
- ✅ Data preserved for historical analysis
- ✅ Can be reattached if needed
- ✅ Safe (partitions already detached)

**Disadvantages:**
- ❌ More complex
- ❌ Manual file management
- ❌ Query performance degraded for old data

## What We Already Have

QuestDB is **already configured** to use both hot and cold storage:
- Hot: NVMe (3.6TB) for active queries
- Cold: HDD (7.3TB) for future manual archival

The `cairo.cold.storage.root` setting tells QuestDB **where cold storage is**, but QuestDB won't automatically move data there. You control the lifecycle via SQL commands.

## Production Strategy Recommendation

### For Crypto Trading (High-Frequency Data)

**Keep it simple with DROP:**

```sql
-- Daily cleanup script (run at 2 AM via cron)
-- Keep only data within retention periods

-- Tick data: 365 days (1 year for backtesting)
ALTER TABLE crypto_ticks DROP PARTITION 
WHERE timestamp < dateadd('d', -365, now());

-- Orderbook: 90 days (sufficient for most strategies)
ALTER TABLE orderbook_snapshots DROP PARTITION 
WHERE timestamp < dateadd('d', -90, now());

-- Regime states: 365 days
ALTER TABLE regime_states DROP PARTITION 
WHERE timestamp < dateadd('d', -365, now());
```

**Why DROP instead of archive?**
1. ✅ **Simplicity**: No file management, no corruption risk
2. ✅ **Performance**: Keep hot storage fast and lean
3. ✅ **Cost-effective**: 1 year of data is plenty for backtesting
4. ✅ **Storage**: Even 1 year of crypto ticks fits comfortably in 3.6TB NVMe

**If you need older data:**
- Download fresh historical data from exchanges (faster than managing archives)
- Exchanges provide 3+ years of historical data for free
- Data quality from source is better than aged local archives

## Cron Schedule

```bash
# Add to crontab
crontab -e

# Run retention cleanup daily at 2 AM (low trading activity)
0 2 * * * /home/youssefbahloul/ai-trading-station/QuestDB/scripts/retention-cleanup.sh
```

## Summary

❌ **Don't do this:** Move partition files with `mv` while QuestDB is running  
✅ **Do this:** Use SQL `DROP PARTITION` or `DETACH PARTITION` commands  
✅ **Best practice:** Simple DROP with reasonable retention periods (365 days)  
✅ **We're ready:** Cold storage configured, just need to create retention script when tables are created

