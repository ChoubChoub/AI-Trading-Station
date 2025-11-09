# QuestDB Orderbook Data Ingestion Issue - Technical Analysis

**Date:** November 6, 2025  
**Issue Type:** Data Pipeline - Silent Data Loss  
**Priority:** HIGH - Market data orderbook showing 0 events/sec since November 4, 2025  
**Status:** Under Investigation - Consultation with Senior IT Engineer Required  

## Executive Summary

The market orderbook data pipeline has been failing silently since November 4, 2025. While the batch writer reports successful HTTP 204 responses from QuestDB, no new orderbook data is appearing in the `market_orderbook` table. Trade data continues to flow normally through the same pipeline.

## Current System State

### ✅ Working Components
- **Redis Streams**: Contains 100k+ orderbook messages from Binance
- **Batch Writer Service**: Running without errors, processing messages
- **QuestDB Server**: Responding to HTTP requests (HTTP 204 success)
- **Trade Data Pipeline**: Working normally, ingesting successfully

### ❌ Failing Components  
- **Orderbook Data Persistence**: Data not appearing in QuestDB table despite successful HTTP responses
- **Last Successful Ingestion**: 2025-11-04T01:15:02.325347Z

## Technical Investigation Summary

### 1. Initial Hypothesis: Separate Timer Issue ❌
- **Theory**: Orderbook and trade flush timers were interfering
- **Action**: Implemented separate independent timers
- **Result**: No change - HTTP 204 responses continue but no data persistence

### 2. Network/Port Configuration Investigation ❌
- **Theory**: Wrong QuestDB endpoint (port 9000 vs 9009)
- **Action**: Tested port 9009, reverted to 9000
- **Result**: Port 9000 confirmed correct - HTTP 204 responses working

### 3. Current Focus: Data Format/Schema Mismatch 🔍
- **Observation**: HTTP 204 success but zero persistence suggests schema/format issue
- **Evidence**: Existing data vs ILP format mismatch

## Data Format Analysis

### Existing QuestDB Data Format
```json
{
  "symbol": "BTCUSDT",
  "exchange": "binance_spot", 
  "bid_price": 106623.75,
  "bid_qty": 2.89075,
  "ask_price": 106623.76,
  "ask_qty": 0.0038,
  "timestamp": "2025-11-04T01:15:02.325347Z"
}
```

### Current ILP Format Being Sent
```text
market_orderbook,symbol=BTCUSDT,exchange=binance_spot bid_price=103120.01,bid_qty=4.29387,ask_price=103120.02,ask_qty=0.7818 1762425860225734000
```

### Table Schema
```sql
columns: [
  {"name": "symbol", "type": "SYMBOL"},
  {"name": "exchange", "type": "SYMBOL"}, 
  {"name": "bid_price", "type": "DOUBLE"},
  {"name": "bid_qty", "type": "DOUBLE"},
  {"name": "ask_price", "type": "DOUBLE"},
  {"name": "ask_qty", "type": "DOUBLE"},
  {"name": "timestamp", "type": "TIMESTAMP"}
]
```

## Key Observations

### 1. HTTP Response Pattern
- **Consistent HTTP 204**: All requests return successful status
- **Empty Response Body**: No error messages or content
- **No Connection Errors**: Network communication working

### 2. Timestamp Discrepancy
- **Existing Data**: ISO format timestamps (`2025-11-04T01:15:02.325347Z`)  
- **ILP Format**: Nanosecond epoch timestamps (`1762425860225734000`)
- **Potential Issue**: Timestamp format conversion or precision mismatch

### 3. Data Volume
- **Redis Queue**: 100k+ messages waiting
- **Processing Rate**: ~500-2000 messages/batch
- **Success Rate**: 100% HTTP 204 responses
- **Persistence Rate**: 0% (no data in table)

## Timeline of Events

### November 4, 2025
- **01:15:02**: Last successful orderbook data ingestion
- **Later**: Scripts moved/reorganized in workspace  
- **Result**: Silent failure begins

### November 6, 2025
- **Investigation Started**: Discovered zero orderbook events/sec
- **Root Cause Analysis**: Confirmed Redis data exists, HTTP responses successful
- **Current Status**: Data format/schema investigation ongoing

## Technical Details

### Batch Writer Configuration
- **File**: `/home/youssefbahloul/ai-trading-station/Services/QuestDB/Runtime/redis_to_questdb_v2.py`
- **Protocol**: ILP over HTTP (InfluxDB Line Protocol)
- **Endpoint**: `http://localhost:9000/write`
- **Batch Size**: 2000 records
- **Workers**: 8 concurrent workers

### QuestDB Configuration  
- **Version**: 9.1.0
- **HTTP Port**: 9000 (confirmed working)
- **ILP Port**: 9009 (TCP protocol)
- **Existing Data**: 117M+ historical orderbook records

### Redis Stream Status
```bash
# Stream length check
XLEN market:binance_spot:orderbook:BTCUSDT: ~100,000 messages
XLEN market:binance_spot:orderbook:ETHUSDT: ~100,000 messages
```

## Questions for Senior IT Engineer (Opus)

### 1. QuestDB ILP Protocol
- Is HTTP POST to `/write` endpoint the correct method for ILP ingestion?
- Could there be a schema validation issue causing silent drops?
- Should we use TCP ILP (port 9009) instead of HTTP ILP (port 9000)?

### 2. Timestamp Format
- How should nanosecond epoch timestamps be formatted for QuestDB ILP?
- Could timestamp precision mismatch cause silent data drops?
- Is there a QuestDB configuration for timestamp format validation?

### 3. Data Pipeline Architecture
- Why would HTTP 204 be returned if data isn't being persisted?
- Could there be a QuestDB buffer/commit issue?
- Are there QuestDB logs we should examine for ingestion errors?

### 4. Schema Compatibility
- Could symbol/tag ordering in ILP format affect ingestion?
- Should existing table schema match exactly with ILP field order?
- Is there a way to test ILP format validation separately?

## Debug Information Available

### 1. Real-time Logs
```bash
journalctl -u batch-writer.service -f
# Shows: HTTP 204 responses, successful flush messages, proper ILP format
```

### 2. QuestDB Query Results  
```bash
curl -s -G "http://localhost:9000/exec" --data-urlencode "query=SELECT COUNT(*) FROM market_orderbook WHERE timestamp > '2025-11-04'"
# Returns: 0 rows
```

### 3. Redis Stream Data
```bash  
redis-cli XRANGE market:binance_spot:orderbook:BTCUSDT - + COUNT 1
# Returns: Valid orderbook data with proper format
```

## Immediate Action Required

1. **Schema Validation**: Test ILP format against existing table schema
2. **Manual ILP Test**: Send simple test record to verify basic functionality  
3. **QuestDB Logs**: Examine server logs for ingestion warnings/errors
4. **Timestamp Format**: Verify nanosecond epoch format compatibility
5. **Alternative Protocol**: Consider TCP ILP if HTTP ILP has issues

## Impact Assessment

- **Business Impact**: Real-time orderbook monitoring offline for 2+ days
- **Data Loss**: No permanent loss (data preserved in Redis streams)  
- **Recovery**: Can replay 100k+ messages once issue resolved
- **Trade Data**: Unaffected - continues working normally

---

**Next Steps**: Await Senior IT Engineer analysis and recommendations for QuestDB ILP debugging approach.