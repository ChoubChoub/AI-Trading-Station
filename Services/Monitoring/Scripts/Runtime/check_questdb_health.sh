#!/bin/bash
################################################################################
# QuestDB Health Check Script
# Monitors WAL status, data freshness, and transaction lag
# Author: Combined proposal from Opus + Copilot
# Date: 2025-11-06
################################################################################

LOG_FILE="/home/youssefbahloul/ai-trading-station/Services/QuestDB/logs/health_check.log"
ALERT_FILE="/tmp/questdb_alert_sent"
WEBHOOK_URL="${QUESTDB_ALERT_WEBHOOK:-}"  # Optional: Set via environment variable

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to send alert (optional webhook)
send_alert() {
    local message="$1"
    
    # Prevent alert spam - only send once per hour
    if [ -f "$ALERT_FILE" ]; then
        last_alert=$(stat -c %Y "$ALERT_FILE" 2>/dev/null || echo 0)
        now=$(date +%s)
        elapsed=$((now - last_alert))
        if [ $elapsed -lt 3600 ]; then
            return  # Skip alert, sent too recently
        fi
    fi
    
    # Send to webhook if configured
    if [ -n "$WEBHOOK_URL" ]; then
        curl -s -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"QuestDB Alert: $message\"}" > /dev/null 2>&1
    fi
    
    # Log to syslog
    logger -t questdb_health "ALERT: $message"
    
    # Mark alert as sent
    touch "$ALERT_FILE"
}

# Check 0: System Health - Write Cache & Power Events
log "Checking system health..."

# Check write cache status (should be "write through")
WRITE_CACHE=$(cat /sys/block/nvme0n1/queue/write_cache 2>/dev/null || echo "unknown")
if [ "$WRITE_CACHE" != "write through" ]; then
    log "🔴 CRITICAL: Write cache not in safe mode! Current: $WRITE_CACHE (Expected: write through)"
    send_alert "CRITICAL: Write cache is $WRITE_CACHE instead of 'write through' - WAL corruption risk!"
fi

# Check for recent unclean shutdowns (last 6 hours)
POWER_EVENTS=$(journalctl --since="6 hours ago" 2>/dev/null | grep -i "unclean\|corrupted.*journal" | wc -l)
if [ "$POWER_EVENTS" -gt 0 ]; then
    log "⚠️  WARNING: Detected $POWER_EVENTS unclean shutdown events in last 6 hours"
    # Don't alert every time, just log it
fi

# Check 1: WAL Suspension Status (CRITICAL CHECK)
log "Checking WAL status..."
WAL_DATA=$(curl -s -G "http://localhost:9000/exec" \
    --data-urlencode "query=SELECT name, suspended, writerTxn, sequencerTxn, (sequencerTxn - writerTxn) as lag, errorMessage FROM wal_tables() WHERE name IN ('market_orderbook', 'market_trades')" 2>/dev/null)

if [ $? -ne 0 ]; then
    log "ERROR: Cannot connect to QuestDB"
    send_alert "QuestDB connection failed"
    exit 1
fi

# Parse WAL data
echo "$WAL_DATA" | jq -e '.dataset' > /dev/null 2>&1
if [ $? -ne 0 ]; then
    log "ERROR: Invalid QuestDB response"
    exit 1
fi

# Check for suspended WAL
SUSPENDED_TABLES=$(echo "$WAL_DATA" | jq -r '.dataset[] | select(.[1] == true) | .[0]' 2>/dev/null)

if [ -n "$SUSPENDED_TABLES" ]; then
    log "🔴 CRITICAL: WAL SUSPENDED!"
    echo "$WAL_DATA" | jq -r '.dataset[] | select(.[1] == true) | "Table: \(.[0]), WriterTxn: \(.[2]), SequencerTxn: \(.[3]), Lag: \(.[4]), Reason: \(.[5])"' | tee -a "$LOG_FILE"
    send_alert "WAL SUSPENDED on tables: $SUSPENDED_TABLES"
    exit 1
fi

# Check 2: Transaction Lag (Backlog Indicator)
MAX_LAG=$(echo "$WAL_DATA" | jq -r '.dataset[] | .[4]' | sort -n | tail -1)

if [ "$MAX_LAG" -gt 10000 ]; then
    log "⚠️  WARNING: High WAL transaction lag: $MAX_LAG transactions"
    send_alert "High WAL lag: $MAX_LAG transactions"
fi

# Check 3: Data Freshness (Staleness Detection)
log "Checking data freshness..."
ORDERBOOK_AGE=$(curl -s -G "http://localhost:9000/exec" \
    --data-urlencode "query=SELECT CAST(timestampdiff('minute', MAX(timestamp), now()) AS INT) FROM market_orderbook" 2>/dev/null \
    | jq -r '.dataset[0][0] // 0' 2>/dev/null)

TRADES_AGE=$(curl -s -G "http://localhost:9000/exec" \
    --data-urlencode "query=SELECT CAST(timestampdiff('minute', MAX(timestamp), now()) AS INT) FROM market_trades" 2>/dev/null \
    | jq -r '.dataset[0][0] // 0' 2>/dev/null)

# Handle null/empty values
ORDERBOOK_AGE=${ORDERBOOK_AGE:-0}
TRADES_AGE=${TRADES_AGE:-0}

if [ "$ORDERBOOK_AGE" -gt 5 ] && [ "$ORDERBOOK_AGE" -ne 0 ]; then
    log "⚠️  WARNING: Orderbook data is $ORDERBOOK_AGE minutes old"
    send_alert "Stale orderbook data: ${ORDERBOOK_AGE}min old"
fi

if [ "$TRADES_AGE" -gt 5 ] && [ "$TRADES_AGE" -ne 0 ]; then
    log "⚠️  WARNING: Trades data is $TRADES_AGE minutes old"
fi

# Check 4: Service Status
log "Checking service status..."
SERVICES=("binance-bookticker.service" "binance-trades.service" "batch-writer.service" "questdb.service")
DOWN_SERVICES=""

for service in "${SERVICES[@]}"; do
    if ! systemctl is-active --quiet "$service"; then
        DOWN_SERVICES="$DOWN_SERVICES $service"
    fi
done

if [ -n "$DOWN_SERVICES" ]; then
    log "⚠️  WARNING: Services down:$DOWN_SERVICES"
    send_alert "Services down:$DOWN_SERVICES"
fi

# Check 5: Record Growth Rate (detect ingestion stall)
CURRENT_COUNT=$(curl -s -G "http://localhost:9000/exec" \
    --data-urlencode "query=SELECT COUNT(*) FROM market_orderbook" 2>/dev/null \
    | jq -r '.dataset[0][0]' 2>/dev/null)

LAST_COUNT_FILE="/tmp/questdb_last_count"
if [ -f "$LAST_COUNT_FILE" ]; then
    LAST_COUNT=$(cat "$LAST_COUNT_FILE")
    GROWTH=$((CURRENT_COUNT - LAST_COUNT))
    
    if [ $GROWTH -eq 0 ]; then
        log "⚠️  WARNING: No new records in last 5 minutes (count: $CURRENT_COUNT)"
    else
        log "✅ Records growing: +$GROWTH records in 5 minutes"
    fi
fi
echo "$CURRENT_COUNT" > "$LAST_COUNT_FILE"

# All checks passed
log "✅ QuestDB healthy - Write cache: $WRITE_CACHE, Orderbook age: ${ORDERBOOK_AGE}min, Trades age: ${TRADES_AGE}min, WAL lag: $MAX_LAG txns"

# Clear alert flag if everything is healthy
rm -f "$ALERT_FILE"

exit 0
