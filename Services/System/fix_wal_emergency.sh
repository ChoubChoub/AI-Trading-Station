#!/bin/bash
###############################################################################
# Emergency WAL Recovery Script
# 
# Purpose: Quick recovery from WAL corruption (should rarely be needed now)
# Location: /home/youssefbahloul/ai-trading-station/Scripts/System/fix_wal_emergency.sh
###############################################################################

echo "🔴 Emergency WAL recovery starting..."
echo ""

# Stop all market data services
echo "1️⃣  Stopping all market data services..."
sudo systemctl stop binance-bookticker.service binance-trades.service batch-writer.service
echo "✅ Services stopped"
echo ""

# Define schema
ORDERBOOK_SCHEMA='CREATE TABLE market_orderbook (
    symbol SYMBOL capacity 256 CACHE,
    exchange SYMBOL capacity 256 CACHE,
    bid_price DOUBLE,
    bid_qty DOUBLE,
    ask_price DOUBLE,
    ask_qty DOUBLE,
    timestamp TIMESTAMP
) timestamp(timestamp) PARTITION BY DAY WAL 
DEDUP UPSERT KEYS(timestamp, symbol, exchange)
WITH maxUncommittedRows=100000, o3MaxLag=31536000000000us'

TRADES_SCHEMA='CREATE TABLE market_trades (
    symbol SYMBOL capacity 256 CACHE,
    exchange SYMBOL capacity 256 CACHE,
    price DOUBLE,
    quantity DOUBLE,
    side SYMBOL capacity 2 CACHE,
    trade_id LONG,
    timestamp TIMESTAMP
) timestamp(timestamp) PARTITION BY DAY WAL 
DEDUP UPSERT KEYS(timestamp, symbol, exchange, trade_id)
WITH maxUncommittedRows=100000, o3MaxLag=31536000000000us'

# Drop and recreate tables
echo "2️⃣  Dropping corrupted tables..."
curl -s -G "http://localhost:9000/exec" --data-urlencode "query=DROP TABLE IF EXISTS market_orderbook"
curl -s -G "http://localhost:9000/exec" --data-urlencode "query=DROP TABLE IF EXISTS market_trades"
echo "✅ Tables dropped"
echo ""

echo "3️⃣  Recreating tables with proper configuration..."
ORDERBOOK_RESULT=$(curl -s -G "http://localhost:9000/exec" --data-urlencode "query=$ORDERBOOK_SCHEMA")
TRADES_RESULT=$(curl -s -G "http://localhost:9000/exec" --data-urlencode "query=$TRADES_SCHEMA")

if echo "$ORDERBOOK_RESULT" | grep -q "ddl"; then
    echo "✅ market_orderbook created"
else
    echo "❌ Failed to create market_orderbook"
    echo "$ORDERBOOK_RESULT"
fi

if echo "$TRADES_RESULT" | grep -q "ddl"; then
    echo "✅ market_trades created"
else
    echo "❌ Failed to create market_trades"
    echo "$TRADES_RESULT"
fi
echo ""

# Restart services
echo "4️⃣  Restarting market data services..."
sudo systemctl start binance-bookticker.service binance-trades.service batch-writer.service
sleep 3
echo "✅ Services restarted"
echo ""

# Verify data flow
echo "5️⃣  Verifying data ingestion (waiting 10 seconds)..."
sleep 10
ORDERBOOK_COUNT=$(curl -s -G "http://localhost:9000/exec" \
    --data-urlencode "query=SELECT count() FROM market_orderbook" \
    | jq -r '.dataset[0][0]')
TRADES_COUNT=$(curl -s -G "http://localhost:9000/exec" \
    --data-urlencode "query=SELECT count() FROM market_trades" \
    | jq -r '.dataset[0][0]')

if [ "$ORDERBOOK_COUNT" -gt 0 ]; then
    echo "✅ Orderbook flowing: $ORDERBOOK_COUNT records"
else
    echo "⚠️  Orderbook: $ORDERBOOK_COUNT records (may need more time)"
fi

if [ "$TRADES_COUNT" -gt 0 ]; then
    echo "✅ Trades flowing: $TRADES_COUNT records"
else
    echo "⚠️  Trades: $TRADES_COUNT records (may need more time)"
fi
echo ""

echo "✅ Emergency recovery complete!"
echo ""
echo "📊 Run health check: /home/youssefbahloul/ai-trading-station/Services/Monitoring/Scripts/Runtime/check_questdb_health.sh"
