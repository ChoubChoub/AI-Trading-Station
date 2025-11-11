#!/bin/bash
###############################################################################
# Emergency WAL Recovery Script (CORRECTED)
# 
# Purpose: Quick recovery from WAL corruption using CORRECT archive schemas
# Location: /home/youssefbahloul/ai-trading-station/Services/System/fix_wal_emergency_corrected.sh
###############################################################################

echo "🔴 Emergency WAL recovery starting (CORRECTED SCHEMAS)..."
echo ""

# Stop all market data services AND QuestDB
echo "1️⃣  Stopping all market data services and QuestDB..."
sudo systemctl stop binance-bookticker.service binance-trades.service batch-writer.service
sudo systemctl stop questdb.service
echo "✅ Services stopped (including QuestDB)"
echo ""

# Wait for QuestDB to fully stop
echo "⏳ Waiting for QuestDB to fully stop..."
sleep 5
echo ""

# Wait for QuestDB to fully stop
echo "⏳ Waiting for QuestDB to fully stop..."
sleep 5
echo ""

# Start QuestDB to execute SQL commands
echo "2️⃣  Starting QuestDB..."
sudo systemctl start questdb.service
sleep 10  # Wait for QuestDB to be ready
echo "✅ QuestDB started"
echo ""

# Drop and recreate tables
echo "3️⃣  Dropping corrupted tables..."
curl -s -G "http://localhost:9000/exec" --data-urlencode "query=DROP TABLE IF EXISTS market_orderbook"
curl -s -G "http://localhost:9000/exec" --data-urlencode "query=DROP TABLE IF EXISTS market_trades"
echo "✅ Tables dropped"
echo ""

echo "4️⃣  Recreating tables with CORRECT archive schemas..."
TRADES_SCHEMA='CREATE TABLE market_trades (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 256 CACHE INDEX,
    exchange SYMBOL CAPACITY 32 CACHE INDEX,
    price DOUBLE,
    volume DOUBLE,
    side BOOLEAN
) TIMESTAMP(timestamp) PARTITION BY HOUR WAL
  DEDUP UPSERT KEYS(timestamp, symbol, exchange)'

ORDERBOOK_SCHEMA='CREATE TABLE market_orderbook (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 256 CACHE INDEX,
    exchange SYMBOL CAPACITY 32 CACHE INDEX,
    bid_price DOUBLE,
    bid_qty DOUBLE,
    ask_price DOUBLE,
    ask_qty DOUBLE
) TIMESTAMP(timestamp) PARTITION BY HOUR WAL
  DEDUP UPSERT KEYS(timestamp, symbol, exchange)'

# Drop and recreate tables
echo "2️⃣  Dropping corrupted tables..."
curl -s -G "http://localhost:9000/exec" --data-urlencode "query=DROP TABLE IF EXISTS market_orderbook"
curl -s -G "http://localhost:9000/exec" --data-urlencode "query=DROP TABLE IF EXISTS market_trades"
echo "✅ Tables dropped"
echo ""

echo "3️⃣  Recreating tables with CORRECT archive schemas..."
TRADES_RESULT=$(curl -s -G "http://localhost:9000/exec" --data-urlencode "query=$TRADES_SCHEMA")
ORDERBOOK_RESULT=$(curl -s -G "http://localhost:9000/exec" --data-urlencode "query=$ORDERBOOK_SCHEMA")

if echo "$TRADES_RESULT" | grep -q "ddl"; then
    echo "✅ market_trades created with correct schema"
else
    echo "❌ Failed to create market_trades"
    echo "$TRADES_RESULT"
fi

if echo "$ORDERBOOK_RESULT" | grep -q "ddl"; then
    echo "✅ market_orderbook created with correct schema"
else
    echo "❌ Failed to create market_orderbook"
    echo "$ORDERBOOK_RESULT"
fi
echo ""

# Add indexes from archive schema
echo "4️⃣  Adding indexes..."
curl -s -G "http://localhost:9000/exec" --data-urlencode "query=ALTER TABLE market_trades ALTER COLUMN symbol ADD INDEX"
curl -s -G "http://localhost:9000/exec" --data-urlencode "query=ALTER TABLE market_trades ALTER COLUMN exchange ADD INDEX"
curl -s -G "http://localhost:9000/exec" --data-urlencode "query=ALTER TABLE market_orderbook ALTER COLUMN symbol ADD INDEX"
curl -s -G "http://localhost:9000/exec" --data-urlencode "query=ALTER TABLE market_orderbook ALTER COLUMN exchange ADD INDEX"
echo "✅ Indexes added"
echo ""

# Verify tables are visible in catalog
echo "5️⃣  Verifying tables are registered..."
TABLE_COUNT=$(curl -s -G "http://localhost:9000/exec" \
    --data-urlencode "query=SELECT table_name FROM tables() WHERE table_name LIKE 'market_%'" \
    | jq -r '.count')

if [ "$TABLE_COUNT" = "2" ]; then
    echo "✅ Both tables registered in catalog"
else
    echo "❌ Tables not properly registered (count: $TABLE_COUNT)"
fi
echo ""

# Restart services
echo "6️⃣  Restarting market data services..."
sudo systemctl start binance-bookticker.service binance-trades.service batch-writer.service
sleep 3
echo "✅ Services restarted"
echo ""

# Verify data flow
echo "7️⃣  Verifying data ingestion (waiting 10 seconds)..."
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

echo "✅ Emergency recovery complete with CORRECT schemas!"
echo ""
echo "📊 Run health check: /home/youssefbahloul/ai-trading-station/Services/Monitoring/Scripts/Runtime/check_questdb_health.sh"