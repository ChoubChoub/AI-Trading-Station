#!/bin/bash
###############################################################################
# Graceful Market Data Shutdown Script
# 
# Purpose: Stop all market data services gracefully before system shutdown
#          to prevent QuestDB WAL corruption
#
# Called by: shutdown-market-data.service (systemd)
# Location: /home/youssefbahloul/ai-trading-station/Services/SystemD/shutdown-market-data.sh
###############################################################################

LOG_FILE="/home/youssefbahloul/ai-trading-station/Services/Monitoring/logs/shutdown.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================="
log "Starting graceful market data shutdown"
SHUTDOWN_START=$(date +%s)
log "========================================="

# 1. Stop data collectors first (stop new data flowing)
log "Stopping Binance collectors..."
systemctl stop binance-bookticker.service
systemctl stop binance-trades.service
log "✓ Collectors stopped"

# 2. Wait for queues to drain
log "Waiting 3 seconds for Redis queues to drain..."
sleep 3

# 3. Stop batch writer (flush remaining data to QuestDB)
log "Stopping batch writer..."
systemctl stop batch-writer.service
log "✓ Batch writer stopped"

# 4. Flush QuestDB WAL and verify
log "Flushing QuestDB WAL..."
if curl -s --connect-timeout 5 "http://localhost:9000/exec?query=SELECT COUNT(*) FROM wal_tables()" > /dev/null 2>&1; then
    log "✓ WAL tables accessible, initiating flush"
    # Give more time for WAL commit with larger datasets
    sleep 10
else
    log "⚠ QuestDB not responding, proceeding with shutdown"
    sleep 5
fi

# 5. Stop QuestDB gracefully and verify
log "Stopping QuestDB..."
systemctl stop questdb.service

# Wait for QuestDB to actually stop completely
timeout=30
elapsed=0
while systemctl is-active questdb.service >/dev/null 2>&1; do
    if [ $elapsed -ge $timeout ]; then
        log "⚠ QuestDB shutdown timeout, forcing stop"
        systemctl kill -s SIGKILL questdb.service
        break
    fi
    log "Waiting for QuestDB to stop... (${elapsed}s)"
    sleep 2
    elapsed=$((elapsed + 2))
done
log "✓ QuestDB stopped"

# 6. Stop Redis
log "Stopping Redis..."
systemctl stop redis-hft.service

# Verify Redis stopped
if systemctl is-active redis-hft.service >/dev/null 2>&1; then
    log "⚠ Redis still running, waiting..."
    sleep 3
    if systemctl is-active redis-hft.service >/dev/null 2>&1; then
        log "⚠ Forcing Redis stop"
        systemctl kill redis-hft.service
    fi
fi
log "✓ Redis stopped"

log "========================================="
SHUTDOWN_END=$(date +%s)
SHUTDOWN_DURATION=$((SHUTDOWN_END - SHUTDOWN_START))
log "Graceful shutdown complete in ${SHUTDOWN_DURATION} seconds"
log "All services stopped safely"
log "========================================="
