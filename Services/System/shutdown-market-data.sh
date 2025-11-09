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

# 4. Give QuestDB time to commit WAL
log "Waiting 5 seconds for QuestDB WAL commit..."
sleep 5

# 5. Stop QuestDB gracefully
log "Stopping QuestDB..."
systemctl stop questdb.service
log "✓ QuestDB stopped"

# 6. Stop Redis
log "Stopping Redis..."
systemctl stop redis-hft.service
log "✓ Redis stopped"

log "========================================="
log "Graceful shutdown complete"
log "All services stopped safely"
log "========================================="
