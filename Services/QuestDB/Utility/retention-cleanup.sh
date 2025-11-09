#!/bin/bash
################################################################################
# QuestDB Partition Retention Cleanup (Safe SQL-Based Approach)
# Purpose: Delete old partitions using QuestDB's built-in DROP PARTITION
# Schedule: Run daily at 2 AM via cron
################################################################################

set -e

# Configuration
QUESTDB_HOST="localhost"
QUESTDB_PORT="9000"
LOG_FILE="/home/youssefbahloul/ai-trading-station/Services/QuestDB/logs/retention-$(date +%Y%m%d).log"

# Retention policies (days to keep)
CRYPTO_TICKS_RETENTION=365        # 1 year for backtesting
ORDERBOOK_RETENTION=90            # 3 months
REGIME_STATES_RETENTION=365       # 1 year

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1${NC}" | tee -a "$LOG_FILE"
}

# Execute QuestDB SQL query
execute_sql() {
    local query="$1"
    local response
    
    # Add timeout to prevent hanging
    response=$(curl -s -G --max-time 10 "http://${QUESTDB_HOST}:${QUESTDB_PORT}/exec" \
        --data-urlencode "query=${query}" 2>&1)
    
    if echo "$response" | grep -q '"ddl":"OK"'; then
        return 0
    elif echo "$response" | grep -q "does not exist"; then
        log "Table does not exist yet - skipping"
        return 0
    else
        log_error "Query failed: $query"
        log_error "Response: $response"
        return 1
    fi
}

# Check if QuestDB is running
check_questdb() {
    # Test with a simple SQL query instead of root endpoint
    local response=$(curl -s --max-time 5 -G "http://${QUESTDB_HOST}:${QUESTDB_PORT}/exec" \
        --data-urlencode "query=SELECT 1" 2>/dev/null)
    
    if echo "$response" | grep -q "dataset"; then
        log_success "QuestDB is running and responding to queries"
        return 0
    else
        log_error "QuestDB is not responding on ${QUESTDB_HOST}:${QUESTDB_PORT}"
        return 1
    fi
}

# Get storage usage before cleanup
get_storage_before() {
    STORAGE_BEFORE=$(df -h /home/youssefbahloul/ai-trading-station/Services/QuestDB/data/hot | tail -1 | awk '{print $3}')
    log "Storage before cleanup: ${STORAGE_BEFORE}"
}

# Get storage usage after cleanup
get_storage_after() {
    STORAGE_AFTER=$(df -h /home/youssefbahloul/ai-trading-station/Services/QuestDB/data/hot | tail -1 | awk '{print $3}')
    log "Storage after cleanup: ${STORAGE_AFTER}"
}

# Drop old partitions for a table
drop_old_partitions() {
    local table=$1
    local retention_days=$2
    
    log "========================================"
    log "Cleaning up table: ${table}"
    log "Retention policy: ${retention_days} days"
    log "========================================"
    
    # Calculate cutoff date
    local cutoff_date=$(date -d "${retention_days} days ago" +%Y-%m-%d)
    log "Deleting partitions older than: ${cutoff_date}"
    
    # SQL query to drop old partitions
    local query="ALTER TABLE ${table} DROP PARTITION WHERE timestamp < '${cutoff_date}'"
    
    if execute_sql "$query"; then
        log_success "Cleaned up ${table} (kept last ${retention_days} days)"
    else
        log_error "Failed to clean up ${table}"
        return 1
    fi
}

################################################################################
# Main Execution
################################################################################

log "========================================="
log "QuestDB Partition Retention Cleanup - START"
log "========================================="

# Check QuestDB is running
if ! check_questdb; then
    log_error "Aborting: QuestDB is not running"
    exit 1
fi

# Get storage usage before cleanup
get_storage_before

# Clean up each table according to retention policy
# Note: These will fail gracefully if tables don't exist yet

drop_old_partitions "crypto_ticks" "$CRYPTO_TICKS_RETENTION"
drop_old_partitions "orderbook_snapshots" "$ORDERBOOK_RETENTION"
drop_old_partitions "regime_states" "$REGIME_STATES_RETENTION"

# Get storage usage after cleanup
get_storage_after

log "========================================="
log "QuestDB Partition Retention Cleanup - END"
log "========================================="

# Cleanup old log files (keep last 30 days)
find "$(dirname "$LOG_FILE")" -name "retention-*.log" -mtime +30 -delete 2>/dev/null || true

exit 0
