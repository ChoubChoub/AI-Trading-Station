#!/bin/bash
################################################################################
# QuestDB Data Lifecycle Management
# Purpose: Move old partitions from hot storage (NVMe) to cold storage (HDD)
# Schedule: Run daily via cron at 02:00 AM (low trading activity)
################################################################################

set -e

# Configuration
QUESTDB_HOST="localhost"
QUESTDB_PORT="9000"
HOT_STORAGE="/home/youssefbahloul/ai-trading-station/Services/QuestDB/data/hot"
COLD_STORAGE="/mnt/hdd/questdb/cold"
LOG_FILE="/home/youssefbahloul/ai-trading-station/Services/QuestDB/logs/lifecycle-$(date +%Y%m%d).log"

# Data retention policies (days to keep in hot storage)
TICKS_HOT_DAYS=30
ORDERBOOK_HOT_DAYS=7
REGIME_HOT_DAYS=30

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1${NC}" | tee -a "$LOG_FILE"
}

# Check if QuestDB is running
check_questdb() {
    if ! curl -s "http://${QUESTDB_HOST}:${QUESTDB_PORT}/exec?query=SELECT%201" > /dev/null; then
        log_error "QuestDB is not responding on ${QUESTDB_HOST}:${QUESTDB_PORT}"
        exit 1
    fi
    log_success "QuestDB is running"
}

# Get list of partitions older than specified days
get_old_partitions() {
    local table=$1
    local days=$2
    local cutoff_date=$(date -d "${days} days ago" +%Y-%m-%d)
    
    log "Checking ${table} for partitions older than ${cutoff_date}..."
    
    # Query to find old partitions
    # Note: This is a template - actual implementation depends on QuestDB's partition info tables
    local query="SELECT name FROM table_partitions('${table}') WHERE maxTimestamp < '${cutoff_date}'"
    
    # Execute query (this is placeholder - actual implementation would use QuestDB REST API)
    # For now, we'll use filesystem-based detection
    
    if [ -d "${HOT_STORAGE}/db/${table}" ]; then
        find "${HOT_STORAGE}/db/${table}" -type d -name "*.detached" -o -name "20*" | while read partition; do
            local partition_name=$(basename "$partition")
            # Check if partition is older than cutoff
            if [[ "$partition_name" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
                if [[ "$partition_name" < "$cutoff_date" ]]; then
                    echo "$partition_name"
                fi
            fi
        done
    fi
}

# Move partition from hot to cold storage
move_partition() {
    local table=$1
    local partition=$2
    
    local hot_path="${HOT_STORAGE}/db/${table}/${partition}"
    local cold_path="${COLD_STORAGE}/${table}"
    
    if [ ! -d "$hot_path" ]; then
        log_warning "Partition ${table}/${partition} not found in hot storage"
        return 1
    fi
    
    # Create cold storage directory if needed
    mkdir -p "$cold_path"
    
    # Calculate partition size
    local size=$(du -sh "$hot_path" | cut -f1)
    
    log "Moving ${table}/${partition} (${size}) to cold storage..."
    
    # Move partition (preserving permissions and timestamps)
    if rsync -a --remove-source-files "$hot_path/" "${cold_path}/${partition}/"; then
        # Remove empty source directory
        rmdir "$hot_path" 2>/dev/null || true
        log_success "Moved ${table}/${partition} to cold storage"
        return 0
    else
        log_error "Failed to move ${table}/${partition}"
        return 1
    fi
}

# Main lifecycle management function
manage_table_lifecycle() {
    local table=$1
    local hot_days=$2
    
    log "========================================"
    log "Processing table: ${table}"
    log "Hot retention: ${hot_days} days"
    log "========================================"
    
    local moved_count=0
    local failed_count=0
    local total_size=0
    
    # Get old partitions
    local partitions=$(get_old_partitions "$table" "$hot_days")
    
    if [ -z "$partitions" ]; then
        log "No partitions to move for ${table}"
        return 0
    fi
    
    # Move each partition
    while IFS= read -r partition; do
        if move_partition "$table" "$partition"; then
            ((moved_count++))
        else
            ((failed_count++))
        fi
    done <<< "$partitions"
    
    log_success "Table ${table}: Moved ${moved_count} partitions, ${failed_count} failures"
}

# Storage usage report
storage_report() {
    log "========================================"
    log "Storage Usage Report"
    log "========================================"
    
    local hot_used=$(df -h "${HOT_STORAGE}" | tail -1 | awk '{print $3}')
    local hot_avail=$(df -h "${HOT_STORAGE}" | tail -1 | awk '{print $4}')
    local hot_pct=$(df -h "${HOT_STORAGE}" | tail -1 | awk '{print $5}')
    
    local cold_used=$(df -h "${COLD_STORAGE}" | tail -1 | awk '{print $3}')
    local cold_avail=$(df -h "${COLD_STORAGE}" | tail -1 | awk '{print $4}')
    local cold_pct=$(df -h "${COLD_STORAGE}" | tail -1 | awk '{print $5}')
    
    log "Hot Storage (NVMe):  ${hot_used} used, ${hot_avail} available (${hot_pct})"
    log "Cold Storage (HDD):  ${cold_used} used, ${cold_avail} available (${cold_pct})"
    
    # Alert if hot storage is getting full
    local hot_pct_num=$(echo "$hot_pct" | sed 's/%//')
    if [ "$hot_pct_num" -gt 80 ]; then
        log_warning "Hot storage is ${hot_pct} full - consider moving more data to cold storage"
    fi
}

################################################################################
# Main Execution
################################################################################

log "========================================="
log "QuestDB Data Lifecycle Management - START"
log "========================================="

# Check QuestDB is running
check_questdb

# Initial storage report
storage_report

# Process each table according to retention policy
# Note: Uncomment these when tables are created in production

# manage_table_lifecycle "crypto_ticks" "$TICKS_HOT_DAYS"
# manage_table_lifecycle "orderbook_snapshots" "$ORDERBOOK_HOT_DAYS"
# manage_table_lifecycle "regime_states" "$REGIME_HOT_DAYS"

# Final storage report
storage_report

log "========================================="
log "QuestDB Data Lifecycle Management - END"
log "========================================="

# Cleanup old log files (keep last 30 days)
find "$(dirname "$LOG_FILE")" -name "lifecycle-*.log" -mtime +30 -delete

exit 0
