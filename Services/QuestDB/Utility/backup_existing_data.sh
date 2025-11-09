#!/bin/bash
# =============================================================================
# BACKUP EXISTING QUESTDB DATA
# =============================================================================
# Purpose: Export all existing tables before schema migration
# Date: 2025-10-20
# =============================================================================

set -e

BACKUP_DIR="/home/youssefbahloul/ai-trading-station/Services/QuestDB/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
QUESTDB_URL="http://localhost:9000"

echo "🔄 Starting QuestDB backup at $(date)"
echo "📁 Backup directory: $BACKUP_DIR"
echo ""

# Function to backup a table
backup_table() {
    local table_name=$1
    local output_file="$BACKUP_DIR/${table_name}_backup_${TIMESTAMP}.csv"
    
    echo "📊 Backing up table: $table_name"
    
    # Get row count first
    row_count=$(curl -s -G "$QUESTDB_URL/exec" \
        --data-urlencode "query=SELECT COUNT(*) as count FROM $table_name" \
        | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['dataset'][0][0] if d['dataset'] else 0)")
    
    echo "   Rows: $row_count"
    
    if [ "$row_count" -gt 0 ]; then
        # Export data
        curl -s -G "$QUESTDB_URL/exp" \
            --data-urlencode "query=SELECT * FROM $table_name" \
            > "$output_file"
        
        if [ -f "$output_file" ]; then
            file_size=$(du -h "$output_file" | cut -f1)
            echo "   ✅ Exported to: $output_file ($file_size)"
        else
            echo "   ❌ Export failed"
        fi
    else
        echo "   ⏭️  Table empty, skipping"
    fi
    echo ""
}

# Document current schema
echo "📋 Documenting current schema..."
curl -s -G "$QUESTDB_URL/exec" \
    --data-urlencode "query=SELECT table_name, designatedTimestamp, partitionBy, walEnabled FROM tables()" \
    > "$BACKUP_DIR/schema_metadata_${TIMESTAMP}.json"
echo "   ✅ Schema metadata saved"
echo ""

# Backup each table
backup_table "crypto_ticks"
backup_table "orderbook_snapshots"
backup_table "regime_states"
backup_table "perf_test"

# Create backup summary
echo "📄 Creating backup summary..."
cat > "$BACKUP_DIR/backup_summary_${TIMESTAMP}.txt" << SUMMARY
QuestDB Backup Summary
======================
Date: $(date)
Backup Directory: $BACKUP_DIR

Tables Backed Up:
$(ls -lh $BACKUP_DIR/*_backup_${TIMESTAMP}.csv 2>/dev/null || echo "No data files created (tables may be empty)")

Schema Metadata:
$(ls -lh $BACKUP_DIR/schema_metadata_${TIMESTAMP}.json)

Total Backup Size:
$(du -sh $BACKUP_DIR | cut -f1)
SUMMARY

echo "✅ Backup complete!"
echo ""
echo "📄 Summary saved to: $BACKUP_DIR/backup_summary_${TIMESTAMP}.txt"
echo "🔍 Review backup files:"
ls -lh "$BACKUP_DIR/"*"${TIMESTAMP}"*
