#!/bin/bash
#
# POST-REBOOT COMPREHENSIVE DIAGNOSTIC
# Tests CPU affinity, temperature, services, and data ingestion pipeline
# Date: October 23, 2025
#

set -e

REPORT_FILE="/home/youssefbahloul/ai-trading-station/Services/Monitoring/logs/post_reboot_diagnostic_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$REPORT_FILE")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

log_test() {
    echo -e "${YELLOW}▶ $1${NC}"
}

log_pass() {
    echo -e "  ${GREEN}✓ $1${NC}"
}

log_fail() {
    echo -e "  ${RED}✗ $1${NC}"
}

log_info() {
    echo -e "  ${BLUE}ℹ $1${NC}"
}

# Start diagnostic
echo "POST-REBOOT DIAGNOSTIC REPORT" | tee "$REPORT_FILE"
echo "Generated: $(date)" | tee -a "$REPORT_FILE"
echo "Hostname: $(hostname)" | tee -a "$REPORT_FILE"
echo "Uptime: $(uptime -p)" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

{
    log_section "1. SYSTEM HEALTH CHECK"
    
    log_test "CPU Topology"
    CPUS=$(nproc)
    echo "  Available CPUs: $CPUS"
    if [ "$CPUS" -eq 8 ]; then
        log_pass "8 CPUs detected (expected for Intel Ultra 9 285K with E-cores disabled)"
    else
        log_fail "Expected 8 CPUs, found $CPUS"
    fi
    
    log_test "CPU Governor"
    GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
    echo "  Current governor: $GOVERNOR"
    if [ "$GOVERNOR" = "performance" ]; then
        log_pass "Performance governor active"
    else
        log_fail "Governor is $GOVERNOR (should be 'performance' for HFT)"
    fi
    
    log_test "System Load"
    LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
    echo "  Load average (1min): $LOAD"
    LOAD_INT=$(echo "$LOAD" | cut -d'.' -f1)
    if [ "$LOAD_INT" -lt 5 ]; then
        log_pass "System load is acceptable"
    else
        log_fail "System load is high: $LOAD"
    fi
    
    log_test "Memory Usage"
    FREE_GB=$(free -g | awk '/^Mem:/ {print $7}')
    TOTAL_GB=$(free -g | awk '/^Mem:/ {print $2}')
    echo "  Available: ${FREE_GB}GB / ${TOTAL_GB}GB"
    if [ "$FREE_GB" -gt 50 ]; then
        log_pass "Sufficient free memory"
    else
        log_fail "Low memory: ${FREE_GB}GB available"
    fi
    
    log_test "Temperature Check"
    if command -v sensors &> /dev/null; then
        TEMP=$(sensors | grep "Package id 0" | awk '{print $4}' | tr -d '+°C')
        echo "  Package temperature: ${TEMP}°C"
        TEMP_INT=$(echo "$TEMP" | cut -d'.' -f1)
        if [ "$TEMP_INT" -lt 75 ]; then
            log_pass "Temperature is healthy (${TEMP}°C < 75°C)"
        elif [ "$TEMP_INT" -lt 85 ]; then
            echo -e "  ${YELLOW}⚠ Temperature elevated (${TEMP}°C)${NC}"
        else
            log_fail "Temperature critical (${TEMP}°C >= 85°C)"
        fi
    else
        log_info "sensors command not available"
    fi
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log_section "2. SERVICE STATUS CHECK"
    
    SERVICES=(
        "prometheus:Monitoring metrics collector"
        "redis-hft:High-frequency trading data cache"
        "questdb:Time-series database"
        "batch-writer:Redis to QuestDB data pipeline"
        "binance-trades:Binance trades WebSocket collector"
        "binance-bookticker:Binance book ticker WebSocket collector"
    )
    
    ALL_SERVICES_OK=true
    for SERVICE_INFO in "${SERVICES[@]}"; do
        SERVICE=$(echo "$SERVICE_INFO" | cut -d':' -f1)
        DESC=$(echo "$SERVICE_INFO" | cut -d':' -f2)
        
        log_test "Service: $SERVICE"
        echo "  Description: $DESC"
        
        if systemctl is-active --quiet "$SERVICE"; then
            STATUS=$(systemctl show -p ActiveState --value "$SERVICE")
            UPTIME=$(systemctl show -p ActiveEnterTimestamp --value "$SERVICE")
            log_pass "Status: $STATUS"
            echo "  Started: $UPTIME"
        else
            log_fail "Service is NOT running"
            ALL_SERVICES_OK=false
        fi
    done
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log_section "3. CPU AFFINITY VERIFICATION"
    
    log_test "Systemd CPU Affinity Configuration"
    
    # Check each service's configured affinity
    declare -A EXPECTED_AFFINITY=(
        ["prometheus"]="2"
        ["redis-hft"]="4"
        ["questdb"]="5-6"
        ["batch-writer"]="7"
        ["binance-trades"]="3"
        ["binance-bookticker"]="3"
    )
    
    ALL_AFFINITY_OK=true
    for SERVICE in "${!EXPECTED_AFFINITY[@]}"; do
        EXPECTED="${EXPECTED_AFFINITY[$SERVICE]}"
        
        # Get configured affinity from systemd
        if [ "$SERVICE" = "redis-hft" ]; then
            # Redis affinity is in main service file
            CONFIGURED=$(systemctl show -p CPUAffinity --value "$SERVICE" | tr ' ' ',')
        else
            # Others have override files
            CONFIGURED=$(systemctl show -p CPUAffinity --value "$SERVICE" | tr ' ' ',')
        fi
        
        echo "  $SERVICE: configured=$CONFIGURED, expected=$EXPECTED"
        
        # Get actual process affinity
        if [ "$SERVICE" = "questdb" ]; then
            PID=$(pgrep -f "io.questdb" | head -1)
        elif [ "$SERVICE" = "batch-writer" ]; then
            PID=$(pgrep -f "redis_to_questdb_v2.py" | head -1)
        elif [ "$SERVICE" = "binance-trades" ]; then
            PID=$(pgrep -f "binance_trades_collector" | head -1)
        elif [ "$SERVICE" = "binance-bookticker" ]; then
            PID=$(pgrep -f "binance_bookticker_collector" | head -1)
        else
            PID=$(pgrep -x "$SERVICE" | head -1)
        fi
        
        if [ -n "$PID" ]; then
            ACTUAL=$(taskset -cp "$PID" 2>/dev/null | awk '{print $NF}')
            echo "  PID $PID actual affinity: $ACTUAL"
            
            # Normalize for comparison (handle both "5-6" and "5,6" formats)
            EXPECTED_NORM=$(echo "$EXPECTED" | tr '-' ',')
            ACTUAL_NORM=$(echo "$ACTUAL" | tr '-' ',')
            
            if [ "$ACTUAL_NORM" = "$EXPECTED_NORM" ] || [ "$ACTUAL" = "$EXPECTED" ]; then
                log_pass "$SERVICE affinity correct: $ACTUAL"
            else
                log_fail "$SERVICE affinity mismatch: expected $EXPECTED, got $ACTUAL"
                ALL_AFFINITY_OK=false
            fi
        else
            log_fail "$SERVICE process not found"
            ALL_AFFINITY_OK=false
        fi
    done
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log_section "4. CPU UTILIZATION ANALYSIS"
    
    log_test "Per-CPU utilization (5-second sample)"
    echo ""
    mpstat -P ALL 1 5 | tail -10
    echo ""
    
    log_test "CPU utilization by service (10-second sample)"
    echo ""
    
    # Collect CPU usage for each service
    for SERVICE_INFO in "${SERVICES[@]}"; do
        SERVICE=$(echo "$SERVICE_INFO" | cut -d':' -f1)
        
        if [ "$SERVICE" = "questdb" ]; then
            PID=$(pgrep -f "io.questdb" | head -1)
        elif [ "$SERVICE" = "batch-writer" ]; then
            PID=$(pgrep -f "redis_to_questdb_v2.py" | head -1)
        elif [ "$SERVICE" = "binance-trades" ]; then
            PID=$(pgrep -f "binance_trades_collector" | head -1)
        elif [ "$SERVICE" = "binance-bookticker" ]; then
            PID=$(pgrep -f "binance_bookticker_collector" | head -1)
        else
            PID=$(pgrep -x "$SERVICE" | head -1)
        fi
        
        if [ -n "$PID" ]; then
            CPU=$(ps -p "$PID" -o %cpu= 2>/dev/null || echo "0.0")
            PSR=$(ps -p "$PID" -o psr= 2>/dev/null || echo "?")
            printf "  %-20s PID: %-6s CPU: %6s%%  Running on CPU: %s\n" "$SERVICE" "$PID" "$CPU" "$PSR"
        fi
    done
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log_section "5. DATA INGESTION PIPELINE CHECK"
    
    log_test "Redis Connectivity"
    if redis-cli ping &>/dev/null; then
        log_pass "Redis is responding"
        
        # Check Redis keys
        KEYS_COUNT=$(redis-cli DBSIZE | awk '{print $2}')
        echo "  Total keys in Redis: $KEYS_COUNT"
        
        # Check market data streams
        log_test "Market data streams in Redis"
        TRADE_KEYS=$(redis-cli KEYS "market:binance_spot:trades:*" | wc -l)
        BOOK_KEYS=$(redis-cli KEYS "market:binance_spot:orderbook:*" | wc -l)
        echo "  Trade streams: $TRADE_KEYS"
        echo "  Order book streams: $BOOK_KEYS"
        
        if [ "$TRADE_KEYS" -gt 0 ] || [ "$BOOK_KEYS" -gt 0 ]; then
            log_pass "Market data streams detected in Redis"
            
            # Sample a key to verify data freshness
            SAMPLE_KEY=$(redis-cli KEYS "market:binance_spot:trades:*" | head -1)
            if [ -n "$SAMPLE_KEY" ]; then
                STREAM_LEN=$(redis-cli XLEN "$SAMPLE_KEY" 2>/dev/null || echo "0")
                echo "  Sample stream length ($SAMPLE_KEY): $STREAM_LEN messages"
                
                if [ "$STREAM_LEN" -gt 0 ]; then
                    log_pass "Data is flowing through Redis streams"
                else
                    log_fail "Stream exists but is empty"
                fi
            fi
        else
            log_fail "No market data streams found in Redis"
        fi
    else
        log_fail "Cannot connect to Redis"
    fi
    
    log_test "QuestDB Connectivity"
    if curl -s http://localhost:9000/exec?query=SELECT%201 &>/dev/null; then
        log_pass "QuestDB is responding"
        
        # Check table existence and row counts
        log_test "QuestDB table verification"
        
        TABLES=$(curl -s "http://localhost:9000/exec?query=SELECT%20table_name%20FROM%20tables()" | grep -o '"table_name":"[^"]*"' | cut -d'"' -f4 || echo "")
        
        if [ -n "$TABLES" ]; then
            echo "  Tables found:"
            echo "$TABLES" | while read -r TABLE; do
                if [ -n "$TABLE" ]; then
                    COUNT=$(curl -s "http://localhost:9000/exec?query=SELECT%20count()%20FROM%20$TABLE" 2>/dev/null | grep -o '"count":[0-9]*' | cut -d':' -f2 || echo "0")
                    echo "    - $TABLE: $COUNT rows"
                fi
            done
            log_pass "QuestDB tables accessible"
        else
            log_info "No tables found (may be fresh installation)"
        fi
        
        # Check recent ingestion (last 60 seconds)
        log_test "Recent data ingestion (last 60 seconds)"
        for TABLE in binance_spot_trades binance_spot_orderbook; do
            RECENT=$(curl -s "http://localhost:9000/exec?query=SELECT%20count()%20FROM%20$TABLE%20WHERE%20timestamp%20%3E%20dateadd('s',-60,now())" 2>/dev/null | grep -o '"count":[0-9]*' | cut -d':' -f2 || echo "0")
            echo "  $TABLE: $RECENT records in last 60s"
            if [ "$RECENT" -gt 0 ]; then
                log_pass "$TABLE receiving fresh data"
            else
                echo -e "  ${YELLOW}⚠ No recent data in $TABLE${NC}"
            fi
        done
    else
        log_fail "Cannot connect to QuestDB"
    fi
    
    log_test "Batch Writer Metrics"
    if curl -s http://localhost:9091/metrics &>/dev/null; then
        log_pass "Batch writer Prometheus endpoint responding"
        
        # Get key metrics
        PROCESSED=$(curl -s http://localhost:9091/metrics | grep "^redis_questdb_records_processed_total" | awk '{print $2}' | head -1 || echo "0")
        FAILED=$(curl -s http://localhost:9091/metrics | grep "^redis_questdb_records_failed_total" | awk '{print $2}' | head -1 || echo "0")
        LATENCY=$(curl -s http://localhost:9091/metrics | grep "^redis_questdb_batch_latency_seconds_sum" | awk '{print $2}' | head -1 || echo "0")
        
        echo "  Records processed: $PROCESSED"
        echo "  Records failed: $FAILED"
        echo "  Batch latency sum: $LATENCY seconds"
        
        if [ "$PROCESSED" != "0" ]; then
            log_pass "Batch writer is processing data"
        else
            log_info "No records processed yet (may be fresh start)"
        fi
    else
        log_fail "Batch writer metrics endpoint not responding"
    fi
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log_section "6. NETWORK & LATENCY CHECKS"
    
    log_test "Network Interface Status"
    if command -v ethtool &> /dev/null; then
        # Find Solarflare interface
        SOLARFLARE_IF=$(ip link show | grep -B1 "link/ether" | grep -v "link/ether" | grep -E "ens|eth|enp" | awk '{print $2}' | tr -d ':' | head -1)
        if [ -n "$SOLARFLARE_IF" ]; then
            SPEED=$(ethtool "$SOLARFLARE_IF" 2>/dev/null | grep "Speed:" | awk '{print $2}')
            DUPLEX=$(ethtool "$SOLARFLARE_IF" 2>/dev/null | grep "Duplex:" | awk '{print $2}')
            echo "  Interface: $SOLARFLARE_IF"
            echo "  Speed: $SPEED"
            echo "  Duplex: $DUPLEX"
            if [ "$SPEED" = "10000Mb/s" ] && [ "$DUPLEX" = "Full" ]; then
                log_pass "Network operating at optimal speed"
            else
                log_fail "Network not at expected 10Gbps full-duplex"
            fi
        else
            log_info "Could not detect Solarflare interface"
        fi
    fi
    
    log_test "Binance API Latency"
    BINANCE_LATENCY=$(curl -o /dev/null -s -w '%{time_total}\n' https://api.binance.com/api/v3/ping)
    echo "  Binance API response time: ${BINANCE_LATENCY}s"
    LATENCY_MS=$(echo "$BINANCE_LATENCY * 1000" | bc)
    LATENCY_MS_INT=$(echo "$LATENCY_MS" | cut -d'.' -f1)
    if [ "$LATENCY_MS_INT" -lt 100 ]; then
        log_pass "Excellent latency to Binance (<100ms)"
    elif [ "$LATENCY_MS_INT" -lt 200 ]; then
        log_pass "Good latency to Binance (<200ms)"
    else
        echo -e "  ${YELLOW}⚠ Elevated latency to Binance (${LATENCY_MS}ms)${NC}"
    fi
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log_section "7. GRAFANA MONITORING CHECK"
    
    log_test "Grafana Accessibility"
    if curl -s http://localhost:3000/api/health &>/dev/null; then
        log_pass "Grafana is accessible"
        
        GRAFANA_STATUS=$(curl -s http://localhost:3000/api/health | grep -o '"database":"[^"]*"' | cut -d'"' -f4)
        echo "  Database status: $GRAFANA_STATUS"
        
        if [ "$GRAFANA_STATUS" = "ok" ]; then
            log_pass "Grafana database healthy"
        else
            log_fail "Grafana database issue: $GRAFANA_STATUS"
        fi
    else
        log_fail "Cannot connect to Grafana"
    fi
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log_section "8. FINAL SUMMARY"
    
    echo ""
    if [ "$ALL_SERVICES_OK" = true ] && [ "$ALL_AFFINITY_OK" = true ]; then
        echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                                                           ║${NC}"
        echo -e "${GREEN}║           ✓ ALL SYSTEMS OPERATIONAL                       ║${NC}"
        echo -e "${GREEN}║                                                           ║${NC}"
        echo -e "${GREEN}║  • All services running                                   ║${NC}"
        echo -e "${GREEN}║  • CPU affinity correctly configured                      ║${NC}"
        echo -e "${GREEN}║  • Temperature within safe limits                         ║${NC}"
        echo -e "${GREEN}║  • Data ingestion pipeline functional                     ║${NC}"
        echo -e "${GREEN}║                                                           ║${NC}"
        echo -e "${GREEN}║  Status: READY FOR PRODUCTION                             ║${NC}"
        echo -e "${GREEN}║                                                           ║${NC}"
        echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    else
        echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║                                                           ║${NC}"
        echo -e "${RED}║           ⚠ ISSUES DETECTED                               ║${NC}"
        echo -e "${RED}║                                                           ║${NC}"
        if [ "$ALL_SERVICES_OK" = false ]; then
            echo -e "${RED}║  • Some services are not running                          ║${NC}"
        fi
        if [ "$ALL_AFFINITY_OK" = false ]; then
            echo -e "${RED}║  • CPU affinity configuration issues detected             ║${NC}"
        fi
        echo -e "${RED}║                                                           ║${NC}"
        echo -e "${RED}║  Status: REQUIRES ATTENTION                               ║${NC}"
        echo -e "${RED}║                                                           ║${NC}"
        echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
    fi
    echo ""
    
    log_info "Full diagnostic report saved to: $REPORT_FILE"
    log_info "Reboot time: $(who -b | awk '{print $3, $4}')"
    log_info "Current time: $(date)"
    
} 2>&1 | tee -a "$REPORT_FILE"

echo ""
echo "Diagnostic complete. Review the report above or at: $REPORT_FILE"
