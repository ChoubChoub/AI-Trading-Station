#!/bin/bash
################################################################################
# Market Data Feed Control System (v2)
# Controls: Trades + Orderbook + Batch Writer
# systemd-based architecture with graceful shutdown
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Service names
TRADES_SERVICE="binance-trades.service"
ORDERBOOK_SERVICE="binance-bookticker.service"
BATCH_WRITER_SERVICE="batch-writer.service"
MARKET_DATA_TARGET="market-data.target"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

get_service_status() {
    local service=$1
    if systemctl is-active --quiet "$service"; then
        local pid=$(systemctl show -p MainPID --value "$service")
        local uptime=$(systemctl show -p ActiveEnterTimestamp --value "$service" | xargs -I {} date -d {} +%s)
        local now=$(date +%s)
        local elapsed=$((now - uptime))
        local uptime_str=$(printf '%dd %dh %dm' $((elapsed/86400)) $((elapsed%86400/3600)) $((elapsed%3600/60)))
        echo -e "${GREEN}✓${NC} Running (PID $pid, up $uptime_str)"
    else
        echo -e "${RED}✗${NC} Stopped"
    fi
}

get_memory_usage() {
    local service=$1
    local mem=$(systemctl show -p MemoryCurrent --value "$service")
    if [ "$mem" != "0" ] && [ "$mem" != "[not set]" ]; then
        echo "$((mem / 1024 / 1024))M"
    else
        echo "N/A"
    fi
}

################################################################################
# Status Function
################################################################################

status() {
    print_header "MARKET DATA FEED STATUS"
    
    # Prerequisites
    echo -e "${BLUE}Prerequisites:${NC}"
    if systemctl is-active --quiet redis-hft.service; then
        local redis_pid=$(pgrep redis-server)
        local redis_cpu=$(taskset -cp "$redis_pid" 2>/dev/null | awk '{print $NF}' || echo "N/A")
        echo -e "  ${GREEN}✓${NC} Redis (PID $redis_pid, CPU $redis_cpu)"
    else
        echo -e "  ${RED}✗${NC} Redis (not running)"
    fi
    
    if systemctl is-active --quiet questdb.service; then
        local questdb_pid=$(pgrep -f io.questdb.ServerMain)
        echo -e "  ${GREEN}✓${NC} QuestDB (PID $questdb_pid)"
    else
        echo -e "  ${RED}✗${NC} QuestDB (not running)"
    fi
    
    # Data Collectors
    echo ""
    echo -e "${BLUE}Data Collectors:${NC}"
    echo -n "  Trades:    "
    get_service_status "$TRADES_SERVICE"
    echo "             Memory: $(get_memory_usage "$TRADES_SERVICE")"
    
    echo -n "  Orderbook: "
    get_service_status "$ORDERBOOK_SERVICE"
    echo "             Memory: $(get_memory_usage "$ORDERBOOK_SERVICE")"
    
    echo -n "  BatchWriter: "
    get_service_status "$BATCH_WRITER_SERVICE"
    echo "             Memory: $(get_memory_usage "$BATCH_WRITER_SERVICE")"
    
    # Data Flow
    echo ""
    echo -e "${BLUE}Data Flow (Last 60 seconds):${NC}"
    if redis-cli -p 6379 PING > /dev/null 2>&1; then
        local trades_count=$(redis-cli -p 6379 XLEN "market:binance_spot:trades:BTCUSDT" 2>/dev/null || echo "0")
        local orderbook_count=$(redis-cli -p 6379 XLEN "market:binance_spot:orderbook:BTCUSDT" 2>/dev/null || echo "0")
        echo "  Redis Streams:"
        echo "    Trades:    $trades_count messages (BTCUSDT)"
        echo "    Orderbook: $orderbook_count messages (BTCUSDT)"
    fi
    
    if curl -s -f http://localhost:9000/exec?query=SELECT%201 > /dev/null 2>&1; then
        local stats=$(curl -s -G 'http://localhost:9000/exec' --data-urlencode "query=
SELECT 'Trades' as type, count() as events, round(count()/60.0, 1) as rate
FROM market_trades WHERE timestamp > now() - 60000000
UNION ALL
SELECT 'Orderbook' as type, count() as events, round(count()/60.0, 1) as rate
FROM market_orderbook WHERE timestamp > now() - 60000000
" 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for row in data.get('dataset', []):
    print(f'    {row[0]:10s} {int(row[1]):6d} events ({row[2]:6.1f}/sec)')
" 2>/dev/null || echo "    Query failed")
        
        echo "  QuestDB Tables:"
        echo "$stats"
    fi
    
    # Prometheus Metrics
    echo ""
    echo -e "${BLUE}Prometheus Metrics:${NC}"
    if curl -s http://localhost:9091/metrics > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Available at http://localhost:9091/metrics"
    else
        echo -e "  ${YELLOW}⚠${NC} Metrics server not responding"
    fi
    
    echo ""
}

################################################################################
# Start Function
################################################################################

start() {
    print_header "STARTING MARKET DATA FEED"
    
    # Check prerequisites
    echo "Checking prerequisites..."
    
    if ! systemctl is-active --quiet redis-hft.service; then
        echo -e "${YELLOW}  ⚠ Starting Redis...${NC}"
        sudo systemctl start redis-hft.service
        sleep 2
    fi
    
    if ! systemctl is-active --quiet questdb.service; then
        echo -e "${YELLOW}  ⚠ Starting QuestDB...${NC}"
        sudo systemctl start questdb.service
        sleep 3
    fi
    
    echo -e "${GREEN}  ✓ Prerequisites ready${NC}"
    echo ""
    
    # Start all services via target
    echo "Starting market data collectors..."
    sudo systemctl start "$MARKET_DATA_TARGET"
    
    sleep 3
    
    # Verify
    local all_running=true
    for service in "$TRADES_SERVICE" "$ORDERBOOK_SERVICE" "$BATCH_WRITER_SERVICE"; do
        if systemctl is-active --quiet "$service"; then
            local pid=$(systemctl show -p MainPID --value "$service")
            echo -e "${GREEN}  ✓${NC} ${service%%.*} started (PID $pid)"
        else
            echo -e "${RED}  ✗${NC} ${service%%.*} failed to start"
            all_running=false
        fi
    done
    
    echo ""
    if $all_running; then
        echo -e "${GREEN}Market data feed started successfully${NC}"
    else
        echo -e "${RED}Some services failed to start. Check: journalctl -xe${NC}"
    fi
    echo ""
}

################################################################################
# Stop Function (Graceful)
################################################################################

stop() {
    print_header "STOPPING MARKET DATA FEED (GRACEFUL)"
    
    # Capture stats before shutdown
    echo "Capturing current state..."
    local redis_before=$(redis-cli -p 6379 XLEN "market:binance_spot:trades:BTCUSDT" 2>/dev/null || echo "0")
    local questdb_before=$(curl -s -G 'http://localhost:9000/exec' --data-urlencode "query=SELECT count(*) FROM market_trades WHERE timestamp > now() - 300000000;" 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin)['dataset'][0][0])" 2>/dev/null || echo "0")
    echo "  Redis trades:  $redis_before messages"
    echo "  QuestDB (5m):  $questdb_before rows"
    echo ""
    
    # Stop services gracefully via systemd (SIGTERM handled by Python)
    echo "Stopping services..."
    
    # Stop collectors first (producers)
    for service in "$TRADES_SERVICE" "$ORDERBOOK_SERVICE"; do
        if systemctl is-active --quiet "$service"; then
            echo -n "  Stopping ${service%%.*}..."
            sudo systemctl stop "$service"
            echo -e " ${GREEN}✓${NC}"
        fi
    done
    
    # Wait a bit for queues to drain
    sleep 2
    
    # Stop batch writer last (consumer)
    if systemctl is-active --quiet "$BATCH_WRITER_SERVICE"; then
        echo -n "  Stopping batch writer..."
        sudo systemctl stop "$BATCH_WRITER_SERVICE"
        echo -e " ${GREEN}✓${NC}"
    fi
    
    echo ""
    echo "Verifying data integrity..."
    sleep 2
    local questdb_after=$(curl -s -G 'http://localhost:9000/exec' --data-urlencode "query=SELECT count(*) FROM market_trades WHERE timestamp > now() - 300000000;" 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin)['dataset'][0][0])" 2>/dev/null || echo "0")
    echo "  QuestDB (5m):  $questdb_after rows (was $questdb_before)"
    
    if [ "$questdb_after" -ge "$questdb_before" ]; then
        echo -e "${GREEN}  ✓ Data integrity verified${NC}"
    else
        echo -e "${YELLOW}  ⚠ Row count changed (may be normal)${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}Market data feed stopped successfully${NC}"
    echo ""
}

################################################################################
# Restart Function
################################################################################

restart() {
    print_header "RESTARTING MARKET DATA FEED"
    stop
    sleep 2
    start
}

################################################################################
# Logs Function
################################################################################

logs() {
    local component=${1:-all}
    
    case $component in
        trades)
            echo "Following trades collector logs (Ctrl+C to exit)..."
            sudo journalctl -u "$TRADES_SERVICE" -f
            ;;
        orderbook|book)
            echo "Following orderbook collector logs (Ctrl+C to exit)..."
            sudo journalctl -u "$ORDERBOOK_SERVICE" -f
            ;;
        batch|writer)
            echo "Following batch writer logs (Ctrl+C to exit)..."
            sudo journalctl -u "$BATCH_WRITER_SERVICE" -f
            ;;
        all)
            echo "Following all market data logs (Ctrl+C to exit)..."
            sudo journalctl -u "$TRADES_SERVICE" -u "$ORDERBOOK_SERVICE" -u "$BATCH_WRITER_SERVICE" -f
            ;;
        *)
            echo "Available logs:"
            echo "  datafeed logs trades     - Trade collector logs"
            echo "  datafeed logs orderbook  - Orderbook collector logs"
            echo "  datafeed logs batch      - Batch writer logs"
            echo "  datafeed logs all        - All logs (default)"
            ;;
    esac
}

################################################################################
# Enable/Disable Functions
################################################################################

enable() {
    echo "Enabling market data feed (auto-start on boot)..."
    sudo systemctl enable "$TRADES_SERVICE" "$ORDERBOOK_SERVICE" "$BATCH_WRITER_SERVICE"
    echo -e "${GREEN}✓ Market data feed will start automatically on boot${NC}"
}

disable() {
    echo "Disabling market data feed (no auto-start)..."
    sudo systemctl disable "$TRADES_SERVICE" "$ORDERBOOK_SERVICE" "$BATCH_WRITER_SERVICE"
    echo -e "${YELLOW}⚠ Market data feed will NOT start automatically on boot${NC}"
}

################################################################################
# Test Function
################################################################################

test_connectivity() {
    print_header "CONNECTIVITY TEST"
    
    local all_ok=true
    
    # Test Redis
    echo -n "Testing Redis connection... "
    if redis-cli -p 6379 PING > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
        all_ok=false
    fi
    
    # Test QuestDB
    echo -n "Testing QuestDB HTTP... "
    if curl -s -f http://localhost:9000/exec?query=SELECT%201 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
        all_ok=false
    fi
    
    # Test QuestDB ILP
    echo -n "Testing QuestDB ILP endpoint... "
    if curl -s -f http://localhost:9000/write -X POST -d "test_table,tag=test value=1 $(date +%s%N)" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
        all_ok=false
    fi
    
    # Test Binance API
    echo -n "Testing Binance API... "
    if curl -s -f https://api.binance.com/api/v3/ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
        all_ok=false
    fi
    
    # Test Prometheus metrics
    echo -n "Testing Prometheus metrics... "
    if curl -s http://localhost:9091/metrics > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${YELLOW}⚠ Not available${NC}"
    fi
    
    # Test health endpoints
    echo -n "Testing trades collector health endpoint... "
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${YELLOW}⚠ Not available${NC}"
    fi
    
    echo ""
    if $all_ok; then
        echo -e "${GREEN}All critical connectivity tests passed${NC}"
        return 0
    else
        echo -e "${RED}Some connectivity tests failed${NC}"
        return 1
    fi
}

################################################################################
# Health Function
################################################################################

health() {
    # Exit code 0 if healthy, 1 if unhealthy (for monitoring scripts)
    local healthy=true
    
    # Check if all services are running
    for service in "$TRADES_SERVICE" "$ORDERBOOK_SERVICE" "$BATCH_WRITER_SERVICE"; do
        if ! systemctl is-active --quiet "$service"; then
            healthy=false
        fi
    done
    
    # Check if prerequisites are running
    if ! systemctl is-active --quiet redis-hft.service; then
        healthy=false
    fi
    
    if ! systemctl is-active --quiet questdb.service; then
        healthy=false
    fi
    
    # Check health endpoint if available
    if curl -s http://localhost:8001/health 2>/dev/null | grep -q '"status":"healthy"'; then
        : # Health endpoint says healthy
    else
        # Health endpoint not available or unhealthy - non-critical
        : 
    fi
    
    if $healthy; then
        echo "HEALTHY"
        return 0
    else
        echo "UNHEALTHY"
        return 1
    fi
}

################################################################################
# Metrics Function
################################################################################

metrics() {
    print_header "METRICS SUMMARY"
    
    # Prometheus metrics summary
    if curl -s http://localhost:9091/metrics > /dev/null 2>&1; then
        echo -e "${BLUE}Batch Writer Metrics (Prometheus):${NC}"
        
        # Get total ticks inserted per symbol
        echo "  Ticks inserted by symbol:"
        curl -s http://localhost:9091/metrics 2>/dev/null | grep '^questdb_ticks_inserted_total' | \
            awk '{gsub(/.*symbol="/, ""); gsub(/".*/,""); print $0}' | sort | uniq -c | \
            awk '{printf "    %-10s %s ticks\\n", $2, $1}'
        
        # Get error counts
        local error_count=$(curl -s http://localhost:9091/metrics 2>/dev/null | grep '^questdb_errors_total' | awk '{sum+=$2} END {print sum+0}')
        echo "  Total errors: $error_count"
        
        echo ""
    fi
    
    # Health endpoint metrics
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${BLUE}Trades Collector Health:${NC}"
        curl -s http://localhost:8001/health 2>/dev/null | jq -r '
            "  Status: \(.status)",
            "  Messages written: \(.messages_written)",
            "  Reconnections: \(.reconnections)",
            "  Errors: \(.errors)",
            "  Uptime: \(.uptime_seconds)s"
        '
        echo ""
    fi
    
    # QuestDB metrics
    echo -e "${BLUE}QuestDB Metrics (Last 60 seconds):${NC}"
    curl -s -G 'http://localhost:9000/exec' --data-urlencode "query=
SELECT 
    'Trades' as type, 
    count() as events, 
    round(count()/60.0, 1) as rate_per_sec
FROM market_trades 
WHERE timestamp > now() - 60000000
UNION ALL
SELECT 
    'Orderbook' as type, 
    count() as events, 
    round(count()/60.0, 1) as rate_per_sec  
FROM market_orderbook 
WHERE timestamp > now() - 60000000
" 2>/dev/null | jq -r '.dataset[] | "  \(.[0]): \(.[1]) events (\(.[2])/sec)"'
    
    echo ""
}

################################################################################
# Main
################################################################################

case "${1:-status}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs "${2:-all}"
        ;;
    enable)
        enable
        ;;
    disable)
        disable
        ;;
    test)
        test_connectivity
        ;;
    health)
        health
        ;;
    metrics)
        metrics
        ;;
    *)
        echo "Usage: datafeed {start|stop|restart|status|logs|enable|disable|test|health|metrics}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the market data feed"
        echo "  stop     - Gracefully stop the feed"
        echo "  restart  - Restart the feed"
        echo "  status   - Show feed status"
        echo "  logs     - View logs (trades|orderbook|batch|all)"
        echo "  enable   - Enable auto-start on boot"
        echo "  disable  - Disable auto-start"
        echo "  test     - Run connectivity tests"
        echo "  health   - Health check (exit 0=healthy, 1=unhealthy)"
        echo "  metrics  - Show metrics summary"
        echo ""
        echo "Examples:"
        echo "  datafeed start"
        echo "  datafeed stop"
        echo "  datafeed status"
        echo "  datafeed test"
        echo "  datafeed metrics"
        echo "  datafeed logs trades"
        exit 1
        ;;
esac
