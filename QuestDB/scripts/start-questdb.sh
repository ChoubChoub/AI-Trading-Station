#!/bin/bash
################################################################################
# QuestDB Startup Script
# Optimized for 192GB RAM + 20 Strategy Architecture
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

QUESTDB_HOME="/home/youssefbahloul/ai-trading-station/QuestDB/questdb-9.1.0-rt-linux-x86-64"
QUESTDB_DATA="/home/youssefbahloul/ai-trading-station/QuestDB/data"
QUESTDB_LOGS="/home/youssefbahloul/ai-trading-station/QuestDB/logs"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting QuestDB for Crypto Trading${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if QuestDB is already running
if pgrep -f "questdb.*server.conf" > /dev/null; then
    echo -e "${YELLOW}⚠ QuestDB is already running${NC}"
    echo ""
    PID=$(pgrep -f "questdb.*server.conf")
    echo "Process ID: ${PID}"
    echo ""
    echo "To stop: kill ${PID}"
    echo "To restart: ./stop-questdb.sh && ./start-questdb.sh"
    exit 1
fi

# Pre-flight checks
echo -e "${YELLOW}Pre-flight checks...${NC}"

# Check memory
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_MEM" -lt 180 ]; then
    echo -e "${RED}ERROR: Insufficient memory. Expected 192GB, found ${TOTAL_MEM}GB${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Memory: ${TOTAL_MEM}GB available${NC}"

# Check NVMe space
NVME_AVAIL=$(df -h / | tail -1 | awk '{print $4}')
echo -e "${GREEN}✓ NVMe space available: ${NVME_AVAIL}${NC}"

# Check data directory
if [ ! -d "$QUESTDB_DATA/hot" ]; then
    echo -e "${RED}ERROR: Data directory not found: ${QUESTDB_DATA}/hot${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Data directory: ${QUESTDB_DATA}/hot${NC}"

# Check configuration
if [ ! -f "$QUESTDB_HOME/conf/server.conf" ]; then
    echo -e "${RED}ERROR: Configuration not found: ${QUESTDB_HOME}/conf/server.conf${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Configuration: ${QUESTDB_HOME}/conf/server.conf${NC}"

echo ""

# Set Java options for optimal performance
export JAVA_OPTS="-XX:+UseG1GC \
-XX:MaxGCPauseMillis=100 \
-XX:+UseStringDeduplication \
-XX:+ParallelRefProcEnabled \
-XX:+UnlockExperimentalVMOptions \
-XX:G1NewSizePercent=20 \
-XX:G1ReservePercent=15 \
-XX:InitiatingHeapOccupancyPercent=35 \
-XX:+UseNUMA \
-XX:+AlwaysPreTouch \
-Djava.awt.headless=true"

echo -e "${GREEN}Starting QuestDB...${NC}"
echo "Log file: ${QUESTDB_LOGS}/questdb.log"
echo ""

# Start QuestDB in background
cd "$QUESTDB_HOME"
nohup ./bin/questdb.sh start -d "$QUESTDB_DATA/hot" > "$QUESTDB_LOGS/questdb.log" 2>&1 &

# Wait for startup
sleep 5

# Check if started successfully
if pgrep -f "questdb.*server.conf" > /dev/null; then
    PID=$(pgrep -f "questdb.*server.conf")
    echo -e "${GREEN}✓ QuestDB started successfully!${NC}"
    echo ""
    echo "Process ID: ${PID}"
    echo ""
    echo -e "${GREEN}Endpoints:${NC}"
    echo "  HTTP:       http://localhost:9000"
    echo "  PostgreSQL: localhost:8812"
    echo "  InfluxDB:   localhost:9009"
    echo ""
    echo -e "${GREEN}Web Console: http://localhost:9000${NC}"
    echo ""
    echo "Tail logs: tail -f ${QUESTDB_LOGS}/questdb.log"
    echo "Stop: ./stop-questdb.sh"
    echo ""
else
    echo -e "${RED}ERROR: QuestDB failed to start${NC}"
    echo "Check logs: cat ${QUESTDB_LOGS}/questdb.log"
    exit 1
fi
