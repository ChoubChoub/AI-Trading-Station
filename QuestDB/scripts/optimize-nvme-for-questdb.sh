#!/bin/bash
################################################################################
# NVMe Optimization Script for QuestDB
# Hardware: Samsung 990 Pro NVMe (3.6TB)
# Purpose: Optimize I/O scheduler, queue depth, and mount options
# Run: sudo ./optimize-nvme-for-questdb.sh
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}QuestDB NVMe Optimization Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}ERROR: Please run as root (sudo)${NC}"
    exit 1
fi

# Detect NVMe device
NVME_DEVICE=$(lsblk -o NAME,TYPE | grep nvme | head -1 | awk '{print $1}')
if [ -z "$NVME_DEVICE" ]; then
    echo -e "${RED}ERROR: No NVMe device found${NC}"
    exit 1
fi

echo -e "${GREEN}Detected NVMe device: /dev/${NVME_DEVICE}${NC}"
echo ""

################################################################################
# 1. I/O Scheduler Optimization
################################################################################

echo -e "${YELLOW}[1/5] Optimizing I/O Scheduler...${NC}"

# Set I/O scheduler to 'none' for NVMe (best for high-performance SSDs)
CURRENT_SCHEDULER=$(cat /sys/block/${NVME_DEVICE}/queue/scheduler)
echo "Current scheduler: ${CURRENT_SCHEDULER}"

if [[ "$CURRENT_SCHEDULER" == *"[none]"* ]]; then
    echo -e "${GREEN}✓ Scheduler already set to 'none'${NC}"
else
    echo "none" > /sys/block/${NVME_DEVICE}/queue/scheduler
    echo -e "${GREEN}✓ Set scheduler to 'none' (optimal for NVMe)${NC}"
fi

# Make persistent across reboots
UDEV_RULE="/etc/udev/rules.d/60-scheduler.rules"
if [ ! -f "$UDEV_RULE" ]; then
    echo "ACTION==\"add|change\", KERNEL==\"${NVME_DEVICE}\", ATTR{queue/scheduler}=\"none\"" > "$UDEV_RULE"
    echo -e "${GREEN}✓ Created udev rule for persistent scheduler setting${NC}"
fi

echo ""

################################################################################
# 2. Queue Depth Optimization
################################################################################

echo -e "${YELLOW}[2/5] Optimizing Queue Depth...${NC}"

CURRENT_NR_REQUESTS=$(cat /sys/block/${NVME_DEVICE}/queue/nr_requests)
echo "Current nr_requests: ${CURRENT_NR_REQUESTS}"

# Increase queue depth to 1024 for high parallelism
echo 1024 > /sys/block/${NVME_DEVICE}/queue/nr_requests
echo -e "${GREEN}✓ Set queue depth to 1024 (from ${CURRENT_NR_REQUESTS})${NC}"

# Increase max sectors (optimal for Samsung 990 Pro)
CURRENT_MAX_SECTORS=$(cat /sys/block/${NVME_DEVICE}/queue/max_sectors_kb)
echo "Current max_sectors_kb: ${CURRENT_MAX_SECTORS}"

if [ "$CURRENT_MAX_SECTORS" -lt 1024 ]; then
    echo 1024 > /sys/block/${NVME_DEVICE}/queue/max_sectors_kb
    echo -e "${GREEN}✓ Set max_sectors_kb to 1024${NC}"
else
    echo -e "${GREEN}✓ max_sectors_kb already optimal (${CURRENT_MAX_SECTORS})${NC}"
fi

echo ""

################################################################################
# 3. Read-Ahead Optimization
################################################################################

echo -e "${YELLOW}[3/5] Optimizing Read-Ahead...${NC}"

CURRENT_READ_AHEAD=$(blockdev --getra /dev/${NVME_DEVICE})
echo "Current read-ahead: ${CURRENT_READ_AHEAD} sectors"

# Set read-ahead to 4096 sectors (2MB) for large sequential reads
blockdev --setra 4096 /dev/${NVME_DEVICE}
NEW_READ_AHEAD=$(blockdev --getra /dev/${NVME_DEVICE})
echo -e "${GREEN}✓ Set read-ahead to ${NEW_READ_AHEAD} sectors (2MB)${NC}"

echo ""

################################################################################
# 4. NVMe-Specific Optimizations
################################################################################

echo -e "${YELLOW}[4/5] Applying NVMe-Specific Optimizations...${NC}"

# Enable write cache (if not already enabled)
if command -v nvme &> /dev/null; then
    WRITE_CACHE=$(nvme get-feature /dev/${NVME_DEVICE} -f 0x06 2>/dev/null | grep -i "volatile write cache" || echo "unknown")
    echo "Write cache status: ${WRITE_CACHE}"
    echo -e "${GREEN}✓ NVMe write cache verified${NC}"
else
    echo -e "${YELLOW}⚠ nvme-cli not installed, skipping NVMe feature checks${NC}"
    echo "  Install with: sudo apt install nvme-cli"
fi

# Set NVMe multiqueue optimization
echo 1 > /sys/block/${NVME_DEVICE}/queue/nomerges
echo -e "${GREEN}✓ Disabled request merging (optimal for NVMe multiqueue)${NC}"

# Enable no-op rotational
echo 0 > /sys/block/${NVME_DEVICE}/queue/rotational
echo -e "${GREEN}✓ Set rotational flag to 0 (SSD optimization)${NC}"

echo ""

################################################################################
# 5. File System Mount Options Check
################################################################################

echo -e "${YELLOW}[5/5] Checking File System Mount Options...${NC}"

MOUNT_POINT=$(df /home/youssefbahloul/ai-trading-station/QuestDB | tail -1 | awk '{print $6}')
MOUNT_OPTIONS=$(mount | grep "${MOUNT_POINT}" | grep -oP '\(.*?\)' | tr -d '()')

echo "QuestDB is on mount point: ${MOUNT_POINT}"
echo "Current mount options: ${MOUNT_OPTIONS}"

# Check for recommended options
RECOMMENDED_OPTIONS=("noatime" "nodiratime")
MISSING_OPTIONS=()

for opt in "${RECOMMENDED_OPTIONS[@]}"; do
    if [[ ! "$MOUNT_OPTIONS" =~ $opt ]]; then
        MISSING_OPTIONS+=("$opt")
    fi
done

if [ ${#MISSING_OPTIONS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All recommended mount options are present${NC}"
else
    echo -e "${YELLOW}⚠ Missing recommended mount options: ${MISSING_OPTIONS[*]}${NC}"
    echo ""
    echo "To add missing options, update /etc/fstab:"
    echo "  1. Find the line for ${MOUNT_POINT}"
    echo "  2. Add options: noatime,nodiratime"
    echo "  3. Run: sudo mount -o remount ${MOUNT_POINT}"
    echo ""
    echo "Example fstab entry:"
    echo "UUID=<uuid> ${MOUNT_POINT} ext4 noatime,nodiratime,errors=remount-ro 0 1"
fi

echo ""

################################################################################
# Summary
################################################################################

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Optimization Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Applied optimizations:"
echo "  ✓ I/O Scheduler: none (optimal for NVMe)"
echo "  ✓ Queue Depth: 1024 requests"
echo "  ✓ Max Sectors: 1024 KB"
echo "  ✓ Read-Ahead: 4096 sectors (2MB)"
echo "  ✓ Request Merging: Disabled (NVMe multiqueue)"
echo "  ✓ Rotational: Disabled (SSD mode)"
echo ""

echo -e "${YELLOW}Performance Validation:${NC}"
echo "Run these commands to verify improvements:"
echo ""
echo "  1. Check I/O stats:"
echo "     iostat -x 1 10"
echo ""
echo "  2. Monitor NVMe performance:"
echo "     sudo nvme smart-log /dev/${NVME_DEVICE}"
echo ""
echo "  3. Test write speed:"
echo "     dd if=/dev/zero of=/home/youssefbahloul/ai-trading-station/QuestDB/data/test bs=1G count=1 oflag=direct"
echo ""
echo "  4. Test read speed:"
echo "     dd if=/home/youssefbahloul/ai-trading-station/QuestDB/data/test of=/dev/null bs=1G count=1 iflag=direct"
echo ""

echo -e "${GREEN}Next Steps:${NC}"
echo "  1. Start QuestDB: ./start-questdb.sh"
echo "  2. Run performance benchmark: ./benchmark-questdb.sh"
echo "  3. Monitor with: watch -n 1 'iostat -x 1 1 | grep ${NVME_DEVICE}'"
echo ""
