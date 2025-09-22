#!/bin/bash
#
# Fix IRQ affinity - move IRQs away from isolated cores
# Must be run as root
set -euo pipefail
# Configuration
ISOLATED_CORES="2,3"
HOUSEKEEPING_CORES="0,1"
TRADING_INTERFACES=("enp130s0f0" "enp130s0f1")
# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   exit 1
fi
echo "=== Fixing IRQ Affinity ==="
echo "Moving IRQs away from isolated cores: $ISOLATED_CORES"
echo "Target housekeeping cores: $HOUSEKEEPING_CORES"
echo ""
# Function to set IRQ affinity
set_irq_affinity() {
    local irq=$1
    local cores=$2
    if [[ -f "/proc/irq/$irq/smp_affinity_list" ]]; then
        echo "$cores" > "/proc/irq/$irq/smp_affinity_list" 2>/dev/null && return 0
    fi
    return 1
}
# Fix trading interface IRQs
for iface in "${TRADING_INTERFACES[@]}"; do
    echo "Processing interface: $iface"
    if [[ ! -d "/sys/class/net/$iface" ]]; then
        echo -e "${YELLOW}  Interface $iface not found${NC}"
        continue
    fi
    # Get IRQs for this interface
    irqs=$(grep "$iface" /proc/interrupts | awk '{print $1}' | tr -d ':')
    if [[ -z "$irqs" ]]; then
        echo -e "${YELLOW}  No IRQs found for $iface${NC}"
        continue
    fi
    for irq in $irqs; do
        if set_irq_affinity "$irq" "$HOUSEKEEPING_CORES"; then
            echo -e "  IRQ $irq -> $HOUSEKEEPING_CORES ${GREEN}[FIXED]${NC}"
        else
            echo -e "  IRQ $irq ${RED}[FAILED]${NC}"
        fi
    done
    echo ""
done
# Fix any other IRQs on isolated cores
echo "Checking all other IRQs..."
fixed_count=0
while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]*([0-9]+): ]]; then
        irq="${BASH_REMATCH[1]}"
        if [[ -f "/proc/irq/$irq/smp_affinity_list" ]]; then
            current_affinity=$(cat "/proc/irq/$irq/smp_affinity_list" 2>/dev/null || echo "")
            # Check if current affinity includes isolated cores
            needs_fix=false
            IFS=',' read -ra cores <<< "$current_affinity"
            for core in "${cores[@]}"; do
                if [[ ",$ISOLATED_CORES," == *",$core,"* ]]; then
                    needs_fix=true
                    break
                fi
            done
            if $needs_fix; then
                desc=$(echo "$line" | awk '{for(i=NF-2;i<=NF;i++) printf "%s ", $i}')
                if set_irq_affinity "$irq" "$HOUSEKEEPING_CORES"; then
                    echo -e "  IRQ $irq ($desc) -> $HOUSEKEEPING_CORES ${GREEN}[FIXED]${NC}"
                    ((fixed_count++))
                else
                    echo -e "  IRQ $irq ($desc) ${RED}[FAILED]${NC}"
                fi
            fi
        fi
    fi
done < /proc/interrupts
echo ""
echo "=== Summary ==="
echo -e "${GREEN}Fixed $fixed_count IRQ(s)${NC}"
# Disable irqbalance if running
if systemctl is-active --quiet irqbalance; then
    echo ""
    echo -e "${YELLOW}Warning: irqbalance service is running${NC}"
    echo "It may override manual IRQ affinity settings."
    echo "Consider: sudo systemctl stop irqbalance && sudo systemctl disable irqbalance"
fi
