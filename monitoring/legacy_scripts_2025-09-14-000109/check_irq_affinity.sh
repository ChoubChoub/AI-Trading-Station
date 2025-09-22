#!/bin/bash
#
# Check IRQ affinity for trading interfaces
# Ensures IRQs are not on isolated cores
set -euo pipefail
# Configuration
ISOLATED_CORES="2,3"
HOUSEKEEPING_CORES="0,1"
TRADING_INTERFACES=("enp130s0f0" "enp130s0f1")
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
echo "=== IRQ Affinity Check ==="
echo "Isolated cores: $ISOLATED_CORES"
echo "Housekeeping cores: $HOUSEKEEPING_CORES"
echo ""
# Function to check if core is isolated
is_isolated_core() {
    local core=$1
    [[ ",$ISOLATED_CORES," == *",$core,"* ]]
}
# Check each interface
for iface in "${TRADING_INTERFACES[@]}"; do
    echo "Checking interface: $iface"
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
    violations=0
    for irq in $irqs; do
        if [[ -f "/proc/irq/$irq/smp_affinity_list" ]]; then
            affinity=$(cat "/proc/irq/$irq/smp_affinity_list")
            echo -n "  IRQ $irq affinity: $affinity"
            # Check if any isolated cores are in the affinity
            violation=false
            IFS=',' read -ra cores <<< "$affinity"
            for core in "${cores[@]}"; do
                # Handle ranges like 0-3
                if [[ "$core" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                    for ((c=${BASH_REMATCH[1]}; c<=${BASH_REMATCH[2]}; c++)); do
                        if is_isolated_core "$c"; then
                            violation=true
                            break
                        fi
                    done
                else
                    if is_isolated_core "$core"; then
                        violation=true
                    fi
                fi
            done
            if $violation; then
                echo -e " ${RED}[VIOLATION]${NC}"
                ((violations++))
            else
                echo -e " ${GREEN}[OK]${NC}"
            fi
        fi
    done
    if [[ $violations -eq 0 ]]; then
        echo -e "${GREEN}  ✓ All IRQs properly configured${NC}"
    else
        echo -e "${RED}  ✗ $violations IRQ(s) on isolated cores!${NC}"
    fi
    echo ""
done
# Check for unexpected IRQs on isolated cores
echo "Checking all IRQs on isolated cores..."
isolated_irq_count=0
while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]*([0-9]+): ]]; then
        irq="${BASH_REMATCH[1]}"
        if [[ -f "/proc/irq/$irq/smp_affinity_list" ]]; then
            affinity=$(cat "/proc/irq/$irq/smp_affinity_list" 2>/dev/null || echo "")
            # Check if affinity includes isolated cores
            IFS=',' read -ra cores <<< "$affinity"
            for core in "${cores[@]}"; do
                if [[ "$core" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                    for ((c=${BASH_REMATCH[1]}; c<=${BASH_REMATCH[2]}; c++)); do
                        if is_isolated_core "$c"; then
                            desc=$(echo "$line" | awk '{for(i=NF-2;i<=NF;i++) printf "%s ", $i}')
                            echo -e "${RED}  IRQ $irq on isolated core(s): $affinity - $desc${NC}"
                            ((isolated_irq_count++))
                            break 2
                        fi
                    done
                else
                    if is_isolated_core "$core"; then
                        desc=$(echo "$line" | awk '{for(i=NF-2;i<=NF;i++) printf "%s ", $i}')
                        echo -e "${RED}  IRQ $irq on isolated core(s): $affinity - $desc${NC}"
                        ((isolated_irq_count++))
                        break
                    fi
                fi
            done
        fi
    fi
done < /proc/interrupts
if [[ $isolated_irq_count -eq 0 ]]; then
    echo -e "${GREEN}✓ No IRQs found on isolated cores${NC}"
else
    echo -e "${RED}✗ Found $isolated_irq_count IRQ(s) on isolated cores${NC}"
fi
echo ""
echo "=== Summary ==="
echo "Use 'sudo ./scripts/fix_irq_affinity.sh' to fix any violations"
