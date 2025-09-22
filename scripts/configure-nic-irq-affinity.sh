#!/bin/bash
set -e

# List of trading NICs to pin
NICS=("enp130s0f0" "enp130s0f1")

# Housekeeping CPU cores
CORES=(0 1)

# Check that both housekeeping cores are online
for core in "${CORES[@]}"; do
  if [ ! -d "/sys/devices/system/cpu/cpu$core" ]; then
    echo "ERROR: CPU core $core not found."
    exit 1
  fi
  if [ -f "/sys/devices/system/cpu/cpu$core/online" ]; then
    if [ "$(cat /sys/devices/system/cpu/cpu$core/online)" != "1" ]; then
      echo "ERROR: CPU core $core is offline."
      exit 1
    fi
  fi
done

for NIC in "${NICS[@]}"; do
  if [ ! -d "/sys/class/net/$NIC" ]; then
    echo "WARNING: $NIC does not exist, skipping."
    continue
  fi

  # Find IRQs for this NIC
  IRQS=$(grep "$NIC" /proc/interrupts | cut -d: -f1 | tr -d ' ')
  if [ -z "$IRQS" ]; then
    IRQS=$(grep -E "${NIC}-.*|.*-${NIC}|.*TxRx.*|.*-rx-.*|.*-tx-.*" /proc/interrupts | cut -d: -f1 | tr -d ' ')
  fi
  if [ -z "$IRQS" ]; then
    echo "No IRQs found for $NIC"
    continue
  fi

  # Alternate assignment between core 0 and core 1
  CORE_IDX=0
  for IRQ in $IRQS; do
    MASK=$((1 << ${CORES[$CORE_IDX]}))
    HEXMASK=$(printf "%x" "$MASK")
    if [ -f "/proc/irq/$IRQ/smp_affinity" ]; then
      echo "$HEXMASK" > "/proc/irq/$IRQ/smp_affinity"
      echo "IRQ $IRQ assigned to core ${CORES[$CORE_IDX]} (mask $HEXMASK)"
    else
      echo "WARNING: /proc/irq/$IRQ/smp_affinity not found."
    fi
    CORE_IDX=$((1 - CORE_IDX))
  done

  # Log result for this NIC
  {
    echo "# IRQ Affinity Configuration Log"
    echo "# Date: $(date)"
    echo "# NIC: $NIC"
    echo "# IRQs: $IRQS"
  } >> /var/log/nic-irq-affinity.log

done

echo "IRQ affinity configuration complete."
