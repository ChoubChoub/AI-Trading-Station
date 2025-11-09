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
      
      # Also move the IRQ kernel thread (if it exists with threadirqs)
      IRQ_THREAD_PID=$(ps -eLo pid,comm | grep "irq/$IRQ-" | awk '{print $1}' | head -1)
      if [ -n "$IRQ_THREAD_PID" ]; then
        taskset -cp 0,1 "$IRQ_THREAD_PID" > /dev/null 2>&1 && \
          echo "  → IRQ thread $IRQ_THREAD_PID moved to cores 0,1" || \
          echo "  → WARNING: Failed to move IRQ thread $IRQ_THREAD_PID"
      fi
    else
      echo "WARNING: /proc/irq/$IRQ/smp_affinity not found."
    fi
    CORE_IDX=$((1 - CORE_IDX))
  done

  # Restore XPS configuration to trading cores (2,3) after IRQ changes
  # Some drivers reset XPS when IRQ affinity changes, so we restore it
  TRADING_CORES_MASK="0c"  # Cores 2,3 in hex
  echo "Restoring XPS configuration to trading cores (mask: $TRADING_CORES_MASK)..."
  for xps_file in /sys/class/net/$NIC/queues/tx-*/xps_cpus; do
    if [ -f "$xps_file" ]; then
      echo "$TRADING_CORES_MASK" > "$xps_file" 2>/dev/null || echo "WARNING: Failed to restore XPS for $xps_file"
    fi
  done
  echo "XPS restored for $NIC to trading cores 2,3"

  # Log result for this NIC
  {
    echo "# IRQ Affinity Configuration Log"
    echo "# Date: $(date)"
    echo "# NIC: $NIC"
    echo "# IRQs: $IRQS"
    echo "# XPS restored to trading cores: $TRADING_CORES_MASK"
  } >> /var/log/nic-irq-affinity.log

done

# Also handle NVMe and other storage IRQs that may land on isolated cores
echo "Checking for NVMe/storage IRQs on isolated cores..."
NVME_IRQS=$(grep -E "nvme|ahci" /proc/interrupts | cut -d: -f1 | tr -d ' ')
for IRQ in $NVME_IRQS; do
  # Set hardware IRQ affinity to cores 0,1
  if [ -f "/proc/irq/$IRQ/smp_affinity_list" ]; then
    CURRENT_AFFINITY=$(cat "/proc/irq/$IRQ/smp_affinity_list")
    if echo "$CURRENT_AFFINITY" | grep -qE "2|3"; then
      if echo "0,1" > "/proc/irq/$IRQ/smp_affinity_list" 2>/dev/null; then
        echo "Storage IRQ $IRQ moved from core(s) $CURRENT_AFFINITY to cores 0,1"
      else
        echo "Storage IRQ $IRQ on core(s) $CURRENT_AFFINITY (cannot move - managed by driver)"
      fi
    fi
  fi
  
  # Move IRQ thread
  IRQ_THREAD_PID=$(ps -eLo pid,comm | grep "irq/$IRQ-" | awk '{print $1}' | head -1)
  if [ -n "$IRQ_THREAD_PID" ]; then
    taskset -cp 0,1 "$IRQ_THREAD_PID" > /dev/null 2>&1 && \
      echo "  → Storage IRQ thread $IRQ_THREAD_PID moved to cores 0,1"
  fi
done

echo "IRQ affinity configuration complete."
