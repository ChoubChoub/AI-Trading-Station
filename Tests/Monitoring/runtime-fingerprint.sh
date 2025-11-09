#!/bin/bash
# redis/runtime-fingerprint.sh
#
# Institutional-grade runtime environment fingerprint for HFT Redis/Onload stack.
# Read-only: collects hashes, kernel, memory, IRQ/NIC/NUMA, security, and config state for drift detection and gatekeeping.
# Emits JSON to stdout. No changes to system state. No root required (except for some info fields).
#
# Usage: ./runtime-fingerprint.sh [--pretty]
#
# Fields:
#   redis_binary_hash, redis_config_hash, systemd_unit_hash, kernel_release, kernel_cmdline, thp_enabled, thp_defrag, swappiness,
#   irq_affinity_map, irqbalance_status, cpu_governor, numa_nodes, onload_version, nic_driver, mlock_status, hugepages_status, baseline_compare
#
# See spec for details. All output is read-only and safe for production.

set -euo pipefail

# Helper: hash a file (sha256, fallback to md5sum)
hash_file() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v md5sum >/dev/null 2>&1; then
    md5sum "$f" | awk '{print $1}'
  else
    echo "NO_HASH_TOOL"
  fi
}

# Helper: JSON escape
json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'
}

# Helper: safe command execution
safe_cmd() {
  local cmd="$1"
  local fallback="${2:-UNKNOWN}"
  eval "$cmd" 2>/dev/null || echo "$fallback"
}

# Collect fields - Redis HFT specific paths
REDIS_BIN="$(command -v redis-server || echo /usr/bin/redis-server)"
REDIS_CONF="/opt/redis-hft/config/redis-hft.conf"
SYSTEMD_UNIT="/etc/systemd/system/redis-hft.service"

# Hashes
redis_binary_hash="$(hash_file "$REDIS_BIN" 2>/dev/null || echo "NOT_FOUND")"
if [[ -f "$REDIS_CONF" ]]; then
  if [[ -r "$REDIS_CONF" ]]; then
    redis_config_hash="$(hash_file "$REDIS_CONF" 2>/dev/null || echo "READ_ERROR")"
  else
    # Try with sudo for protected files
    redis_config_hash="$(sudo sha256sum "$REDIS_CONF" 2>/dev/null | awk '{print $1}' || echo "PERMISSION_DENIED")"
  fi
else
  redis_config_hash="NOT_FOUND"
fi
systemd_unit_hash="$(hash_file "$SYSTEMD_UNIT" 2>/dev/null || echo "NOT_FOUND")"

# Kernel
kernel_release="$(uname -r)"
kernel_cmdline="$(cat /proc/cmdline 2>/dev/null | tr -d '\n')"

# Transparent Huge Pages (THP)
thp_enabled="$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null | grep -o '\[.*\]' | tr -d '[]' || echo "UNKNOWN")"
thp_defrag="$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null | grep -o '\[.*\]' | tr -d '[]' || echo "UNKNOWN")"

# Swappiness
swappiness="$(cat /proc/sys/vm/swappiness 2>/dev/null || echo "UNKNOWN")"

# IRQ affinity map (summarized) - clean format
irq_affinity_map="$(awk '/eth|enp|eno|ens/ {print $1,$NF}' /proc/interrupts 2>/dev/null | tr '\n' ';' | sed 's/;$//' || echo "UNKNOWN")"

# IRQ balance - clean status
if systemctl is-active irqbalance >/dev/null 2>&1; then
  irqbalance_status="active"
elif systemctl is-enabled irqbalance >/dev/null 2>&1; then
  irqbalance_status="inactive"
else
  irqbalance_status="not_installed"
fi

# CPU governor (BIOS performance mode if not available)
if [[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
  cpu_governor="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
else
  cpu_governor="BIOS_PERFORMANCE"
fi

# NUMA nodes
numa_nodes="$(numactl --hardware 2>/dev/null | grep available | awk '{print $2}' || echo "UNKNOWN")"

# Onload version - clean single value
if command -v onload >/dev/null 2>&1; then
  onload_version="$(onload --version 2>/dev/null | head -1 | awk '{print $2}' | tr -d '\n' || echo "UNKNOWN")"
  # Clean any extra characters
  onload_version="${onload_version%%[^0-9.]*}"
elif [[ -x /sbin/onload ]]; then
  onload_version="$(/sbin/onload --version 2>/dev/null | head -1 | awk '{print $2}' | tr -d '\n' || echo "UNKNOWN")"
  onload_version="${onload_version%%[^0-9.]*}"
else
  onload_version="NOT_FOUND"
fi

# NIC driver (first eth*)
nic_driver="$(basename $(readlink -f /sys/class/net/$(ls /sys/class/net | grep -E 'eth|enp|eno|ens' | head -1)/device/driver 2>/dev/null) 2>/dev/null || echo "UNKNOWN")"

# mlock status (rlimit)
mlock_status="$(ulimit -l 2>/dev/null || echo "UNKNOWN")"

# Huge pages
hugepages_status="$(cat /proc/meminfo 2>/dev/null | grep HugePages_Total | awk '{print $2}' || echo "UNKNOWN")"

# Baseline compare (optional, placeholder)
baseline_compare="NOT_IMPLEMENTED"

# Tail monitoring configuration (Phase 4B)
tail_gate_enabled="${TAIL_GATE_ENABLED:-false}"
tail_gate_warn_only="${TAIL_GATE_WARN_ONLY:-true}"
p99_9_max_rtt="${P99_9_MAX_RTT:-20.0}"
tail_span_max_rtt="${TAIL_SPAN_MAX_RTT:-8.0}"

# Detect monitor tail mode based on current configuration
monitor_tail_mode="standard"
if [[ "$tail_gate_enabled" == "true" ]]; then
  if [[ -f "State/tail-run.json" ]]; then
    monitor_tail_mode="tail_enabled"
  else
    monitor_tail_mode="tail_configured"
  fi
fi

# Add timestamp
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Create JSON using printf and proper escaping
{
  printf '{\n'
  printf '  "timestamp": "%s",\n' "$timestamp"
  printf '  "redis_binary_hash": "%s",\n' "$redis_binary_hash"
  printf '  "redis_config_hash": "%s",\n' "$redis_config_hash"
  printf '  "systemd_unit_hash": "%s",\n' "$systemd_unit_hash"
  printf '  "kernel_release": "%s",\n' "$kernel_release"
  printf '  "kernel_cmdline": %s,\n' "$(echo "$kernel_cmdline" | json_escape)"
  printf '  "thp_enabled": "%s",\n' "$thp_enabled"
  printf '  "thp_defrag": "%s",\n' "$thp_defrag"
  printf '  "swappiness": "%s",\n' "$swappiness"
  printf '  "irq_affinity_map": %s,\n' "$(echo "$irq_affinity_map" | json_escape)"
  printf '  "irqbalance_status": "%s",\n' "$irqbalance_status"
  printf '  "cpu_governor": "%s",\n' "$cpu_governor"
  printf '  "numa_nodes": "%s",\n' "$numa_nodes"
  printf '  "onload_version": "%s",\n' "$onload_version"
  printf '  "nic_driver": "%s",\n' "$nic_driver"
  printf '  "mlock_status": "%s",\n' "$mlock_status"
  printf '  "hugepages_status": "%s",\n' "$hugepages_status"
  printf '  "monitor_tail_mode": "%s",\n' "$monitor_tail_mode"
  printf '  "tail_thresholds": {\n'
  printf '    "p99_9_max_rtt": %s,\n' "$p99_9_max_rtt"
  printf '    "tail_span_max_rtt": %s,\n' "$tail_span_max_rtt"
  printf '    "gate_enabled": %s,\n' "$tail_gate_enabled"
  printf '    "warn_only": %s\n' "$tail_gate_warn_only"
  printf '  },\n'
  printf '  "baseline_compare": "%s"\n' "$baseline_compare"
  printf '}\n'
} | if [[ "${1:-}" == "--pretty" ]]; then
  python3 -m json.tool
else
  cat
fi
