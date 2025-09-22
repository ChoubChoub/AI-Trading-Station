#!/bin/bash
# System Discovery Script for AI Trading Station Monitoring Setup (v2 + GPU)
# - Validates CPU topology vs isolcpus
# - Detects SMT and virtualization
# - Checks NIC IRQ affinities are pinned to housekeeping cores (0,1)
# - Reports NVIDIA GPU temperature if available
# - Produces a safe config template using actual ONLINE CPUs

set -euo pipefail

OUTPUT_FILE="system_info_$(date +%Y%m%d_%H%M%S).txt"

echo "=== AI Trading Station - System Discovery (v2+GPU) ==="
echo "Gathering system information for monitoring setup..."
echo
echo "Results will be saved to: $OUTPUT_FILE"
echo

log() { echo "$1" | tee -a "$OUTPUT_FILE"; }

expand_cpu_list() {
  local s="$1"
  local out=()
  IFS=',' read -ra PARTS <<< "$s"
  for p in "${PARTS[@]}"; do
    if [[ "$p" =~ ^[0-9]+-[0-9]+$ ]]; then
      local start=${p%-*}
      local end=${p#*-}
      for ((i=start; i<=end; i++)); do out+=("$i"); done
    elif [[ "$p" =~ ^[0-9]+$ ]]; then
      out+=("$p")
    fi
  done
  echo "${out[*]}"
}

log "=== SYSTEM DISCOVERY RESULTS ==="
log "Date: $(date)"
log "Hostname: $(hostname)"
log ""

virt_type="unknown"
if command -v systemd-detect-virt >/dev/null 2>&1; then
  virt_type=$(systemd-detect-virt || echo "none")
else
  if grep -q 'hypervisor' /proc/cpuinfo 2>/dev/null; then virt_type="generic-hypervisor"; else virt_type="none"; fi
fi
log "=== ENVIRONMENT ==="
log "Virtualization: $virt_type"
log ""

log "=== CPU TOPOLOGY ==="
nproc_val=$(nproc)
online_range=$(cat /sys/devices/system/cpu/online 2>/dev/null || echo "unknown")
online_cpus=""
if [[ "$online_range" != "unknown" ]]; then
  online_cpus=$(expand_cpu_list "$online_range")
fi

threads_per_core="unknown"
cores_per_socket="unknown"
sockets="unknown"
physical_cores="unknown"
smt_active="unknown"

if command -v lscpu >/dev/null 2>&1; then
  threads_per_core=$(LC_ALL=C lscpu | awk -F: '/Thread\(s\) per core/ {gsub(/ /,"",$2); print $2}')
  cores_per_socket=$(LC_ALL=C lscpu | awk -F: '/Core\(s\) per socket/ {gsub(/ /,"",$2); print $2}')
  sockets=$(LC_ALL=C lscpu | awk -F: '/Socket\(s\)/ {gsub(/ /,"",$2); print $2}')
  if [[ "$cores_per_socket" =~ ^[0-9]+$ && "$sockets" =~ ^[0-9]+$ ]]; then
    physical_cores=$((cores_per_socket * sockets))
  fi
fi

if [[ -r /sys/devices/system/cpu/smt/active ]]; then
  smt_val=$(cat /sys/devices/system/cpu/smt/active 2>/dev/null || echo "")
  case "$smt_val" in
    0) smt_active="disabled" ;;
    1) smt_active="enabled" ;;
    *) smt_active="$smt_val" ;;
  esac
fi

log "nproc (online logical CPUs): $nproc_val"
log "Online CPU range (/sys/.../online): $online_range"
log "Online CPU IDs: ${online_cpus:-unknown}"
log "Threads per core: $threads_per_core"
log "Cores per socket: $cores_per_socket"
log "Sockets: $sockets"
log "Physical cores (from lscpu): $physical_cores"
log "SMT status: $smt_active"
log ""

log "=== KERNEL ISOLATION (isolcpus) ==="
isol_arg=$(grep -o 'isolcpus=[^ ]*' /proc/cmdline 2>/dev/null || true)
if [[ -n "$isol_arg" ]]; then
  isol_value="${isol_arg#isolcpus=}"
  isol_list=$(expand_cpu_list "$isol_value")
  log "Kernel cmdline isolcpus: $isol_value"
  log "Isolated CPU IDs (expanded): $isol_list"
  invalid_isol=()
  if [[ -n "$online_cpus" ]]; then
    declare -A online_map=()
    for c in $online_cpus; do online_map[$c]=1; done
    for c in $isol_list; do
      [[ -z "${online_map[$c]:-}" ]] && invalid_isol+=("$c")
    done
  fi
  if (( ${#invalid_isol[@]} > 0 )); then
    log "WARNING: isolcpus contains IDs not currently online: ${invalid_isol[*]}"
  fi
else
  log "No isolcpus found in kernel cmdline."
  isol_list=""
fi
log ""

log "=== NETWORK INTERFACE INFORMATION ==="
TRADING_INTERFACES=()
ALL_INTERFACES=()
while IFS= read -r line; do
  if [[ $line =~ ^[0-9]+:\ ([^:]+): ]]; then
    iface="${BASH_REMATCH[1]}"
    [[ "$iface" == "lo" ]] && continue
    if [[ -d "/sys/class/net/$iface" ]]; then
      driver=$(ethtool -i "$iface" 2>/dev/null | awk '/^driver:/ {print $2}' || echo "unknown")
      speed=$(ethtool "$iface" 2>/dev/null | awk -F': ' '/Speed:/ {print $2}' || echo "unknown")
      state=$(cat "/sys/class/net/$iface/operstate" 2>/dev/null || echo "unknown")
      log "  ${iface}: driver=$driver, speed=$speed, state=$state"
      ALL_INTERFACES+=("$iface")
      if [[ "$driver" == "sfc" || "$driver" == mlx* || "$speed" == 10000* || "$speed" == 25000* || "$speed" == 40000* || "$speed" == 100000* ]]; then
        TRADING_INTERFACES+=("$iface")
        log "    RECOMMENDED for trading (high-performance)"
      fi
    fi
  fi
done <<< "$(ip link show)"
if (( ${#TRADING_INTERFACES[@]} > 0 )); then
  printf -v if_json '%s","' "${TRADING_INTERFACES[@]}"
  if_json='["'${if_json%\",\"}'"]'
  log "Suggested trading interfaces: $if_json"
else
  if_json='["eth0"]'
  log "No high-perf NIC detected; default suggestion: $if_json"
fi
log ""

log "=== IRQ AFFINITY VALIDATION (Housekeeping cores expected: 0,1) ==="
housekeeping="0 1"
if [[ -n "$online_cpus" ]]; then
  hk_online=()
  declare -A online_map=()
  for c in $online_cpus; do online_map[$c]=1; done
  for c in $housekeeping; do [[ "${online_map[$c]:-}" == "1" ]] && hk_online+=("$c"); done
  housekeeping="${hk_online[*]}"
fi
if (( ${#TRADING_INTERFACES[@]} > 0 )); then
  for iface in "${TRADING_INTERFACES[@]}"; do
    irqs=($(grep -i "$iface" /proc/interrupts | awk -F: '{print $1}' | tr -d ' '))
    if (( ${#irqs[@]} == 0 )); then
      log "  $iface: No IRQs found in /proc/interrupts (driver offload or interface down?)."
      continue
    fi
    ok=1
    bad=()
    for irq in "${irqs[@]}"; do
      aff_list=$(cat /proc/irq/$irq/smp_affinity_list 2>/dev/null || echo "")
      if [[ -z "$aff_list" ]]; then
        ok=0; bad+=("$irq:none"); continue
      fi
      subset_ok=1
      for cpu in $(expand_cpu_list "$aff_list"); do
        found=0
        for hk in $housekeeping; do [[ "$cpu" == "$hk" ]] && found=1 && break; done
        (( found == 0 )) && subset_ok=0 && break
      done
      (( subset_ok == 0 )) && ok=0 && bad+=("$irq:$aff_list")
    done
    if (( ok == 1 )); then
      log "  $iface: IRQs pinned to housekeeping cores [$housekeeping] - OK"
    else
      log "  $iface: IRQs not fully pinned to housekeeping cores [$housekeeping]; offending IRQs: ${bad[*]}"
    fi
  done
else
  log "  No trading interfaces selected; skipping IRQ validation."
fi
log ""

log "=== ONLOAD ACCELERATION ==="
if lsmod | grep -q "^onload"; then
  log "OnLoad module: loaded"
else
  log "OnLoad module: not loaded"
fi
if command -v onload_stackdump >/dev/null 2>&1; then
  log "OnLoad tools: available"
else
  log "OnLoad tools: NOT found in PATH"
fi
log ""

log "=== GPU INFORMATION ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  lines=$(nvidia-smi --query-gpu=index,name,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -n "$lines" ]]; then
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      idx=$(echo "$line" | cut -d',' -f1 | xargs)
      name=$(echo "$line" | cut -d',' -f2 | xargs)
      temp=$(echo "$line" | cut -d',' -f3 | xargs)
      log "  GPU $idx: $name, temperature: ${temp}°C"
    done <<< "$lines"
  else
    log "  No NVIDIA GPUs detected."
  fi
else
  log "  nvidia-smi not found; cannot query GPU temperatures."
fi
log ""

log "=== TRADING PROCESS DISCOVERY (conservative) ==="
CANDIDATES=("trading_engine" "market_data_feed" "order_manager" "risk_engine")
FOUND=()
for name in "${CANDIDATES[@]}"; do
  if pgrep -x "$name" >/dev/null 2>&1; then
    FOUND+=("\"$name\"")
    log "  Found: $name"
  fi
done
if (( ${#FOUND[@]} > 0 )); then
  proc_json="[$(IFS=, ; echo "${FOUND[*]}")]"
else
  proc_json="[]"
  log "  No known trading processes running now (this is fine)."
fi
log "Suggested trading processes list: $proc_json"
log ""

log "=== CPU GOVERNORS ==="
governors_found=0
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [[ -r "$g" ]] || continue
  cpu_num=$(basename "$(dirname "$g")" | sed 's/cpu//')
  gov=$(cat "$g" 2>/dev/null || echo "unknown")
  log "  CPU $cpu_num: $gov"
  governors_found=1
done
if (( governors_found == 0 )); then
  log "  No per-CPU governor files found (platform may not expose cpufreq)."
fi
log ""

log "=== GENERATED CONFIGURATION TEMPLATE ==="
isol_json="[]"
if [[ -n "$isol_list" && -n "$online_cpus" ]]; then
  declare -A online_map2=()
  for c in $online_cpus; do online_map2[$c]=1; done
  keep=()
  for c in $isol_list; do [[ "${online_map2[$c]:-}" == "1" ]] && keep+=("$c"); done
  if (( ${#keep[@]} > 0 )); then
    printf -v isol_json '[%s]' "$(IFS=, ; echo "${keep[*]}")"
  fi
fi

preferred_isolated="[]"
if [[ -n "$online_cpus" ]]; then
  have2=0; have3=0
  for c in $online_cpus; do [[ "$c" == "2" ]] && have2=1; [[ "$c" == "3" ]] && have3=1; done
  if (( have2==1 && have3==1 )); then
    preferred_isolated="[2,3]"
  elif [[ "$isol_json" != "[]" ]]; then
    preferred_isolated="$isol_json"
  fi
fi

trade_if_up=()
for iface in "${TRADING_INTERFACES[@]}"; do
  state=$(cat "/sys/class/net/$iface/operstate" 2>/dev/null || echo "unknown")
  [[ "$state" == "up" ]] && trade_if_up+=("\"$iface\"")
done
if (( ${#trade_if_up[@]} == 0 )); then
  if (( ${#TRADING_INTERFACES[@]} > 0 )); then
    trade_if_up=("\"${TRADING_INTERFACES[0]}\"")
  else
    trade_if_up=("\"eth0\"")
  fi
fi
trade_if_json="[$(IFS=, ; echo "${trade_if_up[*]}")]"

log ""
log "{"
log "  \"cpu\": {"
log "    \"isolated_cores\": $preferred_isolated,"
log "    \"expected_governor\": \"performance\""
log "  },"
log "  \"network\": {"
log "    \"trading_interfaces\": $trade_if_json"
log "  },"
log "  \"processes\": {"
log "    \"trading\": $proc_json"
log "  }"
log "}"
log ""

log "=== SUMMARY & NOTES ==="
if [[ "$preferred_isolated" == "[]" ]]; then
  log "NOTE: No valid isolated trading cores determined."
  log " - isolcpus on cmdline: ${isol_arg:-none}"
  log " - Online CPUs: ${online_cpus:-unknown}"
  log "Action: Ensure intended trading cores (e.g., 2 and 3) are ONLINE on this host, or adjust isolcpus accordingly."
else
  log "Trading cores candidate (for monitoring isolated_cores): $preferred_isolated"
fi
log "Housekeeping cores expected: [0,1] (validated against online: $housekeeping)"
log "IRQ pinning status was checked per NIC above."
log "Virtualization: $virt_type (if not 'none', discovered CPU counts reflect the guest, not the bare-metal host)."
log ""

echo
echo "✅ Discovery complete (v2+GPU)."
echo "📄 Results saved to: $OUTPUT_FILE"
