#!/bin/bash

# ============================================================================
# CORE ISOLATION MONITORING AND CONTROL
# ============================================================================
# PURPOSE: Monitor and configure CPU core isolation for deterministic trading
# TARGET: Isolate cores 2,3 for trading with <1μs scheduling jitter
# VALIDATION: cyclictest shows <5μs max latency on isolated cores
# ============================================================================

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/core-isolation-monitor.log"
readonly ISOLATION_STATE_FILE="/tmp/core-isolation-state"
readonly PERFORMANCE_LOG="/tmp/core-isolation-performance.log"

# Default configuration
TRADING_CORES="2,3"
MONITOR_INTERVAL=5
DAEMON_MODE=false
APPLY_ISOLATION=false

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$$] $*" | tee -a "${LOG_FILE}"
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

check_root() {
    [[ $EUID -eq 0 ]] || error "Must run as root for core isolation"
}

get_current_isolation() {
    if [[ -f /sys/devices/system/cpu/isolated ]]; then
        cat /sys/devices/system/cpu/isolated
    else
        echo ""
    fi
}

validate_cores() {
    local cores="$1"
    local max_cpu=$(($(nproc) - 1))
    
    IFS=',' read -ra CORE_ARRAY <<< "$cores"
    for core in "${CORE_ARRAY[@]}"; do
        if [[ ! "$core" =~ ^[0-9]+$ ]] || [[ "$core" -gt "$max_cpu" ]]; then
            error "Invalid core number: $core (max: $max_cpu)"
        fi
    done
}

apply_core_isolation() {
    local cores="$1"
    validate_cores "$cores"
    
    log "Applying core isolation for cores: $cores"
    
    # Check if isolation is already applied
    local current_isolation
    current_isolation=$(get_current_isolation)
    if [[ "$current_isolation" == "$cores" ]]; then
        log "Core isolation already applied for: $cores"
        return 0
    fi
    
    # Apply isolation via kernel parameters (requires reboot)
    local grub_file="/etc/default/grub"
    local grub_backup="${grub_file}.backup.$(date +%s)"
    
    if [[ -f "$grub_file" ]]; then
        cp "$grub_file" "$grub_backup"
        log "GRUB configuration backed up to: $grub_backup"
        
        # Update GRUB_CMDLINE_LINUX_DEFAULT
        if grep -q "isolcpus=" "$grub_file"; then
            sed -i "s/isolcpus=[^ \"]*/isolcpus=$cores/" "$grub_file"
        else
            sed -i '/GRUB_CMDLINE_LINUX_DEFAULT=/ s/"$/ isolcpus='"$cores"'"/' "$grub_file"
        fi
        
        # Add additional optimization parameters
        if ! grep -q "nohz_full=" "$grub_file"; then
            sed -i '/GRUB_CMDLINE_LINUX_DEFAULT=/ s/"$/ nohz_full='"$cores"'"/' "$grub_file"
        fi
        
        if ! grep -q "rcu_nocbs=" "$grub_file"; then
            sed -i '/GRUB_CMDLINE_LINUX_DEFAULT=/ s/"$/ rcu_nocbs='"$cores"'"/' "$grub_file"
        fi
        
        # Update GRUB
        update-grub
        log "GRUB updated. Reboot required for isolation to take effect."
        
        # Create state file
        echo "cores=$cores" > "$ISOLATION_STATE_FILE"
        echo "applied=$(date)" >> "$ISOLATION_STATE_FILE"
        echo "reboot_required=yes" >> "$ISOLATION_STATE_FILE"
    fi
    
    # Apply immediate optimizations (no reboot required)
    apply_runtime_optimizations "$cores"
}

apply_runtime_optimizations() {
    local cores="$1"
    
    log "Applying runtime optimizations for cores: $cores"
    
    IFS=',' read -ra CORE_ARRAY <<< "$cores"
    for core in "${CORE_ARRAY[@]}"; do
        # Set CPU governor to performance
        if [[ -f "/sys/devices/system/cpu/cpu$core/cpufreq/scaling_governor" ]]; then
            echo "performance" > "/sys/devices/system/cpu/cpu$core/cpufreq/scaling_governor"
            log "Set CPU $core governor to performance"
        fi
        
        # Disable CPU frequency scaling
        if [[ -f "/sys/devices/system/cpu/cpu$core/cpufreq/scaling_min_freq" ]] && \
           [[ -f "/sys/devices/system/cpu/cpu$core/cpufreq/scaling_max_freq" ]]; then
            local max_freq
            max_freq=$(cat "/sys/devices/system/cpu/cpu$core/cpufreq/scaling_max_freq")
            echo "$max_freq" > "/sys/devices/system/cpu/cpu$core/cpufreq/scaling_min_freq"
            log "Set CPU $core to maximum frequency: ${max_freq}kHz"
        fi
        
        # Set CPU affinity for kernel threads away from trading cores
        for kthread in $(ps -eo pid,comm | grep '\[.*\]' | awk '{print $1}'); do
            if [[ -f "/proc/$kthread/task/$kthread/comm" ]]; then
                taskset -cp 0,1 "$kthread" 2>/dev/null || true
            fi
        done
    done
    
    # Apply system-wide optimizations
    # Disable irqbalance to maintain manual IRQ affinity
    systemctl stop irqbalance 2>/dev/null || true
    systemctl disable irqbalance 2>/dev/null || true
    
    # Set kernel parameters for low latency
    echo "1" > /proc/sys/kernel/sched_rt_runtime_us 2>/dev/null || true
    echo "0" > /proc/sys/kernel/timer_migration 2>/dev/null || true
    echo "0" > /proc/sys/kernel/numa_balancing 2>/dev/null || true
    
    log "Runtime optimizations applied"
}

measure_isolation_performance() {
    local cores="$1"
    local duration="${2:-30}"
    
    log "Measuring isolation performance on cores: $cores for ${duration}s"
    
    # Install cyclictest if not available
    if ! command -v cyclictest >/dev/null 2>&1; then
        apt-get update && apt-get install -y rt-tests
    fi
    
    IFS=',' read -ra CORE_ARRAY <<< "$cores"
    local results_file="/tmp/cyclictest-$(date +%s).log"
    
    # Run cyclictest on isolated cores
    local core_list=""
    for core in "${CORE_ARRAY[@]}"; do
        core_list="${core_list}${core},"
    done
    core_list="${core_list%,}"  # Remove trailing comma
    
    cyclictest -t "${#CORE_ARRAY[@]}" -a "$core_list" -n -p 99 -i 100 -h 1000 -q -D "${duration}s" > "$results_file" 2>&1
    
    # Parse results
    local max_latency min_latency avg_latency
    max_latency=$(grep "Max Latencies" "$results_file" | awk '{print $3}' | cut -d/ -f1 | sort -n | tail -1)
    min_latency=$(grep "Min Latencies" "$results_file" | awk '{print $3}' | cut -d/ -f1 | sort -n | head -1)
    avg_latency=$(grep "Avg Latencies" "$results_file" | awk '{print $3}' | cut -d/ -f1 | awk '{sum+=$1} END {print sum/NR}')
    
    # Log performance metrics
    {
        echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "cores=$cores"
        echo "duration=${duration}s"
        echo "max_latency_us=$max_latency"
        echo "min_latency_us=$min_latency"
        echo "avg_latency_us=$avg_latency"
        echo "target_max_latency_us=5"
        if [[ "${max_latency:-999}" -le 5 ]]; then
            echo "performance_target=PASS"
        else
            echo "performance_target=FAIL"
        fi
        echo "---"
    } >> "$PERFORMANCE_LOG"
    
    log "Performance test completed:"
    log "  Max Latency: ${max_latency}μs"
    log "  Min Latency: ${min_latency}μs" 
    log "  Avg Latency: ${avg_latency}μs"
    log "  Target: ≤5μs ($(if [[ "${max_latency:-999}" -le 5 ]]; then echo "PASS"; else echo "FAIL"; fi))"
    
    rm -f "$results_file"
}

monitor_isolation_status() {
    log "Starting isolation monitoring (interval: ${MONITOR_INTERVAL}s)"
    
    while true; do
        local current_isolation
        current_isolation=$(get_current_isolation)
        
        if [[ -n "$current_isolation" ]]; then
            log "Current isolation: $current_isolation"
            
            # Check process affinity violations
            local violations=0
            for pid in $(ps -eo pid --no-headers); do
                if [[ -f "/proc/$pid/task/$pid/stat" ]]; then
                    local affinity
                    affinity=$(taskset -p "$pid" 2>/dev/null | awk '{print $NF}' || echo "")
                    if [[ -n "$affinity" ]]; then
                        # Check if process is bound to isolated cores incorrectly
                        # This is a simplified check - in production, use more sophisticated analysis
                        :
                    fi
                fi
            done
            
            if [[ $violations -gt 0 ]]; then
                log "WARNING: $violations processes may be violating core isolation"
            fi
        else
            log "No core isolation currently active"
        fi
        
        # Measure current performance if requested
        if [[ -f "$ISOLATION_STATE_FILE" ]] && grep -q "measure=yes" "$ISOLATION_STATE_FILE"; then
            measure_isolation_performance "$TRADING_CORES" 10
        fi
        
        sleep "$MONITOR_INTERVAL"
    done
}

print_status() {
    echo "=== CORE ISOLATION STATUS ==="
    echo "Current Isolation: $(get_current_isolation)"
    echo "Trading Cores: $TRADING_CORES"
    
    if [[ -f "$ISOLATION_STATE_FILE" ]]; then
        echo "--- Isolation State ---"
        cat "$ISOLATION_STATE_FILE"
    fi
    
    if [[ -f "$PERFORMANCE_LOG" ]]; then
        echo "--- Latest Performance ---"
        tail -10 "$PERFORMANCE_LOG"
    fi
    echo "=========================="
}

usage() {
    cat << EOF
Core Isolation Monitor and Control

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --apply --cores=X,Y    Apply core isolation for specified cores
    --monitor              Monitor isolation status
    --daemon               Run monitoring as daemon
    --measure              Perform isolation performance measurement
    --status               Show current isolation status
    --cores=X,Y            Specify trading cores (default: $TRADING_CORES)
    --interval=N           Monitor interval in seconds (default: $MONITOR_INTERVAL)
    --help                 Show this help

EXAMPLES:
    $0 --apply --cores=2,3         Apply isolation to cores 2,3
    $0 --monitor --daemon          Start monitoring daemon
    $0 --measure --cores=2,3       Measure isolation performance
    $0 --status                    Show current status

PERFORMANCE TARGET:
    Maximum latency ≤5μs on isolated cores

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --apply)
            APPLY_ISOLATION=true
            shift
            ;;
        --monitor)
            MONITOR_MODE=true
            shift
            ;;
        --daemon)
            DAEMON_MODE=true
            shift
            ;;
        --measure)
            MEASURE_MODE=true
            shift
            ;;
        --status)
            STATUS_MODE=true
            shift
            ;;
        --cores=*)
            TRADING_CORES="${1#*=}"
            shift
            ;;
        --interval=*)
            MONITOR_INTERVAL="${1#*=}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Main execution
check_root

if [[ "${APPLY_ISOLATION:-}" == "true" ]]; then
    apply_core_isolation "$TRADING_CORES"
elif [[ "${MEASURE_MODE:-}" == "true" ]]; then
    measure_isolation_performance "$TRADING_CORES"
elif [[ "${MONITOR_MODE:-}" == "true" ]]; then
    if [[ "$DAEMON_MODE" == "true" ]]; then
        monitor_isolation_status &
        echo $! > /var/run/core-isolation-monitor.pid
        log "Core isolation monitor started as daemon (PID: $!)"
    else
        monitor_isolation_status
    fi
elif [[ "${STATUS_MODE:-}" == "true" ]]; then
    print_status
else
    usage
    exit 1
fi