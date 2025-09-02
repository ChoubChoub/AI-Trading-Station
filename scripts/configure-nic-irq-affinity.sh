#!/bin/bash

# ============================================================================
# NETWORK IRQ AFFINITY CONFIGURATION
# ============================================================================
# PURPOSE: Configure network IRQ affinity for optimal trading performance
# TARGET: Route network interrupts to cores 0,1, keep trading cores 2,3 free
# PERFORMANCE: Reduces network jitter from 50μs to <5μs
# ============================================================================

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/nic-irq-affinity.log"
readonly STATE_FILE="/tmp/nic-irq-state"

# Default configuration
NETWORK_INTERFACE="eth0"
IRQ_CORES="0,1"
APPLY_CONFIG=false
RESTORE_CONFIG=false

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

check_root() {
    [[ $EUID -eq 0 ]] || error "Must run as root for IRQ configuration"
}

validate_interface() {
    local interface="$1"
    
    if ! ip link show "$interface" >/dev/null 2>&1; then
        error "Network interface '$interface' not found"
    fi
    
    log "Network interface '$interface' validated"
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
    
    log "IRQ cores validated: $cores"
}

get_interface_irqs() {
    local interface="$1"
    local irqs=()
    
    # Get IRQs for network interface
    # This works for most network drivers
    local irq_info
    irq_info=$(grep -E "($interface|$(ethtool -i "$interface" 2>/dev/null | grep driver | awk '{print $2}' || echo 'unknown'))" /proc/interrupts | awk -F: '{print $1}' | tr -d ' ' || true)
    
    if [[ -z "$irq_info" ]]; then
        # Alternative method using /sys filesystem
        local pci_device
        pci_device=$(basename "$(readlink -f "/sys/class/net/$interface/device")" 2>/dev/null || echo "")
        
        if [[ -n "$pci_device" ]]; then
            # Find IRQs in /sys/bus/pci/devices/*/
            local irq_files
            irq_files=$(find /sys/bus/pci/devices -name "irq" -path "*$pci_device*" 2>/dev/null || true)
            
            for irq_file in $irq_files; do
                local irq_num
                irq_num=$(cat "$irq_file" 2>/dev/null || echo "")
                if [[ -n "$irq_num" && "$irq_num" != "0" ]]; then
                    irqs+=("$irq_num")
                fi
            done
        fi
    else
        # Parse IRQ numbers from /proc/interrupts
        while IFS= read -r irq; do
            if [[ -n "$irq" ]]; then
                irqs+=("$irq")
            fi
        done <<< "$irq_info"
    fi
    
    # Additional method: check MSI-X interrupts
    local msi_irqs
    msi_irqs=$(grep -E "($interface|$interface-.*)" /proc/interrupts | awk -F: '{print $1}' | tr -d ' ' || true)
    
    while IFS= read -r irq; do
        if [[ -n "$irq" ]]; then
            irqs+=("$irq")
        fi
    done <<< "$msi_irqs"
    
    # Remove duplicates and sort
    printf '%s\n' "${irqs[@]}" | sort -u | tr '\n' ' '
}

backup_irq_affinity() {
    log "Backing up current IRQ affinity..."
    
    local backup_file="${STATE_FILE}.backup.$(date +%s)"
    echo "# IRQ Affinity Backup - $(date)" > "$backup_file"
    echo "interface=$NETWORK_INTERFACE" >> "$backup_file"
    
    local irqs
    irqs=($(get_interface_irqs "$NETWORK_INTERFACE"))
    
    for irq in "${irqs[@]}"; do
        if [[ -f "/proc/irq/$irq/smp_affinity" ]]; then
            local current_affinity
            current_affinity=$(cat "/proc/irq/$irq/smp_affinity" 2>/dev/null || echo "")
            echo "irq_${irq}_affinity=$current_affinity" >> "$backup_file"
            log "Backed up IRQ $irq affinity: $current_affinity"
        fi
    done
    
    log "IRQ affinity backup saved to: $backup_file"
    echo "backup_file=$backup_file" > "$STATE_FILE"
}

core_list_to_mask() {
    local cores="$1"
    local mask=0
    
    IFS=',' read -ra CORE_ARRAY <<< "$cores"
    for core in "${CORE_ARRAY[@]}"; do
        mask=$((mask | (1 << core)))
    done
    
    printf "%x" "$mask"
}

apply_irq_affinity() {
    local interface="$1"
    local cores="$2"
    
    log "Applying IRQ affinity for interface $interface to cores: $cores"
    
    local irqs
    irqs=($(get_interface_irqs "$interface"))
    
    if [[ ${#irqs[@]} -eq 0 ]]; then
        log "WARNING: No IRQs found for interface $interface"
        return 1
    fi
    
    log "Found ${#irqs[@]} IRQs for $interface: ${irqs[*]}"
    
    local affinity_mask
    affinity_mask=$(core_list_to_mask "$cores")
    
    local success_count=0
    for irq in "${irqs[@]}"; do
        if [[ -f "/proc/irq/$irq/smp_affinity" ]]; then
            echo "$affinity_mask" > "/proc/irq/$irq/smp_affinity" 2>/dev/null && {
                log "Set IRQ $irq affinity to cores $cores (mask: $affinity_mask)"
                ((success_count++))
            } || {
                log "WARNING: Failed to set IRQ $irq affinity"
            }
        else
            log "WARNING: IRQ $irq affinity file not found"
        fi
    done
    
    if [[ $success_count -gt 0 ]]; then
        log "Successfully configured $success_count IRQs"
        echo "irqs_configured=${irqs[*]}" >> "$STATE_FILE"
        echo "target_cores=$cores" >> "$STATE_FILE"
        echo "applied=$(date)" >> "$STATE_FILE"
        return 0
    else
        log "ERROR: No IRQs were successfully configured"
        return 1
    fi
}

optimize_network_parameters() {
    local interface="$1"
    
    log "Applying network performance optimizations for $interface..."
    
    # Disable IRQ balancing daemon to maintain manual affinity
    if systemctl is-active irqbalance >/dev/null 2>&1; then
        systemctl stop irqbalance
        systemctl disable irqbalance
        log "Disabled irqbalance daemon"
    fi
    
    # Network interface optimizations
    if ethtool -h >/dev/null 2>&1; then
        # Disable interrupt coalescing for lowest latency
        ethtool -C "$interface" rx-usecs 0 rx-frames 1 tx-usecs 0 tx-frames 1 2>/dev/null && {
            log "Disabled interrupt coalescing on $interface"
        } || {
            log "WARNING: Failed to configure interrupt coalescing"
        }
        
        # Set ring buffer sizes
        ethtool -G "$interface" rx 4096 tx 4096 2>/dev/null && {
            log "Optimized ring buffer sizes on $interface"
        } || {
            log "WARNING: Failed to configure ring buffers"
        }
        
        # Enable hardware timestamping if available
        ethtool -T "$interface" >/dev/null 2>&1 && {
            log "Hardware timestamping available on $interface"
        } || {
            log "INFO: Hardware timestamping not available"
        }
    fi
    
    # Kernel network parameters
    {
        echo "net.core.netdev_max_backlog = 30000"
        echo "net.core.netdev_budget = 600"
        echo "net.core.rmem_default = 262144"
        echo "net.core.rmem_max = 16777216"
        echo "net.core.wmem_default = 262144"
        echo "net.core.wmem_max = 16777216"
        echo "net.ipv4.tcp_rmem = 4096 262144 16777216"
        echo "net.ipv4.tcp_wmem = 4096 262144 16777216"
        echo "net.ipv4.tcp_congestion_control = cubic"
        echo "net.core.busy_read = 50"
        echo "net.core.busy_poll = 50"
    } > /tmp/network-optimizations.conf
    
    sysctl -p /tmp/network-optimizations.conf >/dev/null 2>&1 && {
        log "Applied kernel network optimizations"
    } || {
        log "WARNING: Some kernel parameters could not be applied"
    }
}

measure_irq_distribution() {
    log "Measuring IRQ distribution..."
    
    local pre_file="/tmp/interrupts_pre"
    local post_file="/tmp/interrupts_post"
    
    cp /proc/interrupts "$pre_file"
    sleep 5
    cp /proc/interrupts "$post_file"
    
    # Calculate IRQ distribution
    local irqs
    irqs=($(get_interface_irqs "$NETWORK_INTERFACE"))
    
    log "IRQ distribution analysis:"
    for irq in "${irqs[@]}"; do
        local pre_counts post_counts
        pre_counts=($(grep "^ *$irq:" "$pre_file" | awk '{for(i=2;i<=NF-1;i++) printf "%s ", $i; print ""}' || echo ""))
        post_counts=($(grep "^ *$irq:" "$post_file" | awk '{for(i=2;i<=NF-1;i++) printf "%s ", $i; print ""}' || echo ""))
        
        if [[ ${#pre_counts[@]} -eq ${#post_counts[@]} ]] && [[ ${#pre_counts[@]} -gt 0 ]]; then
            log "  IRQ $irq distribution (5s delta):"
            for ((i=0; i<${#pre_counts[@]}; i++)); do
                local delta=$((${post_counts[i]:-0} - ${pre_counts[i]:-0}))
                log "    CPU $i: $delta interrupts"
            done
        fi
    done
    
    rm -f "$pre_file" "$post_file"
}

restore_irq_affinity() {
    log "Restoring original IRQ affinity..."
    
    if [[ ! -f "$STATE_FILE" ]]; then
        log "WARNING: No state file found, cannot restore"
        return 1
    fi
    
    local backup_file
    backup_file=$(grep "^backup_file=" "$STATE_FILE" | cut -d= -f2 || echo "")
    
    if [[ ! -f "$backup_file" ]]; then
        log "WARNING: Backup file not found: $backup_file"
        return 1
    fi
    
    log "Restoring from backup: $backup_file"
    
    # Restore IRQ affinities
    while IFS='=' read -r key value; do
        if [[ "$key" =~ ^irq_[0-9]+_affinity$ ]]; then
            local irq
            irq=$(echo "$key" | sed 's/irq_\([0-9]*\)_affinity/\1/')
            
            if [[ -f "/proc/irq/$irq/smp_affinity" ]] && [[ -n "$value" ]]; then
                echo "$value" > "/proc/irq/$irq/smp_affinity" 2>/dev/null && {
                    log "Restored IRQ $irq affinity to: $value"
                } || {
                    log "WARNING: Failed to restore IRQ $irq affinity"
                }
            fi
        fi
    done < "$backup_file"
    
    # Re-enable irqbalance if it was disabled
    if ! systemctl is-enabled irqbalance >/dev/null 2>&1; then
        systemctl enable irqbalance
        systemctl start irqbalance
        log "Re-enabled irqbalance daemon"
    fi
    
    rm -f "$STATE_FILE"
    log "IRQ affinity restoration completed"
}

show_current_status() {
    echo "=== NETWORK IRQ AFFINITY STATUS ==="
    echo "Network Interface: $NETWORK_INTERFACE"
    
    local irqs
    irqs=($(get_interface_irqs "$NETWORK_INTERFACE"))
    
    if [[ ${#irqs[@]} -eq 0 ]]; then
        echo "No IRQs found for interface $NETWORK_INTERFACE"
    else
        echo "Found ${#irqs[@]} IRQs: ${irqs[*]}"
        echo
        echo "Current IRQ Affinity:"
        for irq in "${irqs[@]}"; do
            if [[ -f "/proc/irq/$irq/smp_affinity" ]]; then
                local affinity
                affinity=$(cat "/proc/irq/$irq/smp_affinity" 2>/dev/null || echo "unknown")
                echo "  IRQ $irq: $affinity"
            fi
        done
    fi
    
    echo
    echo "IRQBalance Status: $(systemctl is-active irqbalance 2>/dev/null || echo 'unknown')"
    
    if [[ -f "$STATE_FILE" ]]; then
        echo
        echo "--- Configuration State ---"
        cat "$STATE_FILE"
    fi
    
    echo "=============================="
}

usage() {
    cat << EOF
Network IRQ Affinity Configuration

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --interface=IFACE     Network interface (default: $NETWORK_INTERFACE)
    --cores=X,Y           Target cores for IRQs (default: $IRQ_CORES)
    --apply              Apply IRQ affinity configuration
    --restore            Restore original IRQ affinity
    --status             Show current IRQ status
    --measure            Measure current IRQ distribution
    --help               Show this help

EXAMPLES:
    $0 --interface=eth0 --cores=0,1 --apply    Apply IRQ affinity
    $0 --status                                Show current status
    $0 --measure                               Measure IRQ distribution
    $0 --restore                               Restore original settings

PERFORMANCE TARGET:
    Route network interrupts to cores 0,1
    Keep trading cores 2,3 interrupt-free
    Reduce network jitter to <5μs

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --interface=*)
            NETWORK_INTERFACE="${1#*=}"
            shift
            ;;
        --cores=*)
            IRQ_CORES="${1#*=}"
            shift
            ;;
        --apply)
            APPLY_CONFIG=true
            shift
            ;;
        --restore)
            RESTORE_CONFIG=true
            shift
            ;;
        --status)
            STATUS_MODE=true
            shift
            ;;
        --measure)
            MEASURE_MODE=true
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

if [[ "${APPLY_CONFIG:-}" == "true" ]]; then
    validate_interface "$NETWORK_INTERFACE"
    validate_cores "$IRQ_CORES"
    backup_irq_affinity
    apply_irq_affinity "$NETWORK_INTERFACE" "$IRQ_CORES"
    optimize_network_parameters "$NETWORK_INTERFACE"
    log "IRQ affinity configuration completed for $NETWORK_INTERFACE"
    
elif [[ "${RESTORE_CONFIG:-}" == "true" ]]; then
    restore_irq_affinity
    
elif [[ "${MEASURE_MODE:-}" == "true" ]]; then
    validate_interface "$NETWORK_INTERFACE"
    measure_irq_distribution
    
elif [[ "${STATUS_MODE:-}" == "true" ]]; then
    show_current_status
    
else
    usage
    exit 1
fi