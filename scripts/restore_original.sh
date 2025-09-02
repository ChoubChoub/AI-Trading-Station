#!/bin/bash

# ============================================================================
# SYSTEM RESTORATION UTILITY
# ============================================================================
# PURPOSE: Comprehensive restoration of all trading system optimizations
# SCOPE: Complete system reset to original factory/distribution defaults
# SAFETY: Multi-layer backup and verification system
# ============================================================================

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/system-restore.log"
readonly MASTER_BACKUP_DIR="/tmp/system-restore-backups"
readonly RESTORATION_STATE="/tmp/restoration-state"

# Restoration options
FULL_RESTORE=false
SELECTIVE_RESTORE=false
DRY_RUN=false
FORCE_RESTORE=false
COMPONENTS=()

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

warning() {
    log "WARNING: $*" >&2
}

check_root() {
    [[ $EUID -eq 0 ]] || error "Must run as root for system restoration"
}

create_master_backup() {
    log "Creating master backup before restoration..."
    
    local master_backup="${MASTER_BACKUP_DIR}/pre-restore-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$master_backup"
    
    # System configuration backup
    mkdir -p "$master_backup/etc"
    cp -r /etc/default "$master_backup/etc/" 2>/dev/null || true
    cp -r /etc/systemd "$master_backup/etc/" 2>/dev/null || true
    cp -r /etc/sysctl.d "$master_backup/etc/" 2>/dev/null || true
    cp -r /etc/onload "$master_backup/etc/" 2>/dev/null || true
    
    # System state backup
    mkdir -p "$master_backup/proc"
    cp /proc/cmdline "$master_backup/proc/" 2>/dev/null || true
    cat /proc/cpuinfo > "$master_backup/proc/cpuinfo"
    cat /proc/interrupts > "$master_backup/proc/interrupts"
    cat /proc/meminfo > "$master_backup/proc/meminfo"
    
    # Service states
    mkdir -p "$master_backup/services"
    systemctl list-units --type=service > "$master_backup/services/all-services.txt"
    systemctl is-enabled irqbalance > "$master_backup/services/irqbalance-state.txt" 2>/dev/null || echo "unknown" > "$master_backup/services/irqbalance-state.txt"
    
    # Network configuration
    mkdir -p "$master_backup/network"
    ip addr > "$master_backup/network/interfaces.txt"
    cat /proc/net/dev > "$master_backup/network/stats.txt"
    
    # CPU and performance state
    mkdir -p "$master_backup/cpu"
    for cpu_dir in /sys/devices/system/cpu/cpu*/cpufreq; do
        if [[ -d "$cpu_dir" ]]; then
            local cpu
            cpu=$(basename "$(dirname "$cpu_dir")")
            cat "$cpu_dir/scaling_governor" > "$master_backup/cpu/${cpu}_governor.txt" 2>/dev/null || true
            cat "$cpu_dir/scaling_cur_freq" > "$master_backup/cpu/${cpu}_freq.txt" 2>/dev/null || true
        fi
    done
    
    log "Master backup created: $master_backup"
    echo "master_backup=$master_backup" > "$RESTORATION_STATE"
}

detect_installed_components() {
    log "Detecting installed trading system components..."
    
    local components=()
    
    # Check for core isolation
    if [[ -f /sys/devices/system/cpu/isolated ]] && [[ -n "$(cat /sys/devices/system/cpu/isolated)" ]]; then
        components+=("core_isolation")
        log "Detected: CPU core isolation"
    fi
    
    # Check for OnLoad
    if command -v onload >/dev/null 2>&1; then
        components+=("onload")
        log "Detected: OnLoad acceleration"
    fi
    
    # Check for IRQ affinity modifications
    if [[ -f /tmp/nic-irq-state ]]; then
        components+=("irq_affinity")
        log "Detected: IRQ affinity configuration"
    fi
    
    # Check for performance optimizations
    if grep -q "performance" /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null; then
        components+=("cpu_performance")
        log "Detected: CPU performance optimizations"
    fi
    
    # Check for network optimizations
    if [[ -f /etc/sysctl.d/99-onload-optimization.conf ]]; then
        components+=("network_optimization")
        log "Detected: Network stack optimizations"
    fi
    
    # Check for GRUB modifications
    if grep -E "(isolcpus|nohz_full|rcu_nocbs)" /etc/default/grub 2>/dev/null; then
        components+=("grub_modifications")
        log "Detected: GRUB kernel parameter modifications"
    fi
    
    echo "detected_components=${components[*]}" >> "$RESTORATION_STATE"
    log "Detected components: ${components[*]}"
}

restore_grub_configuration() {
    log "Restoring GRUB to distribution defaults..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would restore GRUB configuration"
        return 0
    fi
    
    local grub_file="/etc/default/grub"
    local grub_restored=false
    
    # Try to find original distribution GRUB config
    local original_grub_candidates=(
        "/etc/default/grub.dpkg-dist"
        "/etc/default/grub.orig"
        "/etc/default/grub.backup"
        "/usr/share/grub/default/grub"
    )
    
    for candidate in "${original_grub_candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            log "Restoring GRUB from: $candidate"
            cp "$grub_file" "${grub_file}.pre-restore.$(date +%s)"
            cp "$candidate" "$grub_file"
            grub_restored=true
            break
        fi
    done
    
    # If no original found, create minimal default config
    if [[ "$grub_restored" == "false" ]]; then
        log "Creating minimal default GRUB configuration"
        cp "$grub_file" "${grub_file}.pre-restore.$(date +%s)"
        
        cat > "$grub_file" << 'EOF'
# GRUB Configuration - Restored to defaults
GRUB_DEFAULT=0
GRUB_TIMEOUT=5
GRUB_DISTRIBUTOR=`lsb_release -i -s 2> /dev/null || echo Debian`
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
GRUB_CMDLINE_LINUX=""
EOF
        grub_restored=true
    fi
    
    if [[ "$grub_restored" == "true" ]]; then
        # Update GRUB
        if update-grub; then
            log "GRUB configuration restored successfully"
            echo "grub_restored=yes" >> "$RESTORATION_STATE"
            echo "reboot_required=yes" >> "$RESTORATION_STATE"
        else
            error "Failed to update GRUB after restoration"
        fi
    fi
}

restore_cpu_configuration() {
    log "Restoring CPU configuration to defaults..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would restore CPU governors and frequency scaling"
        return 0
    fi
    
    # Restore CPU governors to distribution defaults
    local default_governor="ondemand"
    
    # Check what governors are available and select appropriate default
    local available_governors
    if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors ]]; then
        available_governors=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors)
        
        if echo "$available_governors" | grep -q "ondemand"; then
            default_governor="ondemand"
        elif echo "$available_governors" | grep -q "schedutil"; then
            default_governor="schedutil"
        elif echo "$available_governors" | grep -q "powersave"; then
            default_governor="powersave"
        fi
    fi
    
    log "Setting default CPU governor: $default_governor"
    
    for cpu_dir in /sys/devices/system/cpu/cpu*/cpufreq; do
        if [[ -d "$cpu_dir" ]] && [[ -f "$cpu_dir/scaling_governor" ]]; then
            local cpu
            cpu=$(basename "$(dirname "$cpu_dir")")
            
            echo "$default_governor" > "$cpu_dir/scaling_governor" 2>/dev/null && {
                log "Restored $cpu governor to $default_governor"
            } || {
                warning "Failed to restore $cpu governor"
            }
            
            # Reset frequency scaling limits
            if [[ -f "$cpu_dir/scaling_min_freq" ]] && [[ -f "$cpu_dir/cpuinfo_min_freq" ]]; then
                local min_freq
                min_freq=$(cat "$cpu_dir/cpuinfo_min_freq")
                echo "$min_freq" > "$cpu_dir/scaling_min_freq" 2>/dev/null || true
            fi
            
            if [[ -f "$cpu_dir/scaling_max_freq" ]] && [[ -f "$cpu_dir/cpuinfo_max_freq" ]]; then
                local max_freq
                max_freq=$(cat "$cpu_dir/cpuinfo_max_freq")
                echo "$max_freq" > "$cpu_dir/scaling_max_freq" 2>/dev/null || true
            fi
        fi
    done
    
    echo "cpu_restored=yes" >> "$RESTORATION_STATE"
}

restore_irq_configuration() {
    log "Restoring IRQ configuration to defaults..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would restore IRQ affinity and re-enable irqbalance"
        return 0
    fi
    
    # Re-enable and start irqbalance
    systemctl enable irqbalance 2>/dev/null || true
    systemctl start irqbalance 2>/dev/null || true
    
    # Reset all IRQ affinities to all CPUs (let irqbalance handle them)
    local num_cpus
    num_cpus=$(nproc)
    local cpu_mask
    cpu_mask=$(printf "%x" $(((1 << num_cpus) - 1)))
    
    for irq_dir in /proc/irq/*/; do
        local irq
        irq=$(basename "$irq_dir")
        
        if [[ "$irq" =~ ^[0-9]+$ ]] && [[ -f "${irq_dir}smp_affinity" ]]; then
            echo "$cpu_mask" > "${irq_dir}smp_affinity" 2>/dev/null && {
                log "Reset IRQ $irq affinity to all CPUs"
            } || {
                # Some IRQs cannot be changed, this is normal
                :
            }
        fi
    done
    
    echo "irq_restored=yes" >> "$RESTORATION_STATE"
    log "IRQ configuration restored, irqbalance re-enabled"
}

restore_network_configuration() {
    log "Restoring network configuration to defaults..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would restore network parameters and remove OnLoad config"
        return 0
    fi
    
    # Remove OnLoad configuration files
    local onload_configs=(
        "/etc/onload"
        "/etc/sysctl.d/99-onload-optimization.conf"
        "/usr/local/bin/onload-trading"
        "/usr/local/bin/onload-monitor"
    )
    
    for config in "${onload_configs[@]}"; do
        if [[ -e "$config" ]]; then
            rm -rf "$config"
            log "Removed: $config"
        fi
    done
    
    # Reset network sysctl parameters to distribution defaults
    local default_net_params=(
        "net.core.rmem_default=212992"
        "net.core.rmem_max=212992"
        "net.core.wmem_default=212992"
        "net.core.wmem_max=212992"
        "net.core.netdev_max_backlog=1000"
        "net.core.netdev_budget=300"
        "net.core.netdev_budget_usecs=2000"
        "net.ipv4.tcp_rmem=4096 87380 6291456"
        "net.ipv4.tcp_wmem=4096 16384 4194304"
        "net.ipv4.tcp_congestion_control=cubic"
        "net.ipv4.tcp_slow_start_after_idle=1"
        "net.ipv4.tcp_no_metrics_save=0"
        "net.ipv4.tcp_timestamps=1"
        "net.core.busy_read=0"
        "net.core.busy_poll=0"
        "vm.nr_hugepages=0"
    )
    
    for param in "${default_net_params[@]}"; do
        echo "$param" | sysctl -p - >/dev/null 2>&1 && {
            log "Restored network parameter: $param"
        } || {
            warning "Failed to restore parameter: $param"
        }
    done
    
    # Reset network interface settings
    local interfaces
    interfaces=($(ip link show | grep -E "^[0-9]+:" | awk -F: '{print $2}' | tr -d ' ' | grep -v lo))
    
    for interface in "${interfaces[@]}"; do
        if ethtool -h >/dev/null 2>&1; then
            # Reset interrupt coalescing to auto
            ethtool -C "$interface" adaptive-rx on adaptive-tx on 2>/dev/null || true
            
            # Reset ring buffer sizes to defaults
            ethtool -G "$interface" rx 256 tx 256 2>/dev/null || true
            
            log "Reset network interface settings: $interface"
        fi
    done
    
    echo "network_restored=yes" >> "$RESTORATION_STATE"
}

restore_kernel_parameters() {
    log "Restoring kernel runtime parameters to defaults..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would restore kernel runtime parameters"
        return 0
    fi
    
    # Reset kernel parameters to distribution defaults
    local default_kernel_params=(
        "kernel.sched_rt_runtime_us=950000"
        "kernel.timer_migration=1"
        "kernel.numa_balancing=1"
        "vm.min_free_kbytes=67584"
        "vm.swappiness=60"
    )
    
    for param in "${default_kernel_params[@]}"; do
        echo "$param" | sysctl -p - >/dev/null 2>&1 && {
            log "Restored kernel parameter: $param"
        } || {
            warning "Failed to restore parameter: $param"
        }
    done
    
    echo "kernel_params_restored=yes" >> "$RESTORATION_STATE"
}

stop_trading_services() {
    log "Stopping trading-related services and processes..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would stop trading services and monitoring"
        return 0
    fi
    
    # Stop monitoring processes
    local monitoring_processes=(
        "onload-monitor"
        "core-isolation-monitor"
    )
    
    for process in "${monitoring_processes[@]}"; do
        if pgrep -f "$process" >/dev/null 2>&1; then
            pkill -f "$process" && {
                log "Stopped process: $process"
            } || {
                warning "Failed to stop process: $process"
            }
        fi
    done
    
    # Remove PID files
    local pid_files=(
        "/var/run/core-isolation-monitor.pid"
        "/tmp/onload-monitor.pid"
    )
    
    for pid_file in "${pid_files[@]}"; do
        if [[ -f "$pid_file" ]]; then
            rm -f "$pid_file"
            log "Removed PID file: $pid_file"
        fi
    done
    
    echo "services_stopped=yes" >> "$RESTORATION_STATE"
}

cleanup_state_and_logs() {
    log "Cleaning up state files and temporary data..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would clean up state files and logs"
        return 0
    fi
    
    # Remove state files
    local state_files=(
        "/tmp/core-isolation-state"
        "/tmp/nic-irq-state"
        "/tmp/onload-state"
        "/tmp/isolation-performance-tests"
    )
    
    for state_file in "${state_files[@]}"; do
        if [[ -e "$state_file" ]]; then
            rm -rf "$state_file"
            log "Removed: $state_file"
        fi
    done
    
    # Clean up temporary test data
    find /tmp -name "*trading*" -type f -mtime +0 -delete 2>/dev/null || true
    find /tmp -name "*onload*" -type f -mtime +0 -delete 2>/dev/null || true
    find /tmp -name "*isolation*" -type f -mtime +0 -delete 2>/dev/null || true
    
    echo "cleanup_completed=yes" >> "$RESTORATION_STATE"
}

verify_restoration() {
    log "Verifying system restoration..."
    
    local issues=0
    local checks_passed=0
    
    # Check GRUB configuration
    if ! grep -E "(isolcpus|nohz_full|rcu_nocbs)" /etc/default/grub >/dev/null 2>&1; then
        log "✓ GRUB: No isolation parameters found"
        ((checks_passed++))
    else
        warning "✗ GRUB: Still contains isolation parameters"
        ((issues++))
    fi
    
    # Check CPU governors
    local performance_cpus
    performance_cpus=$(grep -l "performance" /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | wc -l || echo "0")
    if [[ $performance_cpus -eq 0 ]]; then
        log "✓ CPU: No CPUs in performance mode"
        ((checks_passed++))
    else
        warning "✗ CPU: $performance_cpus CPUs still in performance mode"
        ((issues++))
    fi
    
    # Check irqbalance
    if systemctl is-active irqbalance >/dev/null 2>&1; then
        log "✓ IRQ: irqbalance is running"
        ((checks_passed++))
    else
        warning "✗ IRQ: irqbalance is not active"
        ((issues++))
    fi
    
    # Check OnLoad configuration
    if [[ ! -d /etc/onload ]]; then
        log "✓ OnLoad: Configuration removed"
        ((checks_passed++))
    else
        warning "✗ OnLoad: Configuration still present"
        ((issues++))
    fi
    
    # Check for trading processes
    local trading_procs
    trading_procs=$(pgrep -f "trading|onload" 2>/dev/null | wc -l || echo "0")
    if [[ $trading_procs -eq 0 ]]; then
        log "✓ Processes: No trading processes running"
        ((checks_passed++))
    else
        warning "✗ Processes: $trading_procs trading processes still running"
        ((issues++))
    fi
    
    log "Verification completed: $checks_passed checks passed, $issues issues found"
    
    echo "verification_passed=$checks_passed" >> "$RESTORATION_STATE"
    echo "verification_issues=$issues" >> "$RESTORATION_STATE"
    
    return $issues
}

generate_restoration_report() {
    log "Generating restoration report..."
    
    local report_file="/tmp/restoration-report-$(date +%s).txt"
    
    cat > "$report_file" << EOF
================================================================================
SYSTEM RESTORATION REPORT
================================================================================
Timestamp: $(date)
Hostname: $(hostname)
Kernel: $(uname -r)
Restoration Type: $(if [[ "$DRY_RUN" == "true" ]]; then echo "DRY RUN"; else echo "COMPLETE"; fi)

COMPONENTS RESTORED:
$(if grep -q "grub_restored=yes" "$RESTORATION_STATE" 2>/dev/null; then echo "✓ GRUB kernel parameters"; else echo "- GRUB kernel parameters"; fi)
$(if grep -q "cpu_restored=yes" "$RESTORATION_STATE" 2>/dev/null; then echo "✓ CPU governors and frequency scaling"; else echo "- CPU governors and frequency scaling"; fi)
$(if grep -q "irq_restored=yes" "$RESTORATION_STATE" 2>/dev/null; then echo "✓ IRQ affinity and irqbalance"; else echo "- IRQ affinity and irqbalance"; fi)
$(if grep -q "network_restored=yes" "$RESTORATION_STATE" 2>/dev/null; then echo "✓ Network stack configuration"; else echo "- Network stack configuration"; fi)
$(if grep -q "kernel_params_restored=yes" "$RESTORATION_STATE" 2>/dev/null; then echo "✓ Kernel runtime parameters"; else echo "- Kernel runtime parameters"; fi)
$(if grep -q "services_stopped=yes" "$RESTORATION_STATE" 2>/dev/null; then echo "✓ Trading services and monitoring"; else echo "- Trading services and monitoring"; fi)
$(if grep -q "cleanup_completed=yes" "$RESTORATION_STATE" 2>/dev/null; then echo "✓ State files and temporary data"; else echo "- State files and temporary data"; fi)

SYSTEM STATUS AFTER RESTORATION:
CPU Isolation: $(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo "none")
CPU Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
IRQBalance: $(systemctl is-active irqbalance 2>/dev/null || echo "unknown")
OnLoad Config: $(if [[ -d /etc/onload ]]; then echo "present"; else echo "removed"; fi)

VERIFICATION RESULTS:
$(if [[ -f "$RESTORATION_STATE" ]]; then
    local passed=$(grep "verification_passed=" "$RESTORATION_STATE" | cut -d= -f2 || echo "0")
    local issues=$(grep "verification_issues=" "$RESTORATION_STATE" | cut -d= -f2 || echo "0")
    echo "Checks Passed: $passed"
    echo "Issues Found: $issues"
    if [[ $issues -eq 0 ]]; then echo "Overall Status: SUCCESS"; else echo "Overall Status: NEEDS ATTENTION"; fi
else
    echo "Verification: Not completed"
fi)

$(if grep -q "reboot_required=yes" "$RESTORATION_STATE" 2>/dev/null; then
echo "REBOOT REQUIRED: Yes - kernel parameter changes need reboot to take effect"
else
echo "REBOOT REQUIRED: No"
fi)

BACKUP LOCATION:
$(grep "master_backup=" "$RESTORATION_STATE" 2>/dev/null | cut -d= -f2 || echo "Not available")

RECOMMENDATIONS:
1. $(if grep -q "reboot_required=yes" "$RESTORATION_STATE" 2>/dev/null; then echo "Reboot the system to complete kernel parameter restoration"; else echo "No reboot required"; fi)
2. Verify that all applications work correctly after restoration
3. Monitor system performance to ensure stability
4. Remove any remaining custom trading applications if no longer needed

================================================================================
EOF
    
    log "Restoration report generated: $report_file"
    cat "$report_file"
}

usage() {
    cat << EOF
System Restoration Utility

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --full               Restore all components (default)
    --selective          Restore only specified components
    --components=LIST    Comma-separated list of components to restore
    --dry-run           Show what would be done without making changes
    --force             Force restoration despite warnings
    --help              Show this help

COMPONENTS:
    grub                GRUB kernel parameters
    cpu                 CPU governors and frequency scaling
    irq                 IRQ affinity and irqbalance
    network             Network stack and OnLoad configuration
    kernel              Kernel runtime parameters
    services            Trading services and monitoring
    cleanup             State files and temporary data

RESTORATION ACTIONS:
    - Restore GRUB to distribution defaults
    - Reset CPU governors and frequency scaling
    - Re-enable irqbalance and reset IRQ affinity
    - Remove OnLoad configuration and reset network parameters
    - Restore kernel parameters to defaults
    - Stop trading services and monitoring
    - Clean up state files and temporary data

EXAMPLES:
    $0                                    # Full system restoration
    $0 --dry-run                         # Show restoration plan
    $0 --selective --components=grub,cpu # Restore only GRUB and CPU
    $0 --force                           # Force restoration

WARNING:
    This will completely remove all trading system optimizations
    and restore the system to distribution defaults.
    
    A system reboot may be required to complete the restoration.

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_RESTORE=true
            shift
            ;;
        --selective)
            SELECTIVE_RESTORE=true
            shift
            ;;
        --components=*)
            IFS=',' read -ra COMPONENTS <<< "${1#*=}"
            SELECTIVE_RESTORE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE_RESTORE=true
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

# Set default to full restore if no options specified
if [[ "$FULL_RESTORE" == "false" ]] && [[ "$SELECTIVE_RESTORE" == "false" ]]; then
    FULL_RESTORE=true
fi

# Main execution
check_root

if [[ "$DRY_RUN" == "true" ]]; then
    log "=== RESTORATION DRY RUN MODE ==="
    log "No changes will be made to the system"
    log "==============================="
fi

log "Starting system restoration to factory defaults..."

create_master_backup
detect_installed_components

# Safety check for running processes
if [[ "$FORCE_RESTORE" != "true" ]]; then
    local running_processes
    running_processes=$(ps aux | grep -E "(trading|onload)" | grep -v grep | wc -l || echo "0")
    
    if [[ $running_processes -gt 0 ]]; then
        warning "$running_processes trading processes are still running"
        error "Stop all trading processes before restoration or use --force"
    fi
fi

# Execute restoration based on mode
if [[ "$FULL_RESTORE" == "true" ]]; then
    log "Performing full system restoration..."
    stop_trading_services
    restore_grub_configuration
    restore_cpu_configuration
    restore_irq_configuration
    restore_network_configuration
    restore_kernel_parameters
    cleanup_state_and_logs
    
elif [[ "$SELECTIVE_RESTORE" == "true" ]]; then
    log "Performing selective restoration: ${COMPONENTS[*]}"
    
    for component in "${COMPONENTS[@]}"; do
        case "$component" in
            "grub")
                restore_grub_configuration
                ;;
            "cpu")
                restore_cpu_configuration
                ;;
            "irq")
                restore_irq_configuration
                ;;
            "network")
                restore_network_configuration
                ;;
            "kernel")
                restore_kernel_parameters
                ;;
            "services")
                stop_trading_services
                ;;
            "cleanup")
                cleanup_state_and_logs
                ;;
            *)
                warning "Unknown component: $component"
                ;;
        esac
    done
fi

# Verification and reporting
if [[ "$DRY_RUN" != "true" ]]; then
    verify_restoration
fi

generate_restoration_report

if [[ "$DRY_RUN" == "true" ]]; then
    log "Dry run completed - no changes made to system"
else
    log "System restoration completed"
    
    if grep -q "reboot_required=yes" "$RESTORATION_STATE" 2>/dev/null; then
        log "IMPORTANT: System reboot required to complete restoration"
        echo
        echo "To complete the restoration, please run: sudo reboot"
        echo
    fi
fi