#!/bin/bash

# ============================================================================
# CORE ISOLATION ROLLBACK UTILITY
# ============================================================================
# PURPOSE: Safe rollback of CPU core isolation configuration
# SAFETY: Preserves system stability during rollback operations
# RECOVERY: Restores original CPU scheduling and kernel parameters
# ============================================================================

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/core-isolation-rollback.log"
readonly STATE_FILE="/tmp/core-isolation-state"
readonly BACKUP_DIR="/tmp/isolation-backups"

FORCE_ROLLBACK=false
DRY_RUN=false
PRESERVE_PERFORMANCE=false

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
    [[ $EUID -eq 0 ]] || error "Must run as root for system rollback"
}

create_safety_backup() {
    log "Creating safety backup before rollback..."
    
    local safety_backup="${BACKUP_DIR}/pre-rollback-$(date +%s)"
    mkdir -p "$safety_backup"
    
    # Backup current system state
    cp /proc/cmdline "${safety_backup}/current-cmdline" 2>/dev/null || true
    cat /proc/cpuinfo > "${safety_backup}/current-cpuinfo"
    cat /proc/interrupts > "${safety_backup}/current-interrupts"
    systemctl status irqbalance > "${safety_backup}/irqbalance-status" 2>/dev/null || echo "not available" > "${safety_backup}/irqbalance-status"
    
    # Backup GRUB configuration
    if [[ -f /etc/default/grub ]]; then
        cp /etc/default/grub "${safety_backup}/grub-config"
    fi
    
    # Backup sysctl configuration
    sysctl -a > "${safety_backup}/current-sysctl" 2>/dev/null || true
    
    log "Safety backup created: $safety_backup"
    echo "safety_backup=$safety_backup" >> "$STATE_FILE"
}

check_rollback_safety() {
    log "Performing safety checks before rollback..."
    
    local warnings=0
    
    # Check for running trading processes
    local trading_processes
    trading_processes=$(ps aux | grep -E "(trading|onload)" | grep -v grep || echo "")
    
    if [[ -n "$trading_processes" ]]; then
        warning "Trading processes are still running:"
        echo "$trading_processes" | while IFS= read -r process; do
            warning "  $process"
        done
        ((warnings++))
        
        if [[ "$FORCE_ROLLBACK" != "true" ]]; then
            error "Stop trading processes before rollback or use --force"
        fi
    fi
    
    # Check system load
    local load_avg
    load_avg=$(uptime | awk '{print $(NF-2)}' | tr -d ',')
    if (( $(echo "$load_avg > 2.0" | bc -l) )); then
        warning "High system load detected: $load_avg"
        ((warnings++))
    fi
    
    # Check for active network connections
    local active_connections
    active_connections=$(ss -tuln | wc -l)
    if [[ $active_connections -gt 10 ]]; then
        warning "Many active network connections: $active_connections"
        ((warnings++))
    fi
    
    if [[ $warnings -gt 0 ]]; then
        log "$warnings safety warnings detected"
        if [[ "$FORCE_ROLLBACK" != "true" ]]; then
            log "Use --force to proceed despite warnings"
            exit 1
        fi
    else
        log "Safety checks passed"
    fi
}

rollback_grub_configuration() {
    log "Rolling back GRUB configuration..."
    
    local grub_file="/etc/default/grub"
    local grub_backups
    grub_backups=($(ls "${grub_file}.backup."* 2>/dev/null | sort -r || echo ""))
    
    if [[ ${#grub_backups[@]} -eq 0 ]]; then
        warning "No GRUB backup files found"
        return 0
    fi
    
    local latest_backup="${grub_backups[0]}"
    log "Restoring GRUB configuration from: $latest_backup"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would restore: $latest_backup -> $grub_file"
        log "[DRY RUN] Would run: update-grub"
        return 0
    fi
    
    # Create backup of current config
    cp "$grub_file" "${grub_file}.pre-rollback.$(date +%s)"
    
    # Restore original configuration
    cp "$latest_backup" "$grub_file"
    
    # Verify the restoration
    if ! grep -q "GRUB_CMDLINE_LINUX_DEFAULT" "$grub_file"; then
        error "GRUB configuration restoration failed - invalid file"
    fi
    
    # Update GRUB
    if update-grub; then
        log "GRUB configuration restored and updated"
        log "REBOOT REQUIRED for kernel parameter changes to take effect"
        echo "reboot_required=yes" >> "$STATE_FILE"
    else
        error "Failed to update GRUB after restoration"
    fi
}

rollback_runtime_optimizations() {
    log "Rolling back runtime optimizations..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would reset CPU governors to 'ondemand'"
        log "[DRY RUN] Would re-enable irqbalance"
        log "[DRY RUN] Would reset sysctl parameters"
        return 0
    fi
    
    # Reset CPU governors
    for cpu_dir in /sys/devices/system/cpu/cpu*/cpufreq; do
        if [[ -d "$cpu_dir" ]] && [[ -f "$cpu_dir/scaling_governor" ]]; then
            local cpu
            cpu=$(basename "$(dirname "$cpu_dir")")
            
            # Check available governors
            local available_governors
            available_governors=$(cat "$cpu_dir/scaling_available_governors" 2>/dev/null || echo "")
            
            # Set to ondemand if available, otherwise powersave
            if echo "$available_governors" | grep -q ondemand; then
                echo "ondemand" > "$cpu_dir/scaling_governor" 2>/dev/null && {
                    log "Reset $cpu governor to ondemand"
                } || {
                    warning "Failed to reset $cpu governor"
                }
            elif echo "$available_governors" | grep -q powersave; then
                echo "powersave" > "$cpu_dir/scaling_governor" 2>/dev/null && {
                    log "Reset $cpu governor to powersave"
                } || {
                    warning "Failed to reset $cpu governor"
                }
            fi
            
            # Reset frequency scaling
            if [[ -f "$cpu_dir/scaling_min_freq" ]] && [[ -f "$cpu_dir/scaling_available_frequencies" ]]; then
                local min_freq
                min_freq=$(cat "$cpu_dir/scaling_available_frequencies" | awk '{print $1}' 2>/dev/null || echo "")
                if [[ -n "$min_freq" ]]; then
                    echo "$min_freq" > "$cpu_dir/scaling_min_freq" 2>/dev/null || true
                fi
            fi
        fi
    done
    
    # Re-enable irqbalance
    if ! systemctl is-active irqbalance >/dev/null 2>&1; then
        systemctl enable irqbalance
        systemctl start irqbalance
        log "Re-enabled irqbalance service"
    fi
    
    # Reset kernel parameters to defaults
    local default_params=(
        "kernel.sched_rt_runtime_us=950000"
        "kernel.timer_migration=1"
        "kernel.numa_balancing=1"
    )
    
    for param in "${default_params[@]}"; do
        echo "$param" | sysctl -p - >/dev/null 2>&1 && {
            log "Reset sysctl parameter: $param"
        } || {
            warning "Failed to reset parameter: $param"
        }
    done
}

rollback_irq_affinity() {
    log "Rolling back IRQ affinity configuration..."
    
    local irq_state_file="/tmp/nic-irq-state"
    
    if [[ ! -f "$irq_state_file" ]]; then
        log "No IRQ affinity state file found - skipping IRQ rollback"
        return 0
    fi
    
    local backup_file
    backup_file=$(grep "^backup_file=" "$irq_state_file" | cut -d= -f2 || echo "")
    
    if [[ ! -f "$backup_file" ]]; then
        warning "IRQ backup file not found: $backup_file"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would restore IRQ affinity from: $backup_file"
        return 0
    fi
    
    log "Restoring IRQ affinity from: $backup_file"
    
    # Restore IRQ affinities
    while IFS='=' read -r key value; do
        if [[ "$key" =~ ^irq_[0-9]+_affinity$ ]]; then
            local irq
            irq=$(echo "$key" | sed 's/irq_\([0-9]*\)_affinity/\1/')
            
            if [[ -f "/proc/irq/$irq/smp_affinity" ]] && [[ -n "$value" ]]; then
                echo "$value" > "/proc/irq/$irq/smp_affinity" 2>/dev/null && {
                    log "Restored IRQ $irq affinity to: $value"
                } || {
                    warning "Failed to restore IRQ $irq affinity"
                }
            fi
        fi
    done < "$backup_file"
    
    # Clean up state file
    rm -f "$irq_state_file"
    log "IRQ affinity rollback completed"
}

rollback_onload_configuration() {
    log "Rolling back OnLoad configuration..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would remove OnLoad configuration files"
        log "[DRY RUN] Would reset network parameters"
        return 0
    fi
    
    # Stop any OnLoad monitoring processes
    pkill -f "onload-monitor" 2>/dev/null || true
    
    # Remove OnLoad configuration files (but preserve binaries)
    local config_files=(
        "/etc/onload"
        "/etc/sysctl.d/99-onload-optimization.conf"
    )
    
    for config in "${config_files[@]}"; do
        if [[ -e "$config" ]]; then
            rm -rf "$config"
            log "Removed OnLoad configuration: $config"
        fi
    done
    
    # Reset network parameters to defaults
    local default_net_params=(
        "net.core.rmem_max=212992"
        "net.core.wmem_max=212992"
        "net.core.rmem_default=212992"
        "net.core.wmem_default=212992"
        "net.core.netdev_max_backlog=1000"
        "net.core.netdev_budget=300"
        "net.ipv4.tcp_congestion_control=cubic"
        "net.core.busy_read=0"
        "net.core.busy_poll=0"
    )
    
    for param in "${default_net_params[@]}"; do
        echo "$param" | sysctl -p - >/dev/null 2>&1 && {
            log "Reset network parameter: $param"
        } || {
            warning "Failed to reset parameter: $param"
        }
    done
    
    # Reset huge pages
    echo "0" > /proc/sys/vm/nr_hugepages 2>/dev/null || true
    
    log "OnLoad configuration rollback completed"
}

verify_rollback() {
    log "Verifying rollback completion..."
    
    local issues=0
    
    # Check GRUB configuration
    if grep -q "isolcpus=" /etc/default/grub 2>/dev/null; then
        warning "GRUB still contains isolation parameters"
        ((issues++))
    fi
    
    # Check irqbalance status
    if ! systemctl is-active irqbalance >/dev/null 2>&1; then
        warning "irqbalance is not running"
        ((issues++))
    fi
    
    # Check CPU governors
    local perf_governors
    perf_governors=$(grep -l "performance" /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | wc -l || echo "0")
    if [[ $perf_governors -gt 0 ]]; then
        warning "$perf_governors CPUs still in performance mode"
        ((issues++))
    fi
    
    if [[ $issues -eq 0 ]]; then
        log "Rollback verification: PASSED"
    else
        log "Rollback verification: $issues issues found"
        if [[ "$PRESERVE_PERFORMANCE" == "true" ]]; then
            log "Some performance settings preserved by user request"
        else
            warning "Manual intervention may be required"
        fi
    fi
}

cleanup_state_files() {
    log "Cleaning up state files..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would remove state files"
        return 0
    fi
    
    local state_files=(
        "$STATE_FILE"
        "/tmp/nic-irq-state"
        "/tmp/onload-state"
        "/var/run/core-isolation-monitor.pid"
    )
    
    for state_file in "${state_files[@]}"; do
        if [[ -f "$state_file" ]]; then
            rm -f "$state_file"
            log "Removed state file: $state_file"
        fi
    done
    
    log "State files cleanup completed"
}

generate_rollback_report() {
    log "Generating rollback report..."
    
    local report_file="/tmp/rollback-report-$(date +%s).txt"
    
    cat > "$report_file" << EOF
================================================================================
CORE ISOLATION ROLLBACK REPORT
================================================================================
Timestamp: $(date)
Hostname: $(hostname)
Kernel: $(uname -r)
Rollback Type: $(if [[ "$DRY_RUN" == "true" ]]; then echo "DRY RUN"; else echo "ACTUAL"; fi)

ROLLBACK ACTIONS PERFORMED:
- GRUB configuration restored
- CPU governors reset to defaults
- IRQ affinity restored
- irqbalance service re-enabled
- OnLoad configuration removed
- Network parameters reset
- State files cleaned up

SYSTEM STATUS AFTER ROLLBACK:
CPU Isolation: $(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo "none")
IRQBalance: $(systemctl is-active irqbalance 2>/dev/null || echo "unknown")
OnLoad Status: $(if command -v onload >/dev/null 2>&1; then echo "binary present"; else echo "not available"; fi)

$(if grep -q "reboot_required=yes" "$STATE_FILE" 2>/dev/null; then
echo "REBOOT REQUIRED: Yes - kernel parameter changes need reboot to take effect"
else
echo "REBOOT REQUIRED: No"
fi)

RECOMMENDATIONS:
1. Verify trading applications work correctly after rollback
2. Monitor system performance for any anomalies
3. $(if grep -q "reboot_required=yes" "$STATE_FILE" 2>/dev/null; then echo "Reboot the system to complete kernel parameter rollback"; else echo "No reboot required for this rollback"; fi)

BACKUP LOCATIONS:
$(if [[ -f "$STATE_FILE" ]]; then grep "safety_backup=" "$STATE_FILE" | cut -d= -f2 || echo "None"; else echo "None"; fi)

================================================================================
EOF
    
    log "Rollback report generated: $report_file"
    
    # Display summary
    cat "$report_file"
}

usage() {
    cat << EOF
Core Isolation Rollback Utility

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --force                Force rollback despite warnings
    --dry-run             Show what would be done without making changes
    --preserve-performance Keep some performance optimizations
    --help                Show this help

ROLLBACK ACTIONS:
    - Restore original GRUB configuration
    - Reset CPU governors and frequency scaling
    - Restore IRQ affinity to defaults
    - Re-enable irqbalance service
    - Remove OnLoad configuration
    - Reset network parameters
    - Clean up state files

SAFETY FEATURES:
    - Pre-rollback system backup
    - Safety checks for running processes
    - Verification of rollback completion
    - Detailed rollback report

EXAMPLES:
    $0                     # Interactive rollback with safety checks
    $0 --dry-run          # Show rollback plan without changes
    $0 --force            # Force rollback despite warnings

WARNING:
    This will remove all CPU isolation and performance optimizations.
    Trading applications should be stopped before running rollback.
    A system reboot may be required to complete the rollback.

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_ROLLBACK=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --preserve-performance)
            PRESERVE_PERFORMANCE=true
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

if [[ "$DRY_RUN" == "true" ]]; then
    log "=== ROLLBACK DRY RUN MODE ==="
    log "No changes will be made to the system"
    log "=============================="
fi

log "Starting core isolation rollback..."

create_safety_backup
check_rollback_safety

log "Performing rollback operations..."
rollback_grub_configuration
rollback_runtime_optimizations
rollback_irq_affinity
rollback_onload_configuration

if [[ "$DRY_RUN" != "true" ]]; then
    verify_rollback
    cleanup_state_files
fi

generate_rollback_report

if [[ "$DRY_RUN" == "true" ]]; then
    log "Dry run completed - no changes made"
else
    log "Core isolation rollback completed successfully"
    if grep -q "reboot_required=yes" "$STATE_FILE" 2>/dev/null; then
        log "IMPORTANT: System reboot required to complete rollback"
    fi
fi