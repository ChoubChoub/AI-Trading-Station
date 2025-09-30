#!/bin/bash
# Phase 2 System Optimizations Rollback Script
# Created: September 28, 2025
# Purpose: Rollback Phase 2 system-level optimizations

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Find most recent backup
find_latest_backup() {
    local backup_dir
    backup_dir=$(find "$SCRIPT_DIR/backups" -name "phase2_system_backup_*" -type d 2>/dev/null | sort -r | head -1)
    
    if [[ -n "$backup_dir" && -d "$backup_dir" ]]; then
        echo "$backup_dir"
    else
        return 1
    fi
}

# Restore system to default HFT-optimized state
restore_defaults() {
    log "Restoring system to default HFT-optimized state..."
    
    # Restore timer migration (enable - default)
    echo 1 | sudo tee /proc/sys/kernel/timer_migration >/dev/null 2>&1 || true
    
    # Re-enable CPU C-states (default behavior)
    for cpu in 2 3 4; do
        for state in /sys/devices/system/cpu/cpu${cpu}/cpuidle/state*/disable; do
            if [[ -f "$state" ]]; then
                echo 0 | sudo tee "$state" >/dev/null 2>&1 || true
            fi
        done
    done
    
    # Restore hung task timeout (default: 120 seconds)
    echo 120 | sudo tee /proc/sys/kernel/hung_task_timeout_secs >/dev/null 2>&1 || true
    
    # Restore RT runtime (default: 950000)
    echo 950000 | sudo tee /proc/sys/kernel/sched_rt_runtime_us >/dev/null 2>&1 || true
    
    success "Default system parameters restored"
}

# Main rollback function
main() {
    echo "🚨 Phase 2 System Optimizations Rollback"
    echo "========================================"
    echo
    
    # Try to find and use backup
    local backup_dir
    if backup_dir=$(find_latest_backup); then
        log "Found backup: $backup_dir"
        
        # Restore from backup
        if [[ -f "$backup_dir/timer_migration" ]]; then
            log "Restoring timer migration..."
            sudo tee /proc/sys/kernel/timer_migration < "$backup_dir/timer_migration" >/dev/null
        fi
        
        # Restore CPU states
        for cpu in 2 3 4; do
            if [[ -d "$backup_dir/cpu${cpu}_states" ]]; then
                log "Restoring CPU $cpu C-states..."
                for state_file in "$backup_dir/cpu${cpu}_states"/*_disable; do
                    if [[ -f "$state_file" ]]; then
                        state_name=$(basename "$state_file" _disable)
                        target_file="/sys/devices/system/cpu/cpu${cpu}/cpuidle/state${state_name}/disable"
                        if [[ -f "$target_file" ]]; then
                            sudo tee "$target_file" < "$state_file" >/dev/null || true
                        fi
                    fi
                done
            fi
        done
        
        # Restore kernel parameters
        if [[ -f "$backup_dir/hung_task_timeout_secs" ]]; then
            log "Restoring hung task timeout..."
            sudo tee /proc/sys/kernel/hung_task_timeout_secs < "$backup_dir/hung_task_timeout_secs" >/dev/null
        fi
        
        if [[ -f "$backup_dir/sched_rt_runtime_us" ]]; then
            log "Restoring RT runtime..."
            sudo tee /proc/sys/kernel/sched_rt_runtime_us < "$backup_dir/sched_rt_runtime_us" >/dev/null
        fi
        
        success "System restored from backup: $backup_dir"
        
    else
        warning "No Phase 2 backup found, restoring to defaults..."
        restore_defaults
    fi
    
    # Verify Redis is still running
    log "Verifying Redis service..."
    if systemctl is-active redis-hft >/dev/null 2>&1; then
        success "Redis service is running"
    else
        warning "Redis service not running, attempting restart..."
        sudo systemctl restart redis-hft
        sleep 3
        if systemctl is-active redis-hft >/dev/null 2>&1; then
            success "Redis service restarted successfully"
        else
            error "Redis service failed to start"
        fi
    fi
    
    # Test Redis connectivity
    log "Testing Redis connectivity..."
    if redis-cli ping >/dev/null 2>&1; then
        success "Redis responding to ping"
    else
        error "Redis not responding"
        exit 1
    fi
    
    # Quick performance test
    log "Quick performance validation..."
    cd "$SCRIPT_DIR"
    if ../monitoring/redis-hft-monitor_to_json.sh >/dev/null 2>&1; then
        success "Performance monitoring working"
    else
        warning "Performance monitoring issues detected"
    fi
    
    echo
    success "Phase 2 rollback complete!"
    echo
    echo "Current system state:"
    echo "  • Timer migration: $(cat /proc/sys/kernel/timer_migration 2>/dev/null || echo 'unknown')"
    echo "  • Hung task timeout: $(cat /proc/sys/kernel/hung_task_timeout_secs 2>/dev/null || echo 'unknown')"
    echo "  • RT runtime: $(cat /proc/sys/kernel/sched_rt_runtime_us 2>/dev/null || echo 'unknown')"
    echo
    echo "To reapply Phase 2 optimizations: ./test-phase2-system.sh"
}

# Run if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi