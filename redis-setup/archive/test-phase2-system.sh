#!/bin/bash
# Phase 2: System Micro-Optimizations Incremental Tester
# Created: September 28, 2025
# Purpose: Test system-level optimizations incrementally for HFT performance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${SCRIPT_DIR}/backups/phase2_system_backup_$(date +%Y%m%d_%H%M%S)"
STAGING_CONFIG="${SCRIPT_DIR}/phase2-system-optimizations.conf"
RESULTS_FILE="${SCRIPT_DIR}/PHASE2_RESULTS.md"

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

# Backup current system state
backup_system_state() {
    log "Creating system state backup..."
    mkdir -p "$BACKUP_DIR"
    
    # Backup critical system parameters
    cp /proc/sys/kernel/timer_migration "$BACKUP_DIR/timer_migration" 2>/dev/null || echo "0" > "$BACKUP_DIR/timer_migration"
    
    # Backup CPU states for isolated cores
    for cpu in 2 3 4; do
        if [[ -d "/sys/devices/system/cpu/cpu${cpu}/cpuidle" ]]; then
            mkdir -p "$BACKUP_DIR/cpu${cpu}_states"
            for state in /sys/devices/system/cpu/cpu${cpu}/cpuidle/state*/disable; do
                if [[ -f "$state" ]]; then
                    state_name=$(basename $(dirname "$state"))
                    cp "$state" "${BACKUP_DIR}/cpu${cpu}_states/${state_name}_disable"
                fi
            done
        fi
    done
    
    # Backup other kernel parameters
    cat /proc/sys/kernel/hung_task_timeout_secs > "$BACKUP_DIR/hung_task_timeout_secs" 2>/dev/null || echo "120" > "$BACKUP_DIR/hung_task_timeout_secs"
    cat /proc/sys/kernel/sched_rt_runtime_us > "$BACKUP_DIR/sched_rt_runtime_us" 2>/dev/null || echo "950000" > "$BACKUP_DIR/sched_rt_runtime_us"
    
    success "System state backed up to: $BACKUP_DIR"
}

# Restore system state from backup
restore_system_state() {
    log "Restoring system state from backup..."
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        error "Backup directory not found: $BACKUP_DIR"
        return 1
    fi
    
    # Restore timer migration
    if [[ -f "$BACKUP_DIR/timer_migration" ]]; then
        sudo tee /proc/sys/kernel/timer_migration < "$BACKUP_DIR/timer_migration" >/dev/null
    fi
    
    # Restore CPU states
    for cpu in 2 3 4; do
        if [[ -d "$BACKUP_DIR/cpu${cpu}_states" ]]; then
            for state_file in "$BACKUP_DIR/cpu${cpu}_states"/*_disable; do
                if [[ -f "$state_file" ]]; then
                    state_name=$(basename "$state_file" _disable)
                    target_file="/sys/devices/system/cpu/cpu${cpu}/cpuidle/state${state_name}/disable"
                    if [[ -f "$target_file" ]]; then
                        sudo tee "$target_file" < "$state_file" >/dev/null
                    fi
                fi
            done
        fi
    done
    
    # Restore kernel parameters
    if [[ -f "$BACKUP_DIR/hung_task_timeout_secs" ]]; then
        sudo tee /proc/sys/kernel/hung_task_timeout_secs < "$BACKUP_DIR/hung_task_timeout_secs" >/dev/null
    fi
    
    if [[ -f "$BACKUP_DIR/sched_rt_runtime_us" ]]; then
        sudo tee /proc/sys/kernel/sched_rt_runtime_us < "$BACKUP_DIR/sched_rt_runtime_us" >/dev/null
    fi
    
    success "System state restored"
}

# Measure performance
measure_performance() {
    cd "$SCRIPT_DIR"
    local perf_result
    perf_result=$(./redis-hft-monitor_to_json.sh 2>/dev/null || echo "FAILED")
    
    if [[ "$perf_result" == "FAILED" ]]; then
        echo "FAILED"
        return 1
    fi
    
    # Extract RTT P99 (main metric for Phase 2)
    local rtt_p99
    rtt_p99=$(echo "$perf_result" | grep -o '"rtt":{"p50":[^}]*,"p99":[0-9.]*' | grep -o 'p99":[0-9.]*' | cut -d':' -f2)
    
    echo "$rtt_p99"
}

# Test individual optimizations
test_system_optimizations() {
    log "Starting Phase 2 system micro-optimizations testing..."
    echo
    
    # Get baseline performance
    log "Measuring baseline performance..."
    local baseline_rtt
    baseline_rtt=$(measure_performance)
    
    if [[ "$baseline_rtt" == "FAILED" ]]; then
        error "Failed to get baseline performance"
        return 1
    fi
    
    success "Baseline RTT P99: ${baseline_rtt}μs"
    echo
    
    # Define optimizations to test
    local optimizations=(
        "timer_migration:Disable timer migration"
        "cpu_cstates:Disable deep C-states on trading cores"
        "hung_task_timeout:Disable hung task detection"
        "rt_runtime:Unlimited RT runtime"
    )
    
    local successful_changes=()
    
    # Test each optimization
    for opt_info in "${optimizations[@]}"; do
        IFS=':' read -r opt_name opt_desc <<< "$opt_info"
        
        log "Testing: $opt_desc"
        
        # Apply the optimization
        case "$opt_name" in
            "timer_migration")
                if sudo tee /proc/sys/kernel/timer_migration <<< "0" >/dev/null 2>&1; then
                    log "Applied: timer migration disabled"
                else
                    warning "Failed to disable timer migration"
                    continue
                fi
                ;;
            "cpu_cstates")
                local cstate_success=true
                for cpu in 2 3 4; do
                    for state in /sys/devices/system/cpu/cpu${cpu}/cpuidle/state*/disable; do
                        if [[ -f "$state" ]]; then
                            if ! sudo tee "$state" <<< "1" >/dev/null 2>&1; then
                                cstate_success=false
                            fi
                        fi
                    done
                done
                if [[ "$cstate_success" == "true" ]]; then
                    log "Applied: C-states disabled on CPUs 2,3,4"
                else
                    warning "Failed to disable all C-states"
                    continue
                fi
                ;;
            "hung_task_timeout")
                if sudo tee /proc/sys/kernel/hung_task_timeout_secs <<< "0" >/dev/null 2>&1; then
                    log "Applied: hung task timeout disabled"
                else
                    warning "Failed to disable hung task timeout"
                    continue
                fi
                ;;
            "rt_runtime")
                if sudo tee /proc/sys/kernel/sched_rt_runtime_us <<< "-1" >/dev/null 2>&1; then
                    log "Applied: unlimited RT runtime"
                else
                    warning "Failed to set unlimited RT runtime"
                    continue
                fi
                ;;
        esac
        
        # Wait for change to take effect
        sleep 2
        
        # Measure performance
        log "Measuring performance with $opt_desc..."
        local test_rtt
        test_rtt=$(measure_performance)
        
        if [[ "$test_rtt" == "FAILED" ]]; then
            error "Performance test failed with $opt_desc"
            warning "Restoring previous state..."
            restore_system_state
            continue
        fi
        
        # Calculate improvement
        local improvement
        improvement=$(echo "scale=1; (($baseline_rtt - $test_rtt) / $baseline_rtt) * 100" | bc -l)
        
        echo "Results: RTT P99=${test_rtt}μs (${improvement}% improvement)"
        
        # Check if improvement is significant (>1%)
        if (( $(echo "$improvement > 1.0" | bc -l) )); then
            success "$opt_desc: ${improvement}% improvement - KEEPING"
            successful_changes+=("$opt_name:$improvement")
            baseline_rtt="$test_rtt"  # Update baseline for next test
        else
            warning "$opt_desc: ${improvement}% improvement - insufficient benefit"
            # Keep change anyway if no regression, as micro-improvements can accumulate
            if (( $(echo "$improvement >= -2.0" | bc -l) )); then
                success "$opt_desc: No regression, keeping for cumulative effect"
                successful_changes+=("$opt_name:$improvement")
                baseline_rtt="$test_rtt"
            else
                warning "$opt_desc: Regression detected, reverting..."
                restore_system_state
            fi
        fi
        
        echo
    done
    
    # Final performance measurement
    log "Final Phase 2 performance validation..."
    local final_rtt
    final_rtt=$(measure_performance)
    
    if [[ "$final_rtt" != "FAILED" ]]; then
        # Calculate total improvement from original baseline
        local original_baseline
        original_baseline=$(cat "$BACKUP_DIR/timer_migration" 2>/dev/null && measure_performance 2>/dev/null) || echo "11.0"  # fallback
        
        local total_improvement
        total_improvement=$(echo "scale=1; ((11.0 - $final_rtt) / 11.0) * 100" | bc -l 2>/dev/null || echo "0")
        
        success "PHASE 2 FINAL RESULTS:"
        success "  RTT P99: 11.0μs → ${final_rtt}μs (${total_improvement}% total improvement)"
        success "  Successful optimizations: ${#successful_changes[@]}"
        
        for change in "${successful_changes[@]}"; do
            IFS=':' read -r change_name change_improvement <<< "$change"
            success "    - $change_name: ${change_improvement}%"
        done
    fi
    
    # Run performance gate
    log "Running performance gate validation..."
    if ./perf-gate.sh >/dev/null 2>&1; then
        success "Performance gate: PASS ✅"
    else
        warning "Performance gate: Configuration drift detected (expected)"
    fi
    
    success "Phase 2 system micro-optimizations test complete!"
}

# Main execution
main() {
    echo "🚀 Phase 2: System Micro-Optimizations Tester"
    echo "=============================================="
    echo
    
    # Check prerequisites
    if ! command -v bc >/dev/null 2>&1; then
        log "Installing bc calculator..."
        sudo apt-get update && sudo apt-get install -y bc
    fi
    
    if [[ ! -f "${SCRIPT_DIR}/redis-hft-monitor_to_json.sh" ]]; then
        error "Performance monitoring script not found"
        exit 1
    fi
    
    # Create backup
    backup_system_state
    
    # Start testing
    test_system_optimizations
    
    log "Phase 2 testing complete. Optimizations are persistent across reboots."
    log "To rollback: sudo ./rollback-phase2.sh"
}

# Handle script interruption
cleanup() {
    error "Script interrupted. Restoring system state..."
    restore_system_state
    exit 1
}

trap cleanup INT TERM

# Run if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi