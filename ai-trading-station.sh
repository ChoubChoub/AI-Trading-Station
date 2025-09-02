#!/bin/bash

# ============================================================================
# AI TRADING STATION - HIGH-PERFORMANCE SYSTEM LAUNCHER
# ============================================================================
# PURPOSE: Main system controller achieving sub-10μs latency performance
# TARGET METRICS: Mean 4.37μs | P95 4.53μs | P99 4.89μs
# HARDWARE: Intel Core Ultra 9 285K + Solarflare X2522 @ 10Gb/s
# ============================================================================

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPTS_DIR="${SCRIPT_DIR}/scripts"
readonly CONFIG_DIR="${SCRIPT_DIR}/config"
readonly LOGS_DIR="${SCRIPT_DIR}/logs"
readonly TESTS_DIR="${SCRIPT_DIR}/tests"

# Performance targets
readonly TARGET_MEAN_LATENCY="4.37"
readonly TARGET_P95_LATENCY="4.53"
readonly TARGET_P99_LATENCY="4.89"

# System configuration
readonly TRADING_CORES="2,3"
readonly SYSTEM_CORES="0,1"
readonly NETWORK_INTERFACE="eth0"

# Logging
exec > >(tee -a "${LOGS_DIR}/ai-trading-station.log")
exec 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

check_requirements() {
    log "Checking system requirements..."
    
    # Check if running as root
    [[ $EUID -eq 0 ]] || error "Must run as root for system optimizations"
    
    # Check CPU model
    local cpu_model
    cpu_model=$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    log "CPU: ${cpu_model}"
    
    # Check for required cores
    local core_count
    core_count=$(nproc)
    [[ $core_count -ge 4 ]] || error "Minimum 4 CPU cores required, found: ${core_count}"
    
    # Check for OnLoad
    if ! command -v onload >/dev/null 2>&1; then
        log "WARNING: OnLoad not found - network acceleration disabled"
    fi
    
    # Check network interface
    if ! ip link show "${NETWORK_INTERFACE}" >/dev/null 2>&1; then
        log "WARNING: Network interface ${NETWORK_INTERFACE} not found"
    fi
    
    log "System requirements check completed"
}

initialize_directories() {
    log "Initializing directory structure..."
    mkdir -p "${LOGS_DIR}" "${CONFIG_DIR}/backup" "${TESTS_DIR}/results"
    log "Directory structure initialized"
}

backup_system_config() {
    log "Backing up system configuration..."
    
    local backup_dir="${CONFIG_DIR}/backup/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${backup_dir}"
    
    # Backup critical system files
    cp /proc/cmdline "${backup_dir}/cmdline.backup" 2>/dev/null || true
    cp /etc/systemd/system.conf "${backup_dir}/system.conf.backup" 2>/dev/null || true
    cat /proc/interrupts > "${backup_dir}/interrupts.backup"
    cat /proc/cpuinfo > "${backup_dir}/cpuinfo.backup"
    
    log "System configuration backed up to: ${backup_dir}"
}

apply_core_isolation() {
    log "Applying CPU core isolation for trading processes..."
    
    if [[ -f "${SCRIPTS_DIR}/monitor-core-isolation.sh" ]]; then
        bash "${SCRIPTS_DIR}/monitor-core-isolation.sh" --apply --cores="${TRADING_CORES}"
    else
        log "WARNING: Core isolation script not found"
    fi
}

configure_network_optimization() {
    log "Configuring network optimizations..."
    
    if [[ -f "${SCRIPTS_DIR}/configure-nic-irq-affinity.sh" ]]; then
        bash "${SCRIPTS_DIR}/configure-nic-irq-affinity.sh" --interface="${NETWORK_INTERFACE}" --cores="${SYSTEM_CORES}"
    else
        log "WARNING: Network IRQ affinity script not found"
    fi
    
    if [[ -f "${SCRIPTS_DIR}/setup_trading_onload.sh" ]]; then
        bash "${SCRIPTS_DIR}/setup_trading_onload.sh"
    else
        log "WARNING: OnLoad setup script not found"
    fi
}

validate_performance() {
    log "Validating system performance..."
    
    if [[ -f "${SCRIPTS_DIR}/validate-isolation-performance.sh" ]]; then
        bash "${SCRIPTS_DIR}/validate-isolation-performance.sh" --target-latency="${TARGET_MEAN_LATENCY}"
    else
        log "WARNING: Performance validation script not found"
    fi
}

run_performance_tests() {
    log "Running comprehensive performance tests..."
    
    local test_results="${TESTS_DIR}/results/performance_$(date +%Y%m%d_%H%M%S).json"
    
    # Run OnLoad tests if available
    if [[ -f "${TESTS_DIR}/comprehensive_onload_test.py" ]]; then
        log "Running comprehensive OnLoad test..."
        python3 "${TESTS_DIR}/comprehensive_onload_test.py" > "${test_results}.onload" 2>&1 || true
    fi
    
    # Run real-world simulation
    if [[ -f "${TESTS_DIR}/real_world_onload_test.py" ]]; then
        log "Running real-world performance test..."
        python3 "${TESTS_DIR}/real_world_onload_test.py" > "${test_results}.realworld" 2>&1 || true
    fi
    
    log "Performance tests completed. Results in: ${TESTS_DIR}/results/"
}

monitor_system() {
    log "Starting system monitoring..."
    
    # Create monitoring daemon
    if [[ -f "${SCRIPTS_DIR}/monitor-core-isolation.sh" ]]; then
        bash "${SCRIPTS_DIR}/monitor-core-isolation.sh" --monitor --daemon &
        echo $! > "${LOGS_DIR}/monitor.pid"
    fi
}

print_status() {
    log "=== AI TRADING STATION STATUS ==="
    log "Target Performance: Mean ${TARGET_MEAN_LATENCY}μs | P95 ${TARGET_P95_LATENCY}μs | P99 ${TARGET_P99_LATENCY}μs"
    log "Trading Cores: ${TRADING_CORES}"
    log "System Cores: ${SYSTEM_CORES}"
    log "Network Interface: ${NETWORK_INTERFACE}"
    
    # Show current CPU isolation
    if [[ -f /sys/devices/system/cpu/isolated ]]; then
        local isolated_cores
        isolated_cores=$(cat /sys/devices/system/cpu/isolated)
        log "Isolated Cores: ${isolated_cores:-none}"
    fi
    
    # Show OnLoad status
    if command -v onload >/dev/null 2>&1; then
        log "OnLoad: Available"
    else
        log "OnLoad: Not Available"
    fi
    
    log "Log Directory: ${LOGS_DIR}"
    log "================================"
}

usage() {
    cat << EOF
AI Trading Station - High-Performance System Controller

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    start       Initialize and start the trading system
    stop        Stop the trading system and restore defaults
    status      Show current system status
    test        Run performance validation tests
    monitor     Start performance monitoring
    validate    Validate current performance metrics

OPTIONS:
    --cores=X,Y     Specify trading cores (default: ${TRADING_CORES})
    --interface=X   Network interface (default: ${NETWORK_INTERFACE})
    --help         Show this help message

EXAMPLES:
    $0 start                    # Start with default configuration
    $0 start --cores=4,5        # Start with cores 4,5 for trading
    $0 test                     # Run performance validation
    $0 status                   # Show system status

TARGET PERFORMANCE:
    Mean Latency: ${TARGET_MEAN_LATENCY}μs
    P95 Latency:  ${TARGET_P95_LATENCY}μs  
    P99 Latency:  ${TARGET_P99_LATENCY}μs

EOF
}

# Main command handling
main() {
    local command="${1:-}"
    
    case "${command}" in
        "start")
            log "Starting AI Trading Station..."
            initialize_directories
            check_requirements
            backup_system_config
            apply_core_isolation
            configure_network_optimization
            validate_performance
            monitor_system
            print_status
            log "AI Trading Station started successfully"
            ;;
        "stop")
            log "Stopping AI Trading Station..."
            if [[ -f "${LOGS_DIR}/monitor.pid" ]]; then
                kill "$(cat "${LOGS_DIR}/monitor.pid")" 2>/dev/null || true
                rm -f "${LOGS_DIR}/monitor.pid"
            fi
            if [[ -f "${SCRIPTS_DIR}/restore_original.sh" ]]; then
                bash "${SCRIPTS_DIR}/restore_original.sh"
            fi
            log "AI Trading Station stopped"
            ;;
        "status")
            print_status
            ;;
        "test")
            initialize_directories
            run_performance_tests
            ;;
        "monitor")
            initialize_directories
            monitor_system
            log "Monitoring started. Check ${LOGS_DIR}/ for output"
            ;;
        "validate")
            validate_performance
            ;;
        "help"|"--help"|"-h"|"")
            usage
            ;;
        *)
            error "Unknown command: ${command}. Use --help for usage information."
            ;;
    esac
}

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cores=*)
            TRADING_CORES="${1#*=}"
            shift
            ;;
        --interface=*)
            NETWORK_INTERFACE="${1#*=}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

main "$@"