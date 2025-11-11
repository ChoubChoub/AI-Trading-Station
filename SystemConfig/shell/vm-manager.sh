#!/bin/bash
# ============================================================================
# AI Trading Station VM Manager
# ============================================================================
# PURPOSE: Simple VM lifecycle management for development environment
# FEATURES: Start, stop, status, and console access to trading VM
# SAFETY: Isolated development environment for safe testing
# ============================================================================

set -euo pipefail

# VM Configuration
VM_NAME="ai-trading-dev"
readonly VM_DIR="$HOME/.ai-trading-vm"
VM_CONFIG_PATH="$VM_DIR/${VM_NAME}.conf"
VM_PID_FILE="$VM_DIR/${VM_NAME}.pid"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

show_banner() {
    cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║            AI Trading Station VM Manager                      ║
║        Development Environment Control Center                 ║
╠════════════════════════════════════════════════════════════════╣
║  Commands: start, stop, status, console, ssh                  ║
║  Purpose: Safe isolated environment for trading development   ║
╚════════════════════════════════════════════════════════════════╝
EOF
}

load_vm_config() {
    if [ ! -f "$VM_CONFIG_PATH" ]; then
        error "VM configuration not found. Please run vm-setup.sh first."
    fi
    
    # Source the configuration file
    source "$VM_CONFIG_PATH"
    
    # Update paths that depend on VM_NAME after config is loaded
    VM_CONFIG_PATH="$VM_DIR/${VM_NAME}.conf"
    VM_PID_FILE="$VM_DIR/${VM_NAME}.pid"
}

check_vm_running() {
    if [ -f "$VM_PID_FILE" ]; then
        local pid=$(cat "$VM_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0  # VM is running
        else
            # PID file exists but process is dead
            rm -f "$VM_PID_FILE"
            return 1  # VM is not running
        fi
    fi
    return 1  # VM is not running
}

start_vm() {
    load_vm_config
    
    if check_vm_running; then
        warn "VM '$VM_NAME' is already running"
        return 0
    fi
    
    info "Starting VM '$VM_NAME'..."
    
    # Check if disk exists
    if [ ! -f "$VM_DISK_PATH" ]; then
        error "VM disk not found: $VM_DISK_PATH. Please run vm-setup.sh first."
    fi
    
    # Create workspace directory if it doesn't exist
    mkdir -p "$WORKSPACE_HOST_PATH"
    
    info "VM Configuration:"
    info "  • Memory: ${VM_MEMORY}MB"
    info "  • CPU Cores: $VM_CORES"
    info "  • Disk: $VM_DISK_PATH"
    info "  • Workspace: $WORKSPACE_HOST_PATH → $WORKSPACE_VM_PATH"
    
    # Check if KVM is available
    local accel_option="tcg"
    local cpu_option="qemu64"
    if [ -c /dev/kvm ]; then
        accel_option="kvm"
        cpu_option="host"
        info "  • Acceleration: KVM (hardware virtualization)"
    else
        warn "  • Acceleration: TCG (software emulation - slower)"
        warn "    To enable KVM: ensure virtualization is enabled in BIOS and run 'sudo modprobe kvm kvm_intel'"
    fi
    
    # Start VM in background
    nohup qemu-system-x86_64 \
        -name "$VM_NAME" \
        -machine type=pc,accel=$accel_option \
        -cpu $cpu_option \
        -smp "$VM_CORES" \
        -m "$VM_MEMORY" \
        -drive file="$VM_DISK_PATH",format=qcow2,if=virtio \
        $QEMU_NETWORK_ARGS \
        $QEMU_9P_ARGS \
        -vga qxl \
        -display none \
        -daemonize \
        -pidfile "$VM_PID_FILE" \
        > "$VM_DIR/${VM_NAME}.log" 2>&1
    
    # Wait a moment and check if VM started successfully
    sleep 3
    
    if check_vm_running; then
        info "✅ VM '$VM_NAME' started successfully"
        info "SSH access: ssh -p 2222 username@localhost"
        info "Console access: ./vm-manager.sh console"
        info "Status check: ./vm-manager.sh status"
    else
        error "Failed to start VM '$VM_NAME'. Check log: $VM_DIR/${VM_NAME}.log"
    fi
}

stop_vm() {
    if ! check_vm_running; then
        warn "VM '$VM_NAME' is not running"
        return 0
    fi
    
    info "Stopping VM '$VM_NAME'..."
    
    local pid=$(cat "$VM_PID_FILE")
    
    # Try graceful shutdown first
    info "Attempting graceful shutdown..."
    kill -TERM "$pid" 2>/dev/null || true
    
    # Wait for graceful shutdown
    local count=0
    while kill -0 "$pid" 2>/dev/null && [ $count -lt 30 ]; do
        sleep 1
        ((count++))
    done
    
    # If still running, force kill
    if kill -0 "$pid" 2>/dev/null; then
        warn "Graceful shutdown failed, forcing stop..."
        kill -KILL "$pid" 2>/dev/null || true
    fi
    
    # Clean up PID file
    rm -f "$VM_PID_FILE"
    
    info "✅ VM '$VM_NAME' stopped"
}

show_status() {
    load_vm_config
    
    echo
    info "VM Status Report:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # VM running status
    if check_vm_running; then
        local pid=$(cat "$VM_PID_FILE")
        info "✅ VM Status: Running (PID: $pid)"
        
        # Get VM resource usage
        if command -v ps &> /dev/null; then
            local mem_usage=$(ps -p "$pid" -o rss= 2>/dev/null | tr -d ' ' || echo "unknown")
            if [ "$mem_usage" != "unknown" ]; then
                mem_usage=$(( mem_usage / 1024 ))  # Convert to MB
                info "Memory Usage: ${mem_usage}MB"
            fi
        fi
    else
        warn "❌ VM Status: Stopped"
    fi
    
    # Configuration info
    info "VM Configuration:"
    info "  • Name: $VM_NAME"
    info "  • Memory: ${VM_MEMORY}MB"
    info "  • CPU Cores: $VM_CORES"
    info "  • Disk: $VM_DISK_PATH"
    
    # Workspace status
    if [ -d "$WORKSPACE_HOST_PATH" ]; then
        local workspace_size=$(du -sh "$WORKSPACE_HOST_PATH" 2>/dev/null | cut -f1 || echo "unknown")
        info "✅ Workspace: $WORKSPACE_HOST_PATH (${workspace_size})"
        
        # Check for key files
        if [ -f "$WORKSPACE_HOST_PATH/ai-trading-station.sh" ]; then
            info "✅ AI Trading Station: Available"
        else
            warn "⚠ AI Trading Station: Not found in workspace"
        fi
        
        if [ -f "$WORKSPACE_HOST_PATH/scripts/onload-trading" ]; then
            info "✅ OnLoad Trading: Available"
        else
            warn "⚠ OnLoad Trading: Not found in workspace"
        fi
    else
        warn "❌ Workspace: Not found ($WORKSPACE_HOST_PATH)"
    fi
    
    # Network info
    if check_vm_running; then
        info "Network Access:"
        info "  • SSH: ssh -p 2222 username@localhost"
        info "  • Console: ./vm-manager.sh console"
    fi
    
    echo
}

open_console() {
    if ! check_vm_running; then
        error "VM '$VM_NAME' is not running. Start it first with: ./vm-manager.sh start"
    fi
    
    info "Opening console for VM '$VM_NAME'..."
    info "Press Ctrl+Alt+G to release mouse, Ctrl+Alt+2 for QEMU monitor"
    
    # Connect to VM monitor (assumes monitor is on TCP port 4444)
    if command -v telnet >/dev/null 2>&1; then
        telnet localhost 4444
    elif command -v nc >/dev/null 2>&1; then
        nc localhost 4444
    else
        error "Neither 'telnet' nor 'nc' (netcat) is installed. Please install one to access the VM monitor."
    fi
}

connect_ssh() {
    if ! check_vm_running; then
        error "VM '$VM_NAME' is not running. Start it first with: ./vm-manager.sh start"
    fi
    
    info "Connecting to VM via SSH..."
    info "Default port: 2222, use your VM username"
    
    # Check if we have a specific username
    local username="${1:-$USER}"
    
    ssh -p 2222 "$username@localhost"
}

show_logs() {
    local log_file="$VM_DIR/${VM_NAME}.log"
    
    if [ ! -f "$log_file" ]; then
        warn "No log file found: $log_file"
        return 0
    fi
    
    info "Showing VM logs (last 50 lines):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    tail -n 50 "$log_file"
}

show_usage() {
    cat << EOF
AI Trading Station VM Manager

USAGE:
    ./vm-manager.sh <command> [options]

COMMANDS:
    start           Start the development VM
    stop            Stop the running VM
    status          Show VM status and configuration
    console         Open VM console (GUI)
    ssh [user]      Connect to VM via SSH (default user: $USER)
    logs            Show VM logs
    help            Show this help message

EXAMPLES:
    # Start the development VM
    ./vm-manager.sh start
    
    # Check VM status and workspace
    ./vm-manager.sh status
    
    # Connect to VM via SSH
    ./vm-manager.sh ssh developer
    
    # Stop the VM
    ./vm-manager.sh stop

WORKSPACE ACCESS:
    Host path: $HOME/ai-trading-station/
    VM path:   /workspace/ai-trading-station/
    
    Inside the VM, you can run:
    cd /workspace/ai-trading-station
    ./ai-trading-station.sh status
    ./scripts/onload-trading --help

ARCHITECTURE:
    This VM provides safe isolation for testing the AI Trading Station
    components without affecting the production ultra-low latency system.

EOF
}

main() {
    local command="${1:-help}"
    
    case "$command" in
        "start")
            show_banner
            echo
            start_vm
            ;;
        "stop")
            show_banner
            echo
            stop_vm
            ;;
        "status")
            show_banner
            show_status
            ;;
        "console")
            open_console
            ;;
        "ssh")
            connect_ssh "${2:-$USER}"
            ;;
        "logs")
            show_logs
            ;;
        "help"|"--help"|"-h")
            show_banner
            echo
            show_usage
            ;;
        *)
            error "Unknown command: $command. Use 'help' for usage information."
            ;;
    esac
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi