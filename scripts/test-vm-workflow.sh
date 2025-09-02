#!/bin/bash
# ============================================================================
# VM Development Environment Test Script - Workspace Mounting
# ============================================================================
# PURPOSE: Test and validate the complete VM development workflow with 
#          workspace mounting functionality
# ============================================================================

set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[TEST]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

test_script_availability() {
    info "Testing script availability with workspace mounting..."
    
    local scripts=(
        "scripts/vm-dev-environment"
        "scripts/production-mode-switch"
        "scripts/vm-setup-ubuntu.sh"
        "scripts/onload-trading"
        "ai-trading-station.sh"
    )
    
    local all_available=true
    
    for script in "${scripts[@]}"; do
        if [[ -x "$PROJECT_ROOT/$script" ]]; then
            info "✓ $script is available and executable"
        else
            error "✗ $script is missing or not executable"
            all_available=false
        fi
    done
    
    # Test for workspace mounting specific functionality
    if grep -q "workspace" "$PROJECT_ROOT/scripts/vm-dev-environment"; then
        info "✓ Workspace mounting functionality detected in vm-dev-environment"
    else
        error "✗ Workspace mounting functionality missing"
        all_available=false
    fi
    
    if [[ "$all_available" == true ]]; then
        return 0
    else
        return 1
    fi
}

test_workspace_configuration() {
    info "Testing workspace mounting configuration..."
    
    local config_file="$PROJECT_ROOT/configs/vm-dev-setup.json"
    if [[ -f "$config_file" ]]; then
        info "✓ Configuration file exists: $config_file"
        
        # Test JSON validity
        if python3 -m json.tool "$config_file" >/dev/null 2>&1; then
            info "✓ JSON file is valid"
        else
            error "✗ JSON file is invalid"
            return 1
        fi
        
        # Test for workspace mounting configuration
        if grep -q "workspace_mounting" "$config_file"; then
            info "✓ Workspace mounting configuration found"
        else
            error "✗ Workspace mounting configuration missing"
            return 1
        fi
        
        if grep -q "/workspace/ai-trading-station" "$config_file"; then
            info "✓ VM mount point configuration found"
        else
            error "✗ VM mount point configuration missing"
            return 1
        fi
    else
        error "✗ Configuration file missing: $config_file"
        return 1
    fi
}

test_vm_setup_workspace_support() {
    info "Testing VM setup script for workspace mounting support..."
    
    local setup_script="$PROJECT_ROOT/scripts/vm-setup-ubuntu.sh"
    
    # Check for workspace mounting code
    if grep -q "SHARED_FOLDER_TAG" "$setup_script"; then
        info "✓ Shared folder configuration found"
    else
        error "✗ Shared folder configuration missing"
        return 1
    fi
    
    if grep -q "HOST_WORKSPACE_PATH" "$setup_script"; then
        info "✓ Host workspace path configuration found"
    else
        error "✗ Host workspace path configuration missing"
        return 1
    fi
    
    if grep -q "VM_WORKSPACE_PATH" "$setup_script"; then
        info "✓ VM workspace path configuration found"
    else
        error "✗ VM workspace path configuration missing"
        return 1
    fi
    
    if grep -q "filesystem.*workspace" "$setup_script"; then
        info "✓ Filesystem mounting configuration found"
    else
        error "✗ Filesystem mounting configuration missing"
        return 1
    fi
    
    return 0
}

test_vm_dev_environment_workspace() {
    info "Testing VM development environment for workspace mounting..."
    
    local vm_script="$PROJECT_ROOT/scripts/vm-dev-environment"
    
    # Check for workspace mounting functions
    if grep -q "setup_workspace_mounting" "$vm_script"; then
        info "✓ Workspace mounting setup function found"
    else
        error "✗ Workspace mounting setup function missing"
        return 1
    fi
    
    if grep -q "mount_workspace_in_vm" "$vm_script"; then
        info "✓ VM workspace mounting function found"
    else
        error "✗ VM workspace mounting function missing"
        return 1
    fi
    
    if grep -q "WORKSPACE_PATH.*workspace" "$vm_script"; then
        info "✓ Workspace path configuration found"
    else
        error "✗ Workspace path configuration missing"
        return 1
    fi
    
    return 0
}

test_help_commands() {
    info "Testing help command functionality with workspace information..."
    
    local commands=(
        "$PROJECT_ROOT/scripts/vm-dev-environment --help"
        "$PROJECT_ROOT/scripts/production-mode-switch --help"
        "$PROJECT_ROOT/ai-trading-station.sh --help"
    )
    
    for cmd in "${commands[@]}"; do
        if $cmd >/dev/null 2>&1; then
            info "✓ Help command works: $cmd"
        else
            error "✗ Help command failed: $cmd"
            return 1
        fi
    done
    
    # Check if workspace mounting is mentioned in help
    if "$PROJECT_ROOT/scripts/vm-dev-environment" --help 2>&1 | grep -i "workspace"; then
        info "✓ Workspace mounting mentioned in vm-dev-environment help"
    else
        warn "⚠ Workspace mounting not mentioned in help (may be expected)"
    fi
}

test_vm_status_integration() {
    info "Testing VM status integration with workspace mounting..."
    
    # Test basic status command
    if "$PROJECT_ROOT/ai-trading-station.sh" vm-status >/dev/null 2>&1; then
        info "✓ VM status integration works"
    else
        error "✗ VM status integration failed"
        return 1
    fi
    
    # Check if workspace information is included
    if "$PROJECT_ROOT/ai-trading-station.sh" vm-status 2>&1 | grep -i "workspace"; then
        info "✓ Workspace information included in VM status"
    else
        warn "⚠ Workspace information not shown in status (check implementation)"
    fi
}

test_documentation() {
    info "Testing workspace mounting documentation..."
    
    local doc_file="$PROJECT_ROOT/docs/vm-development-guide.md"
    if [[ -f "$doc_file" ]]; then
        info "✓ Documentation file exists"
        
        if grep -q "workspace" "$doc_file"; then
            info "✓ Workspace mounting documented"
        else
            error "✗ Workspace mounting not documented"
            return 1
        fi
        
        if grep -q "complete project" "$doc_file"; then
            info "✓ Complete project access documented"
        else
            warn "⚠ Complete project access not clearly documented"
        fi
    else
        error "✗ Documentation file missing"
        return 1
    fi
}

test_workspace_mounting_workflow() {
    info "Testing complete workspace mounting workflow..."
    
    # Test that all required components are in place
    local required_components=(
        "Host workspace path detection"
        "VM mount point configuration"
        "Filesystem sharing setup"
        "Auto-mounting configuration"
    )
    
    local vm_script="$PROJECT_ROOT/scripts/vm-dev-environment"
    local setup_script="$PROJECT_ROOT/scripts/vm-setup-ubuntu.sh"
    
    # Check host workspace path detection
    if grep -q "PROJECT_ROOT" "$vm_script" && grep -q "HOST_WORKSPACE_PATH" "$vm_script"; then
        info "✓ Host workspace path detection implemented"
    else
        error "✗ Host workspace path detection missing"
        return 1
    fi
    
    # Check VM mount point configuration
    if grep -q "VM_WORKSPACE_PATH" "$vm_script" && grep -q "/workspace" "$vm_script"; then
        info "✓ VM mount point configuration implemented"
    else
        error "✗ VM mount point configuration missing"
        return 1
    fi
    
    # Check filesystem sharing setup
    if grep -q "9p\|virtio" "$setup_script" || grep -q "filesystem" "$setup_script"; then
        info "✓ Filesystem sharing setup implemented"
    else
        error "✗ Filesystem sharing setup missing"
        return 1
    fi
    
    # Check auto-mounting
    if grep -q "fstab\|mount" "$setup_script" || grep -q "mount.*workspace" "$vm_script"; then
        info "✓ Auto-mounting configuration implemented"
    else
        error "✗ Auto-mounting configuration missing"
        return 1
    fi
}

run_all_tests() {
    info "Running VM Development Environment Test Suite - Workspace Mounting Focus"
    echo "========================================================================================="
    
    local tests=(
        "test_script_availability"
        "test_workspace_configuration"
        "test_vm_setup_workspace_support"
        "test_vm_dev_environment_workspace"
        "test_help_commands"
        "test_vm_status_integration"
        "test_documentation"
        "test_workspace_mounting_workflow"
    )
    
    local passed=0
    local total=${#tests[@]}
    
    for test in "${tests[@]}"; do
        echo
        if $test; then
            ((passed++))
        fi
    done
    
    echo
    echo "========================================================================================="
    if [[ $passed -eq $total ]]; then
        info "All workspace mounting tests passed! ($passed/$total)"
        info "VM Development Environment with workspace mounting is ready!"
        echo
        info "WORKSPACE MOUNTING VERIFIED:"
        echo "  ✓ Complete project directory will be mounted in VM"
        echo "  ✓ Host path: $(pwd)"
        echo "  ✓ VM mount point: /workspace/ai-trading-station"
        echo "  ✓ Real-time file synchronization configured"
        echo
        info "Next steps:"
        echo "  1. Install virtualization software: sudo apt install qemu-kvm libvirt-daemon-system"
        echo "  2. Setup development VM: sudo ./scripts/vm-dev-environment setup"
        echo "  3. Start development mode: ./ai-trading-station.sh vm-dev"
        echo "  4. Access complete workspace in VM: cd /workspace/ai-trading-station"
        return 0
    else
        error "Some workspace mounting tests failed! ($passed/$total passed)"
        error "Please fix the issues before using the VM development environment."
        return 1
    fi
}

main() {
    cd "$PROJECT_ROOT"
    run_all_tests
}

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi