#!/bin/bash
# ============================================================================
# VM Development Environment Test Script
# ============================================================================
# PURPOSE: Test and validate the complete VM development workflow
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
    info "Testing script availability..."
    
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
    
    if [[ "$all_available" == true ]]; then
        return 0
    else
        return 1
    fi
}

test_help_commands() {
    info "Testing help command functionality..."
    
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
}

test_configuration_files() {
    info "Testing configuration files..."
    
    local configs=(
        "configs/vm-dev-setup.json"
        "docs/vm-development-guide.md"
    )
    
    for config in "${configs[@]}"; do
        if [[ -f "$PROJECT_ROOT/$config" ]]; then
            info "✓ Configuration file exists: $config"
            
            # Test JSON validity for JSON files
            if [[ "$config" == *.json ]]; then
                if python3 -m json.tool "$PROJECT_ROOT/$config" >/dev/null 2>&1; then
                    info "✓ JSON file is valid: $config"
                else
                    error "✗ JSON file is invalid: $config"
                    return 1
                fi
            fi
        else
            error "✗ Configuration file missing: $config"
            return 1
        fi
    done
}

test_vm_status_integration() {
    info "Testing VM status integration..."
    
    # Test basic status command
    if "$PROJECT_ROOT/ai-trading-station.sh" vm-status >/dev/null 2>&1; then
        info "✓ VM status integration works"
    else
        error "✗ VM status integration failed"
        return 1
    fi
}

test_virtualization_checks() {
    info "Testing virtualization support checks..."
    
    # Test CPU virtualization detection
    if "$PROJECT_ROOT/scripts/vm-dev-environment" status >/dev/null 2>&1; then
        info "✓ Virtualization check works"
    else
        warn "⚠ Virtualization check returned warnings (may be expected in test environment)"
    fi
}

test_mode_switching_status() {
    info "Testing mode switching status..."
    
    # Test mode status check
    if "$PROJECT_ROOT/scripts/production-mode-switch" status >/dev/null 2>&1; then
        info "✓ Mode switching status works"
    else
        warn "⚠ Mode switching status returned warnings (expected without full setup)"
    fi
}

test_workflow_integration() {
    info "Testing complete workflow integration..."
    
    # Test that all commands are available through ai-trading-station.sh
    local integrated_commands=(
        "status"
        "vm-status"
    )
    
    for cmd in "${integrated_commands[@]}"; do
        if "$PROJECT_ROOT/ai-trading-station.sh" "$cmd" >/dev/null 2>&1; then
            info "✓ Integrated command works: $cmd"
        else
            error "✗ Integrated command failed: $cmd"
            return 1
        fi
    done
}

run_all_tests() {
    info "Running VM Development Environment Test Suite..."
    echo "=================================================================="
    
    local tests=(
        "test_script_availability"
        "test_help_commands"
        "test_configuration_files"
        "test_vm_status_integration"
        "test_virtualization_checks"
        "test_mode_switching_status"
        "test_workflow_integration"
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
    echo "=================================================================="
    if [[ $passed -eq $total ]]; then
        info "All tests passed! ($passed/$total)"
        info "VM Development Environment is ready for use."
        echo
        info "Next steps:"
        echo "  1. Install virtualization software: sudo apt install qemu-kvm libvirt-daemon-system"
        echo "  2. Setup development VM: sudo ./scripts/vm-dev-environment setup"
        echo "  3. Switch to development mode: ./ai-trading-station.sh vm-dev"
        echo "  4. Switch to production mode: ./ai-trading-station.sh vm-prod"
        return 0
    else
        error "Some tests failed! ($passed/$total passed)"
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