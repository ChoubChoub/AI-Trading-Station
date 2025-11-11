#!/bin/bash
# AI Trading Station System Configuration Deployment Script - Enhanced
# Purpose: Deploy SystemConfig to system locations with selective and test modes
# Author: AI Trading Station
# Created: November 5, 2025
# Updated: November 6, 2025 - Moved to SystemConfig root for better organization

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is now in SystemConfig root
SYSTEMCONFIG_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Global flags
DRY_RUN=false
TEST_MODE=false
SELECTIVE_MODE=false
INTERACTIVE_MODE=false
FILE_MODE=false
SELECTED_COMPONENTS=()
# Specific selection holders
SELECTED_ITEMS=()   # entries like: file:REL_PATH | service:NAME | network:NAME
TEST_DIR=""

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_test() { echo -e "${CYAN}[TEST]${NC} $1"; }
log_select() { echo -e "${MAGENTA}[SELECT]${NC} $1"; }

# Check if running as root (skip in test mode)
check_root() {
    if [[ "$TEST_MODE" == true ]]; then
        log_test "Test mode - skipping root check"
        return 0
    fi
    
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        echo "For testing without root, use: $0 --test"
        exit 1
    fi
}

# Resolve absolute path under SystemConfig; returns empty if not inside tree
resolve_src_under_systemconfig() {
    local input_path="$1"
    local abs_path=""
    if [[ -z "$input_path" ]]; then echo ""; return 0; fi
    if [[ "$input_path" = /* ]]; then
        abs_path="$input_path"
    else
        abs_path="$SYSTEMCONFIG_DIR/$input_path"
    fi
    # Normalize and ensure within SystemConfig
    if [[ -f "$abs_path" ]]; then
        case "$abs_path" in
            "$SYSTEMCONFIG_DIR"/*) echo "$abs_path"; return 0;;
            *) echo ""; return 0;;
        esac
    fi
    echo ""
}

# Show available components for selective deployment
show_components() {
    echo ""
    echo -e "${BOLD}Available Components:${NC}"
    echo ""
    echo "  [1] systemd-services    - SystemD service definitions"
    echo "  [2] systemd-overrides   - CPU affinity and resource limits"
    echo "  [3] kernel              - Kernel module configurations"
    echo "  [4] network             - Network optimization scripts"
    echo "  [5] redis               - Redis HFT configuration"
    echo "  [6] onload              - Solarflare Onload configuration"
    echo "  [7] sysctl              - System tuning parameters"
    echo "  [8] udev                - Hardware device rules"
    echo "  [9] sudoers             - Passwordless sudo configurations"
    echo "  [10] kvm                - KVM virtualization modules"
    echo "  [11] logrotate          - Log rotation configurations"
    echo "  [c] cron                - Scheduled jobs (crontab)"
    echo "  [s] shell               - Shell environment (bashrc.trading)"
    echo "  [a] all                 - Deploy everything"
    echo ""
}

# Interactive component selection
interactive_selection() {
    show_components
    
    echo -e "${MAGENTA}Select components to deploy (space-separated numbers, or 'a' for all):${NC}"
    read -r selection
    
    if [[ "$selection" == "a" ]] || [[ "$selection" == "all" ]]; then
        SELECTED_COMPONENTS=(systemd-services systemd-overrides kernel network redis onload sysctl udev sudoers kvm logrotate cron shell)
        log_select "Selected: ALL components"
    else
        for num in $selection; do
            case "$num" in
                1) SELECTED_COMPONENTS+=(systemd-services) ;;
                2) SELECTED_COMPONENTS+=(systemd-overrides) ;;
                3) SELECTED_COMPONENTS+=(kernel) ;;
                4) SELECTED_COMPONENTS+=(network) ;;
                5) SELECTED_COMPONENTS+=(redis) ;;
                6) SELECTED_COMPONENTS+=(onload) ;;
                7) SELECTED_COMPONENTS+=(sysctl) ;;
                8) SELECTED_COMPONENTS+=(udev) ;;
                9) SELECTED_COMPONENTS+=(sudoers) ;;
                10) SELECTED_COMPONENTS+=(kvm) ;;
                11) SELECTED_COMPONENTS+=(logrotate) ;;
                c|C) SELECTED_COMPONENTS+=(cron) ;;
                s|S) SELECTED_COMPONENTS+=(shell) ;;
                *) log_warning "Unknown selection: $num" ;;
            esac
        done
    fi
    
    if [[ ${#SELECTED_COMPONENTS[@]} -eq 0 ]]; then
        log_error "No valid components selected"
        exit 1
    fi
    
    echo ""
    log_select "Will deploy: ${SELECTED_COMPONENTS[*]}"
    echo ""
    
    if [[ "$TEST_MODE" != true ]]; then
        echo -e "${YELLOW}Press Enter to continue or Ctrl+C to abort...${NC}"
        read -r
    fi
}

# Check if component is selected
is_selected() {
    local component="$1"
    
    # If not in selective mode, deploy everything
    if [[ "$SELECTIVE_MODE" != true ]]; then
        return 0
    fi
    
    # Check if component is in selected list
    for selected in "${SELECTED_COMPONENTS[@]}"; do
        if [[ "$selected" == "$component" ]]; then
            return 0
        fi
    done
    
    return 1
}

# Execute or simulate based on mode
execute_command() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$TEST_MODE" == true ]]; then
        log_test "Would execute: $description"
        if [[ -n "$3" ]]; then
            log_test "  Details: $3"
        fi
        return 0
    elif [[ "$DRY_RUN" == true ]]; then
        log_info "Would execute: $description"
        return 0
    else
        eval "$cmd"
        return $?
    fi
}

# Get destination path (test or real)
get_dest_path() {
    local system_path="$1"
    
    if [[ "$TEST_MODE" == true ]]; then
        echo "${TEST_DIR}${system_path}"
    else
        echo "$system_path"
    fi
}

# Compute destination path, perms and type for a given SystemConfig source
# Echos: DEST_PATH|PERMS|TYPE  (TYPE: systemd, override, network, redis, onload, modprobe, sysctl, udev, grub, unknown)
compute_destination() {
    local src_abs="$1"
    local rel="${src_abs#"$SYSTEMCONFIG_DIR/"}"
    local dest=""; local perms="644"; local type="unknown"

    case "$rel" in
        systemd/services/*.service)
            dest="/etc/systemd/system/${rel#systemd/services/}"
            type="systemd";
            ;;
        systemd/overrides/*.service.d/*.conf)
            dest="/etc/systemd/system/${rel#systemd/overrides/}"
            type="override";
            ;;
        network/*.sh)
            dest="/usr/local/bin/${rel#network/}"
            perms="755"; type="network";
            ;;
        redis/*.conf)
            dest="/opt/redis-hft/config/${rel#redis/}"
            perms="644"; type="redis";
            ;;
        onload/*.conf)
            dest="/etc/modprobe.d/${rel#onload/}"
            perms="644"; type="onload";
            ;;
        kernel/modprobe.d/*.conf)
            dest="/etc/modprobe.d/${rel#kernel/modprobe.d/}"
            perms="644"; type="modprobe";
            ;;
        sysctl.d/*.conf)
            dest="/etc/sysctl.d/${rel#sysctl.d/}"
            perms="644"; type="sysctl";
            ;;
        udev/rules.d/*.rules)
            dest="/etc/udev/rules.d/${rel#udev/rules.d/}"
            perms="644"; type="udev";
            ;;
        logrotate.d/*)
            dest="/etc/logrotate.d/${rel#logrotate.d/}"
            perms="644"; type="logrotate";
            ;;
        kernel/grub/*.conf)
            # Inform-only; manual integration
            dest=""; type="grub";
            ;;
        *)
            dest=""; type="unknown";
            ;;
    esac

    if [[ -n "$dest" ]]; then
        dest="$(get_dest_path "$dest")"
    fi
    echo "$dest|$perms|$type"
}

# Deploy a single file (absolute path under SystemConfig)
deploy_one_file() {
    local src_abs="$1"
    if [[ ! -f "$src_abs" ]]; then
        log_error "Source not found: $src_abs"
        return 1
    fi
    local mapping
    mapping="$(compute_destination "$src_abs")"
    local dest="${mapping%%|*}"; local rest="${mapping#*|}"; local perms="${rest%%|*}"; local type="${rest#*|}"

    if [[ "$type" == "grub" ]]; then
        log_warning "GRUB config requires manual integration: ${src_abs##*/}"
        return 0
    fi
    if [[ -z "$dest" ]]; then
        log_error "Unsupported file path relative to SystemConfig: ${src_abs#"$SYSTEMCONFIG_DIR/"}"
        return 1
    fi

    # Ensure destination directory exists (test mode uses TEST_DIR prefix)
    execute_command "mkdir -p '$(dirname "$dest")'" "Ensure destination directory exists" "$dest"

    copy_file_with_diff "$src_abs" "$dest" "$perms" "$TEST_MODE"

    # Post-copy adjustments
    if [[ "$type" == "redis" && "$TEST_MODE" != true ]]; then
        if id "redis-hft" &>/dev/null; then
            chown redis-hft:redis-hft "$dest" || true
            chmod 640 "$dest" || true
        fi
    fi

    return 0
}

# Deploy specific selections (files/services/scripts)
deploy_specific_files() {
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Deploying Specific Selections"
    log_info "═══════════════════════════════════════════════════════════════"

    local item
    for item in "${SELECTED_ITEMS[@]}"; do
        case "$item" in
            file:*)
                local rel="${item#file:}"
                local src_abs
                src_abs="$(resolve_src_under_systemconfig "$rel")"
                if [[ -z "$src_abs" ]]; then
                    log_error "File not under SystemConfig or not found: $rel"
                else
                    deploy_one_file "$src_abs"
                fi
                ;;
            service:*)
                local name="${item#service:}"
                [[ "$name" != *.service ]] && name+=".service"
                deploy_one_file "$SYSTEMCONFIG_DIR/systemd/services/$name"
                ;;
            network:*)
                local n="${item#network:}"
                [[ "$n" != *.sh ]] && n+=".sh"
                deploy_one_file "$SYSTEMCONFIG_DIR/network/$n"
                ;;
            *)
                log_warning "Unknown selection token: $item"
                ;;
        esac
    done
}

# Backup existing files
backup_existing() {
    local dest_file="$1"
    
    if [[ "$TEST_MODE" == true ]]; then
        # In test mode, no backup needed
        return 0
    fi
    
    if [[ -f "$dest_file" ]]; then
        local backup_file="${dest_file}.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$dest_file" "$backup_file"
        log_info "Backed up: $(basename "$dest_file") → $(basename "$backup_file")"
    fi
}

# Copy file with diff preview
copy_file_with_diff() {
    local src="$1"
    local dest="$2"
    local perms="$3"
    local show_diff="${4:-false}"
    
    # Check if files are different
    local files_differ=false
    if [[ -f "$dest" ]]; then
        if ! diff -q "$src" "$dest" &>/dev/null; then
            files_differ=true
        fi
    else
        files_differ=true
    fi
    
    # Show diff if requested and files differ
    if [[ "$show_diff" == true ]] && [[ "$files_differ" == true ]] && [[ -f "$dest" ]]; then
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${CYAN}Diff for: $(basename "$src")${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        diff -u "$dest" "$src" 2>/dev/null | head -50 || true
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
    fi
    
    if [[ "$TEST_MODE" == true ]]; then
        if [[ "$files_differ" == true ]]; then
            if [[ -f "$dest" ]]; then
                log_test "Would UPDATE: $(basename "$src") → $dest (files differ)"
            else
                log_test "Would CREATE: $(basename "$src") → $dest (new file)"
            fi
        else
            log_success "UNCHANGED: $(basename "$src") (identical to destination)"
        fi
        return 0
    elif [[ "$DRY_RUN" == true ]]; then
        if [[ "$files_differ" == true ]]; then
            log_info "Would copy: $(basename "$src") → $dest"
        fi
        return 0
    else
        if [[ "$files_differ" == true ]]; then
            backup_existing "$dest"
            mkdir -p "$(dirname "$dest")"
            cp "$src" "$dest"
            chmod "$perms" "$dest"
            log_success "Deployed: $(basename "$src")"
        else
            log_success "Unchanged: $(basename "$src") (already up-to-date)"
        fi
        return 0
    fi
}

# Deploy SystemD service files
deploy_systemd_services() {
    if ! is_selected "systemd-services"; then
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Deploying SystemD Service Files"
    log_info "═══════════════════════════════════════════════════════════════"
    
    if [[ -d "$SYSTEMCONFIG_DIR/systemd/services" ]]; then
        for service_file in "$SYSTEMCONFIG_DIR/systemd/services"/*.service; do
            if [[ -f "$service_file" ]]; then
                local service_name=$(basename "$service_file")
                local dest_path=$(get_dest_path "/etc/systemd/system/$service_name")
                
                copy_file_with_diff "$service_file" "$dest_path" "644" "$TEST_MODE"
            fi
        done
    else
        log_warning "No service files found in $SYSTEMCONFIG_DIR/systemd/services"
    fi
}

# Deploy SystemD overrides
deploy_systemd_overrides() {
    if ! is_selected "systemd-overrides"; then
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Deploying SystemD Service Overrides"
    log_info "═══════════════════════════════════════════════════════════════"
    
    if [[ -d "$SYSTEMCONFIG_DIR/systemd/overrides" ]]; then
        for override_dir in "$SYSTEMCONFIG_DIR/systemd/overrides"/*.service.d; do
            if [[ -d "$override_dir" ]]; then
                local service_name=$(basename "$override_dir")
                local dest_dir=$(get_dest_path "/etc/systemd/system/$service_name")
                
                execute_command "mkdir -p '$dest_dir'" "Create override directory: $service_name"
                
                for conf_file in "$override_dir"/*.conf; do
                    if [[ -f "$conf_file" ]]; then
                        local conf_name=$(basename "$conf_file")
                        local dest_path="$dest_dir/$conf_name"
                        
                        copy_file_with_diff "$conf_file" "$dest_path" "644" "$TEST_MODE"
                    fi
                done
            fi
        done
    else
        log_warning "No override files found in $SYSTEMCONFIG_DIR/systemd/overrides"
    fi
}

# Deploy kernel module configurations
deploy_kernel_configs() {
    if ! is_selected "kernel"; then
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Deploying Kernel Module Configurations"
    log_info "═══════════════════════════════════════════════════════════════"
    
    # Deploy modprobe configurations
    if [[ -d "$SYSTEMCONFIG_DIR/kernel/modprobe.d" ]]; then
        for conf_file in "$SYSTEMCONFIG_DIR/kernel/modprobe.d"/*.conf; do
            if [[ -f "$conf_file" ]]; then
                local conf_name=$(basename "$conf_file")
                local dest_path=$(get_dest_path "/etc/modprobe.d/$conf_name")
                
                copy_file_with_diff "$conf_file" "$dest_path" "644" "$TEST_MODE"
            fi
        done
    fi
    
    # Deploy GRUB configurations (requires manual integration)
    if [[ -d "$SYSTEMCONFIG_DIR/kernel/grub" ]]; then
        log_warning "GRUB configs found - manual integration required:"
        for grub_file in "$SYSTEMCONFIG_DIR/kernel/grub"/*.conf; do
            if [[ -f "$grub_file" ]]; then
                log_info "  → $(basename "$grub_file") - Review and integrate into /etc/default/grub"
            fi
        done
    fi
}

# Deploy network scripts
deploy_network_scripts() {
    if ! is_selected "network"; then
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Deploying Network Optimization Scripts"
    log_info "═══════════════════════════════════════════════════════════════"
    
    if [[ -d "$SYSTEMCONFIG_DIR/network" ]]; then
        for script_file in "$SYSTEMCONFIG_DIR/network"/*.sh; do
            if [[ -f "$script_file" ]]; then
                local script_name=$(basename "$script_file")
                local dest_path=$(get_dest_path "/usr/local/bin/$script_name")
                
                copy_file_with_diff "$script_file" "$dest_path" "755" "$TEST_MODE"
            fi
        done
    else
        log_warning "No network scripts found in $SYSTEMCONFIG_DIR/network"
    fi
}

# Deploy Redis configurations
deploy_redis_configs() {
    if ! is_selected "redis"; then
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Deploying Redis HFT Configurations"
    log_info "═══════════════════════════════════════════════════════════════"
    
    if [[ -d "$SYSTEMCONFIG_DIR/redis" ]]; then
        local redis_dir=$(get_dest_path "/opt/redis-hft/config")
        execute_command "mkdir -p '$redis_dir'" "Create Redis config directory"
        
        for conf_file in "$SYSTEMCONFIG_DIR/redis"/*.conf; do
            if [[ -f "$conf_file" ]]; then
                local conf_name=$(basename "$conf_file")
                local dest_path="$redis_dir/$conf_name"
                
                copy_file_with_diff "$conf_file" "$dest_path" "644" "$TEST_MODE"
                
                # Set proper ownership if not in test mode
                if [[ "$TEST_MODE" != true ]] && id "redis-hft" &>/dev/null; then
                    chown redis-hft:redis-hft "$dest_path"
                    chmod 640 "$dest_path"
                fi
            fi
        done
    else
        log_warning "No Redis configs found in $SYSTEMCONFIG_DIR/redis"
    fi
}

# Deploy Onload configurations
deploy_onload_configs() {
    if ! is_selected "onload"; then
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Deploying Solarflare Onload Configurations"
    log_info "═══════════════════════════════════════════════════════════════"
    
    if [[ -d "$SYSTEMCONFIG_DIR/onload" ]]; then
        for conf_file in "$SYSTEMCONFIG_DIR/onload"/*.conf; do
            if [[ -f "$conf_file" ]]; then
                local conf_name=$(basename "$conf_file")
                local dest_path=$(get_dest_path "/etc/modprobe.d/$conf_name")
                
                copy_file_with_diff "$conf_file" "$dest_path" "644" "$TEST_MODE"
            fi
        done
    else
        log_warning "No Onload configs found in $SYSTEMCONFIG_DIR/onload"
    fi
}

# Deploy sysctl configurations
deploy_sysctl_configs() {
    if ! is_selected "sysctl"; then
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Deploying Sysctl System Tuning Parameters"
    log_info "═══════════════════════════════════════════════════════════════"
    
    if [[ -d "$SYSTEMCONFIG_DIR/sysctl.d" ]]; then
        for conf_file in "$SYSTEMCONFIG_DIR/sysctl.d"/*.conf; do
            if [[ -f "$conf_file" ]]; then
                local conf_name=$(basename "$conf_file")
                local dest_path=$(get_dest_path "/etc/sysctl.d/$conf_name")
                
                copy_file_with_diff "$conf_file" "$dest_path" "644" "$TEST_MODE"
            fi
        done
    else
        log_warning "No sysctl configs found in $SYSTEMCONFIG_DIR/sysctl.d"
    fi
}

# Deploy udev rules
deploy_udev_rules() {
    if ! is_selected "udev"; then
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Deploying Udev Hardware Rules"
    log_info "═══════════════════════════════════════════════════════════════"
    
    if [[ -d "$SYSTEMCONFIG_DIR/udev/rules.d" ]]; then
        for rules_file in "$SYSTEMCONFIG_DIR/udev/rules.d"/*.rules; do
            if [[ -f "$rules_file" ]]; then
                local rules_name=$(basename "$rules_file")
                local dest_path=$(get_dest_path "/etc/udev/rules.d/$rules_name")
                
                copy_file_with_diff "$rules_file" "$dest_path" "644" "$TEST_MODE"
            fi
        done
    else
        log_warning "No udev rules found in $SYSTEMCONFIG_DIR/udev/rules.d"
    fi
}

# Deploy sudoers files
deploy_sudoers() {
    if ! is_selected "sudoers"; then
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Deploying Sudoers Configuration"
    log_info "═══════════════════════════════════════════════════════════════"
    
    if [[ -d "$SYSTEMCONFIG_DIR/sudoers.d" ]]; then
        for sudoers_file in "$SYSTEMCONFIG_DIR/sudoers.d"/*; do
            # Skip README and backup files
            if [[ -f "$sudoers_file" ]] && [[ ! "$sudoers_file" =~ README ]] && [[ ! "$sudoers_file" =~ \.backup\. ]]; then
                local sudoers_name=$(basename "$sudoers_file")
                local dest_path=$(get_dest_path "/etc/sudoers.d/$sudoers_name")
                
                # Sudoers files must have 0440 permissions
                copy_file_with_diff "$sudoers_file" "$dest_path" "440" "$TEST_MODE"
                
                # Validate sudoers syntax after deployment
                if [[ "$TEST_MODE" != true ]] && [[ "$DRY_RUN" != true ]]; then
                    if ! visudo -c -f "$dest_path" &>/dev/null; then
                        log_error "Sudoers validation failed for $dest_path - removing file"
                        rm -f "$dest_path"
                        return 1
                    else
                        log_success "Sudoers file validated: $sudoers_name"
                    fi
                fi
            fi
        done
    else
        log_warning "No sudoers.d directory found in $SYSTEMCONFIG_DIR/sudoers.d"
    fi
}

# Deploy cron jobs
deploy_cron_jobs() {
    if ! is_selected "cron"; then
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Deploying Cron Jobs"
    log_info "═══════════════════════════════════════════════════════════════"
    
    if [[ -f "$SYSTEMCONFIG_DIR/cron/crontab.txt" ]]; then
        local crontab_file="$SYSTEMCONFIG_DIR/cron/crontab.txt"
        
        if [[ "$TEST_MODE" == true ]]; then
            log_test "Would install cron jobs from: crontab.txt"
            log_info "Current cron jobs in file:"
            cat "$crontab_file" | grep -v '^#' | grep -v '^$' | while read -r line; do
                log_info "  → $line"
            done
            return 0
        elif [[ "$DRY_RUN" == true ]]; then
            log_info "Would install cron jobs from: crontab.txt"
            return 0
        else
            # Get current user's crontab (or empty if none)
            local temp_cron=$(mktemp)
            crontab -l > "$temp_cron" 2>/dev/null || true
            
            # Backup current crontab
            if [[ -s "$temp_cron" ]]; then
                local backup_cron="/tmp/crontab.backup.$(date +%Y%m%d_%H%M%S)"
                cp "$temp_cron" "$backup_cron"
                log_info "Backed up current crontab to: $backup_cron"
            fi
            
            # Append new cron jobs (avoid duplicates)
            local added=0
            while IFS= read -r line; do
                # Skip comments and empty lines
                [[ "$line" =~ ^# ]] && continue
                [[ -z "$line" ]] && continue
                
                # Check if job already exists
                if ! grep -Fxq "$line" "$temp_cron" 2>/dev/null; then
                    echo "$line" >> "$temp_cron"
                    log_success "Added cron job: $line"
                    ((added++))
                else
                    log_info "Cron job already exists: $line"
                fi
            done < "$crontab_file"
            
            if [[ $added -gt 0 ]]; then
                # Install the updated crontab
                crontab "$temp_cron"
                log_success "Installed $added new cron job(s)"
            else
                log_success "All cron jobs already installed"
            fi
            
            rm -f "$temp_cron"
        fi
    else
        log_warning "No cron jobs found in $SYSTEMCONFIG_DIR/cron/crontab.txt"
    fi
}

# Deploy shell/user bash snippet
deploy_shell() {
    if ! is_selected "shell"; then
        return 0
    fi

    echo ""
    log_info "══════════════════════════════════════════════════════════════="
    log_info "Deploying Shell Scripts and Environment"
    log_info "══════════════════════════════════════════════════════════════="

    # Deploy bashrc.trading
    local src="$SYSTEMCONFIG_DIR/shell/bashrc.trading"
    if [[ -f "$src" ]]; then
        # Determine target user/home
        local target_user=""
        if [[ -n "$SUDO_USER" ]]; then
            target_user="$SUDO_USER"
        else
            # try to detect interactive user
            target_user="$(logname 2>/dev/null || echo "$USER")"
        fi

        if [[ -z "$target_user" ]]; then
            log_warning "Could not determine target user for shell deployment. You may run the script without sudo to deploy to current user."
        else
            local target_home
            target_home=$(eval echo "~$target_user")
            local dest="$target_home/.bashrc.trading"
            local bashrc_file="$target_home/.bashrc"

            if [[ "$TEST_MODE" == true ]]; then
                log_test "Would copy: $src -> $dest"
                log_test "Would ensure: source ~/.bashrc.trading is present in $bashrc_file (for user: $target_user)"
            elif [[ ! -d "$target_home" ]]; then
                log_error "Target home directory does not exist: $target_home"
            else
                # Backup and copy
                backup_existing "$dest"
                cp "$src" "$dest"
                chown "$target_user":"$target_user" "$dest" || true
                chmod 644 "$dest" || true
                log_success "Deployed shell snippet to: $dest"

                # Ensure the user's ~/.bashrc sources the snippet (avoid duplicates)
                if [[ ! -f "$bashrc_file" ]]; then
                    touch "$bashrc_file" || true
                    chown "$target_user":"$target_user" "$bashrc_file" || true
                fi

                local source_line='[ -f "$HOME/.bashrc.trading" ] && source "$HOME/.bashrc.trading"'
                # Write a literal source line into the user's bashrc if missing
                if ! grep -Fxq '[ -f "$HOME/.bashrc.trading" ] && source "$HOME/.bashrc.trading"' "$bashrc_file" 2>/dev/null; then
                    echo "" >> "$bashrc_file" || true
                    echo "# AI Trading Station: source trading shell snippet" >> "$bashrc_file" || true
                    echo '[ -f "$HOME/.bashrc.trading" ] && source "$HOME/.bashrc.trading"' >> "$bashrc_file" || true
                    chown "$target_user":"$target_user" "$bashrc_file" || true
                    log_success "Appended source line to $bashrc_file"
                else
                    log_info "User bashrc already sources .bashrc.trading"
                fi
            fi
        fi
    else
        log_warning "No shell snippet found at: $src"
    fi

    # Deploy vm-manager.sh to /usr/local/bin/vm-manager
    local vm_src="$SYSTEMCONFIG_DIR/shell/vm-manager.sh"
    if [[ -f "$vm_src" ]]; then
        local vm_dest="/usr/local/bin/vm-manager"
        
        if [[ "$TEST_MODE" == true ]]; then
            local test_dest="$TEST_DIR$vm_dest"
            mkdir -p "$(dirname "$test_dest")"
            log_test "Would copy: $vm_src -> $vm_dest"
            cp "$vm_src" "$test_dest"
            chmod +x "$test_dest"
        elif [[ "$DRY_RUN" == true ]]; then
            log_info "Would deploy VM manager script: vm-manager.sh → /usr/local/bin/vm-manager"
        else
            backup_existing "$vm_dest"
            cp "$vm_src" "$vm_dest"
            chmod +x "$vm_dest"
            log_success "Deployed VM manager: $vm_dest"
            log_info "VM can be managed with: vm-manager {start|stop|status|console|ssh}"
        fi
    else
        log_warning "No VM manager script found at: $vm_src"
    fi

    # Deploy datafeed.sh to /usr/local/bin/datafeed
    local datafeed_src="$SYSTEMCONFIG_DIR/shell/datafeed.sh"
    if [[ -f "$datafeed_src" ]]; then
        local datafeed_dest="/usr/local/bin/datafeed"
        
        if [[ "$TEST_MODE" == true ]]; then
            local test_dest="$TEST_DIR$datafeed_dest"
            mkdir -p "$(dirname "$test_dest")"
            log_test "Would copy: $datafeed_src -> $datafeed_dest"
            cp "$datafeed_src" "$test_dest"
            chmod +x "$test_dest"
        elif [[ "$DRY_RUN" == true ]]; then
            log_info "Would deploy datafeed script: datafeed.sh → /usr/local/bin/datafeed"
        else
            backup_existing "$datafeed_dest"
            cp "$datafeed_src" "$datafeed_dest"
            chmod +x "$datafeed_dest"
            log_success "Deployed datafeed manager: $datafeed_dest"
            log_info "Data feeds can be managed with: datafeed {start|stop|status|logs|health|metrics}"
        fi
    else
        log_warning "No datafeed script found at: $datafeed_src"
    fi

    return 0
}

# Deploy KVM modules configuration
deploy_kvm() {
    if ! is_selected "kvm"; then
        return 0
    fi

    echo ""
    log_info "══════════════════════════════════════════════════════════════="
    log_info "Deploying KVM Virtualization Modules"
    log_info "══════════════════════════════════════════════════════════════="

    local src_dir="$SYSTEMCONFIG_DIR/kernel/modules-load.d"
    local dest_dir="/etc/modules-load.d"

    if [[ ! -d "$src_dir" ]]; then
        log_warning "No KVM modules-load.d directory found at: $src_dir"
        return 0
    fi

    if [[ "$TEST_MODE" == true ]]; then
        local test_dest="$TEST_DIR$dest_dir"
        mkdir -p "$test_dest"
        log_test "Would deploy KVM modules to: $dest_dir"
        
        for conf in "$src_dir"/*.conf; do
            [[ -e "$conf" ]] || continue
            local fname=$(basename "$conf")
            log_test "Would copy: $conf -> $dest_dir/$fname"
            cp "$conf" "$test_dest/$fname"
        done
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would deploy KVM modules configuration files:"
        for conf in "$src_dir"/*.conf; do
            [[ -e "$conf" ]] || continue
            local fname=$(basename "$conf")
            log_info "  → $fname"
        done
        return 0
    fi

    # Deploy each .conf file
    local deployed=0
    for conf in "$src_dir"/*.conf; do
        [[ -e "$conf" ]] || continue
        local fname=$(basename "$conf")
        local dest="$dest_dir/$fname"
        
        backup_existing "$dest"
        cp "$conf" "$dest"
        chmod 644 "$dest"
        log_success "Deployed: $fname → $dest_dir/"
        ((deployed++))
    done

    if [[ $deployed -eq 0 ]]; then
        log_warning "No KVM module configuration files found"
    else
        log_info "KVM modules will be loaded automatically on next boot"
        log_info "To load now: sudo modprobe kvm && sudo modprobe kvm_intel"
        log_info "Note: User must be in 'kvm' group (sudo usermod -aG kvm username)"
    fi

    return 0
}

# Deploy logrotate configurations
deploy_logrotate() {
    if ! is_selected "logrotate"; then
        return 0
    fi
    
    echo ""
    log_info "══════════════════════════════════════════════════════════════="
    log_info "Deploying Logrotate Configurations"
    log_info "══════════════════════════════════════════════════════════════="
    
    if [[ -d "$SYSTEMCONFIG_DIR/logrotate.d" ]]; then
        for logrotate_file in "$SYSTEMCONFIG_DIR/logrotate.d"/*; do
            # Skip README and backup files
            if [[ -f "$logrotate_file" ]] && [[ ! "$logrotate_file" =~ README ]] && [[ ! "$logrotate_file" =~ \.backup\. ]]; then
                local logrotate_name=$(basename "$logrotate_file")
                local dest_path=$(get_dest_path "/etc/logrotate.d/$logrotate_name")
                
                copy_file_with_diff "$logrotate_file" "$dest_path" "644" "$TEST_MODE"
                
                # Test logrotate configuration syntax after deployment
                if [[ "$TEST_MODE" != true ]] && [[ "$DRY_RUN" != true ]]; then
                    if logrotate -d "$dest_path" &>/dev/null; then
                        log_success "Logrotate configuration validated: $logrotate_name"
                    else
                        log_warning "Logrotate syntax check had warnings for: $logrotate_name (may still work)"
                    fi
                fi
            fi
        done
    else
        log_warning "No logrotate.d directory found in $SYSTEMCONFIG_DIR/logrotate.d"
    fi
}

# Reload system services
reload_services() {
    if [[ "$TEST_MODE" == true ]]; then
        echo ""
        log_test "Would reload system services:"
        log_test "  → systemctl daemon-reload"
        log_test "  → sysctl --system"
        log_test "  → udevadm control --reload-rules && udevadm trigger"
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Reloading System Services"
    log_info "═══════════════════════════════════════════════════════════════"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would reload systemd, sysctl, and udev"
        return 0
    fi
    
    # Reload systemd
    systemctl daemon-reload
    log_success "SystemD daemon reloaded"
    
    # Apply sysctl settings - only if we deployed sysctl configs
    if is_selected "sysctl" && [[ -d "$SYSTEMCONFIG_DIR/sysctl.d" ]]; then
        if sysctl --system &>/dev/null; then
            log_success "Sysctl settings applied"
        else
            log_warning "Some sysctl settings may require reboot or have errors"
        fi
    fi
    
    # Reload udev rules
    if is_selected "udev" && [[ -d "$SYSTEMCONFIG_DIR/udev/rules.d" ]]; then
        udevadm control --reload-rules
        udevadm trigger
        log_success "Udev rules reloaded"
    fi
}

# Validate deployment
validate_deployment() {
    if [[ "$TEST_MODE" == true ]]; then
        echo ""
        log_test "Validation skipped in test mode"
        return 0
    fi
    
    echo ""
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Validating Deployment"
    log_info "═══════════════════════════════════════════════════════════════"
    
    local issues=0
    
    if is_selected "systemd-services"; then
        # Check critical services
        local critical_services=("redis-hft" "configure-nic-irq-affinity" "ultra-low-latency-nic")
        for service in "${critical_services[@]}"; do
            if systemctl list-unit-files "${service}.service" &>/dev/null; then
                log_success "Infrastructure service available: $service"
            else
                log_warning "Infrastructure service not found: $service"
                ((issues++))
            fi
        done
        
        # Check trading services
        local trading_services=("binance-trades" "binance-bookticker" "batch-writer" "questdb")
        for service in "${trading_services[@]}"; do
            if systemctl list-unit-files "${service}.service" &>/dev/null; then
                log_success "Trading service available: $service"
            else
                log_warning "Trading service not found: $service"
                ((issues++))
            fi
        done
    fi
    
    echo ""
    if [[ $issues -eq 0 ]]; then
        log_success "All validations passed! ✓"
    else
        log_warning "$issues potential issues found - review warnings above"
    fi
}

# Show deployment summary
show_summary() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    if [[ "$TEST_MODE" == true ]]; then
        echo "          AI TRADING STATION DEPLOYMENT TEST COMPLETE"
        echo "════════════════════════════════════════════════════════════════════"
        echo ""
        echo "🧪 Test Mode Summary:"
        echo "   • No files were modified on the system"
        echo "   • All changes written to: $TEST_DIR"
        echo "   • Diffs shown for changed files"
        echo ""
        echo "📂 Test Results:"
        if [[ -d "$TEST_DIR" ]]; then
            echo "   • Review test deployment: ls -lR $TEST_DIR"
            echo "   • Compare files: diff -r $TEST_DIR/etc /etc"
        fi
        echo ""
        echo "✅ Next Steps:"
        echo "   1. Review diffs above for correctness"
        echo "   2. If satisfied, run WITHOUT --test flag"
        echo "   3. Command: sudo $0 [--select] [components...]"
    elif [[ "$DRY_RUN" == true ]]; then
        echo "          AI TRADING STATION DEPLOYMENT DRY-RUN COMPLETE"
        echo "════════════════════════════════════════════════════════════════════"
        echo ""
        echo "📋 Dry-run completed - no changes made"
        echo ""
        echo "✅ To deploy for real:"
        echo "   sudo $0 [--select] [components...]"
    else
        echo "          AI TRADING STATION DEPLOYMENT COMPLETE"
        echo "════════════════════════════════════════════════════════════════════"
        echo ""
        if [[ "$FILE_MODE" == true ]]; then
            echo "📁 Deployed Items:"
            for i in "${SELECTED_ITEMS[@]}"; do echo "   ✓ $i"; done
        else
            echo "📁 Deployed Components:"
            for component in "${SELECTED_COMPONENTS[@]}"; do
                echo "   ✓ $component"
            done
        fi
        echo ""
        echo "🔄 Next Steps:"
        echo "   1. Reload services: systemctl daemon-reload"
        echo "   2. Restart services: systemctl restart redis-hft"
        echo "   3. Check status: systemctl status redis-hft"
        echo "   4. View logs: journalctl -u redis-hft -f"
        echo ""
        echo "📋 Backup files created with .backup.<timestamp> suffixes"
    fi
    echo "════════════════════════════════════════════════════════════════════"
}

# Show help
show_help() {
    cat <<EOF
AI Trading Station System Configuration Deployment Script - Enhanced

Usage: sudo $0 [OPTIONS] [COMPONENTS...]

OPTIONS:
  -h, --help              Show this help message
  -t, --test              Test mode - deploy to temp directory, show diffs
  -d, --dry-run           Dry-run - show what would be deployed
  -s, --select            Interactive component selection
  -i, --interactive       Alias for --select
    --file <path>           Deploy a single file under SystemConfig (relative or absolute)
    --service <name>        Deploy a single systemd service by name (e.g., binance-trades)
    --network-script <name> Deploy a single network script by name (with or without .sh)

COMPONENTS (use with or without --select):
  systemd-services        SystemD service definitions
  systemd-overrides       CPU affinity and resource limits  
  kernel                  Kernel module configurations
  network                 Network optimization scripts
  redis                   Redis HFT configuration
  onload                  Solarflare Onload configuration
  sysctl                  System tuning parameters
  udev                    Hardware device rules
  sudoers                 Passwordless sudo configurations
  kvm                     KVM virtualization modules
  logrotate               Log rotation configurations
  cron                    Scheduled jobs (crontab)
  shell                   Shell environment (bashrc.trading)

EXAMPLES:
  # Full deployment (production)
  sudo $0

  # Test mode - safe testing without touching production
  $0 --test

  # Interactive component selection
  sudo $0 --select

  # Deploy specific components only
  sudo $0 systemd-services network

  # Test specific components
  $0 --test network redis

  # Dry-run with interactive selection
  $0 --dry-run --select

    # Deploy a single service (production)
    sudo $0 --service binance-trades

    # Test a single file from SystemConfig
    $0 --test --file systemd/services/redis-hft.service

    # Test a single network script
    $0 --test --network-script configure-nic-irq-affinity

MODES:
  • Default: Full deployment to system locations (requires root)
  • --test: Deploy to /tmp/systemconfig-test-*, show diffs (no root needed)
  • --dry-run: Show what would be deployed without changes (requires root)
  • --select: Interactively choose components to deploy
    • --file/--service/--network-script: Deploy specific items only

TESTING WORKFLOW:
  1. Test first:     $0 --test
  2. Review diffs and test directory
  3. Deploy:         sudo $0
  
SELECTIVE DEPLOYMENT:
  1. Interactive:    sudo $0 --select
  2. Direct:         sudo $0 network redis sysctl

EOF
}

# Main execution function
main() {
    # Setup test mode if enabled
    if [[ "$TEST_MODE" == true ]]; then
        TEST_DIR=$(mktemp -d /tmp/systemconfig-test-XXXXXX)
        log_test "Test mode enabled - deploying to: $TEST_DIR"
        log_test "Production files will NOT be modified"
        echo ""
    fi
    
    echo "════════════════════════════════════════════════════════════════════"
    if [[ "$TEST_MODE" == true ]]; then
        echo "        AI TRADING STATION DEPLOYMENT - TEST MODE"
    elif [[ "$DRY_RUN" == true ]]; then
        echo "        AI TRADING STATION DEPLOYMENT - DRY RUN"
    else
        echo "        AI TRADING STATION DEPLOYMENT"
    fi
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    
    check_root
    
    log_info "Source directory: $SYSTEMCONFIG_DIR"
    if [[ "$TEST_MODE" == true ]]; then
        log_info "Test directory: $TEST_DIR"
    fi
    echo ""
    
    # Show what will be deployed
    if [[ "$FILE_MODE" == true ]]; then
        log_select "Deploying specific items: ${SELECTED_ITEMS[*]}"
        echo ""
        deploy_specific_files
    else
        if [[ "$SELECTIVE_MODE" == true ]] && [[ ${#SELECTED_COMPONENTS[@]} -gt 0 ]]; then
            log_select "Deploying selected components: ${SELECTED_COMPONENTS[*]}"
            echo ""
        fi
        # Deploy all selected components
        deploy_systemd_services
        deploy_systemd_overrides
        deploy_kernel_configs
        deploy_network_scripts
        deploy_redis_configs
        deploy_onload_configs
        deploy_sysctl_configs
        deploy_udev_rules
        deploy_sudoers
        deploy_kvm
        deploy_logrotate
        deploy_cron_jobs
        deploy_shell
    fi
    
    # Reload and validate
    reload_services
    validate_deployment
    
    # Show summary
    show_summary
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--test)
                TEST_MODE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -s|--select|-i|--interactive)
                SELECTIVE_MODE=true
                INTERACTIVE_MODE=true
                shift
                ;;
            --file)
                shift
                if [[ -z "$1" ]]; then log_error "--file requires a path"; exit 1; fi
                FILE_MODE=true
                SELECTED_ITEMS+=("file:$1")
                shift
                ;;
            --service)
                shift
                if [[ -z "$1" ]]; then log_error "--service requires a name"; exit 1; fi
                FILE_MODE=true
                SELECTED_ITEMS+=("service:$1")
                shift
                ;;
            --network-script)
                shift
                if [[ -z "$1" ]]; then log_error "--network-script requires a name"; exit 1; fi
                FILE_MODE=true
                SELECTED_ITEMS+=("network:$1")
                shift
                ;;
            systemd-services|systemd-overrides|kernel|network|redis|onload|sysctl|udev|sudoers|kvm|logrotate|cron|shell)
                SELECTIVE_MODE=true
                SELECTED_COMPONENTS+=("$1")
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # If selective mode but no components selected, go interactive
    if [[ "$SELECTIVE_MODE" == true ]] && [[ ${#SELECTED_COMPONENTS[@]} -eq 0 ]]; then
        INTERACTIVE_MODE=true
    fi
    
    # Interactive selection if requested
    if [[ "$INTERACTIVE_MODE" == true ]]; then
        interactive_selection
    fi
}

# Entry point
parse_arguments "$@"
main
