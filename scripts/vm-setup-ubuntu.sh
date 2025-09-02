#!/bin/bash
# ============================================================================
# Ubuntu 22.04 LTS VM Setup Script with Workspace Mounting
# ============================================================================
# PURPOSE: Automated VM creation and configuration for AI Trading Station
#          development environment with GitHub Copilot integration and
#          complete workspace mounting functionality
# ============================================================================

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly CONFIG_FILE="$PROJECT_ROOT/configs/vm-dev-setup.json"
readonly LOG_FILE="${LOG_FILE:-$HOME/.vm-setup.log}"

# VM Configuration
readonly UBUNTU_VERSION="22.04"
readonly VM_DISK_SIZE="50G"
readonly VM_MEMORY="8192"
readonly VM_CPUS="4"
readonly BRIDGE_NAME="virbr0"

# Workspace mounting configuration
readonly HOST_WORKSPACE_PATH="$PROJECT_ROOT"
readonly VM_WORKSPACE_PATH="/workspace/ai-trading-station"
readonly SHARED_FOLDER_TAG="ai-trading-workspace"

# Download URLs
readonly UBUNTU_ISO_URL="https://releases.ubuntu.com/22.04/ubuntu-22.04.3-desktop-amd64.iso"
readonly UBUNTU_ISO_NAME="ubuntu-22.04.3-desktop-amd64.iso"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

info() {
    echo -e "\033[0;32m[INFO]\033[0m $*"
}

warn() {
    echo -e "\033[1;33m[WARN]\033[0m $*"
}

error() {
    echo -e "\033[0;31m[ERROR]\033[0m $*" >&2
}

check_prerequisites() {
    info "Checking prerequisites for VM setup..."
    
    local required_commands=("qemu-img" "virt-install" "virsh" "wget")
    local missing_commands=()
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_commands+=("$cmd")
        fi
    done
    
    if [[ ${#missing_commands[@]} -gt 0 ]]; then
        error "Missing required commands: ${missing_commands[*]}"
        info "Install with: sudo apt update && sudo apt install qemu-kvm libvirt-daemon-system libvirt-clients virtinst virt-manager wget"
        return 1
    fi
    
    # Check if user is in libvirt group
    if ! groups | grep -q libvirt; then
        warn "User not in libvirt group. Add with: sudo usermod -aG libvirt $USER"
        info "You may need to log out and back in for changes to take effect"
    fi
    
    # Ensure libvirtd is running
    if ! systemctl is-active --quiet libvirtd; then
        info "Starting libvirtd service..."
        sudo systemctl start libvirtd
        sudo systemctl enable libvirtd
    fi
    
    info "✓ Prerequisites check passed"
}

create_vm_directory() {
    local vm_name="$1"
    local vm_dir="$HOME/vm-images/$vm_name"
    
    if [[ ! -d "$vm_dir" ]]; then
        info "Creating VM directory: $vm_dir"
        mkdir -p "$vm_dir"
    fi
    
    echo "$vm_dir"
}

download_ubuntu_iso() {
    local vm_dir="$1"
    local iso_path="$vm_dir/$UBUNTU_ISO_NAME"
    
    if [[ -f "$iso_path" ]]; then
        info "Ubuntu ISO already exists: $iso_path"
        echo "$iso_path"
        return 0
    fi
    
    info "Downloading Ubuntu 22.04 ISO..."
    info "This may take several minutes depending on your internet connection."
    
    if ! wget -O "$iso_path" "$UBUNTU_ISO_URL"; then
        error "Failed to download Ubuntu ISO"
        rm -f "$iso_path"
        return 1
    fi
    
    info "✓ Ubuntu ISO downloaded: $iso_path"
    echo "$iso_path"
}

create_vm_disk() {
    local vm_name="$1"
    local vm_dir="$2"
    local disk_path="$vm_dir/$vm_name.qcow2"
    
    if [[ -f "$disk_path" ]]; then
        warn "VM disk already exists: $disk_path"
        read -p "Overwrite existing disk? [y/N]: " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$disk_path"
        else
            info "Using existing disk"
            echo "$disk_path"
            return 0
        fi
    fi
    
    info "Creating VM disk: $disk_path (${VM_DISK_SIZE})"
    qemu-img create -f qcow2 "$disk_path" "$VM_DISK_SIZE"
    
    echo "$disk_path"
}

create_cloud_init_config() {
    local vm_dir="$1"
    local user_data="$vm_dir/user-data"
    local meta_data="$vm_dir/meta-data"
    
    # Create user-data for automated installation with workspace mounting support
    cat > "$user_data" << EOF
#cloud-config
autoinstall:
  version: 1
  locale: en_US.UTF-8
  keyboard:
    layout: us
  ssh:
    install-server: true
    allow-pw: true
  packages:
    - ubuntu-desktop-minimal
    - openssh-server
    - curl
    - wget
    - git
    - vim
    - htop
    - build-essential
    - python3-pip
    - nodejs
    - npm
    - linux-modules-extra-\$(uname -r)
  user-data:
    users:
      - name: developer
        groups: [adm, sudo]
        lock_passwd: false
        passwd: '\$6\$rounds=4096\$aQ7CNhGhXOQPe\$l0E5KDSY0YdMvO7CEh3BUBjHkG6PVNHmFVZ4wJnWYmf7KdkOg1PzC7wPGhxu2OKLaBu4wPQGp3KzKJ4J5K6zg1'
        shell: /bin/bash
        sudo: ALL=(ALL) NOPASSWD:ALL
        ssh_authorized_keys:
          - ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC7... # Placeholder
  late-commands:
    - 'echo "developer:developer" | chpasswd'
    - 'systemctl enable ssh'
    - 'mkdir -p $VM_WORKSPACE_PATH'
    - 'chown developer:developer $VM_WORKSPACE_PATH'
    - 'echo "9pnet_virtio" >> /etc/modules'
EOF

    # Create meta-data
    cat > "$meta_data" << EOF
instance-id: ai-trading-dev-vm-001
local-hostname: ai-trading-dev
EOF

    info "✓ Cloud-init configuration created with workspace mounting support"
}

install_vm_with_workspace() {
    local vm_name="$1"
    local iso_path="$2"
    local disk_path="$3"
    
    info "Installing VM with workspace mounting capability: $vm_name"
    info "This process will take 20-30 minutes..."
    
    # Create VM with virt-install including filesystem device for workspace mounting
    virt-install \
        --name="$vm_name" \
        --memory="$VM_MEMORY" \
        --vcpus="$VM_CPUS" \
        --disk path="$disk_path",format=qcow2,size=50 \
        --cdrom="$iso_path" \
        --os-variant=ubuntu22.04 \
        --network bridge="$BRIDGE_NAME" \
        --graphics vnc,listen=0.0.0.0,port=5901 \
        --console pty,target_type=serial \
        --extra-args="console=ttyS0,115200n8" \
        --filesystem "$HOST_WORKSPACE_PATH,$SHARED_FOLDER_TAG,type=mount,accessmode=passthrough" \
        --noautoconsole \
        --wait=-1
    
    info "✓ VM installation completed with workspace mounting capability"
}

setup_vm_post_install() {
    local vm_name="$1"
    
    info "Configuring VM post-installation..."
    
    # Wait for VM to be accessible
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if virsh domstate "$vm_name" 2>/dev/null | grep -q "running"; then
            break
        fi
        
        sleep 10
        ((attempt++))
        
        if [[ $attempt -eq $max_attempts ]]; then
            error "VM failed to start properly"
            return 1
        fi
    done
    
    # Wait a bit more for the system to fully boot
    info "Waiting for VM system to fully initialize..."
    sleep 30
    
    info "✓ VM post-installation setup completed"
}

create_setup_scripts() {
    local vm_dir="$1"
    local vm_name="$2"
    
    # Create GitHub Copilot setup script with workspace mounting instructions
    cat > "$vm_dir/setup-copilot.sh" << 'EOF'
#!/bin/bash
# GitHub Copilot setup script for AI Trading Station development VM
# Includes workspace mounting and project setup

set -euo pipefail

info() {
    echo -e "\033[0;32m[INFO]\033[0m $*"
}

error() {
    echo -e "\033[0;31m[ERROR]\033[0m $*" >&2
}

# Setup workspace mounting
setup_workspace() {
    info "Setting up AI Trading Station workspace..."
    
    # Check if workspace is already mounted
    if mountpoint -q "/workspace/ai-trading-station"; then
        info "✓ Workspace already mounted at /workspace/ai-trading-station"
    else
        # Ensure mount point exists
        sudo mkdir -p /workspace/ai-trading-station
        
        # Try to mount the workspace
        if sudo mount -t 9p -o trans=virtio,version=9p2000.L ai-trading-workspace /workspace/ai-trading-station; then
            info "✓ Workspace mounted successfully"
        else
            error "Failed to mount workspace. Check if VM was configured with workspace mounting."
            return 1
        fi
    fi
    
    # Create symbolic link in home directory
    if [[ ! -L "$HOME/ai-trading-station" ]]; then
        ln -sf /workspace/ai-trading-station "$HOME/ai-trading-station"
        info "✓ Created symbolic link at $HOME/ai-trading-station"
    fi
    
    # Set proper ownership
    sudo chown -R developer:developer /workspace/ai-trading-station 2>/dev/null || true
    
    info "✓ Workspace setup complete!"
    info "  • Complete AI Trading Station project available at /workspace/ai-trading-station"
    info "  • Convenient link available at $HOME/ai-trading-station"
    
    # Show workspace contents
    if [[ -d "/workspace/ai-trading-station" ]]; then
        echo
        info "Available project files:"
        ls -la /workspace/ai-trading-station | head -10
    fi
}

# Install VS Code
install_vscode() {
    info "Installing Visual Studio Code..."
    
    # Add Microsoft GPG key
    wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
    sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
    
    # Add VS Code repository
    echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list
    
    # Update package list and install
    sudo apt update
    sudo apt install -y code
    
    info "✓ VS Code installed"
}

# Install GitHub Copilot extension
install_copilot_extension() {
    info "Installing GitHub Copilot extension..."
    
    code --install-extension GitHub.copilot
    code --install-extension GitHub.copilot-chat
    
    info "✓ GitHub Copilot extensions installed"
    info "Please authenticate with GitHub Copilot in VS Code"
}

# Setup development environment
setup_development_env() {
    info "Setting up development environment..."
    
    # Install additional development tools
    sudo apt update
    sudo apt install -y \
        git \
        build-essential \
        cmake \
        gdb \
        valgrind \
        htop \
        curl \
        wget \
        vim \
        nano \
        python3-pip \
        nodejs \
        npm \
        docker.io \
        docker-compose
    
    # Add user to docker group
    sudo usermod -aG docker "$USER"
    
    # Install GitHub CLI
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list
    sudo apt update
    sudo apt install -y gh
    
    info "✓ Development environment configured"
}

# Setup AI Trading Station development environment
setup_trading_station_env() {
    info "Configuring AI Trading Station development environment..."
    
    # Change to workspace directory
    cd /workspace/ai-trading-station || {
        error "Workspace not available. Ensure VM workspace mounting is configured."
        return 1
    }
    
    # Make scripts executable
    if [[ -d "scripts" ]]; then
        chmod +x scripts/*
        info "✓ Made scripts executable"
    fi
    
    # Show current status
    if [[ -x "ai-trading-station.sh" ]]; then
        info "Testing AI Trading Station status..."
        ./ai-trading-station.sh status || true
    fi
    
    info "✓ AI Trading Station environment configured"
}

main() {
    info "Setting up AI Trading Station development environment with workspace..."
    
    setup_workspace
    install_vscode
    install_copilot_extension
    setup_development_env
    setup_trading_station_env
    
    info "✓ Development environment setup complete!"
    echo
    info "===================================================="
    info "AI Trading Station VM Development Environment Ready!"
    info "===================================================="
    echo
    info "Workspace Access:"
    info "  • Full project: /workspace/ai-trading-station"
    info "  • Home link: $HOME/ai-trading-station"
    echo
    info "Next steps:"
    info "1. Authenticate with GitHub: gh auth login"
    info "2. Open VS Code: code /workspace/ai-trading-station"
    info "3. Sign in to GitHub Copilot in VS Code"
    info "4. Start developing with full project access!"
    echo
    info "Development Commands:"
    info "  cd /workspace/ai-trading-station"
    info "  ./ai-trading-station.sh status      # Check system"
    info "  ./ai-trading-station.sh demo        # Run demo"
    info "  code .                              # Open in VS Code"
}

main "$@"
EOF

    chmod +x "$vm_dir/setup-copilot.sh"
    
    # Create VM connection script with workspace information
    cat > "$vm_dir/connect-vm.sh" << EOF
#!/bin/bash
# Connect to AI Trading Station development VM

VM_NAME="$vm_name"

# Check VM status
status=\$(virsh domstate "\$VM_NAME" 2>/dev/null || echo "undefined")

if [[ "\$status" == "undefined" ]]; then
    echo "Error: VM '\$VM_NAME' not found"
    exit 1
elif [[ "\$status" == "shut off" ]]; then
    echo "Starting VM '\$VM_NAME'..."
    virsh start "\$VM_NAME"
    echo "Waiting for VM to boot..."
    sleep 30
fi

# Get VM IP address
VM_IP=\$(virsh domifaddr "\$VM_NAME" | grep -oP '192\.168\.\d+\.\d+' | head -1)

if [[ -n "\$VM_IP" ]]; then
    echo "=============================================="
    echo "AI Trading Station Development VM"
    echo "=============================================="
    echo "VM IP: \$VM_IP"
    echo "Workspace: /workspace/ai-trading-station"
    echo "Home link: ~/ai-trading-station"
    echo "=============================================="
    echo
    echo "Connecting to VM..."
    ssh developer@\$VM_IP
else
    echo "Could not determine VM IP address"
    echo "Try connecting via VNC: vncviewer localhost:5901"
fi
EOF

    chmod +x "$vm_dir/connect-vm.sh"
    
    info "✓ Setup scripts created in $vm_dir with workspace mounting support"
}

show_completion_message() {
    local vm_name="$1"
    local vm_dir="$2"
    
    cat << EOF

╔════════════════════════════════════════════════════════════════╗
║              VM Setup Completed with Workspace Mounting!      ║
╚════════════════════════════════════════════════════════════════╝

VM Name: $vm_name
VM Directory: $vm_dir
VNC Console: localhost:5901 (password may be required)

WORKSPACE MOUNTING CONFIGURATION:
  Host Project Path:  $HOST_WORKSPACE_PATH
  VM Mount Point:     $VM_WORKSPACE_PATH
  Shared Folder Tag:  $SHARED_FOLDER_TAG

The COMPLETE AI Trading Station project directory is now accessible 
inside the VM for full development work!

Next Steps:
1. Start the VM:
   virsh start $vm_name

2. Connect to the VM:
   $vm_dir/connect-vm.sh

3. Setup development environment inside VM:
   ./setup-copilot.sh

4. Access the complete project workspace:
   cd /workspace/ai-trading-station
   ls -la  # See all project files

5. Start development:
   code /workspace/ai-trading-station

6. Access VM via VNC if needed:
   vncviewer localhost:5901

7. Switch modes using vm-dev-environment:
   ./scripts/vm-dev-environment start    # Development mode
   ./scripts/vm-dev-environment production  # Production mode

The VM is configured with:
- Ubuntu 22.04 LTS Desktop
- User: developer / Password: developer
- SSH enabled on port 22
- VNC enabled on port 5901
- COMPLETE workspace mounted at $VM_WORKSPACE_PATH
- Ready for GitHub Copilot integration

IMPORTANT - WORKSPACE ACCESS:
✓ All project files (scripts, configs, modules) available in VM
✓ Changes made in VM are reflected on host and vice versa
✓ Complete development environment with full project access
✓ No need to clone repository - it's already mounted!

Production Performance:
- VM can be completely disabled for zero-overhead production
- scripts/onload-trading still delivers 4.37μs latency when needed

EOF
}

main() {
    local vm_name="${1:-ai-trading-dev-vm}"
    
    log "Starting VM setup with workspace mounting for: $vm_name"
    
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    # Validate host workspace path
    if [[ ! -d "$HOST_WORKSPACE_PATH" ]]; then
        error "Host workspace path does not exist: $HOST_WORKSPACE_PATH"
        exit 1
    fi
    
    info "Host workspace path validated: $HOST_WORKSPACE_PATH"
    
    # Create VM directory
    local vm_dir
    vm_dir=$(create_vm_directory "$vm_name")
    
    # Download Ubuntu ISO
    info "Preparing Ubuntu 22.04 installation..."
    local iso_path
    iso_path=$(download_ubuntu_iso "$vm_dir")
    
    # Create VM disk
    local disk_path
    disk_path=$(create_vm_disk "$vm_name" "$vm_dir")
    
    # Create cloud-init configuration
    create_cloud_init_config "$vm_dir"
    
    # Install VM with workspace mounting
    install_vm_with_workspace "$vm_name" "$iso_path" "$disk_path"
    
    # Post-installation setup
    setup_vm_post_install "$vm_name"
    
    # Create helper scripts
    create_setup_scripts "$vm_dir" "$vm_name"
    
    # Show completion message
    show_completion_message "$vm_name" "$vm_dir"
    
    log "VM setup with workspace mounting completed successfully"
}

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi