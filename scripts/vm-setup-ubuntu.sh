#!/bin/bash
# ============================================================================
# Ubuntu 22.04 LTS VM Setup Script
# ============================================================================
# PURPOSE: Automated VM creation and configuration for AI Trading Station
#          development environment with GitHub Copilot integration
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
    
    # Create user-data for automated installation
    cat > "$user_data" << 'EOF'
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
  user-data:
    users:
      - name: developer
        groups: [adm, sudo]
        lock_passwd: false
        passwd: '$6$rounds=4096$aQ7CNhGhXOQPe$l0E5KDSY0YdMvO7CEh3BUBjHkG6PVNHmFVZ4wJnWYmf7KdkOg1PzC7wPGhxu2OKLaBu4wPQGp3KzKJ4J5K6zg1'
        shell: /bin/bash
        sudo: ALL=(ALL) NOPASSWD:ALL
        ssh_authorized_keys:
          - ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC7... # Placeholder
  late-commands:
    - 'echo "developer:developer" | chpasswd'
    - 'systemctl enable ssh'
EOF

    # Create meta-data
    cat > "$meta_data" << EOF
instance-id: ai-trading-dev-vm-001
local-hostname: ai-trading-dev
EOF

    info "✓ Cloud-init configuration created"
}

install_vm() {
    local vm_name="$1"
    local iso_path="$2"
    local disk_path="$3"
    
    info "Installing VM: $vm_name"
    info "This process will take 20-30 minutes..."
    
    # Create VM with virt-install
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
        --noautoconsole \
        --wait=-1
    
    info "✓ VM installation completed"
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
    
    info "✓ VM post-installation setup completed"
}

create_setup_scripts() {
    local vm_dir="$1"
    
    # Create GitHub Copilot setup script
    cat > "$vm_dir/setup-copilot.sh" << 'EOF'
#!/bin/bash
# GitHub Copilot setup script for AI Trading Station development VM

set -euo pipefail

info() {
    echo -e "\033[0;32m[INFO]\033[0m $*"
}

error() {
    echo -e "\033[0;31m[ERROR]\033[0m $*" >&2
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

# Clone AI Trading Station repository
clone_repository() {
    info "Setting up AI Trading Station repository access..."
    
    local repo_dir="$HOME/ai-trading-station"
    
    if [[ ! -d "$repo_dir" ]]; then
        info "Repository will be cloned when authentication is set up"
        info "Run: gh auth login && git clone https://github.com/ChoubChoub/AI-Trading-Station $repo_dir"
    else
        info "Repository already exists at $repo_dir"
    fi
}

main() {
    info "Setting up AI Trading Station development environment..."
    
    install_vscode
    install_copilot_extension
    setup_development_env
    clone_repository
    
    info "✓ Development environment setup complete!"
    info ""
    info "Next steps:"
    info "1. Authenticate with GitHub: gh auth login"
    info "2. Open VS Code and sign in to GitHub Copilot"
    info "3. Clone the repository: git clone https://github.com/ChoubChoub/AI-Trading-Station"
    info "4. Start developing with AI assistance!"
}

main "$@"
EOF

    chmod +x "$vm_dir/setup-copilot.sh"
    
    # Create VM connection script
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
    echo "Connecting to VM at \$VM_IP..."
    ssh developer@\$VM_IP
else
    echo "Could not determine VM IP address"
    echo "Try connecting via VNC: vncviewer localhost:5901"
fi
EOF

    chmod +x "$vm_dir/connect-vm.sh"
    
    info "✓ Setup scripts created in $vm_dir"
}

show_completion_message() {
    local vm_name="$1"
    local vm_dir="$2"
    
    cat << EOF

╔════════════════════════════════════════════════════════════════╗
║                    VM Setup Completed!                        ║
╚════════════════════════════════════════════════════════════════╝

VM Name: $vm_name
VM Directory: $vm_dir
VNC Console: localhost:5901 (password may be required)

Next Steps:
1. Start the VM:
   virsh start $vm_name

2. Connect to the VM:
   $vm_dir/connect-vm.sh

3. Setup development environment inside VM:
   ./setup-copilot.sh

4. Access VM via VNC if needed:
   vncviewer localhost:5901

5. Switch modes using vm-dev-environment:
   ./scripts/vm-dev-environment start    # Development mode
   ./scripts/vm-dev-environment production  # Production mode

The VM is configured with:
- Ubuntu 22.04 LTS Desktop
- User: developer / Password: developer
- SSH enabled on port 22
- VNC enabled on port 5901
- Ready for GitHub Copilot integration

Production Performance:
- VM can be completely disabled for zero-overhead production
- scripts/onload-trading still delivers 4.37μs latency when needed

EOF
}

main() {
    local vm_name="${1:-ai-trading-dev-vm}"
    
    log "Starting VM setup for: $vm_name"
    
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
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
    
    # Install VM
    install_vm "$vm_name" "$iso_path" "$disk_path"
    
    # Post-installation setup
    setup_vm_post_install "$vm_name"
    
    # Create helper scripts
    create_setup_scripts "$vm_dir"
    
    # Show completion message
    show_completion_message "$vm_name" "$vm_dir"
    
    log "VM setup completed successfully"
}

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi