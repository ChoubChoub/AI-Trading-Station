#!/bin/bash
# ============================================================================
# AI Trading Station VM Development Environment Setup
# ============================================================================
# PURPOSE: Create isolated VM environment for safe development and testing
# FEATURES: KVM/QEMU virtualization with 9p workspace sharing
# SAFETY: Complete isolation from production trading system
# ============================================================================

set -euo pipefail

# VM Configuration
readonly VM_NAME="ai-trading-dev"
readonly VM_MEMORY="32768"  # 32GB in MB
readonly VM_CORES="8"
readonly VM_DISK_SIZE="100G"
readonly VM_OS="ubuntu22.04"

# Paths and directories
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly VM_DIR="$HOME/.ai-trading-vm"
readonly VM_DISK_PATH="$VM_DIR/${VM_NAME}.qcow2"
readonly VM_CONFIG_PATH="$VM_DIR/${VM_NAME}.conf"
readonly WORKSPACE_HOST_PATH="$HOME/ai-trading-station"
readonly WORKSPACE_VM_PATH="/workspace/ai-trading-station"

# Ubuntu 22.04 ISO URL
readonly UBUNTU_ISO_URL="https://releases.ubuntu.com/22.04/ubuntu-22.04.3-live-server-amd64.iso"
readonly UBUNTU_ISO_PATH="$VM_DIR/ubuntu-22.04.3-live-server-amd64.iso"

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
    exit 1
}

show_banner() {
    cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║            AI Trading Station VM Setup                        ║
║        Safe Development Environment Configuration              ║
╠════════════════════════════════════════════════════════════════╣
║  Purpose: Isolated VM for testing trading components safely   ║
║  Specs: 32GB RAM, 8 cores, 100GB disk, Ubuntu 22.04          ║
║  Workspace: ~/ai-trading-station → /workspace/ai-trading-station║
╚════════════════════════════════════════════════════════════════╝
EOF
}

check_dependencies() {
    info "Checking system dependencies..."
    
    local missing_packages=()
    
    # Check for KVM support
    if ! grep -q -E 'vmx|svm' /proc/cpuinfo; then
        error "CPU does not support hardware virtualization (VT-x/AMD-V)"
    fi
    
    # Check for required packages
    local required_packages=("qemu-kvm" "libvirt-daemon-system" "libvirt-clients" "bridge-utils" "virt-manager" "wget" "curl")
    
    for package in "${required_packages[@]}"; do
        if ! command -v "$package" &> /dev/null && ! dpkg -l | grep -q "^ii  $package "; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        warn "Missing required packages: ${missing_packages[*]}"
        info "Installing missing packages..."
        sudo apt-get update
        sudo apt-get install -y "${missing_packages[@]}"
    fi
    
    # Add user to libvirt group
    if ! groups "$USER" | grep -q libvirt; then
        info "Adding user to libvirt group..."
        sudo usermod -a -G libvirt "$USER"
        warn "Please log out and log back in for group changes to take effect"
    fi
    
    info "✓ All dependencies satisfied"
}

setup_directories() {
    info "Setting up VM directories..."
    
    mkdir -p "$VM_DIR"
    
    # Create workspace directory if it doesn't exist
    if [ ! -d "$WORKSPACE_HOST_PATH" ]; then
        info "Creating workspace directory: $WORKSPACE_HOST_PATH"
        mkdir -p "$WORKSPACE_HOST_PATH"
        
        # Copy current repository content to workspace
        if [ -f "$SCRIPT_DIR/ai-trading-station.sh" ]; then
            info "Copying current AI Trading Station to workspace..."
            cp -r "$SCRIPT_DIR"/* "$WORKSPACE_HOST_PATH/"
        fi
    fi
    
    info "✓ Directories configured"
}

download_ubuntu_iso() {
    if [ -f "$UBUNTU_ISO_PATH" ]; then
        info "Ubuntu ISO already exists: $UBUNTU_ISO_PATH"
        return 0
    fi
    
    info "Downloading Ubuntu 22.04 Server ISO..."
    info "This may take several minutes depending on your internet connection..."
    
    if ! wget -O "$UBUNTU_ISO_PATH" "$UBUNTU_ISO_URL"; then
        error "Failed to download Ubuntu ISO"
    fi
    
    info "✓ Ubuntu ISO downloaded successfully"
}

create_vm_disk() {
    if [ -f "$VM_DISK_PATH" ]; then
        warn "VM disk already exists: $VM_DISK_PATH"
        read -p "Do you want to recreate it? This will destroy existing VM data. (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
        rm -f "$VM_DISK_PATH"
    fi
    
    info "Creating VM disk ($VM_DISK_SIZE)..."
    
    if ! qemu-img create -f qcow2 "$VM_DISK_PATH" "$VM_DISK_SIZE"; then
        error "Failed to create VM disk"
    fi
    
    info "✓ VM disk created: $VM_DISK_PATH"
}

create_vm_config() {
    info "Creating VM configuration..."
    
    cat > "$VM_CONFIG_PATH" << EOF
# AI Trading Station VM Configuration
VM_NAME="$VM_NAME"
VM_MEMORY="$VM_MEMORY"
VM_CORES="$VM_CORES"
VM_DISK_PATH="$VM_DISK_PATH"
UBUNTU_ISO_PATH="$UBUNTU_ISO_PATH"
WORKSPACE_HOST_PATH="$WORKSPACE_HOST_PATH"
WORKSPACE_VM_PATH="$WORKSPACE_VM_PATH"

# QEMU Arguments for 9p filesystem sharing
QEMU_9P_ARGS="-fsdev local,security_model=passthrough,id=fsdev0,path=$WORKSPACE_HOST_PATH -device virtio-9p-pci,id=fs0,fsdev=fsdev0,mount_tag=workspace"

# Network configuration
QEMU_NETWORK_ARGS="-netdev user,id=network0,hostfwd=tcp::2222-:22 -device e1000,netdev=network0"
EOF
    
    info "✓ VM configuration saved: $VM_CONFIG_PATH"
}

create_vm_install_script() {
    local install_script_path="$VM_DIR/install-vm.sh"
    
    info "Creating VM installation script..."
    
    cat > "$install_script_path" << 'EOF'
#!/bin/bash
# Automated VM installation script
# This script will be used to install Ubuntu in the VM

# Source VM configuration
source "$HOME/.ai-trading-vm/ai-trading-dev.conf"

echo "Starting VM installation..."
echo "VM Name: $VM_NAME"
echo "Memory: ${VM_MEMORY}MB"
echo "Cores: $VM_CORES"
echo "Disk: $VM_DISK_PATH"
echo "ISO: $UBUNTU_ISO_PATH"

# Launch VM with installation ISO
qemu-system-x86_64 \
    -name "$VM_NAME" \
    -machine type=pc,accel=kvm \
    -cpu host \
    -smp "$VM_CORES" \
    -m "$VM_MEMORY" \
    -drive file="$VM_DISK_PATH",format=qcow2,if=virtio \
    -cdrom "$UBUNTU_ISO_PATH" \
    -boot d \
    $QEMU_NETWORK_ARGS \
    $QEMU_9P_ARGS \
    -vga qxl \
    -display gtk \
    -monitor stdio

echo "VM installation completed. Use vm-manager.sh to manage the VM."
EOF
    
    chmod +x "$install_script_path"
    info "✓ VM installation script created: $install_script_path"
}

create_vm_post_install_script() {
    local post_install_script_path="$VM_DIR/post-install-setup.sh"
    
    info "Creating VM post-installation setup script..."
    
    cat > "$post_install_script_path" << 'EOF'
#!/bin/bash
# Post-installation setup script for AI Trading Station VM
# Run this inside the VM after Ubuntu installation is complete

set -euo pipefail

echo "=== AI Trading Station VM Post-Installation Setup ==="

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install essential packages
sudo apt-get install -y \
    build-essential \
    git \
    vim \
    htop \
    curl \
    wget \
    net-tools \
    openssh-server \
    python3 \
    python3-pip

# Enable SSH service
sudo systemctl enable ssh
sudo systemctl start ssh

# Create workspace mount point
sudo mkdir -p /workspace/ai-trading-station
sudo chown $USER:$USER /workspace

# Configure 9p filesystem mounting
echo "workspace /workspace/ai-trading-station 9p trans=virtio,version=9p2000.L,rw,_netdev,msize=104857600 0 0" | sudo tee -a /etc/fstab

# Mount workspace
sudo mount /workspace/ai-trading-station

# Create symbolic link for convenience
ln -sf /workspace/ai-trading-station ~/ai-trading-station

echo "=== Setup completed successfully ==="
echo "Workspace mounted at: /workspace/ai-trading-station"
echo "Symbolic link created: ~/ai-trading-station"
echo "SSH service enabled for remote access"
echo ""
echo "You can now:"
echo "1. Access trading station files in /workspace/ai-trading-station"
echo "2. Run ./ai-trading-station.sh from the mounted workspace"
echo "3. Test scripts/onload-trading safely in this isolated environment"
EOF
    
    chmod +x "$post_install_script_path"
    info "✓ VM post-installation setup script created: $post_install_script_path"
}

main() {
    show_banner
    echo
    
    info "Starting AI Trading Station VM development environment setup..."
    
    # Run setup steps
    check_dependencies
    setup_directories
    download_ubuntu_iso
    create_vm_disk
    create_vm_config
    create_vm_install_script
    create_vm_post_install_script
    
    echo
    info "✅ VM setup completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Run: $VM_DIR/install-vm.sh"
    echo "2. Install Ubuntu 22.04 in the VM (follow on-screen prompts)"
    echo "3. After installation, copy and run post-install-setup.sh inside the VM"
    echo "4. Use vm-manager.sh to manage the VM lifecycle"
    echo
    echo "VM Configuration:"
    echo "  • Name: $VM_NAME"
    echo "  • Memory: ${VM_MEMORY}MB (32GB)"
    echo "  • CPU Cores: $VM_CORES"
    echo "  • Disk: $VM_DISK_SIZE"
    echo "  • Workspace: $WORKSPACE_HOST_PATH → $WORKSPACE_VM_PATH"
    echo
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi