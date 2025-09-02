# VM Development Environment Setup Guide

## Overview

This guide provides comprehensive setup and usage instructions for the AI Trading Station VM development environment. The system enables safe development and optimization work while preserving the ultra-low latency production environment (4.37μs mean latency).

## Architecture

The VM development environment works alongside the existing AI Trading Station components:

```
AI Trading Station Architecture:
├── Production Environment
│   ├── scripts/onload-trading          ← Core performance (4.37μs latency)  
│   ├── CPU isolation (cores 2,3)       ← Dedicated trading cores
│   └── OnLoad kernel bypass           ← Zero-latency networking
│
├── VM Development Environment  
│   ├── scripts/vm-dev-environment      ← VM management
│   ├── scripts/vm-setup-ubuntu.sh      ← Automated VM setup
│   ├── scripts/production-mode-switch  ← Mode switching
│   └── configs/vm-dev-setup.json       ← VM configuration
│
└── Integration Layer
    ├── ai-trading-station.sh           ← Unified monitoring
    └── Mode switching automation       ← Zero-impact production
```

## Key Features

### 🔧 VM Development Environment
- **Automated VM Setup**: Ubuntu 22.04 LTS with development tools
- **GitHub Copilot Integration**: VS Code + AI-powered development
- **Performance Isolation**: VM runs on cores 4-7, production on cores 2-3
- **Safe Testing Environment**: Mirror production structure for realistic testing

### ⚡ Zero-Impact Production Mode
- **Complete VM Shutdown**: All VMs stopped and services disabled
- **Kernel Module Unloading**: VM-related modules unloaded for minimal overhead
- **State Verification**: Comprehensive checks ensure clean production state
- **Performance Preservation**: OnLoad trading wrapper remains fully functional (4.37μs)

### 🔄 Automated Mode Switching
- **Development Mode**: VM services active, development environment available
- **Production Mode**: Zero VM overhead, maximum performance
- **Safety Mechanisms**: Automatic backups and validation before switching
- **Status Monitoring**: Real-time mode and system status checking

## Prerequisites

### Hardware Requirements
- **Minimum**: 16GB RAM, 8+ CPU cores, 100GB available disk space
- **Recommended**: 32GB RAM, 12+ CPU cores, 200GB SSD storage
- **CPU Features**: VT-x/AMD-V, VT-d/AMD-Vi, EPT support

### Software Requirements
- **Operating System**: Ubuntu 22.04 LTS (host)
- **Virtualization**: KVM/QEMU with libvirt
- **Network**: Bridge networking capability
- **Permissions**: sudo access for VM management

### Existing AI Trading Station Components
- **OnLoad Drivers**: Solarflare OnLoad for kernel bypass
- **CPU Isolation**: Cores 2,3 isolated via GRUB configuration  
- **scripts/onload-trading**: Core performance wrapper (must be preserved)

## Installation

### 1. Install Virtualization Stack

```bash
# Install KVM/QEMU and management tools
sudo apt update
sudo apt install qemu-kvm libvirt-daemon-system libvirt-clients virtinst virt-manager

# Add user to libvirt group
sudo usermod -aG libvirt $USER

# Enable and start libvirt service
sudo systemctl enable libvirtd
sudo systemctl start libvirtd

# Log out and back in for group changes to take effect
```

### 2. Verify Virtualization Support

```bash
# Check CPU virtualization features
egrep -c '(vmx|svm)' /proc/cpuinfo  # Should be > 0

# Check KVM acceleration
kvm-ok  # Should show "KVM acceleration can be used"

# Verify libvirt installation
virsh list --all
```

### 3. Clone and Setup AI Trading Station

```bash
# If not already cloned
git clone https://github.com/ChoubChoub/AI-Trading-Station
cd AI-Trading-Station

# Verify existing components
ls -la scripts/
./ai-trading-station.sh status
```

### 4. Setup VM Development Environment

```bash
# Create and configure development VM
sudo ./scripts/vm-dev-environment setup

# This will:
# - Download Ubuntu 22.04 ISO (~4GB)
# - Create VM with 8GB RAM, 4 CPU cores, 50GB disk
# - Install Ubuntu with development tools
# - Configure SSH and VNC access
```

## Daily Workflow

### Starting Development Work

```bash
# 1. Switch to development mode
./scripts/production-mode-switch development

# 2. Start development VM
./scripts/vm-dev-environment start

# 3. Connect to VM
ssh developer@$(virsh domifaddr ai-trading-dev-vm | grep -oP '192\.168\.\d+\.\d+')

# Or use VNC for GUI access
vncviewer localhost:5901
```

### Inside the Development VM

```bash
# Setup GitHub Copilot and development tools
./setup-copilot.sh

# Authenticate with GitHub
gh auth login

# Clone repository for development
git clone https://github.com/ChoubChoub/AI-Trading-Station
cd AI-Trading-Station

# Start VS Code with Copilot
code .
```

### Switching to Production Mode

```bash
# Stop all development work
./scripts/vm-dev-environment stop

# Enable zero-overhead production mode
sudo ./scripts/production-mode-switch production

# Verify clean state
./scripts/production-mode-switch status

# Production trading is now available with full performance
sudo ./scripts/onload-trading --mode=strict ./your-trading-app
```

## Command Reference

### VM Management (`scripts/vm-dev-environment`)

```bash
# VM lifecycle management
./scripts/vm-dev-environment setup      # Create new development VM
./scripts/vm-dev-environment start      # Start development mode
./scripts/vm-dev-environment stop       # Stop development mode  
./scripts/vm-dev-environment status     # Show system status

# Advanced options
./scripts/vm-dev-environment --debug setup
./scripts/vm-dev-environment --vm-name custom-vm setup
```

### Mode Switching (`scripts/production-mode-switch`)

```bash
# Mode switching
./scripts/production-mode-switch development  # Enable development mode
./scripts/production-mode-switch production   # Enable production mode (zero overhead)
./scripts/production-mode-switch status       # Show current mode

# Validation and maintenance
./scripts/production-mode-switch validate     # Run performance tests
./scripts/production-mode-switch backup       # Create system backup

# Force mode changes
./scripts/production-mode-switch production --force
```

### Integrated Monitoring (`ai-trading-station.sh`)

```bash
# System monitoring (works in both modes)
./ai-trading-station.sh status    # Overall system status
./ai-trading-station.sh demo      # Performance demonstration
./ai-trading-station.sh monitor   # Real-time monitoring

# Production performance (production mode only)
./ai-trading-station.sh launch strict ./trading-app
```

## Configuration

### VM Specifications (`configs/vm-dev-setup.json`)

```json
{
  "vm_configurations": {
    "development": {
      "specs": {
        "memory_mb": 8192,      // 8GB RAM
        "cpu_cores": 4,         // 4 CPU cores  
        "disk_gb": 50,          // 50GB disk
        "network_mode": "bridged"
      },
      "performance_isolation": {
        "vm_cpu_affinity": [4, 5, 6, 7],    // VM uses cores 4-7
        "production_cpu_cores": [2, 3],     // Production uses cores 2-3
        "system_cpu_cores": [0, 1]          // System uses cores 0-1
      }
    }
  }
}
```

### CPU Isolation Configuration

The system requires proper CPU isolation for optimal performance:

```bash
# Check current isolation
cat /proc/cmdline
cat /sys/devices/system/cpu/isolated

# Expected configuration in /etc/default/grub:
GRUB_CMDLINE_LINUX_DEFAULT="isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3"
```

## Performance Validation

### Production Mode Verification

```bash
# Verify zero VM overhead
./scripts/production-mode-switch status

# Expected output:
# ✓ Clean production state verified
# ✓ OnLoad performance verified - 4.37μs latency capable
# • Running VMs: 0
# • Active VM services: none
```

### Development Mode Testing

```bash
# Test VM functionality
./scripts/vm-dev-environment status

# Expected output:
# Current Mode: development
# VM Status: running
# ✓ Virtualization support verified
# ✓ System resources: sufficient
```

### OnLoad Performance Testing

```bash
# Test core performance component
./scripts/onload-trading --version
./scripts/onload-trading --list-modes

# Production performance test
sudo ./scripts/onload-trading --mode=strict echo "Performance test"
```

## Troubleshooting

### Common Issues

#### VM Setup Issues

**Problem**: "Virtualization not supported"
```bash
# Check BIOS settings
sudo dmesg | grep -i virtualization

# Enable VT-x/AMD-V in BIOS
# Enable VT-d/AMD-Vi if available
```

**Problem**: "Permission denied accessing libvirt"
```bash
# Add user to libvirt group
sudo usermod -aG libvirt $USER

# Logout and login again
sudo systemctl restart libvirtd
```

**Problem**: "VM fails to start"
```bash
# Check system resources
./scripts/vm-dev-environment status

# Check VM logs
sudo virsh dominfo ai-trading-dev-vm
tail -f ~/.vm-dev-environment.log
```

#### Performance Issues

**Problem**: "Production mode still shows VM overhead"
```bash
# Force clean production state
sudo ./scripts/production-mode-switch production --force

# Verify clean state
./scripts/production-mode-switch verify
ps aux | grep qemu
lsmod | grep kvm
```

**Problem**: "OnLoad performance degraded"
```bash
# Check OnLoad status
onload_tool reload
sudo systemctl status onload

# Verify CPU isolation
cat /sys/devices/system/cpu/isolated

# Test performance wrapper
./scripts/onload-trading --mode=auto true
```

#### Network Issues

**Problem**: "Cannot connect to VM"
```bash
# Check VM network interface
sudo virsh domifaddr ai-trading-dev-vm

# Check bridge networking
ip addr show virbr0
sudo systemctl status libvirtd
```

**Problem**: "VM has no network access"
```bash
# Restart networking in VM
sudo systemctl restart systemd-networkd

# Check bridge configuration
sudo virsh net-list --all
sudo virsh net-start default
```

### Log Files and Debugging

```bash
# VM management logs
tail -f ~/.vm-dev-environment.log

# VM setup logs  
tail -f ~/.vm-setup.log

# Mode switching logs
tail -f ~/.production-mode-switch.log

# OnLoad performance logs
tail -f ~/.onload-trading.log

# AI Trading Station logs
tail -f ~/.ai-trading-station.log
```

### Recovery Procedures

#### Restore from Backup

```bash
# List available backups
ls -la ~/.ai-trading-station-backups/

# View backup contents
cat ~/.ai-trading-station-backups/backup-20231201-143022/system-state.txt

# Manual recovery (if needed)
sudo cp ~/.ai-trading-station-backups/backup-*/grub /etc/default/grub
sudo update-grub
```

#### Reset to Production Mode

```bash
# Emergency production mode restoration
sudo ./scripts/production-mode-switch production --force

# Stop all VMs manually
sudo virsh destroy $(sudo virsh list --name)

# Disable all VM services
sudo systemctl stop libvirtd libvirt-guests
sudo systemctl disable libvirtd libvirt-guests

# Unload VM modules
sudo modprobe -r kvm_intel kvm_amd kvm vhost_net vhost
```

#### Complete Environment Reset

```bash
# Remove all VMs and start fresh
sudo virsh destroy ai-trading-dev-vm
sudo virsh undefine ai-trading-dev-vm --remove-all-storage

# Clean VM images
rm -rf ~/vm-images/

# Recreate development environment
sudo ./scripts/vm-dev-environment setup
```

## GitHub Copilot Integration

### Setup in Development VM

```bash
# Inside the VM, install VS Code and Copilot
./setup-copilot.sh

# Authenticate with GitHub
gh auth login

# Launch VS Code with Copilot
code .
```

### Using Copilot for Trading Optimizations

```bash
# Open AI Trading Station project
cd ~/ai-trading-station
code .

# Copilot can help with:
# - Performance optimization suggestions
# - Code completion for trading algorithms
# - OnLoad configuration tuning
# - System optimization recommendations
```

### Best Practices

1. **Development Isolation**: Always develop in VM to avoid impacting production
2. **Performance Testing**: Use VM for initial testing, bare metal for final validation  
3. **Code Review**: Use Copilot suggestions as starting points, not final solutions
4. **Production Deployment**: Always test on production hardware before going live

## Integration with Existing System

### Preserving OnLoad Performance

The VM development environment is designed to coexist with the existing OnLoad trading system:

- **Production Mode**: Zero VM overhead, full OnLoad performance (4.37μs)
- **Development Mode**: VM isolated on different CPU cores
- **Mode Switching**: Automated transition with verification

### Working with Existing Scripts

```bash
# ai-trading-station.sh continues to work
./ai-trading-station.sh status        # Shows both VM and trading status
./ai-trading-station.sh launch strict # Works in production mode

# onload-trading remains unchanged  
sudo ./scripts/onload-trading --mode=strict ./trading-app

# New VM capabilities added
./scripts/vm-dev-environment status
./scripts/production-mode-switch status
```

### Module Integration

The VM development environment enhances the existing module system:

- **Module 1**: Enhanced with VM-based development environment
- **Module 2-4**: Remain unchanged, work in both modes
- **New VM Module**: Provides development capabilities without production impact

## Security Considerations

### VM Network Security

- VMs use bridged networking with firewall rules
- SSH access limited to development network
- Production network isolation maintained

### Data Protection

- Development work isolated in VM
- Production data never accessed from VM
- Automatic backups before mode changes

### Access Control

- sudo required for production mode changes
- VM access through standard authentication
- Audit logs for all mode switches

## Performance Benchmarks

### Production Mode Performance
- **OnLoad Latency**: 4.37μs mean latency maintained
- **CPU Overhead**: Zero (all VM services disabled)
- **Memory Overhead**: Zero (VM memory released)
- **Process Overhead**: Zero (no VM processes running)

### Development Mode Resources
- **VM Memory**: 8GB (isolated from production)
- **VM CPU**: 4 cores (cores 4-7, isolated from production cores 2-3)
- **Disk I/O**: Minimal impact on NVMe SSD
- **Network**: Bridge interface with minimal overhead

### Mode Switch Performance
- **Development → Production**: ~30 seconds (VM shutdown + verification)
- **Production → Development**: ~60 seconds (VM startup + service initialization)
- **Verification Time**: ~5 seconds (clean state check)

## Future Enhancements

### Planned Features
- **Remote Development**: Access VM from multiple workstations
- **Container Integration**: Docker support within VM
- **Performance Profiling**: Integrated profiling tools for optimization
- **Automated Testing**: CI/CD pipeline integration

### Optimization Opportunities
- **VM Templates**: Pre-configured snapshots for faster setup
- **Resource Scaling**: Dynamic CPU/memory allocation
- **Storage Optimization**: Copy-on-write snapshots for quick resets
- **Network Optimization**: SR-IOV for high-performance networking

## Support and Maintenance

### Regular Maintenance Tasks

```bash
# Weekly VM maintenance
./scripts/vm-dev-environment stop
sudo virsh pool-refresh default
sudo virsh vol-list default

# Monthly performance validation
./scripts/production-mode-switch validate
./ai-trading-station.sh demo

# Quarterly full backup
tar -czf ai-trading-station-backup-$(date +%Y%m%d).tar.gz ~/.ai-trading-station-backups/
```

### Updates and Upgrades

```bash
# Update VM development tools
# (Run inside VM)
sudo apt update && sudo apt upgrade
code --update-extensions

# Update host virtualization stack
sudo apt update
sudo apt upgrade qemu-kvm libvirt-daemon-system

# Update AI Trading Station components
git pull origin main
chmod +x scripts/*
```

## Conclusion

The VM development environment provides a comprehensive solution for safe development while preserving the ultra-low latency production capabilities of the AI Trading Station. Key benefits:

✅ **Zero Production Impact**: Complete VM isolation with verified clean states
✅ **Enhanced Development**: GitHub Copilot integration with professional tools  
✅ **Automated Management**: One-command mode switching with safety checks
✅ **Performance Preservation**: Full OnLoad trading capability (4.37μs) maintained
✅ **Comprehensive Documentation**: Complete setup and troubleshooting guidance

The system enables confident development and optimization work while ensuring the critical production trading performance is never compromised.