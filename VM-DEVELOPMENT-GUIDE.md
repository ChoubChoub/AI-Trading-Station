# VM Development Environment Guide

This guide provides comprehensive instructions for setting up and using the AI Trading Station VM development environment for safe testing and development.

## 🎯 Purpose

The VM development environment provides:
- **Complete isolation** from production trading systems
- **Safe testing environment** for trading components
- **Full workspace access** to all AI Trading Station files
- **Development-friendly setup** with Ubuntu 22.04

## 🚀 Quick Start

### Prerequisites

- **Host System**: Linux with KVM support (Intel VT-x or AMD-V)
- **Memory**: At least 40GB total system RAM (32GB for VM + 8GB for host)
- **Storage**: At least 120GB free disk space
- **CPU**: 10+ cores recommended (8 for VM + 2 for host)
- **Network**: Internet connection for Ubuntu ISO download

### Installation Steps

1. **Initial Setup**
   ```bash
   # Make scripts executable
   chmod +x vm-setup.sh vm-manager.sh
   
   # Run VM setup (downloads ISO, creates VM disk, configures environment)
   ./vm-setup.sh
   ```

2. **VM Installation**
   ```bash
   # Start VM installation process
   ~/.ai-trading-vm/install-vm.sh
   
   # Follow Ubuntu installation prompts:
   # - Select "Ubuntu Server" installation
   # - Configure network, users, and packages as needed
   # - Enable SSH server when prompted
   # - Complete installation and reboot
   ```

3. **Post-Installation Setup**
   ```bash
   # After VM boots, copy and run the post-install script inside the VM:
   
   # From host, copy script to VM (via SSH or shared folder)
   # Then inside VM:
   chmod +x post-install-setup.sh
   ./post-install-setup.sh
   ```

4. **Start Development**
   ```bash
   # Start VM for development
   ./vm-manager.sh start
   
   # Check status
   ./vm-manager.sh status
   
   # Connect to VM
   ./vm-manager.sh ssh your_username
   ```

## 🛠️ VM Management Commands

### Basic Operations

```bash
# Start the development VM
./vm-manager.sh start

# Stop the VM
./vm-manager.sh stop

# Check VM status and configuration
./vm-manager.sh status

# View VM logs
./vm-manager.sh logs
```

### Access Methods

```bash
# SSH access to VM (recommended)
./vm-manager.sh ssh username

# Direct console access (GUI)
./vm-manager.sh console

# Manual SSH connection
ssh -p 2222 username@localhost
```

## 📁 Workspace Access

### Host System
- **Path**: `~/ai-trading-station/`
- **Contents**: Complete AI Trading Station repository
- **Access**: Full read/write access for development

### Inside VM
- **Path**: `/workspace/ai-trading-station/`
- **Mount**: Automatically mounted via 9p filesystem
- **Symlink**: `~/ai-trading-station` → `/workspace/ai-trading-station`
- **Access**: Full read/write access to host files

### File Synchronization
All changes made in either location are immediately synchronized:
- Edit files on host → Changes appear in VM instantly
- Edit files in VM → Changes appear on host instantly
- No manual sync required

## 🧪 Testing Trading Components

### Safe Testing Environment

The VM provides complete isolation for testing:

```bash
# Inside VM, navigate to workspace
cd /workspace/ai-trading-station

# Test main trading station script
./ai-trading-station.sh status
./ai-trading-station.sh demo

# Test core performance component (safe in VM)
./scripts/onload-trading --help
./scripts/onload-trading --list-modes

# Test with development applications
./scripts/onload-trading --mode=auto your-test-app
```

### Development Workflow

1. **Develop on Host**: Use your preferred IDE/editor on the host system
2. **Test in VM**: Run and test components safely in the isolated VM
3. **Iterate**: Make changes on host, test immediately in VM
4. **Deploy**: When satisfied, deploy to production system

## ⚙️ VM Specifications

### Hardware Configuration
- **Memory**: 32GB (32,768MB)
- **CPU Cores**: 8 dedicated cores
- **Storage**: 100GB virtual disk (QCOW2 format)
- **Network**: NAT with SSH port forwarding (host:2222 → VM:22)

### Software Configuration
- **OS**: Ubuntu 22.04 LTS Server
- **Virtualization**: KVM/QEMU with hardware acceleration
- **Filesystem**: 9p virtio for workspace sharing
- **SSH**: Enabled for remote access

### Performance Characteristics
- **Boot Time**: ~30-60 seconds
- **File Access**: Near-native performance via 9p
- **Network**: Full network access through NAT
- **Isolation**: Complete separation from host processes

## 🔧 Advanced Configuration

### Custom VM Settings

Edit `~/.ai-trading-vm/ai-trading-dev.conf` to customize:

```bash
# VM Resource Allocation
VM_MEMORY="32768"    # Memory in MB
VM_CORES="8"         # Number of CPU cores

# Workspace Paths
WORKSPACE_HOST_PATH="/path/to/your/workspace"
WORKSPACE_VM_PATH="/workspace/ai-trading-station"
```

### Network Configuration

Default network setup:
- **Type**: User mode networking (NAT)
- **SSH Port**: Host 2222 → VM 22
- **Internet**: Full access from VM

For advanced networking, modify `QEMU_NETWORK_ARGS` in the config file.

### Storage Management

```bash
# Check VM disk usage
qemu-img info ~/.ai-trading-vm/ai-trading-dev.qcow2

# Resize VM disk (power off VM first)
qemu-img resize ~/.ai-trading-vm/ai-trading-dev.qcow2 +50G
```

## 🐛 Troubleshooting

### Common Issues

**VM Won't Start**
```bash
# Check KVM support
grep -E 'vmx|svm' /proc/cpuinfo

# Check libvirt group membership
groups $USER

# Check VM logs
./vm-manager.sh logs
```

**Workspace Not Mounting**
```bash
# Inside VM, check 9p mount
mount | grep 9p

# Manually mount if needed
sudo mount -t 9p -o trans=virtio workspace /workspace/ai-trading-station
```

**SSH Connection Failed**
```bash
# Check VM is running
./vm-manager.sh status

# Check port forwarding
netstat -ln | grep 2222

# Test connection
telnet localhost 2222
```

**Performance Issues**
```bash
# Verify KVM acceleration
./vm-manager.sh logs | grep -i kvm

# Check host resources
htop
free -h
```

### Log Files

- **VM Logs**: `~/.ai-trading-vm/ai-trading-dev.log`
- **VM PID**: `~/.ai-trading-vm/ai-trading-dev.pid`
- **VM Config**: `~/.ai-trading-vm/ai-trading-dev.conf`

## 🔒 Security Considerations

### Isolation Benefits
- **Process Isolation**: VM processes cannot affect host system
- **Network Isolation**: VM network is isolated from host network
- **File System Isolation**: Only workspace is shared, system files are separate
- **Resource Isolation**: VM resources are contained and limited

### Security Best Practices
- **Regular Updates**: Keep VM OS updated
- **SSH Keys**: Use SSH keys for authentication
- **Firewall**: Configure VM firewall as needed
- **Backup**: Regular backup of important development work

## 📊 Performance Comparison

| Environment | Trading Latency | Safety | Development Speed |
|-------------|----------------|---------|-------------------|
| **Production** | 4.37μs | ⚠️ High Risk | ❌ Slow |
| **VM Development** | ~50-100μs | ✅ Safe | ✅ Fast |
| **Standard Linux** | 50-100μs | ⚠️ Medium Risk | ⚠️ Medium |

The VM environment provides the optimal balance of safety and development speed for trading system development.

## 🚦 Getting Started Checklist

- [ ] Verify KVM support and install dependencies
- [ ] Run `./vm-setup.sh` to create VM infrastructure
- [ ] Install Ubuntu 22.04 in the VM
- [ ] Run post-installation setup inside VM
- [ ] Test workspace mounting and file access
- [ ] Verify AI Trading Station scripts work in VM
- [ ] Set up development workflow

## 📞 Support

For issues with the VM development environment:
1. Check the troubleshooting section above
2. Review log files in `~/.ai-trading-vm/`
3. Ensure all prerequisites are met
4. Verify KVM and virtualization support

The VM development environment enables safe, efficient development of ultra-low latency trading components while maintaining complete isolation from production systems.