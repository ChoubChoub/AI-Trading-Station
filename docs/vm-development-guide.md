# VM Development Environment Setup Guide - Workspace Mounting

## Overview

This guide describes the VM development environment with **complete workspace mounting** functionality. The system enables safe development and optimization work while preserving the ultra-low latency production environment (4.37μs mean latency).

## Key Fix: Complete Workspace Mounting

**FIXED**: The VM now mounts the **ENTIRE AI Trading Station project directory** inside the VM, providing complete access to all files, scripts, configurations, and modules.

### Workspace Configuration

- **Host Path**: `$(pwd)` (complete project directory)
- **VM Mount Point**: `/workspace/ai-trading-station/`
- **Mount Technology**: 9p virtio filesystem sharing
- **Access**: Complete read-write access to all project files

## Quick Start

### 1. Setup VM with Workspace Mounting

```bash
# Install virtualization stack
sudo apt install qemu-kvm libvirt-daemon-system libvirt-clients virtinst

# Setup development VM with complete workspace mounting
sudo ./scripts/vm-dev-environment setup

# This automatically configures:
# - Ubuntu 22.04 VM with development tools
# - Complete project workspace mounted at /workspace/ai-trading-station
# - GitHub Copilot integration ready
# - All AI Trading Station files accessible in VM
```

### 2. Start Development Mode

```bash
# Switch to development mode (enables VM + mounts workspace)
./ai-trading-station.sh vm-dev

# Connect to VM
ssh developer@$(virsh domifaddr ai-trading-dev-vm | grep -oP '192\.168\.\d+\.\d+')

# Inside VM - access complete workspace
cd /workspace/ai-trading-station
ls -la  # See all AI Trading Station files!

# Start development with VS Code
code /workspace/ai-trading-station
```

### 3. Development Workflow

Inside the VM, you have access to the **COMPLETE** AI Trading Station project:

```bash
# All scripts available
./ai-trading-station.sh status
./scripts/onload-trading --help

# All configuration files accessible
cat configs/vm-dev-setup.json

# All modules available
ls -la "Module 1" "Module 2" "Module 3" "Module 4"

# Complete development environment
# No need to clone repository - it's already mounted!
```

### 4. Switch to Production Mode

```bash
# Switch to production mode (zero overhead)
./ai-trading-station.sh vm-prod

# Production trading available with full performance
sudo ./scripts/onload-trading --mode=strict ./your-trading-app
```

## Workspace Mounting Benefits

✅ **Complete Project Access**: All files, scripts, configs available in VM  
✅ **Real-time Synchronization**: Changes sync between host and VM instantly  
✅ **No Repository Cloning**: Complete workspace already mounted  
✅ **Full Development Context**: Access to all modules and dependencies  
✅ **GitHub Copilot Integration**: AI assistance with complete codebase context  

## Architecture Integration

```
AI Trading Station with Complete Workspace Mounting:
├── Production Environment
│   ├── scripts/onload-trading          ← Core performance (4.37μs latency)  
│   ├── CPU isolation (cores 2,3)       ← Dedicated trading cores
│   └── OnLoad kernel bypass           ← Zero-latency networking
│
├── VM Development Environment  
│   ├── scripts/vm-dev-environment      ← VM management with workspace mounting
│   ├── scripts/vm-setup-ubuntu.sh      ← VM setup with filesystem sharing
│   ├── scripts/production-mode-switch  ← Mode switching
│   └── configs/vm-dev-setup.json       ← VM config with workspace settings
│
└── Complete Workspace Mounting
    ├── Host: /path/to/ai-trading-station/     ← Complete project
    ├── VM: /workspace/ai-trading-station/     ← Mounted workspace  
    └── Sync: Real-time file synchronization  ← Instant updates
```

## Commands Reference

### VM Management

```bash
# Setup VM with workspace mounting
sudo ./scripts/vm-dev-environment setup

# Start development mode (VM + workspace)
./scripts/vm-dev-environment start

# Stop development mode
./scripts/vm-dev-environment stop

# Show status and workspace info
./scripts/vm-dev-environment status
```

### Integrated Commands

```bash
# Check VM and workspace status
./ai-trading-station.sh vm-status

# Enable development mode with workspace
./ai-trading-station.sh vm-dev

# Enable production mode (zero overhead)
./ai-trading-station.sh vm-prod
```

### Inside VM Development

```bash
# Access complete workspace
cd /workspace/ai-trading-station

# Use any project script
./ai-trading-station.sh demo
./scripts/onload-trading --list-modes

# Open complete project in VS Code
code /workspace/ai-trading-station

# Setup GitHub Copilot with full workspace context
./setup-copilot.sh
```

## Troubleshooting

### Workspace Not Mounted

```bash
# Check if filesystem is attached to VM
sudo virsh dumpxml ai-trading-dev-vm | grep workspace

# Manual mount inside VM
sudo mount -t 9p -o trans=virtio ai-trading-workspace /workspace/ai-trading-station
```

### Permission Issues

```bash
# Fix permissions inside VM
sudo chown -R developer:developer /workspace/ai-trading-station
```

### Performance Verification

```bash
# Verify production mode is clean
./scripts/production-mode-switch status

# Test OnLoad performance
sudo ./scripts/onload-trading --mode=strict echo "test"
```

## Key Features

### Complete Workspace Access
- **All Files**: Scripts, configs, modules, documentation
- **Real-time Sync**: Changes instantly available in both environments
- **No Duplication**: No need to clone or copy files

### Development Integration
- **GitHub Copilot**: AI assistance with full project context
- **VS Code**: Complete workspace available in IDE
- **Debugging**: Full access to all source files and configurations

### Production Safety
- **Zero Impact**: Complete VM shutdown for production mode
- **Performance Preserved**: OnLoad trading maintains 4.37μs latency
- **Clean State**: Verified zero VM overhead in production

This implementation solves the workspace mounting issue by providing complete, real-time access to the entire AI Trading Station project directory inside the development VM.