# Desktop Mode Management Guide

## Overview

The AI Trading Station operates in **server mode** (multi-user.target) by default, following industry best practices for algorithmic trading systems. This configuration minimizes resource consumption and maximizes trading performance.

Desktop mode (graphical.target) with Xfce4 is available on-demand for troubleshooting, configuration, and maintenance tasks.

---

## System Architecture

### Default Configuration
- **Default Boot Mode**: Server Mode (multi-user.target)
- **Display Manager**: LightDM (available but not auto-started)
- **Desktop Environment**: Xfce4 (lightweight, installed)
- **Access Method**: SSH (primary), Desktop (on-demand)

### Resource Impact
- **Server Mode**: ~92MB RAM saved, CPU cycles available for trading
- **Desktop Mode**: Display manager and GUI services consume system resources

---

## Management Script

Location: `/home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh`

### Quick Reference

```bash
# Check current mode
sudo ./toggle-desktop-mode.sh status

# Set server mode as default (recommended)
sudo ./toggle-desktop-mode.sh server

# Start desktop temporarily without changing default
sudo ./toggle-desktop-mode.sh start

# Stop desktop
sudo ./toggle-desktop-mode.sh stop

# Set desktop as default (not recommended for production)
sudo ./toggle-desktop-mode.sh desktop

# Show help
sudo ./toggle-desktop-mode.sh help
```

---

## Common Operations

### 1. Check Current System Mode

**Command**:
```bash
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh status
```

**Expected Output (Server Mode)**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current System Mode:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mode: Server Mode (multi-user.target) [Recommended]
Display Manager: inactive
```

### 2. Temporary Desktop Access (Recommended Workflow)

When you need desktop access without changing the default boot mode:

**Start Desktop**:
```bash
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh start
```

**Access Methods**:
- Physical monitor connection
- Remote desktop (VNC/RDP if configured)
- X11 forwarding over SSH: `ssh -X user@host`

**Stop Desktop When Done**:
```bash
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh stop
```

**Benefits**:
- Desktop available immediately
- Server mode remains default (survives reboots)
- Resources freed immediately when stopped
- No reboot required

### 3. Set Server Mode as Default

This is the **recommended configuration** for production trading:

```bash
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh server
```

**What This Does**:
- Sets `multi-user.target` as default boot target
- Stops LightDM display manager immediately
- Frees ~92MB RAM and associated CPU cycles
- Takes full effect on next reboot
- Desktop still available via `start` command

### 4. Set Desktop Mode as Default (Not Recommended)

Only use this if you frequently need desktop access:

```bash
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh desktop
```

**Note**: This increases baseline resource consumption and is not recommended for production trading environments.

---

## Manual Operations (Advanced)

### Check System Target Directly
```bash
systemctl get-default
```

**Output**:
- `multi-user.target` = Server mode
- `graphical.target` = Desktop mode

### Check Display Manager Status
```bash
systemctl status lightdm.service
```

### Manual Mode Changes (Without Script)

**Set Server Mode**:
```bash
sudo systemctl set-default multi-user.target
sudo systemctl stop lightdm.service
```

**Set Desktop Mode**:
```bash
sudo systemctl set-default graphical.target
sudo systemctl start lightdm.service
```

**Temporary Desktop Start**:
```bash
sudo systemctl start lightdm.service
```

**Temporary Desktop Stop**:
```bash
sudo systemctl stop lightdm.service
```

---

## Use Cases

### ✅ When to Use Desktop Mode

1. **Initial System Setup**: Configure network, drivers, hardware
2. **Troubleshooting**: Visual inspection of GUI applications
3. **Development**: Testing trading dashboard UIs
4. **Monitoring**: Grafana visualization on local display
5. **Maintenance**: System configuration with GUI tools

### ⚠️ When to Stay in Server Mode

1. **Production Trading**: All live trading operations
2. **Backtesting**: Long-running computational tasks
3. **Remote Access**: When SSH is sufficient
4. **24/7 Operations**: Minimize resource overhead
5. **High-Frequency Trading**: Maximum performance required

---

## Best Practices

### For Production Trading Systems

1. ✅ **Default to Server Mode**: Keep `multi-user.target` as default
2. ✅ **On-Demand Desktop**: Use `start` command when needed
3. ✅ **Stop When Done**: Free resources with `stop` command
4. ✅ **Monitor Resources**: Check impact if desktop runs during trading
5. ✅ **Document Usage**: Log when desktop was started/stopped

### Resource Management

```bash
# Before starting trading session
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh stop

# After trading session, if desktop needed
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh start
```

---

## Troubleshooting

### Desktop Won't Start

**Check LightDM Status**:
```bash
sudo systemctl status lightdm.service
```

**View Logs**:
```bash
sudo journalctl -u lightdm.service -n 50
```

**Restart LightDM**:
```bash
sudo systemctl restart lightdm.service
```

### Desktop Consuming Too Many Resources

**Check Display Manager Memory**:
```bash
systemctl status lightdm.service | grep Memory
```

**Check X Server Processes**:
```bash
ps aux | grep -E 'Xorg|lightdm' | grep -v grep
```

**Solution**: Stop desktop when not needed:
```bash
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh stop
```

### Can't Access Desktop After Start

**Verify Service is Running**:
```bash
systemctl is-active lightdm.service
```

**Check Display Variable**:
```bash
echo $DISPLAY
```

**For SSH X11 Forwarding**:
```bash
ssh -X youssefbahloul@ai-trading-station
export DISPLAY=:0
```

---

## Integration with Trading Operations

### Pre-Trading Checklist

```bash
# 1. Ensure server mode for optimal performance
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh status

# 2. Stop desktop if running
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh stop

# 3. Verify system resources
free -h
htop
```

### Post-Trading Maintenance

```bash
# 1. Trading session complete - safe to start desktop
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh start

# 2. Perform maintenance tasks via GUI

# 3. Stop desktop before next session
sudo /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh stop
```

---

## Security Considerations

### Server Mode (Default)
- ✅ Smaller attack surface (no GUI services exposed)
- ✅ Fewer running processes
- ✅ Standard for production servers

### Desktop Mode
- ⚠️ Additional services running (LightDM, Xorg)
- ⚠️ Potential security updates required for GUI packages
- ℹ️ Only accessible locally or via explicitly configured remote desktop

### Recommendations
1. Keep server mode as default
2. Use SSH with key authentication for remote access
3. Only start desktop when physically present or via secure remote desktop
4. Stop desktop when not actively using it

---

## Related Documentation

- **System Scripts**: `/home/youssefbahloul/ai-trading-station/Services/System/README.md`
- **CPU Affinity**: `cpu-affinity-reference.md` (this directory)
- **Monitoring**: `redis-health-monitoring.md` (this directory)
- **Architecture**: `/home/youssefbahloul/ai-trading-station/Documentation/Architecture/`

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-09 | 1.0 | Initial documentation - Desktop mode toggle implementation |

---

## Summary

**Default Configuration**: Server mode for optimal trading performance  
**Desktop Availability**: On-demand via simple script  
**Recommended Workflow**: Start desktop only when needed, stop when done  
**Industry Standard**: Matches best practices for algorithmic trading infrastructure
