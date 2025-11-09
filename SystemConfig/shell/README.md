# AI Trading Station - Shell Configuration

## Purpose

This directory contains trading-specific shell environment configurations that must be added to user shell profiles for the trading system to function correctly.

## Files

| File | Purpose | Deployment Location |
|------|---------|---------------------|
| `bashrc.trading` | Trading environment variables, PATH modifications, and aliases | Append to `~/.bashrc` or source as separate file |

## Quick Deployment

### Option 1: Direct Append (Recommended)

```bash
# Append trading config to your ~/.bashrc
cat /home/youssefbahloul/ai-trading-station/SystemConfig/shell/bashrc.trading >> ~/.bashrc

# Reload your shell
source ~/.bashrc
```

### Option 2: Separate File (Cleaner)

```bash
# Copy to home directory
cp /home/youssefbahloul/ai-trading-station/SystemConfig/shell/bashrc.trading ~/.bashrc.trading

# Add this line to your ~/.bashrc:
echo 'source ~/.bashrc.trading' >> ~/.bashrc

# Reload your shell
source ~/.bashrc
```

## What Gets Configured

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `NIC_INTERFACE` | `enp130s0f0np0` | Network interface for tuning scripts |
| `PATH` | Adds `/opt/onload/bin` | Onload utilities |
| `PATH` | Adds `/usr/local/cuda-13.0/bin` | CUDA toolkit |
| `PATH` | Adds `$HOME/ai-trading-station/Services/Trading` | `onload-trading` wrapper |
| `LD_LIBRARY_PATH` | Adds `/opt/onload/lib64` | Onload libraries |
| `LD_LIBRARY_PATH` | Adds `/usr/local/cuda-13.0/lib64` | CUDA libraries |
| `TRADING_PRODUCTION_MODE` | `false` | Monitoring mode toggle |

### Aliases

| Alias | Command | Purpose |
|-------|---------|---------|
| `monitordash` | Launches monitor_dashboard_complete.py with Onload | Real-time trading system dashboard |
| `set-monitoring-mode` | Sources set_monitoring_mode.sh | Toggle production/monitoring mode |
| `trading-mode` | `set-monitoring-mode trading` | Enable production trading mode |
| `monitor-mode` | `set-monitoring-mode notrading` | Enable safe monitoring mode |
| `monitoring-status` | `set-monitoring-mode status` | Show current monitoring mode |

## Verification

After deployment, verify the configuration:

```bash
# Check environment variables
echo $NIC_INTERFACE
echo $TRADING_PRODUCTION_MODE

# Verify onload-trading is in PATH
which onload-trading

# Test aliases
monitoring-status
```

## Critical Dependencies

### Files Referenced by bashrc.trading

1. **`$HOME/ai-trading-station/Services/Trading/onload-trading`**
   - Onload wrapper script
   - Used by: `monitordash` alias
   - **CRITICAL**: If this file moves, update `monitordash` alias

2. **`$HOME/ai-trading-station/Services/Monitoring/Scripts/Utility/set_monitoring_mode.sh`**
   - Monitoring mode toggle script
   - Used by: `set-monitoring-mode` alias

3. **`$HOME/ai-trading-station/Services/Monitoring/Scripts/Runtime/monitor_dashboard_complete.py`**
   - Monitoring dashboard application
   - Used by: `monitordash` alias

## Security Notes

**⚠️ IMPORTANT: GITHUB_TOKEN**

The `bashrc.trading` file does NOT include GitHub tokens or other secrets. If you previously had `GITHUB_TOKEN` in your `~/.bashrc`, consider moving it to:

```bash
# Create a secure secrets file
touch ~/.trading-secrets
chmod 600 ~/.trading-secrets

# Add your token there
echo 'export GITHUB_TOKEN="your_token_here"' >> ~/.trading-secrets

# Source it from ~/.bashrc
echo 'source ~/.trading-secrets' >> ~/.bashrc
```

## Customization

### If Your NIC Name Differs

Edit `bashrc.trading` and change:
```bash
export NIC_INTERFACE='your_nic_name_here'
```

### If You Move onload-trading Wrapper

If you move the wrapper from `Services/Trading/` to `Services/Onload/`, update:
```bash
# In bashrc.trading, change PATH:
export PATH="$HOME/ai-trading-station/Services/Onload:$PATH"

# And update monitordash alias:
alias monitordash='... $HOME/ai-trading-station/Services/Onload/onload-trading python3 ...'
```

## Integration with deploy-system-config-enhanced.sh

The shell configuration is intentionally **NOT** auto-deployed by the system deployment script because:

1. **User-specific** - Each user may have different shell preferences
2. **Session-dependent** - Requires shell reload to take effect
3. **Risk of corruption** - Automated edits to `~/.bashrc` can break the shell

**Manual deployment is required** - Follow the Quick Deployment steps above.

## Maintenance

### Update from Production

If you modify your `~/.bashrc` and want to update the SystemConfig copy:

```bash
# Extract only trading-specific lines (manual review required)
grep -E 'NIC_INTERFACE|onload|cuda|Trading|monitordash|TRADING_PRODUCTION_MODE' ~/.bashrc
```

**Do NOT blindly copy** - Review and extract only trading-related configuration.

### Verify Consistency

Check if your active shell config matches SystemConfig:

```bash
# Compare PATH components
echo $PATH | tr ':' '\n' | grep -E 'onload|cuda|Trading'

# Compare aliases
alias | grep -E 'monitor|trading'

# Compare environment
env | grep -E 'NIC_INTERFACE|TRADING_PRODUCTION_MODE'
```

## Troubleshooting

### Issue: `onload-trading: command not found`

**Solution**: Trading PATH not set. Run:
```bash
export PATH="$HOME/ai-trading-station/Services/Trading:$PATH"
```

### Issue: `monitordash` alias not found

**Solution**: Shell config not loaded. Run:
```bash
source ~/.bashrc
```

### Issue: Network scripts can't find NIC

**Solution**: NIC_INTERFACE not set or wrong. Check:
```bash
echo $NIC_INTERFACE
ip link show
```

### Issue: Onload commands fail

**Solution**: Onload paths not set. Run:
```bash
export PATH="/opt/onload/bin:$PATH"
export LD_LIBRARY_PATH="/opt/onload/lib64:$LD_LIBRARY_PATH"
```

---

*Last Updated: November 9, 2025*  
*Part of AI Trading Station SystemConfig infrastructure*
