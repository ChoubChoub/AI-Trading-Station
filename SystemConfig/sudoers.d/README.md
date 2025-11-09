# Sudoers Configuration for AI Trading Station

## Purpose

This directory contains passwordless sudo configurations for trading system operations that require root privileges. These configurations enable monitoring, system management, and desktop mode control without password prompts during critical trading operations.

---

## Important: File Permissions

**SystemConfig copies (this directory):**
- Ownership: `youssefbahloul:youssefbahloul`
- Permissions: `0644` (rw-r--r--)
- Purpose: Version control, editing, portability

**Deployed files in `/etc/sudoers.d/`:**
- Ownership: `root:root`
- Permissions: `0440` (r--r-----)
- Purpose: Security (sudo requires these strict permissions)

The deployment script automatically sets correct ownership and permissions when deploying to `/etc/sudoers.d/`.

---

## Files

| File | Purpose | Commands Allowed |
|------|---------|------------------|
| `desktop-toggle` | Desktop mode management | `/home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh` |
| `redis-monitoring` | Redis health monitoring | Redis inspection commands |
| `lspci-monitoring` | GPU/PCIe monitoring | `lspci` hardware inspection |

---

## Deployment

### Using Deployment Script (Recommended)

The enhanced deployment script will handle sudoers deployment:

```bash
cd /home/youssefbahloul/ai-trading-station/SystemConfig/

# Test first
./deploy-system-config-enhanced.sh --test sudoers

# Deploy
sudo ./deploy-system-config-enhanced.sh sudoers
```

### Manual Deployment

```bash
# Copy all sudoers files
sudo cp SystemConfig/sudoers.d/desktop-toggle /etc/sudoers.d/
sudo cp SystemConfig/sudoers.d/redis-monitoring /etc/sudoers.d/
sudo cp SystemConfig/sudoers.d/lspci-monitoring /etc/sudoers.d/

# Set correct permissions (CRITICAL)
sudo chmod 0440 /etc/sudoers.d/desktop-toggle
sudo chmod 0440 /etc/sudoers.d/redis-monitoring
sudo chmod 0440 /etc/sudoers.d/lspci-monitoring

# Validate syntax
sudo visudo -c
```

---

## File Contents

### desktop-toggle

Allows passwordless execution of desktop mode toggle script:

```bash
# Allow passwordless execution of desktop mode toggle script
# AI Trading Station - Desktop Mode Management
youssefbahloul ALL=(ALL) NOPASSWD: /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh
```

**Used by aliases:**
- `desktop-status`
- `desktop-start`
- `desktop-stop`
- `desktop-server`
- `desktop-mode`

### redis-monitoring

Allows passwordless Redis monitoring commands for health checks.

### lspci-monitoring

Allows passwordless GPU/PCIe inspection for hardware monitoring.

---

## Security Considerations

### File Permissions

**CRITICAL:** Sudoers files MUST have `0440` permissions (read-only for root):

```bash
sudo chmod 0440 /etc/sudoers.d/*
```

**Why:** Incorrect permissions cause sudo to ignore the file for security reasons.

### Validation

Always validate sudoers syntax after deployment:

```bash
sudo visudo -c
```

Expected output:
```
/etc/sudoers: parsed OK
/etc/sudoers.d/README: parsed OK
/etc/sudoers.d/desktop-toggle: parsed OK
/etc/sudoers.d/lspci-monitoring: parsed OK
/etc/sudoers.d/redis-monitoring: parsed OK
```

### Path Specificity

All sudoers rules use **absolute paths** to scripts:
- ✅ Secure: Only specific script at specific location can run
- ❌ Insecure: Wildcards or partial paths would allow privilege escalation

### User Specificity

Rules are tied to specific username (`youssefbahloul`):
- **Portable Deployment:** Update username when deploying to different systems
- **Multi-User Systems:** Create separate sudoers files per user

---

## Updating Sudoers Files

### After Modifying SystemConfig Copy

```bash
# 1. Edit file in SystemConfig
nano SystemConfig/sudoers.d/desktop-toggle

# 2. Test deployment
./deploy-system-config-enhanced.sh --test sudoers

# 3. Deploy to system
sudo ./deploy-system-config-enhanced.sh sudoers

# 4. Validate
sudo visudo -c
```

### After Modifying System File

```bash
# 1. Edit system file
sudo visudo -f /etc/sudoers.d/desktop-toggle

# 2. Validate
sudo visudo -c

# 3. Copy back to SystemConfig for version control
sudo cp /etc/sudoers.d/desktop-toggle SystemConfig/sudoers.d/
```

---

## Adding New Sudoers Rules

### Example: Add New Trading Script

1. **Create sudoers file in SystemConfig:**

```bash
cat > SystemConfig/sudoers.d/new-script << 'EOF'
# Allow passwordless execution of new trading script
youssefbahloul ALL=(ALL) NOPASSWD: /home/youssefbahloul/ai-trading-station/Services/System/new-script.sh
EOF
```

2. **Test deployment:**

```bash
./deploy-system-config-enhanced.sh --test sudoers
```

3. **Deploy:**

```bash
sudo ./deploy-system-config-enhanced.sh sudoers
```

4. **Validate:**

```bash
sudo visudo -c
```

5. **Test the command:**

```bash
sudo /home/youssefbahloul/ai-trading-station/Services/System/new-script.sh
# Should run without password prompt
```

---

## Troubleshooting

### Problem: Sudo still asks for password

**Causes:**
1. Incorrect file permissions
2. Syntax error in sudoers file
3. Wrong username in sudoers file
4. Script path doesn't match exactly

**Solution:**

```bash
# Check permissions
ls -la /etc/sudoers.d/desktop-toggle
# Should show: -r--r----- root root

# Validate syntax
sudo visudo -c

# Check username
whoami
grep youssefbahloul /etc/sudoers.d/desktop-toggle

# Verify script path
ls -la /home/youssefbahloul/ai-trading-station/Services/System/toggle-desktop-mode.sh
```

### Problem: visudo reports syntax error

**Solution:**

```bash
# Check which file has the error
sudo visudo -c

# Edit the problematic file
sudo visudo -f /etc/sudoers.d/desktop-toggle

# Common issues:
# - Missing NOPASSWD keyword
# - Incorrect path format
# - Trailing whitespace
```

### Problem: File ignored by sudo

**Solution:**

```bash
# Sudoers files must NOT contain '.' or '~' in filename
# Bad: desktop-toggle.conf, desktop-toggle~
# Good: desktop-toggle

# Also check permissions
sudo chmod 0440 /etc/sudoers.d/desktop-toggle
```

---

## Integration with Trading System

### Desktop Mode Management

The `desktop-toggle` sudoers file enables passwordless desktop mode control, critical for:

1. **Pre-Trading Setup**: Ensure server mode without manual intervention
2. **Automated Scripts**: Toggle desktop via cron or startup scripts
3. **Remote Management**: SSH-based desktop control without password
4. **Monitoring Dashboards**: Display desktop status in monitoring tools

### Monitoring Operations

The `redis-monitoring` and `lspci-monitoring` sudoers files enable:

1. **Health Check Scripts**: Automated Redis and GPU monitoring
2. **Dashboard Data**: Real-time hardware stats without sudo prompts
3. **Alerting Systems**: Automated checks for system issues
4. **Performance Tracking**: Continuous hardware monitoring

---

## Best Practices

1. ✅ **Always validate**: Run `sudo visudo -c` after any changes
2. ✅ **Use absolute paths**: Never use relative paths or wildcards
3. ✅ **Specific commands**: Grant access to specific scripts only
4. ✅ **Document purpose**: Add comments explaining each rule
5. ✅ **Version control**: Keep SystemConfig copies updated
6. ✅ **Test first**: Use `--test` mode before deployment
7. ✅ **Backup**: Deployment script creates `.backup` files automatically

---

## Related Documentation

- **Deployment Guide**: `SystemConfig/README.md`
- **Desktop Mode Management**: `Documentation/Operations/Component_Guides/desktop-mode-management.md`
- **System Services**: `Services/System/README.md`

---

*Last Updated: November 9, 2025*  
*Part of AI Trading Station SystemConfig infrastructure*
