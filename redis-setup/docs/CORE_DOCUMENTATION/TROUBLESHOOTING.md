# Redis HFT Troubleshooting Guide
**Problem Diagnosis & Resolution for AI Trading Station**

**Date**: September 28, 2025  
**Purpose**: Comprehensive troubleshooting guide for Redis HFT issues

---

## 🚨 **Emergency Response Flowchart**

```
Performance Issue Detected
           ↓
    Is Redis responding?
           ↓
    NO → Redis Service Issue (Section 1)
           ↓
    YES → Check Performance Gate
           ↓
    FAIL → Performance Degradation (Section 2)
           ↓
    PASS → Environmental Issue (Section 3)
```

---

## 1️⃣ **Redis Service Issues**

### **Symptoms**
- `redis-cli ping` fails or times out
- Performance gate shows connection errors
- Trading applications cannot connect

### **Diagnosis Steps**
```bash
# Check Redis service status
systemctl status redis-hft

# Check if Redis port is listening
ss -tulpn | grep :6379

# Check Redis logs
journalctl -u redis-hft -n 50

# Check system resources
free -h && df -h
```

### **Common Issues & Solutions**

#### **Service Not Running**
```bash
# Start Redis service
sudo systemctl start redis-hft

# Enable auto-start
sudo systemctl enable redis-hft

# Verify startup
systemctl status redis-hft
```

#### **Port Already in Use**
```bash
# Find process using port 6379
sudo lsof -i :6379

# Kill conflicting process (if safe)
sudo kill -9 <PID>

# Restart Redis
sudo systemctl restart redis-hft
```

#### **Memory Issues**
```bash
# Check available memory
free -h

# Check Redis memory usage
redis-cli info memory

# If memory exhausted, consider restart
sudo systemctl restart redis-hft
```

---

## 2️⃣ **Performance Degradation**

### **Symptoms**
- P99 RTT > 15μs consistently
- Performance gate failure
- High jitter (>3μs)
- Low throughput (<10k ops/sec)

### **Performance Diagnosis**
```bash
# Get detailed performance metrics
./monitoring/redis-hft-monitor_to_json.sh | jq .

# Run comprehensive health check
./monitoring/health-report.sh

# Check tail analysis for burst classification
./production/extended_tail_sampler.py --duration 300
```

### **Common Performance Issues**

#### **High CPU Load**
**Symptoms**: P99 > 20μs, system sluggish
```bash
# Check CPU usage
htop

# Verify CPU isolation
cat /proc/cmdline | grep isolcpus

# Check Redis CPU affinity
ps -eo pid,comm,psr | grep redis
```

**Solutions**:
- Ensure Redis runs on isolated CPU
- Check for background processes on isolated cores
- Verify CPU governor is set to "performance"

#### **Memory Pressure**
**Symptoms**: High latency, system swapping
```bash
# Check memory usage
free -h

# Check swap usage
swapon --show

# Check Redis memory
redis-cli info memory
```

**Solutions**:
- Clear unnecessary processes
- Check for memory leaks in Redis
- Consider Redis restart if memory >50MB

#### **Disk I/O Issues**
**Symptoms**: Intermittent high latency
```bash
# Check disk I/O
iostat -x 1

# Check disk space
df -h

# Check for disk errors
dmesg | grep -i error
```

**Solutions**:
- Ensure sufficient free disk space
- Check for failing drives
- Move logs to different disk if needed

---

## 3️⃣ **Environmental Issues**

### **Configuration Drift**
**Symptoms**: Environment fingerprint warnings
```bash
# Check for configuration changes
./production/runtime-fingerprint.sh --pretty

# Compare with baseline
# Check kernel parameters
cat /proc/cmdline

# Verify system settings
cat /proc/sys/vm/swappiness
```

**Solutions**:
- Restore original kernel parameters
- Verify CPU isolation settings
- Check system configuration files

### **Network Issues**
**Symptoms**: Connection timeouts, intermittent failures
```bash
# Test local connection
redis-cli ping

# Check network interface
ip addr show

# Check for network errors
ss -i | grep :6379
```

**Solutions**:
- Restart network service if needed
- Check firewall settings
- Verify localhost resolution

---

## 4️⃣ **Monitoring System Issues**

### **Monitoring Scripts Failing**
**Symptoms**: Health reports fail, dashboard not updating

#### **Script Permission Issues**
```bash
# Check executable permissions
ls -la monitoring/*.sh production/*.sh

# Fix permissions if needed
chmod +x monitoring/*.sh production/*.sh
```

#### **Missing Dependencies**
```bash
# Check Python dependencies
python3 -c "import redis, json, psutil"

# Check jq availability
which jq

# Install missing tools
sudo apt-get install jq
```

### **State File Corruption**
**Symptoms**: Tail monitoring fails, gate decisions not logged
```bash
# Check state file integrity
python3 -c "import json; print(json.load(open('state/tail-run.json')))"

# Backup and reset if corrupted
mv state/tail-run.json state/tail-run.json.backup
echo '{"recent_windows": []}' > state/tail-run.json
```

---

## 5️⃣ **Advanced Troubleshooting**

### **Performance Profiling**
For persistent performance issues:
```bash
# Extended tail analysis
./production/extended_tail_sampler.py --duration 1800 --samples 10000

# System-wide profiling
perf top -p $(pgrep redis-server)

# Network latency analysis
./production/network_latency_harness.py --comprehensive
```

### **Log Analysis**
```bash
# Redis logs with timestamps
journalctl -u redis-hft --since "1 hour ago"

# System logs for errors
journalctl --since "1 hour ago" | grep -i error

# Check dmesg for hardware issues
dmesg -T | tail -50
```

---

## 🔧 **Recovery Procedures**

### **Quick Recovery** (Performance issues)
```bash
# 1. Restart Redis service
sudo systemctl restart redis-hft

# 2. Wait for startup
sleep 10

# 3. Validate system
./production/perf-gate.sh --metrics-only

# 4. Verify performance
./monitoring/redis-hft-monitor_to_json.sh | jq .rtt
```

### **Full System Recovery** (Major issues)
```bash
# 1. Stop Redis
sudo systemctl stop redis-hft

# 2. Clear any locks/temp files
sudo rm -f /tmp/redis*.sock

# 3. Check system resources
free -h && df -h

# 4. Restart Redis
sudo systemctl start redis-hft

# 5. Full validation
./monitoring/health-report.sh
```

### **Configuration Recovery**
If system configuration is corrupted:
```bash
# 1. Check current settings
./production/runtime-fingerprint.sh --pretty

# 2. Compare with known good configuration
# 3. Apply system optimizations if needed
sudo sysctl -p /etc/sysctl.conf

# 4. Verify CPU isolation
cat /proc/cmdline | grep isolcpus

# 5. Restart if kernel parameters changed
# (Reboot required for isolcpus changes)
```

---

## 📊 **Troubleshooting Decision Matrix**

| Symptom | Most Likely Cause | First Action | Advanced Action |
|---------|-------------------|--------------|-----------------|
| **No Redis connection** | Service down | `systemctl start redis-hft` | Check logs, restart system |
| **P99 > 20μs** | CPU/Memory pressure | Check `htop`, `free -h` | Profile with `perf`, check isolation |
| **High jitter** | System interference | Check CPU isolation | Analyze IRQ affinity |
| **Low throughput** | Configuration issue | Check Redis config | Analyze network stack |
| **Intermittent failures** | Environment drift | Run fingerprint check | Compare with baseline |

---

## 🚨 **When to Escalate**

### **Immediate Escalation** (Stop Trading)
- Performance gate consistently failing (>5 minutes)
- P99 RTT > 25μs sustained
- Redis service cannot be restored
- System hardware failures detected

### **Schedule Maintenance** (Non-urgent)
- Gradual performance degradation over days
- Configuration drift warnings
- Disk space approaching limits
- Memory usage slowly growing

---

## ✅ **Recovery Validation Checklist**

After any troubleshooting:
- [ ] Redis service: `systemctl status redis-hft` → **active**
- [ ] Connection test: `redis-cli ping` → **PONG**
- [ ] Performance gate: `./production/perf-gate.sh --metrics-only` → **PASS**
- [ ] Performance check: P99 RTT < 15μs
- [ ] System health: `./monitoring/health-report.sh` → All green
- [ ] Monitoring: All scripts working
- [ ] State files: No corruption detected

**When all checks pass, system is ready to resume trading operations.** ✅