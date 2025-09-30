# Redis HFT Operations Handbook
**Daily Operations Guide for AI Trading Station**

**Date**: September 28, 2025  
**Purpose**: Comprehensive guide for daily operations and maintenance

---

## 🎯 **Daily Operations Workflow**

### **Morning Startup Routine (5 minutes)**
```bash
cd /home/youssefbahloul/ai-trading-station/redis-setup

# 1. Performance Gate - Go/No-Go Decision
./production/perf-gate.sh --metrics-only

# 2. System Health Check
./monitoring/health-report.sh

# 3. Quick Performance Verification
./monitoring/redis-hft-monitor_to_json.sh | jq .rtt.p99
```

**Expected Results**:
- Performance Gate: ✅ **PASS**
- P99 RTT: **10-15μs** (excellent for HFT)
- System Status: All green indicators

---

## 📊 **Performance Monitoring During Trading**

### **Continuous Monitoring Options**

#### **Option 1: Live Dashboard** (Recommended)
```bash
# Real-time monitoring with color-coded alerts
./monitoring/live-dashboard.sh
```

#### **Option 2: Periodic Checks** (Every 30 minutes)
```bash
# Quick performance snapshot
./monitoring/redis-hft-monitor_to_json.sh | jq '{rtt: .rtt.p99, ops: .health.ops_per_sec}'
```

#### **Option 3: Background Monitoring**
```bash
# Continuous tail monitoring (runs in background)
./production/extended_tail_sampler.py --duration 3600 --samples 5000 &
```

### **Performance Thresholds**
| Metric | Excellent | Good | Warning | **STOP TRADING** |
|--------|-----------|------|---------|------------------|
| **P99 RTT** | <12μs | 12-15μs | 15-20μs | **>20μs** |
| **Jitter** | <2μs | 2-3μs | 3-5μs | **>5μs** |
| **Memory** | <5MB | 5-20MB | 20-50MB | **>50MB** |
| **Ops/sec** | >20k | 15-20k | 10-15k | **<10k** |

---

## 🔧 **Configuration Management**

### **Current System Configuration**
- **Performance Thresholds**: `setup/configs/thresholds/perf-thresholds.env`
- **Tail Monitoring**: `setup/configs/thresholds/tail-thresholds.env`  
- **Redis Config**: Optimized for HFT with CPU isolation
- **CPU Isolation**: Cores 2,3 isolated for trading workloads

### **Configuration Verification**
```bash
# Check CPU isolation
cat /proc/cmdline | grep isolcpus

# Verify Redis service
systemctl status redis-hft

# Check performance thresholds
cat setup/configs/thresholds/perf-thresholds.env
```

---

## 📋 **Backup & State Management**

### **State Files** (Automatic)
- **Tail Monitoring**: `state/tail-run.json` (auto-updated)
- **Gate Decisions**: `logs/gate-decisions.log` (audit trail)
- **Performance History**: Captured in monitoring logs

### **Manual Backup** (Weekly)
```bash
# Backup critical state files
mkdir -p backups/$(date +%Y%m%d)
cp state/*.json backups/$(date +%Y%m%d)/
cp logs/*.log backups/$(date +%Y%m%d)/
```

---

## 🚨 **Alert Response Procedures**

### **Performance Gate FAIL**
**Immediate Actions**:
1. **Stop Trading** - System not ready
2. **Check Redis**: `systemctl status redis-hft`
3. **Check Performance**: `./monitoring/health-report.sh`
4. **Investigate**: `./production/runtime-fingerprint.sh --pretty`

### **High P99 Latency (>15μs)**
**Response Steps**:
1. **Check system load**: `htop` 
2. **Verify CPU isolation**: `cat /proc/cmdline | grep isolcpus`
3. **Check memory**: `free -h`
4. **Run tail analysis**: `./production/extended_tail_sampler.py --duration 300`

### **Redis Connection Issues**
**Diagnostic Steps**:
1. **Test connection**: `redis-cli ping`
2. **Check service**: `systemctl status redis-hft`
3. **Check port**: `ss -tulpn | grep :6379`
4. **Review logs**: `journalctl -u redis-hft -n 50`

---

## 📈 **Performance Optimization Maintenance**

### **Weekly Maintenance** (End of Week)
```bash
# 1. Performance trend analysis
python3 production/gate_decision_ledger.py --show-recent 50

# 2. Environment drift check
./production/runtime-fingerprint.sh --pretty

# 3. Clean old logs (keep last 30 days)
find logs/ -name "*.log" -mtime +30 -delete

# 4. System resource review
./monitoring/health-report.sh
```

### **Monthly Review**
- Review performance trends from tail monitoring data
- Check for any configuration drift
- Update documentation if procedures change
- Validate backup integrity

---

## 🎯 **Key Performance Indicators (KPIs)**

### **Daily KPIs to Track**
- **Morning Gate Status**: Must be PASS before trading
- **Average P99 RTT**: Target <12μs consistently  
- **Peak P99 RTT**: Should not exceed 15μs during trading
- **System Uptime**: Redis service availability
- **Memory Stability**: No growing memory usage

### **Weekly KPIs to Review**
- **Performance Gate Success Rate**: >95%
- **Tail Event Classification**: Monitor SCHED/IRQ/ALLOC patterns
- **System Configuration Stability**: No drift warnings

---

## 📞 **Emergency Contacts & Escalation**

### **System Recovery Priority**
1. **Performance Gate Failure**: Immediate trading halt
2. **Redis Service Down**: Critical system recovery needed  
3. **High Latency**: Monitor closely, may need intervention
4. **Memory Issues**: Check for leaks, consider restart

### **Recovery Procedures**
```bash
# Redis service restart (if needed)
sudo systemctl restart redis-hft
sleep 5
./production/perf-gate.sh --metrics-only

# Full system validation after any restart
./monitoring/health-report.sh
```

---

## ✅ **Operations Success Checklist**

### **Daily Success Criteria**
- [ ] Performance Gate: PASS
- [ ] P99 RTT: <15μs consistently
- [ ] No system alerts or warnings
- [ ] Redis service: Active and responsive
- [ ] Memory usage: Stable and <20MB

### **System Health Indicators**
- [ ] CPU isolation active and effective
- [ ] No configuration drift detected
- [ ] Monitoring systems operational
- [ ] Audit trails being captured

**When all criteria are met, your Redis HFT system is ready for institutional-grade trading operations.** 🚀