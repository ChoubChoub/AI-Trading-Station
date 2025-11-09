# CPU Affinity Management - Quick Reference Card
**Version:** 1.0 (Enhanced with Opus Feedback)  
**Date:** October 22, 2025

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Backup (2 min)
cd ~/ai-trading-station/Monitoring/Scripts && ./backup_systemd_services.sh

# 2. Configure (5 min - includes pre-flight checks & user confirmation)
./configure_cpu_affinity.sh

# 3. Restart Services (2 min)
sudo systemctl restart market-data.target

# 4. Verify (1 min)
./verify_cpu_affinity.sh
```

---

## 📋 CPU Allocation Map

```
CPU 0-1: Network IRQs (kernel)            ← Existing (don't change)
CPU 2:   Prometheus                       ← Already configured
CPU 3:   WebSocket Collectors (both)      ← NEW
CPU 4:   Redis HFT                        ← Already configured  
CPU 5:   QuestDB                          ← NEW
CPU 6-7: Batch Writer (8 workers)         ← NEW
```

---

## 🔍 Essential Commands

### Check Current State
```bash
# Quick status check
systemctl status market-data.target

# Verify CPU affinity for all services
~/ai-trading-station/Monitoring/Scripts/verify_cpu_affinity.sh

# Check actual process affinity
for pid in $(pgrep -f 'binance|batch-writer|questdb'); do
  echo "PID $pid: $(taskset -cp $pid 2>/dev/null | awk '{print $NF}')"
done
```

### Monitor Performance
```bash
# CPU usage per core (refresh every 2 seconds)
mpstat -P ALL 2

# Check Grafana dashboard
xdg-open http://localhost:3000  # or just open in browser

# Watch capture rate
watch -n 5 'curl -s http://localhost:3000/api/health'
```

### Rollback (If Needed)
```bash
cd ~/ai-trading-station/Monitoring/Scripts
./rollback_cpu_affinity.sh
```

---

## ⚠️ Key Monitoring Points (First 24 Hours)

### Critical Metrics
- **Capture Rate:** Must stay >99% (Grafana dashboard)
- **QuestDB GC:** Watch for pauses >500ms
- **Batch Writer CPU:** Should be <80% sustained
- **Service Status:** All "active (running)"

### Check Commands
```bash
# QuestDB GC monitoring
sudo journalctl -u questdb.service --since "1 hour ago" | grep -i "gc"

# Batch writer CPU utilization
pidstat -u 5 3 -p $(pgrep -f batch-writer)

# WebSocket collectors context switches
pidstat -w 5 3 -p $(pgrep -f 'binance.*collector')
```

---

## 🚨 Troubleshooting

### Issue: Service Won't Start
```bash
# Check logs
sudo journalctl -u <service-name> -n 50

# Temporarily disable CPU affinity
sudo systemctl edit <service>.service
# Add: CPUAffinity=
sudo systemctl daemon-reload
sudo systemctl restart <service>
```

### Issue: Poor Performance
```bash
# Check CPU migration events
perf stat -e migrations -p $(pgrep -f <process>) sleep 60

# If migrations >10, check CPU affinity configuration
systemctl show -p CPUAffinity <service>.service
```

### Issue: Need to Rollback
```bash
cd ~/ai-trading-station/Monitoring/Scripts
./rollback_cpu_affinity.sh
# Services will restart and float across CPUs again
```

---

## 📚 Documentation References

- **Full Audit & Plan:** `Documentation/CPU_AFFINITY_AUDIT_AND_PLAN.md`
- **Implementation Checklist:** `Documentation/CPU_AFFINITY_IMPLEMENTATION_CHECKLIST.md`
- **Phase 3 Report:** `Documentation/PHASE3_FINAL_REPORT.md`
- **Smart Speed Architecture:** `Smart-Speed-Trading-Architecture-Crypto.md`

---

## 🎯 Success Indicators

### Immediate (Within 10 minutes)
- ✅ All services "active (running)"
- ✅ Processes pinned to correct CPUs
- ✅ Capture rate >99%
- ✅ No errors in logs

### Short-term (Within 24 hours)
- ✅ No service crashes
- ✅ Stable performance metrics
- ✅ QuestDB GC acceptable
- ✅ CPU utilization balanced

---

## 📞 Quick Help

**Script Locations:**
```
~/ai-trading-station/Monitoring/Scripts/
├── backup_systemd_services.sh
├── configure_cpu_affinity.sh
├── verify_cpu_affinity.sh
└── rollback_cpu_affinity.sh
```

**Check Script Status:**
```bash
ls -lh ~/ai-trading-station/Monitoring/Scripts/*cpu*.sh
```

**Make Scripts Executable (if needed):**
```bash
chmod +x ~/ai-trading-station/Monitoring/Scripts/{backup_systemd_services,configure_cpu_affinity,verify_cpu_affinity,rollback_cpu_affinity}.sh
```

---

**Card Version:** 1.0  
**Last Updated:** October 22, 2025  
**Print this card for quick reference during implementation!**
# CPU Affinity Management Scripts

**Location:** `/home/youssefbahloul/ai-trading-station/Monitoring/Scripts/`  
**Purpose:** Centralized CPU affinity management for market data services  
**Created:** October 22, 2025  
**Status:** Production Ready ✅

---

## 📋 Available Scripts

### 1. `backup_systemd_services.sh`
- **Purpose:** Automated backup of all systemd service files
- **When to use:** Before making any CPU affinity changes
- **Output:** Timestamped backup in `/home/youssefbahloul/ai-trading-station/Archive/`
- **Duration:** 2 minutes
- **Usage:**
  ```bash
  cd ~/ai-trading-station/Monitoring/Scripts
  ./backup_systemd_services.sh
  ```

### 2. `configure_cpu_affinity.sh`
- **Purpose:** Configure CPU affinity for all market data services
- **Features:**
  - Pre-flight system checks
  - User confirmation required
  - Creates systemd override directories
  - Comprehensive post-configuration instructions
- **Duration:** 5 minutes
- **Usage:**
  ```bash
  cd ~/ai-trading-station/Monitoring/Scripts
  ./configure_cpu_affinity.sh
  # Follow prompts and confirm changes
  ```

### 3. `verify_cpu_affinity.sh`
- **Purpose:** Verify CPU affinity configuration and system status
- **Features:**
  - Systemd configuration check
  - Actual process affinity verification
  - Service dependency display
  - CPU governor check
  - Memory usage reporting
  - Advanced monitoring suggestions
- **Duration:** 1 minute
- **Usage:**
  ```bash
  cd ~/ai-trading-station/Monitoring/Scripts
  ./verify_cpu_affinity.sh
  ```

### 4. `rollback_cpu_affinity.sh`
- **Purpose:** Emergency rollback to remove CPU affinity configuration
- **When to use:** If issues arise after implementation
- **Features:**
  - User confirmation required
  - Removes CPU affinity overrides
  - Restarts all services
  - Preserves Prometheus and Redis configurations
- **Duration:** 2 minutes
- **Usage:**
  ```bash
  cd ~/ai-trading-station/Monitoring/Scripts
  ./rollback_cpu_affinity.sh
  # Confirm rollback when prompted
  ```

---

## 🚀 Quick Implementation

### Standard Implementation (Recommended)
```bash
cd ~/ai-trading-station/Monitoring/Scripts

# Step 1: Backup
./backup_systemd_services.sh

# Step 2: Configure (includes user confirmation)
./configure_cpu_affinity.sh

# Step 3: Restart services
sudo systemctl restart market-data.target

# Step 4: Verify
./verify_cpu_affinity.sh

# Step 5: Monitor Grafana
# URL: http://localhost:3000
# Ensure capture rate stays >99%
```

### Emergency Rollback
```bash
cd ~/ai-trading-station/Monitoring/Scripts
./rollback_cpu_affinity.sh
```

---

## 📊 CPU Allocation After Implementation

```
CPU 0-1: Network IRQs (kernel)            ← Existing (unchanged)
CPU 2:   Prometheus                       ← Already configured
CPU 3:   WebSocket Collectors             ← NEW
         - binance-trades.service
         - binance-bookticker.service
CPU 4:   Redis HFT                        ← Already configured
CPU 5:   QuestDB                          ← NEW
         - Monitor JVM/GC (first 24 hours)
CPU 6-7: Batch Writer                     ← NEW
         - 8 workers on 2 cores
         - Monitor CPU utilization
```

---

## 📚 Documentation

### Full Documentation
- **Audit & Plan:** `~/ai-trading-station/Documentation/CPU_AFFINITY_AUDIT_AND_PLAN.md`
- **Implementation Checklist:** `~/ai-trading-station/Documentation/CPU_AFFINITY_IMPLEMENTATION_CHECKLIST.md`
- **Quick Reference:** `~/ai-trading-station/Documentation/CPU_AFFINITY_QUICK_REFERENCE.md`
- **Delivery Summary:** `~/ai-trading-station/Documentation/CPU_AFFINITY_DELIVERY_SUMMARY.md`

### Quick Commands
```bash
# View all CPU affinity documentation
ls -lh ~/ai-trading-station/Documentation/CPU_AFFINITY_*.md

# View quick reference
cat ~/ai-trading-station/Documentation/CPU_AFFINITY_QUICK_REFERENCE.md

# View implementation checklist
cat ~/ai-trading-station/Documentation/CPU_AFFINITY_IMPLEMENTATION_CHECKLIST.md
```

---

## ✅ Pre-Implementation Checklist

Before running scripts, ensure:

- [ ] Grafana dashboard accessible (`curl http://localhost:3000/api/health`)
- [ ] All market data services running (`systemctl status market-data.target`)
- [ ] Current capture rate documented (baseline for comparison)
- [ ] Disk space available for backups (`df -h ~/ai-trading-station/Archive`)
- [ ] Team notified of maintenance window

---

## 🎯 Success Criteria

### Immediate (Within 10 minutes)
- ✅ All services "active (running)"
- ✅ All processes pinned to correct CPUs
- ✅ Capture rate >99%
- ✅ No errors in service logs

### Short-term (Within 24 hours)
- ✅ No service crashes or restarts
- ✅ Capture rate consistently >99%
- ✅ QuestDB GC pauses acceptable (<500ms)
- ✅ Batch writer CPU utilization reasonable (<80%)

---

## 🔍 Monitoring Commands

### Check Service Status
```bash
# All services
systemctl status market-data.target

# Individual service
systemctl status <service-name>.service

# View logs
sudo journalctl -u <service-name> -n 50
```

### Verify CPU Affinity
```bash
# Run verification script
~/ai-trading-station/Monitoring/Scripts/verify_cpu_affinity.sh

# Check specific process
taskset -cp $(pgrep -f <process-name>)

# Watch CPU usage per core
mpstat -P ALL 2
```

### Monitor Performance
```bash
# Check Grafana dashboard
# URL: http://localhost:3000

# Monitor QuestDB GC
sudo journalctl -u questdb.service --since "1 hour ago" | grep -i "gc"

# Monitor batch-writer CPU
pidstat -u 5 3 -p $(pgrep -f batch-writer)

# Check context switches
pidstat -w 5 3 -p $(pgrep -f 'binance.*collector')
```

---

## 🚨 Troubleshooting

### Service Won't Start
```bash
# Check logs
sudo journalctl -u <service-name> -n 50

# Check CPU affinity config
systemctl show -p CPUAffinity <service-name>.service

# Temporarily disable affinity
sudo systemctl edit <service>.service
# Add: CPUAffinity=
sudo systemctl daemon-reload
sudo systemctl restart <service>
```

### Performance Issues
```bash
# Check CPU migration
perf stat -e migrations -p $(pgrep -f <process>) sleep 60

# Check cache efficiency
perf stat -e cache-misses,cache-references -p $(pgrep -f <process>) sleep 60

# If persistent issues, rollback
./rollback_cpu_affinity.sh
```

---

## 📞 Support

### Documentation References
- Full audit and plan in `Documentation/CPU_AFFINITY_AUDIT_AND_PLAN.md`
- Step-by-step checklist in `Documentation/CPU_AFFINITY_IMPLEMENTATION_CHECKLIST.md`
- Quick reference card in `Documentation/CPU_AFFINITY_QUICK_REFERENCE.md`

### Emergency Procedures
1. Check Grafana dashboard for issues
2. Review service logs: `sudo journalctl -u <service> -n 100`
3. Verify CPU affinity: `./verify_cpu_affinity.sh`
4. Rollback if necessary: `./rollback_cpu_affinity.sh`

---

## 🔐 Security Notes

- All scripts require user confirmation for destructive operations
- Backup procedures preserve original configurations
- Rollback procedure available at all times
- No passwords or sensitive data in scripts
- Service files backed up with proper permissions

---

## 📝 Version History

- **v1.0** (2025-10-22): Initial release with Opus feedback incorporated
  - 4 production-ready scripts
  - Comprehensive documentation
  - Full backup and rollback procedures
  - 24-hour monitoring plan

---

**Status:** ✅ Production Ready  
**Last Updated:** October 22, 2025  
**Maintained By:** AI Trading Station Team


---

# CPU Affinity Scripts Documentation


# CPU Affinity Management Scripts

**Location:** `/home/youssefbahloul/ai-trading-station/Monitoring/Scripts/`  
**Purpose:** Centralized CPU affinity management for market data services  
**Created:** October 22, 2025  
**Status:** Production Ready ✅

---

## 📋 Available Scripts

### 1. `backup_systemd_services.sh`
- **Purpose:** Automated backup of all systemd service files
- **When to use:** Before making any CPU affinity changes
- **Output:** Timestamped backup in `/home/youssefbahloul/ai-trading-station/Archive/`
- **Duration:** 2 minutes
- **Usage:**
  ```bash
  cd ~/ai-trading-station/Monitoring/Scripts
  ./backup_systemd_services.sh
  ```

### 2. `configure_cpu_affinity.sh`
- **Purpose:** Configure CPU affinity for all market data services
- **Features:**
  - Pre-flight system checks
  - User confirmation required
  - Creates systemd override directories
  - Comprehensive post-configuration instructions
- **Duration:** 5 minutes
- **Usage:**
  ```bash
  cd ~/ai-trading-station/Monitoring/Scripts
  ./configure_cpu_affinity.sh
  # Follow prompts and confirm changes
  ```

### 3. `verify_cpu_affinity.sh`
- **Purpose:** Verify CPU affinity configuration and system status
- **Features:**
  - Systemd configuration check
  - Actual process affinity verification
  - Service dependency display
  - CPU governor check
  - Memory usage reporting
  - Advanced monitoring suggestions
- **Duration:** 1 minute
- **Usage:**
  ```bash
  cd ~/ai-trading-station/Monitoring/Scripts
  ./verify_cpu_affinity.sh
  ```

### 4. `rollback_cpu_affinity.sh`
- **Purpose:** Emergency rollback to remove CPU affinity configuration
- **When to use:** If issues arise after implementation
- **Features:**
  - User confirmation required
  - Removes CPU affinity overrides
  - Restarts all services
  - Preserves Prometheus and Redis configurations
- **Duration:** 2 minutes
- **Usage:**
  ```bash
  cd ~/ai-trading-station/Monitoring/Scripts
  ./rollback_cpu_affinity.sh
  # Confirm rollback when prompted
  ```

---

## 🚀 Quick Implementation

### Standard Implementation (Recommended)
```bash
cd ~/ai-trading-station/Monitoring/Scripts

# Step 1: Backup
./backup_systemd_services.sh

# Step 2: Configure (includes user confirmation)
./configure_cpu_affinity.sh

# Step 3: Restart services
sudo systemctl restart market-data.target

# Step 4: Verify
./verify_cpu_affinity.sh

# Step 5: Monitor Grafana
# URL: http://localhost:3000
# Ensure capture rate stays >99%
```

### Emergency Rollback
```bash
cd ~/ai-trading-station/Monitoring/Scripts
./rollback_cpu_affinity.sh
```

---

## 📊 CPU Allocation After Implementation

```
CPU 0-1: Network IRQs (kernel)            ← Existing (unchanged)
CPU 2:   Prometheus                       ← Already configured
CPU 3:   WebSocket Collectors             ← NEW
         - binance-trades.service
         - binance-bookticker.service
CPU 4:   Redis HFT                        ← Already configured
CPU 5:   QuestDB                          ← NEW
         - Monitor JVM/GC (first 24 hours)
CPU 6-7: Batch Writer                     ← NEW
         - 8 workers on 2 cores
         - Monitor CPU utilization
```

---

## 📚 Documentation

### Full Documentation
- **Audit & Plan:** `~/ai-trading-station/Documentation/CPU_AFFINITY_AUDIT_AND_PLAN.md`
- **Implementation Checklist:** `~/ai-trading-station/Documentation/CPU_AFFINITY_IMPLEMENTATION_CHECKLIST.md`
- **Quick Reference:** `~/ai-trading-station/Documentation/CPU_AFFINITY_QUICK_REFERENCE.md`
- **Delivery Summary:** `~/ai-trading-station/Documentation/CPU_AFFINITY_DELIVERY_SUMMARY.md`

### Quick Commands
```bash
# View all CPU affinity documentation
ls -lh ~/ai-trading-station/Documentation/CPU_AFFINITY_*.md

# View quick reference
cat ~/ai-trading-station/Documentation/CPU_AFFINITY_QUICK_REFERENCE.md

# View implementation checklist
cat ~/ai-trading-station/Documentation/CPU_AFFINITY_IMPLEMENTATION_CHECKLIST.md
```

---

## ✅ Pre-Implementation Checklist

Before running scripts, ensure:

- [ ] Grafana dashboard accessible (`curl http://localhost:3000/api/health`)
- [ ] All market data services running (`systemctl status market-data.target`)
- [ ] Current capture rate documented (baseline for comparison)
- [ ] Disk space available for backups (`df -h ~/ai-trading-station/Archive`)
- [ ] Team notified of maintenance window

---

## 🎯 Success Criteria

### Immediate (Within 10 minutes)
- ✅ All services "active (running)"
- ✅ All processes pinned to correct CPUs
- ✅ Capture rate >99%
- ✅ No errors in service logs

### Short-term (Within 24 hours)
- ✅ No service crashes or restarts
- ✅ Capture rate consistently >99%
- ✅ QuestDB GC pauses acceptable (<500ms)
- ✅ Batch writer CPU utilization reasonable (<80%)

---

## 🔍 Monitoring Commands

### Check Service Status
```bash
# All services
systemctl status market-data.target

# Individual service
systemctl status <service-name>.service

# View logs
sudo journalctl -u <service-name> -n 50
```

### Verify CPU Affinity
```bash
# Run verification script
~/ai-trading-station/Monitoring/Scripts/verify_cpu_affinity.sh

# Check specific process
taskset -cp $(pgrep -f <process-name>)

# Watch CPU usage per core
mpstat -P ALL 2
```

### Monitor Performance
```bash
# Check Grafana dashboard
# URL: http://localhost:3000

# Monitor QuestDB GC
sudo journalctl -u questdb.service --since "1 hour ago" | grep -i "gc"

# Monitor batch-writer CPU
pidstat -u 5 3 -p $(pgrep -f batch-writer)

# Check context switches
pidstat -w 5 3 -p $(pgrep -f 'binance.*collector')
```

---

## 🚨 Troubleshooting

### Service Won't Start
```bash
# Check logs
sudo journalctl -u <service-name> -n 50

# Check CPU affinity config
systemctl show -p CPUAffinity <service-name>.service

# Temporarily disable affinity
sudo systemctl edit <service>.service
# Add: CPUAffinity=
sudo systemctl daemon-reload
sudo systemctl restart <service>
```

### Performance Issues
```bash
# Check CPU migration
perf stat -e migrations -p $(pgrep -f <process>) sleep 60

# Check cache efficiency
perf stat -e cache-misses,cache-references -p $(pgrep -f <process>) sleep 60

# If persistent issues, rollback
./rollback_cpu_affinity.sh
```

---

## 📞 Support

### Documentation References
- Full audit and plan in `Documentation/CPU_AFFINITY_AUDIT_AND_PLAN.md`
- Step-by-step checklist in `Documentation/CPU_AFFINITY_IMPLEMENTATION_CHECKLIST.md`
- Quick reference card in `Documentation/CPU_AFFINITY_QUICK_REFERENCE.md`

### Emergency Procedures
1. Check Grafana dashboard for issues
2. Review service logs: `sudo journalctl -u <service> -n 100`
3. Verify CPU affinity: `./verify_cpu_affinity.sh`
4. Rollback if necessary: `./rollback_cpu_affinity.sh`

---

## 🔐 Security Notes

- All scripts require user confirmation for destructive operations
- Backup procedures preserve original configurations
- Rollback procedure available at all times
- No passwords or sensitive data in scripts
- Service files backed up with proper permissions

---

## 📝 Version History

- **v1.0** (2025-10-22): Initial release with Opus feedback incorporated
  - 4 production-ready scripts
  - Comprehensive documentation
  - Full backup and rollback procedures
  - 24-hour monitoring plan

---

**Status:** ✅ Production Ready  
**Last Updated:** October 22, 2025  
**Maintained By:** AI Trading Station Team
