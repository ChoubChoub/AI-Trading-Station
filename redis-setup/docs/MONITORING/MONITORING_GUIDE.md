# Redis HFT Monitoring & Performance Guide
**Comprehensive System Health & KPI Monitoring for AI Trading Station**

**Date**: September 28, 2025  
**Purpose**: Complete guide to monitor, understand, and act on system performance indicators

---

## 🎯 **Executive Summary**

Your Redis HFT system has **4 layers of monitoring** that work together to provide complete visibility:

1. **Basic Performance Monitoring** (redis-hft-monitor scripts)
2. **Performance Gates** (perf-gate.sh) 
3. **Advanced Tail Monitoring** (P99.9 precision)
4. **Environment Drift Detection** (runtime fingerprinting)

---

## 📊 **The Complete Monitoring Stack**

```
┌─────────────────────────────────────────────────────────────┐
│                    MONITORING HIERARCHY                    │
├─────────────────────────────────────────────────────────────┤
│ 🔴 CRITICAL ALERTS    │ Performance Gate (perf-gate.sh)    │
│                       │ • P99 > 20μs = FAIL               │
│                       │ • Environment drift = FAIL         │
├─────────────────────────────────────────────────────────────┤
│ 🟡 ADVANCED METRICS   │ Tail Monitoring (P99.9 precision) │
│                       │ • Burst classification             │
│                       │ • Long-term trend analysis         │
├─────────────────────────────────────────────────────────────┤
│ 🟢 BASIC PERFORMANCE  │ JSON Monitoring Scripts           │
│                       │ • P50/P95/P99 latency             │
│                       │ • Throughput & health              │
├─────────────────────────────────────────────────────────────┤
│ 🔵 SYSTEM STATUS      │ Redis Server Health               │
│                       │ • Memory, connections, ops/sec     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 **Daily Monitoring Workflow**

### **Morning Health Check (2 minutes)**
```bash
# 1. Quick system status
cd /home/youssefbahloul/ai-trading-station/redis-setup

# 2. Performance gate - GO/NO-GO decision
./production/perf-gate.sh --metrics-only

# 3. Current performance snapshot
./monitoring/redis-hft-monitor_to_json.sh | jq '{rtt, health}'
```

**Expected Output**:
```
✅ GATE: PASS - System ready for institutional-grade trading
{
  "rtt": {
    "p50": 9.71,    # Target: <10μs
    "p95": 10.07,   # Target: <15μs  
    "p99": 10.83,   # Target: <20μs
    "jitter": 1.12  # Target: <5μs
  },
  "health": {
    "mem_used_human": "1.62M",  # Target: <10M
    "clients": 1,               # Normal: 1-10
    "ops_per_sec": 17443       # Target: >10,000
  }
}
```

---

## 📈 **Key Performance Indicators (KPIs) Guide**

### **🔴 CRITICAL ALERTS - Act Immediately**

| Metric | Threshold | Meaning | Action Required |
|--------|-----------|---------|-----------------|
| **Performance Gate** | FAIL | System degraded | Investigate immediately |
| **P99 RTT** | >20μs | 99% of requests too slow | Check system load, CPU usage |
| **Memory Usage** | >50MB | Potential memory leak | Restart Redis, check config |
| **Redis Connection** | FAIL | Server down/unreachable | Check service status |

### **🟡 WARNING INDICATORS - Monitor Closely**

| Metric | Threshold | Meaning | Action |
|--------|-----------|---------|--------|
| **P99 RTT** | >15μs | Performance declining | Monitor trend, check logs |
| **P95 RTT** | >12μs | Some requests slow | Check for background processes |
| **Jitter** | >3μs | Inconsistent performance | Check CPU isolation |
| **P99.9 Tail Span** | >8μs | Rare extreme latencies | Run tail analysis |

### **🟢 HEALTHY RANGES - Normal Operations**

| Metric | Healthy Range | Meaning |
|--------|---------------|---------|
| **P50 RTT** | 8-10μs | Median performance excellent |
| **P95 RTT** | 9-12μs | 95% of requests fast |
| **P99 RTT** | 10-15μs | 99% of requests acceptable |
| **Jitter** | <2μs | Very consistent performance |
| **Memory** | 1-5MB | Efficient memory usage |
| **Ops/sec** | >15,000 | High throughput maintained |

---

## 🔍 **Monitoring Tools Deep Dive**

### **1. Basic Performance Monitoring**

#### **redis-hft-monitor_to_json.sh** (Most Important)
```bash
./monitoring/redis-hft-monitor_to_json.sh
```

**What it measures**:
- **RTT (Round Trip Time)**: Real Redis performance
- **SET operations**: Write performance  
- **XADD operations**: Stream write performance
- **Health metrics**: Memory, connections, throughput

**When to use**: Every few minutes during trading hours

**Key fields explained**:
```json
{
  "rtt": {
    "p50": 9.71,     // 50% of requests complete in 9.71μs
    "p95": 10.07,    // 95% of requests complete in 10.07μs  
    "p99": 10.83,    // 99% of requests complete in 10.83μs
    "jitter": 1.12   // Standard deviation (consistency)
  },
  "health": {
    "mem_used_human": "1.62M",  // Redis memory usage
    "clients": 1,               // Active connections
    "ops_per_sec": 17443       // Operations per second
  }
}
```

#### **redis-hft-monitor_v2.sh** (Enhanced Version)
```bash
./monitoring/redis-hft-monitor_v2.sh
```
**Additional features**: System metrics, more detailed analysis

### **2. Performance Gate (Critical Validation)**

#### **perf-gate.sh** (GO/NO-GO Decision)
```bash
# Daily validation
./production/perf-gate.sh --metrics-only

# Full validation (with environment check)
./production/perf-gate.sh
```

**What it does**:
- **Performance Check**: Runs monitoring and validates against thresholds
- **Environment Check**: Detects system configuration drift
- **Decision**: PASS/FAIL for trading readiness

**Exit codes**:
- `0` = PASS (ready for trading)
- `1` = SOFT FAIL (performance regression)
- `2` = HARD FAIL (environment drift)
- `3` = CONFIG ERROR (missing files)

**Thresholds** (in `setup/configs/thresholds/perf-thresholds.env`):
```bash
P99_RTT_MAX_US=20.0      # P99 must be <20μs
P95_RTT_MAX_US=15.0      # P95 must be <15μs  
JITTER_MAX_US=5.0        # Jitter must be <5μs
MIN_OPS_PER_SEC=10000    # Throughput must be >10k ops/sec
```

### **3. Advanced Tail Monitoring (P99.9 Precision)**

#### **extended_tail_sampler.py** (Long-term Analysis)
```bash
# Run 5-minute tail analysis
./production/extended_tail_sampler.py --duration 300 --samples 5000

# Background monitoring 
./production/extended_tail_sampler.py --duration 1800 --samples 5000 &
```

**What it provides**:
- **P99.9 precision**: Captures rare extreme latencies
- **Burst classification**: Identifies cause of tail spikes
- **Historical tracking**: Long-term performance trends
- **Confidence scoring**: Validates measurement reliability

**Output interpretation**:
```json
{
  "p99_9": 12.45,           // 99.9% of requests <12.45μs
  "tail_span": 2.87,        // Difference between P99.9 and P99
  "classification": "SCHED", // Burst type: SCHED/IRQ/ALLOC/UNKNOWN
  "confidence": "HIGH"       // Sample size reliability
}
```

**Burst classifications**:
- **SCHED**: CPU scheduling delays (check isolation)
- **IRQ**: Interrupt handling delays (check IRQ affinity)  
- **ALLOC**: Memory allocation delays (check memory)
- **UNKNOWN**: No clear system correlation

#### **tail_aware_gate.py** (Tail Validation) 
```bash
./production/tail_aware_gate.py state/test_metrics.json
```
**Purpose**: Validates if tail metrics are within acceptable ranges

### **4. Environment Drift Detection**

#### **runtime-fingerprint.sh** (Configuration Monitoring)
```bash
./production/runtime-fingerprint.sh --pretty
```

**What it monitors**:
- **Kernel parameters**: CPU isolation, memory settings
- **Redis configuration**: Config file changes
- **System state**: IRQ affinity, CPU governors
- **Binary integrity**: Redis version consistency

---

## 🚨 **Alert Scenarios & Response Guide**

### **Scenario 1: Performance Gate FAIL**
```bash
./production/perf-gate.sh
# Output: ❌ GATE: FAIL - Performance regression detected
```

**Immediate Actions**:
1. Check current performance: `./monitoring/redis-hft-monitor_to_json.sh`
2. Look for system issues: `htop`, `iostat -x 1`
3. Check Redis status: `systemctl status redis-hft`
4. Review recent changes: `./production/runtime-fingerprint.sh`

### **Scenario 2: High P99 Latency (>15μs)**
```json
{"rtt": {"p99": 18.5}}  // Too high!
```

**Investigation Steps**:
1. **Check CPU usage**: `htop` - Look for high CPU on isolated cores
2. **Check system load**: `uptime` - Load should be <1.0
3. **Check memory**: `free -h` - Ensure no swapping
4. **Run tail analysis**: `./production/extended_tail_sampler.py --duration 300`

### **Scenario 3: Tail Span Warning (>8μs)**
```json
{"tail_span": 12.3, "classification": "SCHED"}
```

**Analysis & Action**:
- **SCHED classification**: CPU scheduling issue
- **Check**: `cat /proc/cmdline | grep isolcpus`
- **Verify**: Redis running on isolated CPU
- **Action**: May need to adjust CPU affinity

### **Scenario 4: Memory Usage Growing**
```json
{"health": {"mem_used_human": "15.2M"}}  // Usually ~1-2M
```

**Investigation**:
1. Check for memory leaks: `redis-cli info memory`
2. Check keyspace size: `redis-cli info keyspace`  
3. Consider Redis restart if >50MB

---

## 📊 **Comprehensive Monitoring Dashboard Commands**

### **One-Line System Status**
```bash
echo "=== REDIS HFT STATUS ===" && \
./production/perf-gate.sh --metrics-only && \
echo "Current Performance:" && \
./monitoring/redis-hft-monitor_to_json.sh | jq '{rtt, health}' && \
echo "Server Status:" && \
systemctl is-active redis-hft
```

### **Complete Health Report**
```bash
# Save this as monitoring/health-report.sh
#!/bin/bash
echo "=== REDIS HFT HEALTH REPORT ===" 
echo "Time: $(date)"
echo

echo "🎯 Performance Gate:"
./production/perf-gate.sh --metrics-only

echo "📊 Current Metrics:"  
./monitoring/redis-hft-monitor_to_json.sh | jq '{rtt, health}'

echo "🔍 System Status:"
echo "Redis Service: $(systemctl is-active redis-hft)"
echo "Memory: $(free -h | grep Mem | awk '{print $3"/"$2}')"
echo "CPU Load: $(uptime | awk -F'load average:' '{print $2}')"

echo "⚡ Recent Performance (last 5 gate decisions):"
python3 production/gate_decision_ledger.py --show-recent 5
```

### **Continuous Monitoring (Background)**
```bash
# Terminal 1: Continuous performance gate (every 5 minutes)
while true; do 
  ./production/perf-gate.sh --metrics-only
  sleep 300
done

# Terminal 2: Continuous tail monitoring  
./production/extended_tail_sampler.py --duration 3600 --samples 5000

# Terminal 3: Basic metrics (every 30 seconds)
while true; do
  ./monitoring/redis-hft-monitor_to_json.sh | jq '{ts, rtt: .rtt.p99, health: .health.ops_per_sec}'
  sleep 30  
done
```

---

## 📋 **Monitoring Checklist**

### **Daily (Start of Trading)**
- [ ] Run `./production/perf-gate.sh --metrics-only`
- [ ] Verify P99 < 15μs with `./monitoring/redis-hft-monitor_to_json.sh`
- [ ] Check Redis service status
- [ ] Review overnight performance in logs

### **Hourly (During Trading)**  
- [ ] Quick performance check: `./monitoring/redis-hft-monitor_to_json.sh | jq .rtt`
- [ ] Monitor ops/sec staying >10,000
- [ ] Watch for memory growth

### **End of Day**
- [ ] Run full performance gate: `./production/perf-gate.sh`
- [ ] Review tail analysis: Check `state/tail-run.json` 
- [ ] Check decision ledger: `python3 production/gate_decision_ledger.py --show-recent 10`

### **Weekly**
- [ ] Analyze performance trends from tail data
- [ ] Review environment fingerprint for drift
- [ ] Clean up old log files

---

## 🎯 **Quick Reference**

### **Emergency Commands**
```bash
# System down? Quick diagnosis
systemctl status redis-hft
./monitoring/redis-hft-monitor_to_json.sh
./production/perf-gate.sh --metrics-only

# Performance problem? Deep analysis  
./production/extended_tail_sampler.py --duration 300
./production/runtime-fingerprint.sh --pretty

# Historical analysis
python3 production/gate_decision_ledger.py --show-recent 20
cat state/tail-run.json | jq '.recent_windows[-10:]'
```

### **Key Files Locations**
- **Thresholds**: `setup/configs/thresholds/perf-thresholds.env`
- **State History**: `state/tail-run.json`  
- **Decision Log**: `logs/gate-decisions.log`
- **Monitoring Scripts**: `monitoring/`
- **Production Gates**: `production/`

---

## 🏆 **Success Indicators**

**Your system is healthy when**:
- ✅ Performance gate: PASS
- ✅ P99 RTT: 10-15μs consistently  
- ✅ Jitter: <2μs
- ✅ Memory: <5MB
- ✅ Ops/sec: >15,000
- ✅ No environment drift warnings

**You have elite-level HFT infrastructure when all these indicators stay green consistently!** 🚀

---

This monitoring guide gives you complete visibility into your Redis HFT system. All the tools work together to provide early warning of issues and detailed analysis when needed.