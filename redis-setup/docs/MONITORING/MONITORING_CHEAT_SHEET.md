# Redis HFT Monitoring Scripts Guide

**Clear Script Breakdown:**

## **Your 4 Monitoring Scripts - What Each Does:**

### **1. `redis-hft-monitor_to_json.sh`** 
- **Purpose**: Get current performance numbers (P99 latency, memory, etc.)
- **When**: Check performance anytime
- **Output**: Raw JSON data
- **Use**: `./monitoring/redis-hft-monitor_to_json.sh | jq .rtt`

### **2. `perf-gate.sh`**
- **Purpose**: Pass/Fail decision - "Is system ready for trading?"
- **When**: Once before trading starts (8:30 AM)
- **Output**: ✅ PASS or ❌ FAIL
- **Use**: `./production/perf-gate.sh --metrics-only`

### **3. `health-report.sh`** (New - I created this)
- **Purpose**: Complete system overview with interpretation
- **When**: Troubleshooting or daily summary
- **Output**: Formatted report with health assessment
- **Use**: `./monitoring/health-report.sh`

### **4. `live-dashboard.sh`** (New - I created this)
- **Purpose**: Real-time monitoring display (updates every 10 seconds)
- **When**: Active monitoring during trading
- **Output**: Live updating dashboard
- **Use**: `./monitoring/live-dashboard.sh`

---

## **How Scripts Call Each Other:**

### **Script 3 (`health-report.sh`) calls:**
- ✅ **Script 1** (`redis-hft-monitor_to_json.sh`) - to get performance data
- ✅ **Script 2** (`perf-gate.sh`) - to get Pass/Fail status
- **PLUS**: Adds system info (memory, CPU, Redis service status)

### **Script 4 (`live-dashboard.sh`) calls:**
- ✅ **Script 1** (`redis-hft-monitor_to_json.sh`) - to get performance data  
- ✅ **Script 2** (`perf-gate.sh`) - to get Pass/Fail status
- **PLUS**: Updates every 10 seconds in a loop with colors/formatting

## **So you have:**

**Scripts 1 & 2**: Raw tools (get data, get decision)  
**Scripts 3 & 4**: User-friendly wrappers that call 1 & 2 + add formatting

---

## **What to actually run:**

```bash
# Easy option: Use the wrappers
./monitoring/health-report.sh      # Calls 1+2, shows everything nicely
./monitoring/live-dashboard.sh     # Calls 1+2 in loop, live updates

# Direct option: Use raw tools  
./monitoring/redis-hft-monitor_to_json.sh  # Just data
./production/perf-gate.sh --metrics-only   # Just Pass/Fail
```

**Scripts 3&4 are convenience wrappers around 1&2.**

---

## **Simple Daily Workflow:**

```bash
# Morning: Go/No-Go decision
./production/perf-gate.sh --metrics-only

# During trading: Quick performance check
./monitoring/redis-hft-monitor_to_json.sh | jq .rtt.p99

# Problem investigation: Full analysis
./monitoring/health-report.sh

# Active monitoring: Real-time dashboard
./monitoring/live-dashboard.sh
```

**Bottom line**: Scripts 1&2 existed, I created 3&4 to make monitoring easier for you.