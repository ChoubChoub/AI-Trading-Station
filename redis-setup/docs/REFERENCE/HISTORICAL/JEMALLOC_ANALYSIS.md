# Current jemalloc Configuration Analysis

**Phase 3A - Week 1: jemalloc Research Results**  
*Date: September 28, 2025*

---

## 🎯 **Key Discovery: Redis Already Uses jemalloc 5.3.0**

### **Current Configuration Analysis**

#### **jemalloc Version & Build**
- **Version**: 5.3.0-0-g54eaed1d (stable release)
- **Build Type**: Distribution package (not HFT-optimized)
- **Profile Support**: ✅ Available (`prof: true`)
- **Stats Support**: ✅ Available (`stats: true`)

#### **Current Runtime Configuration**
| Parameter | Current Value | HFT Optimal | Impact |
|-----------|---------------|-------------|---------|
| `narenas` | 1 | 2-4 | 🟡 Too low for multi-core |
| `background_thread` | false | false/true | ✅ Good for determinism |
| `metadata_thp` | disabled | auto/always | 🟡 Missing THP benefits |
| `dirty_decay_ms` | 10000 | 1000-5000 | 🟡 Too slow for HFT |
| `muzzy_decay_ms` | 0 | 0 | ✅ Good |
| `tcache` | true | true | ✅ Good |

### **Performance Characteristics Analysis**

#### **Memory Utilization**
- **Used Memory**: 1.71MB (Redis data)
- **RSS Memory**: 14.75MB (system allocated)
- **Allocator Efficiency**: 2.51MB allocated vs 5.50MB resident
- **Fragmentation Ratio**: 8.71 (HIGH - indicates optimization opportunity)

#### **Allocation Patterns**
- **Small allocations** (16-3072 bytes): Heavy usage, well-cached
- **Large allocations** (16KB-262KB): Moderate usage
- **Arena Utilization**: Single arena heavily used (not optimal for multi-core)

### **HFT Optimization Opportunities**

#### **1. Arena Configuration**
```bash
# Current: narenas=1 (single arena bottleneck)
# Optimal: narenas=2-4 (reduced contention)
MALLOC_CONF="narenas:2"
```

#### **2. Decay Timing Optimization**
```bash
# Current: dirty_decay_ms=10000 (10 seconds)
# Optimal: dirty_decay_ms=1000 (1 second for faster cleanup)
MALLOC_CONF="dirty_decay_ms:1000"
```

#### **3. Transparent Huge Pages**
```bash
# Current: metadata_thp=disabled
# Optimal: metadata_thp=auto (reduce TLB pressure)
MALLOC_CONF="metadata_thp:auto"
```

#### **4. Background Thread Control**
```bash
# Current: background_thread=false (runtime: true)
# Decision: Keep false for determinism or enable for cleanup efficiency
```

---

## 🎯 **Phase 3A Strategy Revision**

### **Original Plan**: Build custom jemalloc
### **Revised Plan**: Optimize existing jemalloc configuration

#### **Advantages of Revision**
1. **Lower Risk**: No Redis rebuild required
2. **Faster Implementation**: Configuration changes vs compilation
3. **Easier Rollback**: Runtime parameter changes
4. **Proven Compatibility**: Same jemalloc version, different config

#### **Optimization Approach**
1. **Step 1**: Test runtime configuration changes
2. **Step 2**: Measure performance improvements  
3. **Step 3**: Optimize MALLOC_CONF parameters
4. **Step 4**: Consider custom build only if needed

### **Expected Improvements**
| Optimization | Expected Impact | Risk Level |
|--------------|-----------------|------------|
| Arena count increase | 10-20% allocation efficiency | Low |
| Decay timing optimization | 5-15% memory cleanup speed | Low |
| THP metadata | 5-10% TLB performance | Medium |
| Combined effect | **15-25% total improvement** | **Low-Medium** |

---

## 🚀 **Next Steps (Week 1 Continued)**

### **Immediate Actions**
1. **Baseline Performance**: Measure current allocation performance
2. **MALLOC_CONF Testing**: Test optimized configuration parameters
3. **A/B Comparison**: Compare current vs optimized settings
4. **Performance Validation**: Confirm improvements in RTT metrics

### **Testing Framework**
```bash
# Current configuration test
taskset -c 4 ./redis-hft-monitor_to_json.sh > baseline_current.json

# Optimized configuration test  
MALLOC_CONF="narenas:2,dirty_decay_ms:1000,metadata_thp:auto" \
systemctl restart redis-server
taskset -c 4 ./redis-hft-monitor_to_json.sh > optimized_test.json
```

### **Success Criteria**
- **RTT P99**: Target 7.4-8.3μs (15-25% improvement from 9.9μs)
- **Memory Fragmentation**: Reduce from 8.71 to <5.0
- **Allocation Efficiency**: Improve RSS/allocated ratio
- **System Stability**: Maintain current reliability

---

## 📊 **Risk Assessment**

### **Low Risk Optimizations** ✅
- Runtime configuration parameter changes
- Arena count optimization
- Decay timing adjustments

### **Medium Risk Optimizations** ⚠️
- Transparent huge page changes
- Background thread modifications

### **Rollback Plan**
```bash
# Immediate rollback to current configuration
unset MALLOC_CONF
systemctl restart redis-server
```

---

**Status**: jemalloc research complete - pivot to configuration optimization  
**Next**: Implement and test MALLOC_CONF parameter optimization  
**Timeline**: Accelerated due to avoiding custom build complexity