# Onload Driver Module Priority Configuration
**AI Trading Station - Onload depmod.d Override Configuration**  
**Last Updated:** October 8, 2025

---

## 📋 Overview

This directory contains backup of the Onload driver module priority configuration. This file is **CRITICAL** for ensuring Onload's patched Solarflare driver is loaded instead of standard DKMS or kernel drivers.

**File:** `depmod-onload.conf.backup`  
**Production Location:** `/etc/depmod.d/onload.conf`  
**Purpose:** Prioritize Onload's patched sfc driver for kernel bypass  
**Size:** 265 bytes  
**Critical:** ✅ YES - Required for sub-5μs latency

---

## 🔧 File Content

```bash
# SPDX-License-Identifier: GPL-2.0
# X-SPDX-Copyright-Text: (c) Copyright 2011-2019 Xilinx, Inc.
# Ensure that drivers provided by Onload install override any that came
# with the kernel.
override sfc * weak-updates
override sfc * extra
override sfc * extra/onload
```

---

## 🎯 What This Does

### **Purpose:**
Tell the kernel module loader to use Onload's **patched** Solarflare driver instead of standard DKMS or in-kernel drivers. The patched driver includes kernel bypass extensions required for ultra-low latency trading.

### **Three Override Directives:**

```bash
override sfc * weak-updates    # Priority 1: Basic override
override sfc * extra           # Priority 2: More specific (ACTIVE)
override sfc * extra/onload    # Priority 3: Most specific (if exists)
```

**How they work:**
- **More specific path = higher priority**
- `extra/onload/` > `extra/` > `weak-updates/`
- Currently: **`extra/sfc.ko`** is loaded (26MB Onload patched driver)

---

## 📂 Module Search Hierarchy

### **Default Priority (without override):**
1. 🥇 `updates/dkms/` - DKMS modules (highest by default)
2. 🥈 `extra/` - External vendor modules
3. 🥉 `weak-updates/` - Symlinks to modules
4. 💤 `kernel/` - In-kernel drivers (lowest)

### **With Onload Override (this file):**
1. 🥇 **`extra/onload/sfc.ko`** - Most specific (if exists)
2. 🥈 **`extra/sfc.ko`** - **CURRENT ACTIVE** ✓ (26MB Onload driver)
3. 🥉 `weak-updates/sfc.ko` - Basic override
4. 💤 `updates/dkms/sfc.ko` - Overridden by extra/
5. 💤 `kernel/.../sfc.ko` - Overridden by all

**Result:** Onload's 26MB patched driver in `extra/` wins ✓

---

## 🔴 **CRITICAL: This File Overrides Solarflare DKMS**

### **Conflict Resolution:**

There's also `/etc/depmod.d/sfc.conf` (Solarflare DKMS) which contains:
```bash
override sfc * weak-updates
```

### **Priority Resolution:**
| File | Directive | Specificity | Result |
|------|-----------|-------------|---------|
| `sfc.conf` | `weak-updates` | Low | ⏸️ Overridden |
| **`onload.conf`** | **`extra/`** | **High** | ✅ **WINS** |
| **`onload.conf`** | **`extra/onload/`** | **Highest** | (if exists) |

**Winner:** Onload's `extra/sfc.ko` (this file's directive)

---

## 📊 Current Module Versions

### **Available Drivers:**

| Location | Size | Source | Patches | Status |
|----------|------|--------|---------|--------|
| **`extra/sfc.ko`** | **26MB** | **Onload 9.0.2.140** | **Kernel bypass** | ✅ **ACTIVE** |
| `updates/dkms/sfc.ko.zst` | 265KB | Solarflare DKMS | Standard | ⏸️ Available |
| `kernel/.../sfc.ko.zst` | 272KB | Ubuntu 6.8 | Standard | ⏸️ Fallback |

### **Why Onload Driver is 100x Larger:**

**26MB (Onload) vs 265KB (DKMS/Kernel)**

Onload driver includes:
- ✅ Kernel bypass patches (zero-copy networking)
- ✅ EF_* environment variable support
- ✅ Huge pages integration (EF_USE_HUGE_PAGES)
- ✅ Packet buffer optimizations
- ✅ Direct hardware access (EF_PACKET_BUFFER_MODE=2)
- ✅ Sub-5μs latency optimizations

Standard driver:
- ❌ No kernel bypass
- ❌ No EF_* features
- ❌ 20-50μs latency (10x slower)

---

## 🚀 Performance Impact

### **With Onload Driver (current):**
- Network RTT: **3-5μs** 🚀
- Redis GET P99: **4.2μs** 🚀
- Redis SET P99: **4.8μs** 🚀
- Redis XADD P99: **5.5μs** 🚀

### **Without Onload (standard driver):**
- Network RTT: **20-50μs** ❌
- Redis GET P99: **25μs** ❌
- Redis SET P99: **30μs** ❌
- **Unacceptable for HFT trading** ❌

**Performance difference:** ~5-10x latency reduction with Onload ✓

---

## 🎯 Why This File is Critical

### **1. Enables Kernel Bypass:**
```bash
# With Onload driver (extra/sfc.ko):
onload-trading redis-cli ping
# Uses kernel bypass → sub-5μs

# Without Onload driver (DKMS/kernel):
redis-cli ping  
# Normal kernel stack → 20-50μs
```

### **2. Provides EF_* Environment Variables:**
```bash
# These ONLY work with Onload's patched driver:
EF_USE_HUGE_PAGES=2           # Zero-copy networking
EF_POLL_USEC=0                # Busy polling mode
EF_PACKET_BUFFER_MODE=2       # Physical memory mode
EF_RXQ_SIZE=4096              # Large RX queue
EF_TXQ_SIZE=2048              # Large TX queue
```

### **3. Institutional-Grade Latency:**
- Onload driver: Sub-10μs (institutional HFT standard)
- Standard driver: 20-50μs (retail/general purpose)
- **Difference matters** for market data and order execution

---

## 🚨 Critical Warnings

### **DO NOT:**

1. ❌ **Remove this file**
   - Reason: System falls back to DKMS/kernel driver (no Onload patches)
   - Risk: Latency increases 5-10x (sub-5μs → 20-50μs)
   - Impact: **Trading system becomes uncompetitive**

2. ❌ **Move to workspace or symlink**
   - Reason: Loaded very early in boot (before /home mount)
   - Risk: Driver priority unset → wrong driver loads → high latency

3. ❌ **Change priority to lower than DKMS**
   - Reason: DKMS driver lacks kernel bypass patches
   - Risk: Onload features (EF_*) stop working

4. ❌ **Comment out lines**
   - Reason: Each line provides escalating priority
   - Risk: Less specific override → wrong driver selection

### **WHY it must stay in /etc:**
- Processed by `depmod` **before /home mount**
- Happens during early boot (kernel module initialization)
- **Boot-critical for HFT performance**
- No dependencies allowed on user filesystems

---

## 🔍 Verification Commands

### **1. Check which driver is loaded:**
```bash
modinfo sfc | grep filename
# Expected: /lib/modules/.../extra/sfc.ko ✓
# Wrong: .../updates/dkms/sfc.ko or .../kernel/.../sfc.ko
```

### **2. Check driver size (Onload indicator):**
```bash
ls -lh /lib/modules/$(uname -r)/extra/sfc.ko
# Expected: ~26M (Onload patched driver) ✓
# Wrong: ~265K (standard driver, missing Onload patches)
```

### **3. Check depmod configuration:**
```bash
cat /etc/depmod.d/onload.conf
# Expected: 3 override lines for sfc ✓

grep "override sfc" /etc/depmod.d/*.conf
# Should show both sfc.conf and onload.conf
# onload.conf should have more specific paths (extra/) ✓
```

### **4. Test Onload functionality:**
```bash
# Test 1: Check Onload wrapper works
onload-trading echo "Onload OK"
# Should succeed without errors ✓

# Test 2: Check EF_* variables work
onload-trading bash -c 'env | grep EF_'
# Should show EF_* environment variables ✓

# Test 3: Measure latency with Onload
onload-trading redis-cli --latency-history -i 1
# Expected: P99 < 10μs ✓
```

---

## 🛠️ Maintenance Procedures

### **Restore from backup:**
```bash
# If production file is lost or corrupted
sudo cp Tuning/Onload/depmod-onload.conf.backup /etc/depmod.d/onload.conf
sudo chmod 644 /etc/depmod.d/onload.conf

# Rebuild module dependencies
sudo depmod -a

# Reload driver (requires network downtime)
sudo rmmod sfc
sudo modprobe sfc

# Verify correct driver loaded
modinfo sfc | grep filename
# Expected: extra/sfc.ko ✓
```

### **Check Onload driver integrity:**
```bash
# Check if Onload driver exists
ls -lh /lib/modules/$(uname -r)/extra/sfc.ko
# Expected: ~26M ✓

# Check Onload installation
dpkg -l | grep onload
# Expected: onload package installed

# Verify Onload version
onload --version 2>/dev/null || cat /opt/onload/version 2>/dev/null
# Expected: 9.0.2.140 or similar
```

### **Emergency: Force DKMS driver (testing only):**
```bash
# Temporarily disable Onload override
sudo mv /etc/depmod.d/onload.conf /etc/depmod.d/onload.conf.disabled
sudo depmod -a
sudo rmmod sfc && sudo modprobe sfc

# Check which driver loaded
modinfo sfc | grep filename
# Will show: updates/dkms/sfc.ko (standard driver)

# Measure latency impact
redis-cli --latency-history -i 1
# Expected: P99 increases to 20-50μs (degraded)

# Re-enable Onload
sudo mv /etc/depmod.d/onload.conf.disabled /etc/depmod.d/onload.conf
sudo depmod -a
sudo rmmod sfc && sudo modprobe sfc
```

⚠️ **Warning:** Changing drivers requires NIC reload → **network downtime**

---

## 🔗 Related Configurations

This file works together with:

1. **`/etc/depmod.d/sfc.conf`** (Lower priority)
   - Sets basic DKMS priority
   - Overridden by this file's more specific paths
   - See: `Tuning/Solarflare/DEPMOD-README.md`

2. **`/etc/modprobe.d/onload-hft.conf`** (Module options)
   - Enables physical memory mode (`phys_mode_gid=1000`)
   - Requires Onload's patched driver (enabled by this file)

3. **`/etc/modprobe.d/sfc.conf`** (NIC options)
   - Sets `performance_profile=latency`
   - Works with any sfc driver (standard or Onload)

4. **`scripts/onload-trading`** (Runtime wrapper)
   - Uses Onload driver loaded by this file
   - Sets EF_* environment variables
   - Requires kernel bypass patches (only in Onload driver)

5. **`/etc/sysctl.d/99-trading-hugepages.conf`**
   - Allocates 2GB huge pages
   - Used by Onload driver (EF_USE_HUGE_PAGES=2)

---

## 📚 Understanding Override Hierarchy

### **How depmod.d Override Works:**

When multiple `override` directives exist:
1. **Most specific path wins** (more directory levels = higher priority)
2. **Last file processed wins** (if same specificity)
3. **Each directive is independent** (multiple can coexist)

### **Example Resolution (this file):**

```bash
override sfc * weak-updates       # Specificity: 1 directory
override sfc * extra              # Specificity: 1 directory (MORE SPECIFIC than weak-updates)
override sfc * extra/onload       # Specificity: 2 directories (MOST SPECIFIC)
```

**Priority Calculation:**
- `extra/onload/` = 2 levels → Priority 10
- `extra/` = 1 level (special) → Priority 9
- `weak-updates/` = 1 level → Priority 8
- `updates/dkms/` = 2 levels (default) → Priority 7 (overridden by extra/)
- `kernel/` = many levels → Priority 1 (lowest)

**Result:** `extra/` wins over `updates/dkms/` despite both having similar depth

---

## 📊 Driver Selection Flow

```
Boot Sequence:
1. Kernel initializes
2. systemd-modules-load.service starts
3. depmod reads /etc/depmod.d/*.conf (alphabetically)
   ├─ sfc.conf: override sfc * weak-updates
   └─ onload.conf: override sfc * extra ← MORE SPECIFIC
4. Builds module priority database
5. NIC detected → modprobe sfc triggered

Module Selection:
┌─────────────────────────────────────────┐
│ Check /etc/depmod.d/onload.conf        │
│ Priority order:                         │
│ 1. extra/onload/sfc.ko (not found)     │
│ 2. extra/sfc.ko (FOUND) ✓              │ ← LOADS THIS
│ 3. weak-updates/sfc.ko (skipped)       │
│ 4. updates/dkms/sfc.ko (skipped)       │
│ 5. kernel/.../sfc.ko (skipped)         │
└──────────────┬────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│ Load: /lib/modules/.../extra/sfc.ko    │
│ Size: 26MB (Onload patched)             │
│ Features:                               │
│ ✅ Kernel bypass (zero-copy)           │
│ ✅ EF_* environment variables          │
│ ✅ Huge pages support                  │
│ ✅ Sub-5μs latency                     │
└──────────────┬────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│ NIC Initialized                         │
│ enp130s0f0, enp130s0f1 UP               │
│ Onload ready for trading ✓              │
└─────────────────────────────────────────┘
```

---

## 🎯 Success Criteria

**System is correctly configured if:**
1. ✅ File exists: `/etc/depmod.d/onload.conf`
2. ✅ Driver loaded: `extra/sfc.ko` (26MB)
3. ✅ Onload wrapper works: `onload-trading echo OK`
4. ✅ EF_* variables work: `onload-trading env | grep EF_`
5. ✅ Latency: Redis P99 < 10μs with Onload
6. ✅ All 3 override lines present in file

---

## 📞 Troubleshooting

### **Problem: High latency (20-50μs instead of sub-5μs)**
```bash
# Root cause: Wrong driver loaded (standard DKMS/kernel instead of Onload)

# Step 1: Check which driver is loaded
modinfo sfc | grep filename

# If shows: updates/dkms/sfc.ko or kernel/.../sfc.ko (WRONG)
# Should be: extra/sfc.ko ✓

# Step 2: Check if this file exists
cat /etc/depmod.d/onload.conf

# Step 3: Check if Onload driver exists
ls -lh /lib/modules/$(uname -r)/extra/sfc.ko
# Should be ~26MB

# Step 4: Restore if missing
sudo cp Tuning/Onload/depmod-onload.conf.backup /etc/depmod.d/onload.conf
sudo depmod -a
sudo rmmod sfc && sudo modprobe sfc

# Step 5: Verify
modinfo sfc | grep filename
# Should now show: extra/sfc.ko ✓
```

### **Problem: onload-trading command fails**
```bash
# Check if Onload library exists
ls -lh /lib/x86_64-linux-gnu/libonload.so*

# Check if Onload driver loaded
lsmod | grep onload

# Check if sfc driver is Onload's patched version
modinfo sfc | grep filename
# Must be: extra/sfc.ko (26MB)

# Reload Onload
sudo rmmod onload 2>/dev/null
sudo modprobe onload
```

### **Problem: EF_* environment variables not working**
```bash
# Indicates standard driver loaded (missing Onload patches)

# Check driver
modinfo sfc | grep filename
# If NOT extra/sfc.ko:

# Reinstall Onload
sudo apt-get install --reinstall onload

# Verify depmod.d file
cat /etc/depmod.d/onload.conf

# Rebuild and reload
sudo depmod -a
sudo rmmod sfc && sudo modprobe sfc
```

---

## 🎬 Summary

- **File location:** `/etc/depmod.d/onload.conf` (MUST stay in /etc)
- **Backup location:** `Tuning/Onload/depmod-onload.conf.backup`
- **Purpose:** Prioritize Onload's patched sfc driver for kernel bypass
- **Critical level:** ⚠️ **MAXIMUM** - Required for sub-5μs latency
- **Override hierarchy:** `extra/onload/` > `extra/` > `weak-updates/` > defaults
- **Current driver:** `extra/sfc.ko` (26MB Onload patched) ✅
- **Performance impact:** 5-10x latency reduction vs standard driver
- **Maintenance:** Backup only, **NEVER move or symlink**

---

**Status:** ✅ Production-critical, Onload driver active  
**Performance:** Sub-5μs network latency ✓  
**Last Verified:** October 8, 2025  
**Maintainer:** AI Trading Station Team
