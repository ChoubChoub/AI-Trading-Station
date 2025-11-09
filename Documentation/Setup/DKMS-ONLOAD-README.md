# DKMS Onload Build Configuration
**AI Trading Station - Dynamic Kernel Module Support for Onload**  
**Last Updated:** October 8, 2025

---

## 📋 Overview

This directory contains backup of the DKMS (Dynamic Kernel Module Support) configuration file for Onload. This file controls how DKMS builds Onload kernel modules when new kernel versions are installed.

**File:** `dkms-onload.conf.backup`  
**Production Location:** `/etc/dkms/onload.conf`  
**Purpose:** Custom DKMS build parameters for Onload kernel modules  
**Size:** 13 bytes  
**Critical:** ⚠️ YES - Required for automatic kernel module rebuilds

---

## 🔧 File Content

```bash
MAKE[0]+=" "
```

---

## 🎯 What This Does

### **Purpose:**
Configure DKMS build system to compile Onload kernel modules with custom parameters when new Linux kernels are installed via system updates.

### **Directive Explained:**
```bash
MAKE[0]+=" "
```

- **`MAKE[0]`** - First make command in DKMS build sequence (indexed array)
- **`+=" "`** - Append empty space to make arguments
- **Effect:** Prevents DKMS from passing invalid flags to Onload's custom build system

### **Why This Is Needed:**

Onload uses a **custom build system** (not standard kernel Makefile). The default DKMS framework passes flags that Onload's build system doesn't understand, causing build failures. This configuration **neutralizes those incompatible flags** by adding empty space.

---

## 🔍 DKMS System Overview

### **What is DKMS?**

**Dynamic Kernel Module Support** - A framework that automatically rebuilds out-of-tree kernel modules when you install new kernel versions.

**Without DKMS:**
```
1. Install new kernel via apt upgrade
2. Reboot
3. Onload modules missing (built for old kernel)
4. Manual rebuild: cd /usr/src/onload-9.0.2.140 && ./scripts/onload_install
5. System broken until manual fix
```

**With DKMS:**
```
1. Install new kernel via apt upgrade
2. DKMS automatically rebuilds Onload for new kernel
3. Reboot
4. Onload modules ready ✓
5. System works immediately
```

---

## 📊 Current DKMS Status

### **Installed Onload Version:**
```bash
$ dkms status onload
onload/9.0.2.140, 6.8.0-83-generic, x86_64: installed
onload/9.0.2.140, 6.8.0-84-generic, x86_64: installed
onload/9.0.2.140, 6.8.0-85-generic, x86_64: installed
```

**Interpretation:**
- ✅ Onload 9.0.2.140 built for 3 kernel versions
- ✅ Automatically rebuilds when new kernels installed
- ⚠️ WARNING messages about diff = Onload uses patched sfc driver (expected)

### **Source Location:**
- DKMS source: `/usr/src/onload-9.0.2.140/`
- Build output: `/var/lib/dkms/onload/9.0.2.140/<kernel>/`
- Installed modules: `/lib/modules/<kernel>/updates/`

---

## 🚀 How DKMS Builds Onload

### **Build Process:**

1. **Trigger:** New kernel installed via `apt upgrade linux-image-...`
2. **DKMS detects:** Kernel version changed
3. **DKMS reads:** `/etc/dkms/onload.conf` (this file)
4. **DKMS builds:** 
   ```bash
   cd /usr/src/onload-9.0.2.140
   ./scripts/onload_build --kernel <version>
   ```
5. **DKMS installs:** Built modules to `/lib/modules/<kernel>/updates/`
6. **Result:** Onload ready for new kernel ✓

### **Modules Built by DKMS:**

```bash
$ ls /lib/modules/$(uname -r)/updates/onload/
onload.ko        # Main Onload kernel bypass module
sfc.ko           # Patched Solarflare driver (265KB DKMS version)
sfc_affinity.ko  # IRQ affinity helper
sfc_resource.ko  # Resource management
```

**Note:** The actual active sfc driver is from `/opt/onload-9.0.2.140/extra/sfc.ko` (26MB patched), not the DKMS version. DKMS builds as backup.

---

## 🔗 Integration with Other Configs

This DKMS config works with the driver priority system:

### **Driver Loading Priority:**

1. **`/etc/depmod.d/onload.conf`** - Sets Onload's `extra/` directory as highest priority
   ```bash
   override sfc * extra
   override onload * extra/onload
   ```
   → Result: `/opt/onload-9.0.2.140/extra/sfc.ko` (26MB patched) loads first

2. **DKMS builds** - Modules in `/lib/modules/<kernel>/updates/`
   → Priority: Medium (below `extra/`, above kernel built-in)
   → Result: Used as fallback if `extra/` fails

3. **Kernel built-in** - `/lib/modules/<kernel>/kernel/drivers/net/ethernet/sfc/`
   → Priority: Lowest
   → Result: Never used (overridden by depmod.d)

### **Why We Keep DKMS Despite Using extra/?**

**DKMS benefits:**
- ✅ **Automatic rebuilds** when kernel updates
- ✅ **Fallback option** if extra/ drivers fail
- ✅ **Version tracking** - `dkms status` shows compatibility
- ✅ **Clean uninstall** - `dkms remove` handles cleanup

**extra/ benefits:**
- ✅ **Full performance** - 26MB patched driver with all optimizations
- ✅ **HFT-specific patches** - Not in DKMS build
- ✅ **Tested configuration** - Vendor-validated for trading

**Conclusion:** Keep both. DKMS for automation, extra/ for performance.

---

## 📝 Configuration Details

### **File Location: `/etc/dkms/onload.conf`**

**Why it must stay in /etc/dkms/:**
- Read by DKMS framework during kernel updates
- Must be available before user filesystems mount
- Part of system-wide DKMS configuration
- **Cannot be symlinked** to workspace

### **Why It's This Simple:**

Many DKMS modules have complex configs:
```bash
# Example: Nvidia driver DKMS config
PACKAGE_NAME="nvidia"
PACKAGE_VERSION="535.183.01"
BUILT_MODULE_NAME[0]="nvidia"
BUILT_MODULE_NAME[1]="nvidia-uvm"
DEST_MODULE_LOCATION[0]="/kernel/drivers/video"
MAKE[0]="make -j$(nproc) KERNEL_UNAME=${kernelver}"
```

**Onload's config is minimal because:**
- Onload has its own `dkms.conf` in `/usr/src/onload-9.0.2.140/dkms.conf`
- That file defines all module names, locations, build commands
- `/etc/dkms/onload.conf` only **overrides MAKE flags**
- Everything else comes from package's dkms.conf

---

## 🔍 Verification Commands

### **1. Check DKMS status:**
```bash
dkms status onload
# Expected: installed for current kernel ✓
```

### **2. Check which sfc driver is active:**
```bash
modinfo sfc | grep -E "filename|vermagic"
# Expected: /opt/onload-9.0.2.140/extra/sfc.ko (26MB version)
```

### **3. Verify DKMS built modules exist:**
```bash
ls -lh /lib/modules/$(uname -r)/updates/onload/
# Expected: sfc.ko (265KB), onload.ko, sfc_affinity.ko, sfc_resource.ko
```

### **4. Test DKMS build manually:**
```bash
# Remove and rebuild (safe test - doesn't affect running system)
sudo dkms remove onload/9.0.2.140 -k $(uname -r)
sudo dkms build onload/9.0.2.140 -k $(uname -r)
sudo dkms install onload/9.0.2.140 -k $(uname -r)

# Verify rebuild successful
dkms status onload
# Expected: installed ✓
```

### **5. Check build logs:**
```bash
sudo cat /var/lib/dkms/onload/9.0.2.140/build/make.log | tail -50
# Shows build output, errors if any
```

---

## 🚨 Critical Warnings

### **DO NOT:**

1. ❌ **Remove this file**
   - Reason: DKMS builds will fail with invalid make flags
   - Risk: New kernel updates break Onload
   - Impact: Manual rebuild required after each kernel update

2. ❌ **Edit without understanding**
   - Reason: Wrong flags can break DKMS builds
   - Risk: Onload modules won't rebuild automatically
   - Impact: System broken after kernel updates

3. ❌ **Move to workspace or symlink**
   - Reason: DKMS reads from /etc/dkms/ only
   - Risk: Configuration ignored
   - Impact: Builds fail with default (broken) flags

4. ❌ **Remove DKMS Onload package**
   - Command to avoid: `sudo dkms remove onload/9.0.2.140 --all`
   - Reason: Removes automatic rebuild capability
   - Risk: Kernel updates break Onload
   - Impact: Manual builds required

5. ❌ **Ignore DKMS warnings**
   - Warning: "Diff between built and installed module"
   - **This is EXPECTED** for Onload (uses extra/ driver)
   - Only worry if build **fails**, not warnings

---

## 🛠️ Maintenance Procedures

### **Restore from backup:**
```bash
# If production file is lost
sudo cp Tuning/Onload/dkms-onload.conf.backup /etc/dkms/onload.conf
sudo chmod 644 /etc/dkms/onload.conf

# Rebuild Onload for current kernel
sudo dkms remove onload/9.0.2.140 -k $(uname -r)
sudo dkms install onload/9.0.2.140 -k $(uname -r)

# Verify
dkms status onload
```

### **Test DKMS build after config change:**
```bash
# Edit config
sudo nano /etc/dkms/onload.conf

# Remove old build
sudo dkms remove onload/9.0.2.140 -k $(uname -r)

# Test rebuild
sudo dkms build onload/9.0.2.140 -k $(uname -r)

# If successful, install
sudo dkms install onload/9.0.2.140 -k $(uname -r)

# If failed, restore backup
sudo cp Tuning/Onload/dkms-onload.conf.backup /etc/dkms/onload.conf
```

### **Handle kernel updates:**
```bash
# When new kernel is installed, DKMS auto-rebuilds
# Check after reboot:
dkms status onload | grep $(uname -r)
# Expected: installed ✓

# If not installed, manual build:
sudo dkms install onload/9.0.2.140 -k $(uname -r)

# Verify Onload still works
onload --version
# Expected: 9.0.2.140 ✓
```

### **Clean old kernel modules:**
```bash
# List all DKMS builds
dkms status onload

# Remove old kernel versions no longer installed
sudo dkms remove onload/9.0.2.140 -k 6.8.0-83-generic

# Keep current and previous kernel only
```

---

## 🔗 Related Configurations

This DKMS config works together with:

1. **`/usr/src/onload-9.0.2.140/dkms.conf`**
   - Main DKMS configuration from Onload package
   - Defines module names, versions, build commands
   - `/etc/dkms/onload.conf` overrides MAKE flags only

2. **`/etc/depmod.d/onload.conf`**
   - Sets driver priority (extra/ > updates/ > kernel/)
   - Ensures Onload's patched driver loads
   - See: `Tuning/Onload/DEPMOD-README.md`

3. **`/opt/onload-9.0.2.140/`**
   - Full Onload installation with extra/ drivers
   - Contains 26MB patched sfc driver (actual active driver)
   - DKMS builds are fallback, not primary

4. **`/etc/modprobe.d/onload-hft.conf`**
   - Module options (phys_mode_gid=1000)
   - Applied to loaded modules (from extra/ or DKMS)
   - See: `Tuning/Onload/MODPROBE-README.md`

---

## 📚 Understanding the Build System

### **Onload's Build Architecture:**

**Standard kernel module:**
```bash
# Normal DKMS build
cd /usr/src/module-1.0/
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
# Uses kernel's Makefile system
```

**Onload build:**
```bash
# Custom build system
cd /usr/src/onload-9.0.2.140/
./scripts/onload_build --kernel $(uname -r)
# Uses Onload's proprietary build scripts
# Builds 4 modules: onload, sfc, sfc_affinity, sfc_resource
```

### **Why MAKE[0]+=" " Is Needed:**

**Without `/etc/dkms/onload.conf`:**
```bash
# DKMS passes incompatible flags:
MAKE[0]="make -j8 KERNEL_UNAME=6.8.0-85-generic KVER=6.8.0-85"

# Onload's build system doesn't understand KVER flag
# Build fails: "Unknown option: KVER"
```

**With `/etc/dkms/onload.conf`:**
```bash
# After appending empty space:
MAKE[0]="make -j8 KERNEL_UNAME=6.8.0-85-generic KVER=6.8.0-85 "

# Empty space neutralizes invalid flags (implementation detail)
# Onload's build system ignores them
# Build succeeds ✓
```

**Technical reason:** Onload's Makefile doesn't process these flags. Adding space changes how shell parses arguments, causing incompatible flags to be silently ignored instead of causing errors.

---

## 📊 DKMS vs. Manual Build Comparison

| Aspect | DKMS (with this config) | Manual Build |
|--------|------------------------|--------------|
| **Rebuild on kernel update** | ✅ Automatic | ❌ Manual |
| **Build quality** | ✅ Same (uses Onload scripts) | ✅ Same |
| **Maintenance** | ✅ Low (hands-off) | ❌ High (manual work) |
| **Recovery** | ✅ Easy (dkms install) | ⚠️ Moderate (run scripts) |
| **Version tracking** | ✅ Yes (dkms status) | ❌ Manual tracking |
| **Disk space** | ⚠️ ~500MB per kernel | ✅ ~200MB (one build) |
| **Active driver** | ⚠️ No (extra/ used) | ✅ Yes (if no depmod) |
| **HFT suitability** | ✅ Production-ready | ✅ Production-ready |

**Recommendation:** Use DKMS with this config. Automatic rebuilds worth the disk space.

---

## 🎯 Success Criteria

**System is correctly configured if:**
1. ✅ File exists: `/etc/dkms/onload.conf`
2. ✅ Content: `MAKE[0]+=" "`
3. ✅ DKMS status: `onload/9.0.2.140, <kernel>, x86_64: installed`
4. ✅ Active driver: `/opt/onload-9.0.2.140/extra/sfc.ko` (26MB version)
5. ✅ DKMS fallback: `/lib/modules/<kernel>/updates/onload/sfc.ko` (265KB) exists
6. ✅ Onload works: `onload --version` shows 9.0.2.140

---

## 📞 Troubleshooting

### **Problem: DKMS build fails after kernel update**
```bash
# Check DKMS status
dkms status onload
# Shows: "build failed"

# Check build log
sudo cat /var/lib/dkms/onload/9.0.2.140/build/make.log

# If error mentions invalid flags:
# 1. Verify /etc/dkms/onload.conf exists
cat /etc/dkms/onload.conf
# Expected: MAKE[0]+=" "

# 2. Restore if missing
sudo cp Tuning/Onload/dkms-onload.conf.backup /etc/dkms/onload.conf

# 3. Retry build
sudo dkms remove onload/9.0.2.140 -k $(uname -r)
sudo dkms install onload/9.0.2.140 -k $(uname -r)
```

### **Problem: Kernel update broke Onload**
```bash
# Symptoms: Onload commands fail after kernel update
onload --version
# Error: module not loaded

# Check DKMS built for new kernel
dkms status onload | grep $(uname -r)

# If missing, manual install:
sudo dkms install onload/9.0.2.140 -k $(uname -r)

# Load modules
sudo modprobe sfc
sudo modprobe onload

# Verify
onload --version
# Expected: 9.0.2.140 ✓
```

### **Problem: DKMS warnings about diff**
```bash
# Symptom:
dkms status onload
# Shows: "WARNING! Diff between built and installed module!"

# This is NORMAL for Onload:
# - DKMS builds 265KB sfc.ko in updates/
# - extra/sfc.ko (26MB patched) is actually loaded
# - depmod.d gives extra/ priority
# - DKMS detects difference, warns (but it's intentional)

# Action: IGNORE this warning, it's expected ✓
```

### **Problem: System won't boot after kernel update**
```bash
# Rare case: DKMS build failed, Onload broken, services can't start

# Boot to recovery mode (hold Shift during boot)
# Select: Advanced options → Recovery mode → Root shell

# Check what failed
dkms status onload

# If DKMS build failed, use extra/ driver (always available)
depmod -a
modprobe sfc
modprobe onload

# Reboot normally
reboot

# After boot, fix DKMS:
sudo dkms remove onload/9.0.2.140 -k $(uname -r)
sudo dkms install onload/9.0.2.140 -k $(uname -r)
```

---

## 🎬 Summary

- **File location:** `/etc/dkms/onload.conf` (MUST stay in /etc)
- **Backup location:** `Tuning/Onload/dkms-onload.conf.backup`
- **Purpose:** Fix DKMS make flags for Onload's custom build system
- **Content:** `MAKE[0]+=" "` (13 bytes)
- **Impact:** Enables automatic Onload rebuilds on kernel updates
- **Integration:** Works with depmod.d (extra/ priority) and modprobe.d (physical mode)
- **Critical level:** ⚠️ **HIGH** - Required for kernel update compatibility
- **Maintenance:** Backup only, **NEVER move or remove**

---

**Status:** ✅ Production-critical, DKMS auto-rebuilds enabled  
**DKMS builds:** 3 kernel versions (6.8.0-83, -84, -85) ✓  
**Active driver:** extra/sfc.ko (26MB patched) via depmod priority ✓  
**Fallback driver:** DKMS sfc.ko (265KB) available ✓  
**Last Verified:** October 8, 2025  
**Maintainer:** AI Trading Station Team
