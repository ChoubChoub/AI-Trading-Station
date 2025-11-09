# MANDATORY PRE-MIGRATION PROCEDURE

**Status:** ENFORCED - No exceptions  
**Last Updated:** 2025-11-01  
**Reason:** Critical failure in Tuning migration where grep-only search missed symlinks

---

## Overview

This document defines the **MANDATORY** procedure that **MUST** be followed before moving **ANY** file or folder in the ai-trading-station workspace. 

**No file or folder may be declared "safe to move" until ALL FIVE verification layers return clean.**

---

## Why This Exists

### The Failure That Made This Necessary

During the Tuning folder migration on 2025-11-01, the following occurred:

**What Agent Did:**
```bash
# Used grep to search for text references
grep -r "99-solarflare-trading.conf" /etc/
grep -r "99-trading-nic-irq.rules" /etc/
# Both returned empty → declared "safe to move"
```

**What Agent Missed:**
```bash
# Actual filesystem state:
/etc/sysctl.d/99-solarflare-trading.conf -> .../Tuning/Network/99-solarflare-trading.conf
/etc/udev/rules.d/99-trading-nic-irq.rules -> .../Tuning/Network/99-trading-nic-irq.rules
# Both were SYMLINKS - grep doesn't detect filesystem pointers
```

**Impact if Executed:**
- 🔴 Network IRQ affinity would stop working
- 🔴 Solarflare NIC optimizations would be lost
- 🔴 10-100x latency increase in production

**What Saved Us:**
User asked critical questions:
- "Are these copies or originals?"
- "How is system reading these files from Tuning/Network?"

This triggered proper symlink detection and prevented catastrophic failure.

---

## Phase 0: Comprehensive Audit (MANDATORY)

### Step 1: Run Comprehensive Audit Script

**BEFORE** proposing any file move:

```bash
./Scripts/System/comprehensive-filesystem-audit.sh <target_path>
```

**Example:**
```bash
./Scripts/System/comprehensive-filesystem-audit.sh Tuning/Network/99-solarflare-trading.conf
```

**Script Output:**
- ✅ Green: All layers clean → Safe to move
- ❌ Red: Dependencies found → NOT safe to move
- Exit code 0 = safe, Exit code 1 = dependencies found

### Step 2: Multi-Layer Verification

The audit script checks **FIVE LAYERS**:

#### Layer 1: Text References
**Purpose:** Find text mentions of filename/path  
**Method:** grep searches in:
- `/etc/` (all system configs)
- `~/.bashrc`, `~/.bash_aliases`, `~/.bash_profile`
- `/usr/local/bin/`, `/usr/bin/` scripts
- Systemd unit files (`systemctl cat --all`)
- Cron jobs (`/etc/cron*`, `crontab -l`)

**Output:** Text references requiring updates

#### Layer 2: Symlinks (CRITICAL - Most Often Missed)
**Purpose:** Find filesystem pointers to target  
**Method:** 
```bash
find /etc/ -type l -exec readlink -f {} \; | grep "<target>"
find /usr/local/bin/ -type l -exec readlink -f {} \; | grep "<target>"
find /opt/ -type l -exec readlink -f {} \; | grep "<target>"
```

**Why Critical:** grep only finds TEXT, not filesystem symlinks  
**Output:** All symlinks pointing to target (with source→destination)

#### Layer 3: Open File Handles
**Purpose:** Detect files currently in use by processes  
**Method:** `lsof | grep "<target>"`  
**Output:** Processes with open handles to target

#### Layer 4: Hardlinks
**Purpose:** Find files with multiple filesystem pointers  
**Method:** `stat -c "%h" <file>` (should be 1)  
**If >1:** Find all with `find / -inum <inode>`  
**Output:** Hardlink count and locations

#### Layer 5: Service & Timer Deep Scan
**Purpose:** Find service/timer/path units referencing target  
**Method:** Content search in all systemd units:
```bash
/etc/systemd/system/*.{service,timer,path}
/usr/lib/systemd/system/*.{service,timer,path}
```
**Output:** Unit files with path references

---

## Step 3: User Approval and Action Plan

### If ANY Layer Shows Dependencies:

**STOP.** File is **NOT safe to move** until all dependencies are addressed.

**Required Actions:**

1. **Document ALL Findings**
   Create summary table:
   
   | Dependency Source | Type | Path | Action Required |
   |-------------------|------|------|-----------------|
   | /etc/sysctl.d/99-solarflare-trading.conf | Symlink | Tuning/Network/... | Update symlink after move |
   | /etc/systemd/system/redis-hft.service | Text (ExecStart) | Tuning/Redis/... | Update service file |
   | /usr/local/bin/script.sh | Text reference | Tuning/Network/... | Update script |

2. **Present to User**
   - Show complete findings
   - Propose atomic update procedure
   - Request explicit approval

3. **Create Atomic Migration Plan**
   ```bash
   # WRONG - Separated operations:
   mv file.conf new/location/
   # <-- System broken here if someone accesses file
   sudo ln -sf new/location/file.conf /etc/...
   
   # RIGHT - Atomic operation:
   mv file.conf new/location/ && sudo ln -sf new/location/file.conf /etc/.../file.conf
   ```

### If ALL Layers Clean:

**Proceed**, but still:
1. Show user the clean audit results
2. Request approval before move
3. Create backup before execution

---

## Step 4: Atomic Execution

**Golden Rule:** Never separate file move from dependency updates.

### For Symlinked Files:

```bash
# Single atomic operation
NEW_PATH="SystemConfig/sysctl.d/99-solarflare-trading.conf"
OLD_SYMLINK="/etc/sysctl.d/99-solarflare-trading.conf"

mv Tuning/Network/99-solarflare-trading.conf "$NEW_PATH" && \
sudo ln -sf "$(realpath $NEW_PATH)" "$OLD_SYMLINK" && \
echo "✓ Move and symlink update complete"
```

### For Service-Referenced Files:

```bash
# Move file
mv Tuning/Network/script.sh Scripts/Network/script.sh

# Update service immediately
sudo sed -i 's|Tuning/Network/script.sh|Scripts/Network/script.sh|g' \
  /etc/systemd/system/service-name.service

# Reload daemon
sudo systemctl daemon-reload
```

### Validation After Each Move:

```bash
# Verify symlink points to new location
readlink -f /etc/sysctl.d/99-solarflare-trading.conf
# Should show: .../SystemConfig/sysctl.d/99-solarflare-trading.conf

# Verify service references new location
systemctl cat service-name.service | grep Scripts/Network/
# Should show new path

# Test functionality
sudo systemctl restart service-name.service
systemctl status service-name.service
```

---

## Step 5: Post-Migration Validation

**MANDATORY** after any migration:

### Re-Run All Five Layers

```bash
# Check old location has no remaining references
./Scripts/System/comprehensive-filesystem-audit.sh Tuning/Network/

# Should show zero dependencies (all moved/updated)
```

### Functional Testing

**For Config Files:**
```bash
# Reload configurations
sudo sysctl --system  # For sysctl configs
sudo udevadm control --reload-rules  # For udev rules
```

**For Scripts:**
```bash
# Test script execution
/usr/local/bin/script.sh --test
```

**For Services:**
```bash
# Restart and check status
sudo systemctl restart service-name
systemctl status service-name
# Check logs for errors
journalctl -u service-name -n 50
```

### User Confirmation

**Before cleanup:**
1. Show validation results to user
2. Demonstrate functionality preserved
3. Request final approval
4. Only then remove old location/backups

---

## The "Safe to Move" Checklist

A file/folder is **ONLY** safe to move when **ALL** of these are true:

- [ ] Layer 1 (Text References): ✅ No grep hits in /etc/, bash configs, systemd, cron
- [ ] Layer 2 (Symlinks): ✅ No symlinks from /etc/, /usr/, /opt/ pointing to target
- [ ] Layer 3 (Open Handles): ✅ No lsof hits (no processes have file open)
- [ ] Layer 4 (Hardlinks): ✅ Link count = 1 (or all hardlinks documented)
- [ ] Layer 5 (Services): ✅ No systemd unit files reference target path
- [ ] User Approval: ✅ User has reviewed audit results and approved move
- [ ] Backup Created: ✅ Full backup exists and verified
- [ ] Atomic Plan: ✅ Move and all dependency updates scripted together
- [ ] Rollback Ready: ✅ Procedure to restore from backup documented

**If ANY checkbox is unchecked → NOT SAFE TO MOVE.**

---

## What NOT To Do

### ❌ NEVER: Trust Grep Alone

```bash
# WRONG - Only finds text, misses symlinks
grep -r "filename" /etc/
# Empty result ≠ No dependencies
```

### ❌ NEVER: Assume Documentation is Complete

```bash
# WRONG - Assuming no docs = no usage
ls Documentation/ | grep filename
# Not found ≠ Not in use
```

### ❌ NEVER: Separate Move from Updates

```bash
# WRONG - Creates broken window
mv file.conf new/location/
# <-- System broken here
sudo ln -sf new/location/file.conf /etc/...
```

### ❌ NEVER: Skip User Approval

```bash
# WRONG - Acting on incomplete information
# "I think this is safe to move..."
mv file.conf new/location/
```

---

## Emergency Rollback

If a migration breaks something:

### Immediate Restoration

```bash
# Restore from backup
sudo cp -a Archive/Backup_YYYYMMDD_HHMMSS/Tuning/ ./

# Restore symlinks
sudo ln -sf $(pwd)/Tuning/Network/file.conf /etc/.../file.conf

# Restore service files
sudo cp Archive/Backup_YYYYMMDD_HHMMSS/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Reload configurations
sudo sysctl --system
sudo udevadm control --reload-rules

# Restart services
sudo systemctl restart service-name
```

### Post-Rollback

1. Document what broke
2. Update audit script if needed
3. Re-plan migration with correct dependencies
4. Do NOT re-attempt without user approval

---

## Summary: Trust But Verify (Actually, Just Verify)

### The Core Principle

**"No grep results" does NOT mean "no dependencies"**

Text search finds TEXT references only.  
Filesystem tools find FILESYSTEM pointers.  
Both are required.

### The Enforcement

This procedure is **MANDATORY** because:

1. grep-only searches missed production symlinks
2. User questions prevented production failure
3. We cannot afford this type of error

### The Commitment

Before ANY file move, I (the agent) will:

1. ✅ Run comprehensive-filesystem-audit.sh on target
2. ✅ Verify ALL FIVE layers clean or documented
3. ✅ Present complete findings to user
4. ✅ Create atomic migration procedure
5. ✅ Request user approval before execution
6. ✅ Validate after migration
7. ✅ Request final user confirmation

**No exceptions. No shortcuts. No assumptions.**

---

## Questions to Always Ask

Before declaring "safe to move":

1. **"Did I check for symlinks with `find -type l` and `readlink`?"**
2. **"Did I check /etc/ AND /usr/ AND /opt/ directories?"**
3. **"Did I verify with lsof that no process has this open?"**
4. **"Did I search actual systemd unit file content, not just service names?"**
5. **"Can I show the user proof from all five layers?"**

If ANY answer is "no" → **STOP** and complete verification.

---

**This procedure exists because user skepticism saved us once. We cannot rely on that again. The system must be foolproof.**
