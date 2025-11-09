# AI Trading Station - External Dependencies Audit
**Date:** October 29, 2025  
**Purpose:** Pre-migration inventory of all out-of-workspace dependencies  
**Goal:** Enable clean repository-driven deployment to new hardware/cloud/recovery environment

---

## Executive Summary

This audit identified **123 external dependencies** across the ai-trading-station workspace that currently rely on resources outside the repository. These dependencies span system services, kernel configurations, external storage, system binaries, OS-level optimizations, and security hardening that are critical for HFT performance.

### Risk Categories
- 🔴 **CRITICAL (35):** System will not function without these - systemd services, kernel configs, storage mounts, GPU blacklists, core utilities
- 🟡 **HIGH (54):** Performance degradation or feature loss - CPU affinity, network tuning, GPU optimization, Onload binaries, operator tools
- 🟢 **MEDIUM (34):** Monitoring/operational/security - Grafana, Prometheus, logs, system hardening, backup files

### Migration Complexity
- **Simple:** 45 items (copy files, includes backups and symlinks)
- **Moderate:** 48 items (requires configuration updates, symlink recreation)
- **Complex:** 30 items (system-level integration, kernel parameters, hardware-specific, module installation)

---

## 1. MONITORING FOLDER DEPENDENCIES

### 1.1 System Services (Systemd) - 🔴 CRITICAL

#### External Location: `/etc/systemd/system/`

| Service File | Purpose | Impact if Missing | Migration Notes |
|-------------|---------|-------------------|-----------------|
| `binance-trades.service` | WebSocket trade collector | No trade data ingestion | Service file must be recreated on new system |
| `binance-bookticker.service` | OrderBook data collector | No orderbook data | Service file must be recreated |
| `batch-writer.service` | Redis→QuestDB batch writer | Data pipeline breaks | Service file must be recreated |
| `questdb.service` | QuestDB time-series database | Complete data storage loss | Service file must be recreated |
| `prometheus.service` | Metrics collection | No monitoring | Service file must be recreated |
| `redis-hft.service` | HFT-optimized Redis | Cache layer fails | Service file must be recreated |
| `market-data.target` | Orchestrates all market data services | Cannot start/stop pipeline as unit | Service file must be recreated |
| `dual-gpu-trading-config.service` | GPU clock locking at boot | Performance variance 2-5ms | Located in workspace but needs systemd install |

**Scripts Referencing These:**
- `Monitoring/Scripts/backup_systemd_services.sh` (lines 19-25)
- `Monitoring/Scripts/configure_cpu_affinity.sh` (creates override directories)
- `Monitoring/Scripts/verify_cpu_affinity.sh` (checks service status)
- `QuestDB/scripts/datafeed_v2.sh` (start/stop/status commands)

**Why Out of Tree:**  
Systemd requires service files in `/etc/systemd/system/` for system-level service management, auto-restart, boot persistence.

**What Breaks:**  
- Cannot use `systemctl start/stop/restart` commands
- No auto-restart on failure
- No boot persistence
- Entire datafeed control system non-functional

**Migration Strategy:**
1. Export all service files from `/etc/systemd/system/` to `Archive/systemd_services/` ✅ (backup script exists)
2. Create installation script that deploys service files on new system
3. Update ExecStart paths to be workspace-relative or use environment variables
4. Create systemd service template generator

---

### 1.2 CPU Affinity Overrides - 🟡 HIGH

#### External Location: `/etc/systemd/system/*.service.d/`

CPU affinity configurations for HFT latency optimization:

| Override Directory | CPU Assignment | Performance Impact | Script References |
|-------------------|----------------|-------------------|-------------------|
| `/etc/systemd/system/prometheus.service.d/cpu-affinity.conf` | CPU 2 | Monitoring jitter | `configure_cpu_affinity.sh` line 89 |
| `/etc/systemd/system/redis-hft.service.d/cpu-affinity.conf` | CPU 4 | 200-500μs latency increase | `configure_cpu_affinity.sh` line 132 |
| `/etc/systemd/system/binance-trades.service.d/cpu-affinity.conf` | CPU 3 | WebSocket latency increase | `configure_cpu_affinity.sh` line 98 |
| `/etc/systemd/system/binance-bookticker.service.d/cpu-affinity.conf` | CPU 3 | OrderBook latency increase | `configure_cpu_affinity.sh` line 105 |
| `/etc/systemd/system/questdb.service.d/cpu-affinity.conf` | CPU 5 | JVM GC pauses, write latency | `configure_cpu_affinity.sh` line 115 |
| `/etc/systemd/system/batch-writer.service.d/cpu-affinity.conf` | CPU 6-7 | Ingestion throughput drops | `configure_cpu_affinity.sh` line 125 |

**Special Case - questdb.service.d/ Override Directory:**

The QuestDB service has **multiple override files** with configuration conflicts:
- `questdb.service.d/jvm-tuning.conf` - ❌ **DISABLED** (conflicts with main service file)
- `questdb.service.d/cpu-affinity.conf` - Status unknown (may be active or disabled)
- `questdb.service.d/hugepages.conf` - Status unknown (may be active or disabled)

**Issue:** The override directory exists but at least one file (`jvm-tuning.conf`) was disabled to prevent conflicts with settings in the main service file. This creates confusion about which overrides are active.

**Risk:** 🟡 HIGH - During migration, if all override files are copied without understanding which are disabled, conflicts may reappear causing QuestDB startup failures.

**Migration Strategy for QuestDB Overrides:**
1. Audit current active overrides: `sudo systemctl cat questdb.service` (shows merged config)
2. Document which overrides are actually used vs disabled
3. Either:
   - **Option A:** Merge active overrides into main service file, delete override directory
   - **Option B:** Clearly mark disabled overrides with `.disabled` extension
   - **Option C:** Document override conflicts in deployment guide
4. Test QuestDB startup after migration to ensure no conflicts

**Why Out of Tree:**  
Systemd override mechanism requires these in `/etc/systemd/system/` to apply CPU pinning at service startup.

**What Breaks:**  
Without CPU affinity:
- Cross-core cache invalidation adds 200-500μs per operation
- Trading services compete for CPU time
- Unpredictable latency spikes during heavy load
- Cannot meet <10ms P99 latency SLA

**Migration Strategy:**
1. Backup all `.service.d/` directories (script exists)
2. Create CPU topology detection script for new hardware
3. Generate affinity configs based on detected topology
4. Apply during deployment

---

### 1.3 Grafana Installation - 🟡 HIGH

#### External Locations: `/etc/grafana/`, `/var/lib/grafana/`, `/etc/apt/`

| Component | Path | Purpose | Script |
|-----------|------|---------|--------|
| Grafana config | `/etc/grafana/grafana.ini` | Main configuration | `install_grafana.sh` line 43 |
| Data sources | `/etc/grafana/provisioning/datasources/` | Prometheus, QuestDB connections | `install_grafana.sh` lines 55-97 |
| Dashboard provisioning | `/etc/grafana/provisioning/dashboards/` | Auto-load dashboards | `install_grafana.sh` line 103 |
| Dashboard storage | `/var/lib/grafana/dashboards/` | JSON dashboard files | `setup_dashboards.sh` line 13 |
| APT repository | `/etc/apt/sources.list.d/grafana.list` | Package source | `install_grafana.sh` line 34 |
| GPG key | `/etc/apt/keyrings/grafana.gpg` | Package verification | `install_grafana.sh` line 32 |

**Why Out of Tree:**  
System-wide monitoring service following FHS (Filesystem Hierarchy Standard) for Debian-based systems.

**What Breaks:**
- No visualization of market data metrics
- Cannot monitor capture rate, latency, errors
- Lose 7 configured alert rules
- No operational dashboard for trading

**Migration Strategy:**
1. Export Grafana provisioning configs
2. Backup dashboards (already in workspace: `Monitoring/*.json`)
3. Create Grafana bootstrap script with:
   - APT repository setup
   - Config deployment
   - Dashboard import
   - Data source auto-configuration

---

### 1.4 RAPL Power Monitoring - 🟢 MEDIUM

#### External Location: `/etc/udev/rules.d/90-rapl-permissions.rules`

**Purpose:** Allow non-root access to Intel RAPL (Running Average Power Limit) CPU power monitoring.

**Script:** `Monitoring/Config/90-rapl-permissions.rules`
```
SUBSYSTEM=="powercap", ACTION=="add", RUN+="/bin/chmod -R a+r /sys/class/powercap/intel-rapl/"
```

**Why Out of Tree:**  
Udev rules must be in `/etc/udev/rules.d/` to modify `/sys` permissions at boot.

**What Breaks:**
- Monitoring scripts cannot read CPU power consumption
- `PerformanceGate/extended_tail_sampler.py` (line 140) loses power metrics
- Thermal monitoring incomplete

**Migration Strategy:**
1. Copy rule file to new system `/etc/udev/rules.d/`
2. Reload udev: `udevadm control --reload-rules && udevadm trigger`

---

### 1.5 Performance Gate Baselines - 🟡 HIGH

#### External Location: `/opt/redis-hft/`

**Directory Structure:** 17 files across 6 subdirectories

| Path | Purpose | Size | Status |
|------|---------|------|--------|
| **config/** | Redis configuration | ~500KB | ACTIVE |
| `├─ redis-hft.conf` | **Active Redis config** (used by redis-hft.service) | 4KB | 🔴 CRITICAL |
| `├─ redis-hft.conf.backup_20251008` | Config backup | 4KB | Backup |
| `├─ redis.env` | Environment variables | 1KB | ACTIVE |
| `├─ redis-auth.txt` | Authentication credentials | <1KB | 🔴 CRITICAL |
| `└─ redis-pass.txt` | Password file | <1KB | 🔴 CRITICAL |
| **scripts/** | Monitoring/management scripts | ~100KB | ACTIVE |
| `├─ redis-hft-cli` | CLI wrapper (symlinked from /usr/local/bin) | 20KB | 🟡 HIGH |
| `└─ redis-hft-monitor.sh` | **Active monitoring script** | 20KB | 🟡 HIGH |
| **metrics/** | Live performance metrics | ~1MB | ACTIVE |
| `├─ redis-metrics.json` | Current metrics (updated by monitor) | Variable | ACTIVE |
| `├─ redis-metrics.json.tmp.*` | Temporary metric files | Variable | Transient |
| `├─ current-fingerprint.json` | System fingerprint | ~100KB | ACTIVE |
| `└─ network-ull-metrics.json` | Network ULL metrics | ~50KB | ACTIVE |
| **baselines/** | Performance baselines | ~500KB | ACTIVE |
| `├─ fingerprint-baseline.json` | **Performance baseline** | ~200KB | 🟡 HIGH |
| `└─ fingerprint-baseline.json.backup_kernel_6.8.0-84` | Kernel-specific backup | ~200KB | Backup |
| **archive/** | Old configurations | ~2MB | Archive |
| `├─ redis-hft.conf.backup.20250930_125659` | Historical backup | 4KB | Archive |
| `├─ redis-hft.conf.backup` | Generic backup | 4KB | Archive |
| `└─ redis-hft-memory-patch-fixed.conf` | Patched config archive | 4KB | Archive |
| **data/** | Redis persistence files | ~10MB | ACTIVE |
| **logs/** | Redis log files | ~35MB | ACTIVE |
| `CONFIG_README.md` | Documentation | 5KB | Documentation |

**Total Size:** ~50MB

**Why Out of Tree:**  
- Persistent location for institutional-grade performance tracking
- Survives workspace updates/git operations
- Root-owned for security (config hashing)
- Standardized location for Redis HFT infrastructure

**What Breaks:**
- **redis-hft.service fails to start** - Cannot find `/opt/redis-hft/config/redis-hft.conf`
- **Authentication fails** - redis-auth.txt missing
- **Performance gate cannot validate system** - Baselines missing
- **No runtime monitoring** - redis-hft-monitor.sh not found
- **Cannot detect environment drift** - Lose fingerprint comparisons
- **Lose latency SLA tracking** - Baseline comparisons impossible

**Scripts Using These:**
- `redis-hft.service` ExecStart → `/opt/redis-hft/config/redis-hft.conf` (🔴 CRITICAL)
- `/opt/redis-hft/scripts/redis-hft-monitor.sh` → Updates `/opt/redis-hft/metrics/` (🟡 HIGH)
- `Monitoring/Scripts/PerformanceGate/perf-gate.sh` → Reads `/opt/redis-hft/baselines/` (lines 22-23, 317, 476)
- `Monitoring/Scripts/PerformanceGate/runtime-fingerprint.sh` → Reads config (line 44-45)
- `Monitoring/Scripts/Redis/redis-hft-monitor_to_json.sh` → Calls monitor script (line 9)

**Critical Files (Must Migrate):**
1. 🔴 `config/redis-hft.conf` - Redis service dependency
2. 🔴 `config/redis-auth.txt` - Authentication
3. 🟡 `scripts/redis-hft-monitor.sh` - Active monitoring
4. 🟡 `baselines/fingerprint-baseline.json` - Performance validation

**Optional Files (Can Regenerate):**
- `metrics/*.json` - Regenerated at runtime
- `archive/*` - Historical, not needed for operation
- `data/` - Ephemeral (no persistence mode)
- `logs/` - Regenerated

**Migration Strategy:**
1. **Option A - Full Migration (Recommended):**
   ```bash
   sudo tar -czf opt_redis_hft_backup.tar.gz /opt/redis-hft/
   # On new system:
   sudo tar -xzf opt_redis_hft_backup.tar.gz -C /
   sudo chown -R redis:redis /opt/redis-hft/
   ```
   
2. **Option B - Selective Migration:**
   ```bash
   # Migrate only critical files
   sudo mkdir -p /opt/redis-hft/{config,scripts,baselines}
   sudo cp /opt/redis-hft/config/redis-hft.conf /opt/redis-hft/config/
   sudo cp /opt/redis-hft/config/redis-auth.txt /opt/redis-hft/config/
   sudo cp /opt/redis-hft/scripts/redis-hft-monitor.sh /opt/redis-hft/scripts/
   sudo cp /opt/redis-hft/baselines/fingerprint-baseline.json /opt/redis-hft/baselines/
   ```

3. **Option C - Workspace Integration (Future):**
   - Move entire `/opt/redis-hft/` into workspace as `Redis/external/`
   - Symlink `/opt/redis-hft/` → workspace
   - Update service file to reference workspace
   - **Risk:** High - requires service reconfiguration and testing

4. **Validation After Migration:**
   ```bash
   sudo systemctl start redis-hft
   /opt/redis-hft/scripts/redis-hft-monitor.sh
   ls -la /opt/redis-hft/metrics/redis-metrics.json  # Should be updated
   ```

---

### 1.6 System Binaries (/usr/local/bin/) - 🟡 HIGH / 🔴 CRITICAL

#### External Location: `/usr/local/bin/`

This directory contains 20 files: workspace symlinks, standalone system scripts, Onload binaries, and backup files.

#### **A. Workspace Symlinks (4 files) - 🟡 HIGH**

| Symlink | Target | Purpose | Risk |
|---------|--------|---------|------|
| `datafeed` | `QuestDB/scripts/datafeed_v2.sh` | **Primary operator interface** for data pipeline control | 🟡 HIGH - Operator workflows break |
| `ultra-low-latency-nic.sh` | `Tuning/Network/ultra-low-latency-nic.sh` | NIC tuning (used by systemd service) | 🟡 HIGH - Network optimization lost |
| `vm-manager.sh` | `vm-manager.sh` (workspace root) | VM/system management utility | 🟢 MEDIUM - Convenience tool |
| `redis-hft-cli` | `/opt/redis-hft/scripts/redis-hft-cli` | Redis CLI wrapper (external → external symlink) | 🟢 MEDIUM - Convenience tool |

**Why Symlinked:**  
Allows operator commands (`datafeed start`, `datafeed status`) from any directory without full paths.

**What Breaks:**
- `datafeed` command not found - operators can't control pipeline easily
- System services fail if they reference symlinked scripts
- Workflow documentation becomes outdated

**Migration Strategy:**
1. Recreate symlinks on new system pointing to new workspace path
2. OR add workspace scripts to PATH via `/etc/profile.d/trading-station.sh`
3. Update any hardcoded references to `/usr/local/bin/datafeed`

---

#### **B. Standalone System Scripts (6 files) - 🔴 CRITICAL / 🟡 HIGH**

| Script | Purpose | Used By | Risk |
|--------|---------|---------|------|
| **configure-nic-irq-affinity.sh** | Pins NIC IRQs to CPUs 0-1 (CPU core assignment) | **configure-nic-irq-affinity.service** (systemd) | 🔴 CRITICAL |
| **post_boot_core_isolation_verification.sh** | Validates CPU isolation after boot | Manual/monitoring | 🟡 HIGH |
| **toggle_trading_mode_enhanced.sh** | Switches between trading/dev mode | Operators | 🟡 HIGH |
| **setup_solarflare.sh** | Solarflare NIC initial setup | One-time setup | 🟢 MEDIUM |
| **prevent_snap_firefox.sh** | Disables snap packages (system optimization) | Boot/manual | 🟢 MEDIUM |
| **toggle_trading_mode.sh.backup** | Backup of trading mode script | Archive | 🟢 LOW |

**Note on configure-nic-irq-affinity.sh:**
- **Location**: Only in workspace `Tuning/Network/` - referenced directly by systemd service
- **NOT A DUPLICATE**: Works alongside `ultra-low-latency-nic.sh` (which IS symlinked)
- **Different purposes**:
  - `configure-nic-irq-affinity.sh`: CPU core assignment (which CPU handles NIC interrupts)
  - `ultra-low-latency-nic.sh`: NIC hardware tuning (interrupt coalescing, ring buffers, offloads)
- Both run at boot via separate systemd services, both production-active

**What Breaks:**
- **configure-nic-irq-affinity.service fails** - NIC IRQs not pinned, 50-200μs latency variance
- No post-boot validation - cannot verify CPU isolation is working
- Cannot switch trading modes - system stuck in one configuration

**Migration Strategy:**
1. **configure-nic-irq-affinity.sh**: Already in workspace at `Tuning/Network/`, referenced by systemd service - just update service file path
2. Include `post_boot_core_isolation_verification.sh` in deployment bundle
3. Copy `toggle_trading_mode_enhanced.sh` for operational flexibility
4. Archive `setup_solarflare.sh` (one-time use, keep for reference)

---

#### **C. Onload Binaries (7 files) - 🟡 HIGH**

These are part of the Solarflare Onload installation, NOT workspace-managed:

| Binary | Purpose | Part Of |
|--------|---------|---------|
| `onload_cp_client` | Onload control plane client | Onload |
| `onload_cp_server` | Onload control plane server | Onload |
| `onload_fuser` | Show processes using Onload | Onload |
| `onload_mibdump` | Dump Onload MIB statistics | Onload |
| `onload_tcpdump.bin` | Onload-aware packet capture | Onload |
| `onload_tool` | Onload configuration utility | Onload |
| Additional Onload utilities | Various debugging/monitoring | Onload |

**Why Out of Tree:**  
Installed by Onload installer, system-level kernel bypass networking infrastructure.

**What Breaks:**
- Cannot debug Onload issues
- No visibility into accelerated network stack
- Cannot verify which processes are using Onload
- Troubleshooting becomes blind

**Migration Strategy:**
1. These are installed automatically by Onload installer
2. Include Onload installer in deployment bundle
3. Run Onload installation during system setup (Phase 3)
4. Verify binaries present: `ls /usr/local/bin/onload_*`
5. **DO NOT** try to copy binaries manually - must use installer

---

#### **D. Backup Files (3 files) - 🟢 LOW**

| Backup File | Original | Date | Risk |
|-------------|----------|------|------|
| `configure-nic-irq-affinity.sh.bak-20250901-200453` | configure-nic-irq-affinity.sh | 2025-09-01 | 🟢 LOW |
| `ultra-low-latency-nic.sh.backup_v3_20251002` | ultra-low-latency-nic.sh | 2025-10-02 | 🟢 LOW |
| `ultra-low-latency-nic-v2.0.backup` | ultra-low-latency-nic.sh | Unknown | 🟢 LOW |

**Why These Exist:**  
Manual backups before script modifications, left in `/usr/local/bin/`.

**What Breaks:**  
Nothing - these are inactive backups.

**Migration Strategy:**
1. **Do NOT migrate** - these are stale backups
2. Move to Archive on current system: `sudo mv /usr/local/bin/*.backup* /home/youssefbahloul/ai-trading-station/Archive/`
3. Clean up clutter in system directory

---

#### **Summary: /usr/local/bin/ Dependencies**

| Category | Count | Risk | Migration Complexity |
|----------|-------|------|---------------------|
| Workspace symlinks | 4 | 🟡 HIGH | Simple (recreate symlinks) |
| Standalone scripts | 6 | 🔴 CRITICAL | Moderate (copy + decide on duplicates) |
| Onload binaries | 7 | 🟡 HIGH | Complex (Onload installer required) |
| Backup files | 3 | 🟢 LOW | N/A (do not migrate) |
| **TOTAL** | **20** | **Mixed** | **Moderate overall** |

**Key Action Items:**
1.  Recreate 4 symlinks on new system with updated workspace path
2. 🟡 Include Onload installer in deployment (installs 7 binaries automatically)
3. 🟡 Copy standalone scripts (post_boot_core_isolation_verification.sh, toggle_trading_mode_enhanced.sh, etc.)
4. 🟢 Clean up 3 backup files (archive on current system, don't migrate)

---

## 2. QUESTDB FOLDER DEPENDENCIES

### 2.1 QuestDB Database Storage - 🔴 CRITICAL

#### Primary Storage: `/home/youssefbahloul/ai-trading-station/QuestDB/data/`
#### Cold Storage: `/mnt/hdd/questdb/cold`

| Storage Tier | Path | Purpose | Capacity | Scripts |
|-------------|------|---------|----------|---------|
| Hot (NVMe) | `QuestDB/data/` | Last 30 days, active queries | 512GB | All QuestDB scripts |
| Cold (HDD) | `/mnt/hdd/questdb/cold` | Archive >30 days | 4TB | `lifecycle-manager.sh` lines 13-14 |

**Scripts Referencing External Storage:**
- `QuestDB/scripts/lifecycle-manager.sh` - Moves partitions to cold storage (line 14)
- `Monitoring/Scripts/monitor_questdb.py` - Monitors both storage tiers (lines 22-23)
- `QuestDB/scripts/retention-cleanup.sh` - Storage usage tracking (lines 77, 83)

**Why Split:**
- Hot: Ultra-low latency NVMe for trading queries (<2ms)
- Cold: Cost-effective HDD for compliance/backtesting (5-20ms acceptable)
- Automatic tiering saves ~$5K/year in storage costs

**What Breaks:**
- Cold storage mount missing: Archive data inaccessible, lifecycle manager fails
- Hot storage path change: All ingestion stops, existing data orphaned
- Both: Complete data loss for time-series analytics

**Migration Strategy:**
1. **Option A - Single Tier (Simple):** Keep all data in workspace, remove cold storage (easier migration, higher cost)
2. **Option B - Dual Tier (Recommended):** 
   - Document cold storage mount requirements
   - Create mount point detection script
   - Make cold storage path configurable via env var
   - Update `lifecycle-manager.sh` with fallback to single-tier mode
3. Export cold storage configuration to `config/storage.conf`

---

### 2.2 Python Path Hardcoding - 🟡 HIGH

Multiple scripts contain hardcoded workspace paths:

| Script | Hardcoded Path | Line | Reason |
|--------|---------------|------|--------|
| `benchmark_tiered_cache.py` | `/home/youssefbahloul/ai-trading-station/QuestDB/scripts` | 24 | sys.path import |
| `test_cache_activity.py` | `/home/youssefbahloul/ai-trading-station/QuestDB/scripts` | 9 | sys.path import |
| `test_cache_integration.py` | `/home/youssefbahloul/ai-trading-station/QuestDB/scripts` | 13 | sys.path import |
| `tiered_cache_service.py` | `/home/youssefbahloul/ai-trading-station/QuestDB/scripts` | 27, 75 | sys.path + logging |
| `comprehensive_latency_baseline.py` | `/home/youssefbahloul/ai-trading-station/QuestDB/scripts` | 368 | sys.path import |
| `comprehensive_latency_baseline.py` | `/home/youssefbahloul/ai-trading-station/Documentation/` | 494 | Output file |

**Why Hardcoded:**  
Quick development, needed to import shared modules across test scripts.

**What Breaks:**
- Scripts fail to find modules on new system
- Import errors prevent testing/benchmarking
- Output files written to wrong location

**Migration Strategy:**
1. Replace all hardcoded paths with:
   ```python
   import os
   WORKSPACE_ROOT = os.getenv('AI_TRADING_WORKSPACE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   sys.path.insert(0, os.path.join(WORKSPACE_ROOT, 'QuestDB/scripts'))
   ```
2. Create `setup_env.sh` to export workspace location
3. Update all scripts in single pass

---

### 2.3 QuestDB Binary Installation - 🔴 CRITICAL

#### Location: `QuestDB/questdb-9.1.0-rt-linux-x86-64/`

**Status:** ✅ Already in workspace (portable)

The QuestDB binary is self-contained within the workspace, BUT:

**External Dependencies:**
- Java 11+ runtime (not in workspace)
- System libraries: `libc`, `libpthread`, `librt`
- Optional: `jemalloc` (for memory allocation)

**Script:** `QuestDB/questdb-9.1.0-rt-linux-x86-64/bin/questdb.sh`
- Checks for Java in `/usr/libexec/java_home` (line 128)
- Looks for jemalloc in `$BASE/libjemalloc*` (line 152)
- Default data dir: `/usr/local/var/questdb` (line 73-74)

**Migration Strategy:**
1. Document Java 11+ requirement
2. Create Java detection/installation script
3. Override data directory via `QDB_ROOT` environment variable
4. Include jemalloc in deployment bundle for performance

---

## 3. TRADING FOLDER DEPENDENCIES

### 3.1 Onload Acceleration - 🟡 HIGH

#### Location: `Trading/Wrapper/onload-trading`

**Status:** Binary wrapper in workspace, but depends on system-installed Onload kernel module.

**External Dependencies:**
- Solarflare Onload kernel module: `/lib/modules/$(uname -r)/extra/sfc.ko`
- Onload user-space library: `/usr/lib/x86_64-linux-gnu/libonload.so`
- Onload binary: `/usr/bin/onload` or `/sbin/onload`

**Referenced in:**
- `Monitoring/Scripts/PerformanceGate/runtime-fingerprint.sh` (lines 101-107) - Version detection
- `Tuning/Onload/*.conf.backup` - Module configuration

**Why Out of Tree:**
Kernel module must be installed system-wide for low-level network stack bypass.

**What Breaks:**
- Onload not installed: 50-100μs latency increase, network bypass disabled
- Module version mismatch: Trading services fail to start
- No graceful fallback to standard networking

**Migration Strategy:**
1. **Critical:** Document exact Onload version (currently detected at runtime)
2. Include Onload installer/tarball in deployment bundle
3. Create Onload installation verification script
4. Add fallback to standard sockets if Onload unavailable (development mode)

---

## 4. TUNING FOLDER DEPENDENCIES

### 4.1 Kernel Parameters (sysctl) - 🔴 CRITICAL

#### External Location: `/etc/sysctl.d/99-solarflare-trading.conf`

**File:** `Tuning/Network/99-solarflare-trading.conf` (backup in workspace)

**Critical Parameters:**
```conf
net.core.rmem_max = 268435456          # 256MB socket buffers
net.core.wmem_max = 268435456
net.core.busy_read = 50                # Busy polling (reduces interrupt latency)
net.core.busy_poll = 50
net.ipv4.tcp_timestamps = 0            # Disable for lower latency
net.ipv4.tcp_low_latency = 1           # Prioritize latency over throughput
vm.swappiness = 1                      # Minimize swapping
net.core.somaxconn = 65535             # Redis connection queue
vm.overcommit_memory = 1               # Redis memory allocation
```

**Why Out of Tree:**  
Must be in `/etc/sysctl.d/` to apply kernel parameters at boot before any services start.

**What Breaks Without These:**
- Packet drops under burst traffic (10-30% data loss)
- Network latency increase 100-500μs
- Redis connection failures under load
- OOM killer may terminate Redis
- TCP buffering issues

**Migration Strategy:**
1. ✅ Backup exists in `Tuning/Network/99-solarflare-trading.conf`
2. Create deployment script:
   ```bash
   sudo cp Tuning/Network/99-solarflare-trading.conf /etc/sysctl.d/
   sudo sysctl -p /etc/sysctl.d/99-solarflare-trading.conf
   ```
3. Add validation: Check if parameters applied correctly
4. Document reboot requirement for full effect

---

### 4.2 Huge Pages Configuration - 🟡 HIGH

#### External Location: `/etc/sysctl.d/99-trading-hugepages.conf`

**File:** `Tuning/Onload/99-trading-hugepages.conf` (backup in workspace)

**Purpose:** Pre-allocate huge pages for QuestDB and Redis to reduce TLB misses (5-10% performance gain).

**Parameters:**
```conf
vm.nr_hugepages = 4096                 # 8GB of 2MB pages
vm.hugetlb_shm_group = 1000           # Allow trading user group
```

**Scripts Using Huge Pages:**
- QuestDB JVM (automatically uses huge pages if available)
- Redis (can use huge pages with compile flag)

**What Breaks:**
- Memory allocation slower (10-20μs per allocation)
- Higher CPU usage for page table walks
- Slight performance regression (5-10%)

**Migration Strategy:**
1. Include in system tuning bundle
2. Make optional with detection: `grep -q Hugepagesize /proc/meminfo`
3. Auto-configure based on available RAM

---

### 4.3 Network IRQ Affinity - 🔴 CRITICAL

#### External Locations:
- `/etc/udev/rules.d/99-trading-nic-irq.rules` - Triggers affinity script on NIC up
- `/proc/irq/*/smp_affinity` - Runtime IRQ → CPU mapping (ephemeral)

**File:** `Tuning/Network/99-trading-nic-irq.rules`
```
SUBSYSTEM=="net", ACTION=="add", KERNEL=="enp130s0f*", 
RUN+="/bin/bash /home/youssefbahloul/ai-trading-station/Tuning/Network/configure-nic-irq-affinity.sh"
```

**Scripts:**
- `Tuning/Network/configure-nic-irq-affinity.sh` - Pins NIC IRQs to CPUs 0-1
- `Tuning/Network/ultra-low-latency-nic.sh` - Additional NIC tuning

**Why Out of Tree:**
- Udev rule: Must be in `/etc/udev/rules.d/` to trigger on hardware events
- `/proc/irq/`: Kernel interface, cannot be relocated

**What Breaks:**
- NIC IRQs not pinned: Spread across all CPUs
- IRQ handling interferes with trading cores
- Latency variance 50-200μs
- Network processing jitter

**Migration Strategy:**
1. ✅ Script is in workspace, udev rule is not
2. Copy udev rule to `/etc/udev/rules.d/`
3. **CRITICAL:** Update hardcoded path in udev rule:
   ```
   RUN+="/bin/bash /home/youssefbahloul/ai-trading-station/..."
   ```
   Change to:
   ```
   RUN+="/bin/bash /opt/ai-trading-station/Tuning/Network/configure-nic-irq-affinity.sh"
   ```
4. Reload udev after deployment

---

### 4.4 GPU Configuration - 🟡 HIGH

#### External: `/etc/systemd/system/dual-gpu-trading-config.service`

**File:** `Tuning/GPU/dual-gpu-trading-config.service` (in workspace)

**Purpose:** Lock GPU clocks at boot to eliminate frequency scaling (2-5ms variance reduction).

**Script:** `Tuning/GPU/gpu_clock_lock.sh`
- Sets persistence mode
- Locks clocks to 3090MHz (graphics) / 14001MHz (memory)
- Requires NVIDIA drivers and `nvidia-smi`

**ExecStart Path:**
```
ExecStart=/home/youssefbahloul/ai-trading-station/Tuning/GPU/gpu_clock_lock.sh
```

**Why Out of Tree:**  
Systemd service for boot-time execution.

**What Breaks:**
- GPU frequency scaling causes 2-5ms inference variance
- Thermal throttling during trading hours
- Unpredictable model inference times

**Migration Strategy:**
1. ✅ Service file in workspace
2. Update ExecStart to new workspace path during deployment
3. Install service: `sudo systemctl enable dual-gpu-trading-config.service`
4. Document NVIDIA driver version requirement

---

### 4.5 Redis HFT Configuration - 🔴 CRITICAL

#### External Location: `/opt/redis-hft/config/redis-hft.conf`

**File:** `Tuning/Redis/redis-hft.conf` (backup in workspace)

**Critical Settings:**
```conf
port 6379
maxmemory 8gb
maxmemory-policy noeviction
save ""                          # No disk persistence for latency
appendonly no
dir /var/lib/redis-hft           # External data directory
```

**Why Out of Tree:**
- Referenced by systemd service ExecStart
- Separate from workspace for security (protected by file permissions)
- Data directory in `/var/lib/` follows FHS

**What Breaks:**
- Redis starts with default config (persistence enabled = latency spikes)
- Wrong memory limit (OOM killer)
- Persistence enabled (100-500ms fsync stalls)
- Data directory wrong (permission issues)

**Migration Strategy:**
1. ✅ Config backup exists
2. Create `/opt/redis-hft/config/` during deployment
3. Copy config with proper permissions (0640, redis:redis)
4. Update systemd service to reference correct config path
5. Create `/var/lib/redis-hft/` with correct ownership

---

### 4.6 Solarflare Driver Configuration - 🟡 HIGH

#### External Locations:
- `/etc/modprobe.d/sfc.conf` - Driver module parameters
- `/etc/depmod.d/sfc-depmod.conf` - Module dependency resolution

**Files:** `Tuning/Solarflare/*.conf.backup` (backups in workspace)

**Purpose:** Configure Solarflare NIC driver for HFT:
- Large receive rings
- Interrupt moderation off
- Hardware timestamping

**Why Out of Tree:**  
Kernel module configuration must be in `/etc/modprobe.d/` to load at boot.

**What Breaks:**
- Driver loads with default settings
- Smaller ring buffers (packet drops under burst)
- Interrupt coalescing enabled (60-150μs latency)

**Migration Strategy:**
1. ✅ Backups exist
2. Copy to `/etc/modprobe.d/` and `/etc/depmod.d/`
3. Rebuild module dependencies: `sudo depmod -a`
4. Reload driver: `sudo modprobe -r sfc && sudo modprobe sfc`

---

### 4.7 Onload Module Configuration - 🟡 HIGH

#### External Locations:
- `/etc/modprobe.d/onload.conf` - Onload kernel module parameters
- `/etc/depmod.d/onload.conf` - Dependency resolution
- `/etc/dkms/framework.conf` - DKMS (Dynamic Kernel Module Support) for Onload

**Files:** `Tuning/Onload/*.conf.backup` (backups in workspace)

**Purpose:** Configure Onload for kernel bypass networking.

**Why Out of Tree:**  
Kernel module infrastructure requires system-level configuration.

**What Breaks:**
- Onload may not load automatically
- Incorrect module parameters (performance degradation)
- DKMS fails to rebuild after kernel updates

**Migration Strategy:**
1. Include Onload installer in deployment bundle
2. Use installer to set up configs correctly
3. Document kernel version compatibility
4. Test DKMS rebuild after deployment

---

### 4.8 System Hardening Configurations - 🟢 MEDIUM

#### External Location: `/etc/sysctl.d/` (10-* prefix files)

These are system security and stability configurations, NOT directly HFT-related but important for production system integrity.

| Config File | Purpose | Impact | Risk |
|-------------|---------|--------|------|
| `10-kernel-hardening.conf` | Kernel security hardening (ASLR, stack protection) | Enhanced security | 🟢 MEDIUM |
| `10-network-security.conf` | Network stack security (SYN cookies, RP filter) | DDoS protection | 🟢 MEDIUM |
| `10-ptrace.conf` | Restrict ptrace (prevent process inspection) | Security | 🟢 LOW |
| `10-bufferbloat.conf` | TCP congestion control (reduce bufferbloat) | Network stability | 🟢 MEDIUM |
| `10-ipv6-privacy.conf` | IPv6 privacy extensions | Privacy | 🟢 LOW |
| `10-map-count.conf` | `vm.max_map_count` for memory-mapped files | Application stability | 🟢 MEDIUM |
| `10-zeropage.conf` | Transparent hugepages settings | Memory management | 🟢 LOW |
| `10-magic-sysrq.conf` | Magic SysRq key configuration | Emergency recovery | 🟢 LOW |
| `10-console-messages.conf` | Kernel console log level | Logging | 🟢 LOW |

**Why Out of Tree:**  
System-wide security policies, applied at boot before any services start.

**What Breaks:**
- Less secure system (but still functions)
- Potential vulnerability to DDoS/attacks
- Memory-intensive apps (Redis/QuestDB) may hit limits if `max_map_count` not set
- More kernel log spam if console messages not filtered

**Migration Strategy:**
1. **Option A (Recommended):** Include in deployment bundle as optional "system hardening layer"
   ```bash
   sudo cp configs/sysctl.d/10-*.conf /etc/sysctl.d/
   sudo sysctl --system
   ```
2. **Option B:** Document but don't enforce - let target system use its own security policies
3. Check if target system already has hardening - don't overwrite
4. **Critical Check:** Verify `10-map-count.conf` is high enough for Redis/QuestDB (usually `vm.max_map_count=262144`)

**Backup Status:**  
⚠️ These files are NOT currently backed up in workspace. 

**Action Required:**
```bash
sudo cp /etc/sysctl.d/10-*.conf ~/ai-trading-station/Tuning/System-Hardening/
```

---

### 4.9 Kernel Module Blacklists - 🔴 CRITICAL (GPU) / 🟢 LOW (Others)

#### External Location: `/etc/modprobe.d/` (blacklist-* files)

Module blacklists prevent conflicting kernel modules from loading, especially critical for NVIDIA GPU operation.

| Blacklist File | Prevents Loading | Purpose | Risk |
|----------------|------------------|---------|------|
| **blacklist-nouveau.conf** | **nouveau driver** | **Prevents open-source NVIDIA driver conflict** | 🔴 **CRITICAL** |
| `blacklist-framebuffer.conf` | Generic framebuffer drivers | GPU stability | 🟡 HIGH |
| `blacklist-firewire.conf` | FireWire subsystem | Attack surface reduction | 🟢 LOW |
| `blacklist-rare-network.conf` | Rare network drivers | Clean system, less modules | 🟢 LOW |

#### **CRITICAL: blacklist-nouveau.conf**

**Content:**
```conf
blacklist nouveau
options nouveau modeset=0
```

**Why CRITICAL:**  
- Nouveau (open-source NVIDIA driver) conflicts with proprietary NVIDIA driver
- If nouveau loads, NVIDIA driver fails to initialize
- GPU becomes unusable for trading inference
- System may not boot into graphical environment

**What Breaks Without It:**
- 🔴 NVIDIA driver load fails
- 🔴 GPU unavailable for inference
- 🔴 `nvidia-smi` command fails
- 🔴 `dual-gpu-trading-config.service` fails (cannot lock GPU clocks)
- 🔴 2-5ms inference variance returns (no clock locking possible)

**Migration Strategy:**
1. **MUST HAVE** for any system with NVIDIA GPUs
2. Copy to target system BEFORE installing NVIDIA drivers:
   ```bash
   sudo cp blacklist-nouveau.conf /etc/modprobe.d/
   sudo update-initramfs -u
   sudo reboot
   ```
3. Verify after reboot: `lsmod | grep nouveau` should return nothing
4. Then install NVIDIA drivers

#### **Other Blacklists**

**blacklist-framebuffer.conf:**
- Prevents generic framebuffer drivers that can interfere with NVIDIA
- Recommended for GPU stability
- Risk if missing: 🟡 HIGH (GPU may work but with glitches)

**blacklist-firewire.conf & blacklist-rare-network.conf:**
- Security/cleanliness optimizations
- Not functional requirements
- Risk if missing: 🟢 LOW (no impact on trading)

**Backup Status:**  
⚠️ These files are NOT currently backed up in workspace.

**Action Required:**
```bash
sudo cp /etc/modprobe.d/blacklist-*.conf ~/ai-trading-station/Tuning/Module-Blacklists/
```

**Migration Priority:**
1. 🔴 **PHASE 1 (Pre-GPU setup):** blacklist-nouveau.conf - MUST install before NVIDIA drivers
2. 🟡 **PHASE 3 (GPU tuning):** blacklist-framebuffer.conf - Include with GPU setup
3. 🟢 **PHASE 4 (Optional):** blacklist-firewire, blacklist-rare-network - System cleanup

---

## 5. CROSS-CUTTING DEPENDENCIES

### 5.1 Kernel Command Line Parameters - 🔴 CRITICAL

#### External Location: `/etc/default/grub` → `/boot/grub/grub.cfg`

**Parameters Currently Set:**
```
GRUB_CMDLINE_LINUX="isolcpus=2,3,4,5,6,7 nohz_full=2,3,4,5,6,7 rcu_nocbs=2,3,4,5,6,7 intel_pstate=disable processor.max_cstate=1 intel_idle.max_cstate=0"
```

**Purpose:**
- `isolcpus`: Isolate CPUs 2-7 for trading (no general processes)
- `nohz_full`: Disable timer ticks on trading cores (reduce interrupts)
- `rcu_nocbs`: Move RCU callbacks off trading cores
- `intel_pstate=disable`: Use legacy CPU frequency scaling (more predictable)
- `processor.max_cstate=1`: Disable deep sleep states (lower wakeup latency)

**Referenced in:**
- `Monitoring/Scripts/verify_cpu_affinity.sh` (line 150) - Validation
- `Monitoring/Scripts/post_reboot_diagnostic.sh` - Checks applied correctly
- `Monitoring/Scripts/PerformanceGate/runtime-fingerprint.sh` (line 64) - Fingerprinting

**Why Out of Tree:**  
Kernel command line set by bootloader, applied before any filesystems mount.

**What Breaks Without These:**
- 🔴 **CATASTROPHIC:** No CPU isolation
- General OS processes run on trading cores
- Scheduler interrupts on trading cores (1000Hz timer)
- Latency increases 10-100x (microseconds → milliseconds)
- Cannot meet HFT latency requirements

**Migration Strategy:**
1. **Document current kernel cmdline:**
   ```bash
   cat /proc/cmdline > Documentation/KERNEL_CMDLINE.txt
   ```
2. Create GRUB configuration script for new system
3. **Critical:** Verify CPU topology matches (8-core system assumed)
4. Update and rebuild GRUB:
   ```bash
   sudo update-grub
   ```
5. **Requires reboot** - include in deployment checklist
6. Validation script post-reboot

---

### 5.2 CPU Frequency Governor - 🟡 HIGH

#### External Location: `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`

**Current Setting:** `performance` (all CPUs locked at maximum frequency)

**Scripts Checking Governor:**
- `Monitoring/Scripts/verify_cpu_affinity.sh` (lines 86-102)
- `Monitoring/Scripts/post_reboot_diagnostic.sh` (line 62)
- `Monitoring/Scripts/PerformanceGate/runtime-fingerprint.sh` (lines 82-86)

**Why Out of Tree:**  
Kernel sysfs interface for CPU frequency scaling policy.

**What Breaks:**
- CPUs use `powersave` or `schedutil` governor
- Frequency scaling based on load (100-500μs transition latency)
- Unpredictable CPU frequency during trading
- Inconsistent instruction throughput

**Migration Strategy:**
1. Set at boot via systemd service or rc.local:
   ```bash
   for cpu in /sys/devices/system/cpu/cpu[0-7]/cpufreq/scaling_governor; do
       echo performance > $cpu
   done
   ```
2. OR use kernel parameter `intel_pstate=disable` (already in grub)
3. Validate post-boot

---

### 5.3 Network Interface Configuration - 🔴 CRITICAL

#### External: `/etc/network/interfaces` or NetworkManager configs

**Current Setup:**
- Interface: `enp130s0f0` (Solarflare X2522)
- Static IP (likely)
- MTU: 1500 or 9000 (jumbo frames)

**Scripts Assuming This Interface:**
- `Tuning/Network/configure-nic-irq-affinity.sh` - Hardcoded `NIC=enp130s0f0`
- `Tuning/Network/ultra-low-latency-nic.sh` - Hardcoded `INTERFACE=enp130s0f0`
- `Monitoring/Scripts/PerformanceGate/ultra_low_latency_checker.py` - Interface checks

**What Breaks:**
- New system has different interface name (e.g., `eth0`, `eno1`)
- Scripts fail to find interface
- No network optimizations applied
- NIC runs with default (slow) settings

**Migration Strategy:**
1. **Make interface configurable via environment variable:**
   ```bash
   TRADING_NIC=${TRADING_NIC:-enp130s0f0}
   ```
2. Auto-detect Solarflare NIC:
   ```bash
   TRADING_NIC=$(ls /sys/class/net/ | grep -E '^(enp|eth|eno)' | xargs -I {} bash -c 'ethtool -i {} 2>/dev/null | grep -q sfc && echo {}' | head -1)
   ```
3. Update all scripts to use variable instead of hardcoded name
4. Document network setup requirements

---

### 5.4 System Libraries & Dependencies - 🟢 MEDIUM

**Python Packages:** (Not in repository)
- `asyncpg` - QuestDB PostgreSQL driver
- `aiohttp` - Async HTTP for ILP writes
- `redis` - Redis client
- `prometheus_client` - Metrics
- `orjson` - Fast JSON parsing
- `websockets` - Binance WebSocket
- Many others...

**System Packages:**
- `redis-server` - Used for benchmarking/testing
- `postgresql-client` - QuestDB wire protocol
- `curl` - API calls in scripts
- `jq` - JSON processing
- `ethtool` - Network tuning
- `numactl` - NUMA awareness
- `lscpu` - CPU topology detection

**Migration Strategy:**
1. Create `requirements.txt` for Python packages ✅ (likely exists)
2. Create system package list:
   ```bash
   dpkg --get-selections | grep -v deinstall > system_packages.txt
   ```
3. Create installation script:
   ```bash
   ./scripts/install_dependencies.sh
   ```
4. Use virtual environment for Python isolation

---

## 6. MIGRATION PRIORITY MATRIX

### Phase 1: System Foundation (CRITICAL - Day 1)
**Priority:** 🔴 Must complete before any services start

1. ✅ **Module Blacklists** (NEW - **MUST BE FIRST**)
   - `/etc/modprobe.d/blacklist-nouveau.conf` - **BEFORE** NVIDIA drivers
   - `/etc/modprobe.d/blacklist-framebuffer.conf`
   - Update initramfs
   - **Reboot required**
   - **Critical:** GPU won't work without this

2. ✅ **Kernel Parameters** (`/etc/sysctl.d/`, `/etc/default/grub`)
   - Network tuning (99-solarflare-trading.conf)
   - Huge pages (99-trading-hugepages.conf)
   - CPU isolation (GRUB cmdline)
   - System hardening (10-*.conf) - Optional but recommended
   - **Requires reboot**

3. ✅ **System Packages**
   - Redis, Java, Python, ethtool, etc.
   - Install before configuring

4. ✅ **Network Configuration**
   - Identify/configure trading NIC
   - Apply IRQ affinity (configure-nic-irq-affinity.sh to /usr/local/bin)
   - Set static IP

### Phase 2: Core Services (CRITICAL - Day 1)
**Priority:** 🔴 Required for data pipeline

5. ✅ **Systemd Services**
   - Copy all `.service` files (9 files)
   - Update ExecStart paths to new workspace location
   - Create CPU affinity overrides (6 override directories)
   - **Important:** Review questdb.service.d/ for conflicts
   - Enable services

6. ✅ **Redis Configuration**
   - Create `/opt/redis-hft/` directory structure
   - Copy `/opt/redis-hft/config/redis-hft.conf`
   - Copy authentication files (redis-auth.txt, redis-pass.txt)
   - Copy monitoring script (redis-hft-monitor.sh)
   - Create `/var/lib/redis-hft/`
   - Start redis-hft.service

7. ✅ **QuestDB Setup**
   - Verify Java installation
   - Configure data directories (hot/warm/cold)
   - Resolve questdb.service.d/ override conflicts
   - Start questdb.service

8. ✅ **System Binaries** (NEW)
   - Copy standalone scripts to /usr/local/bin/ (6 scripts)
   - Create workspace symlinks in /usr/local/bin/ (4 symlinks: datafeed, ultra-low-latency-nic.sh, vm-manager.sh, redis-hft-cli)
   - Verify symlink targets exist in workspace
   - **DO NOT** copy backup files

### Phase 3: Performance Tuning (HIGH - Day 2)
**Priority:** 🟡 Performance degradation without these

9. ✅ **Solarflare/Onload**
   - Install Onload (installs 7+ binaries to /usr/local/bin automatically)
   - Configure kernel modules (/etc/modprobe.d/)
   - Set up DKMS
   - Verify onload binaries present

10. ✅ **GPU Configuration**
    - **Prerequisite:** blacklist-nouveau.conf already applied (Phase 1)
    - Install NVIDIA drivers
    - Deploy clock locking service (dual-gpu-trading-config.service)
    - Validate locked clocks
    - Test nvidia-smi

11. ✅ **Performance Gate**
    - Migrate `/opt/redis-hft/baselines/` (fingerprint-baseline.json)
    - Copy performance gate scripts if not in workspace
    - Run bootstrap mode: `perf-gate.sh --bootstrap`
    - Validate new baseline captured

### Phase 4: Monitoring (MEDIUM - Day 3)
**Priority:** 🟢 Operational visibility

12. ✅ **Grafana Stack**
    - Install Grafana
    - Deploy provisioning configs
    - Import dashboards
    - Configure data sources

13. ✅ **Prometheus**
    - Configure scrape targets
    - Set up exporters

### Phase 5: Validation (Day 4)
**Priority:** Ensure everything works

14. ✅ **Run Verification Scripts**
    - `verify_cpu_affinity.sh`
    - `post_reboot_diagnostic.sh` (now in /usr/local/bin)
    - `perf-gate.sh --bootstrap`
    - `datafeed test` (symlink from /usr/local/bin)

15. ✅ **Performance Baseline**
    - Capture new baseline
    - Compare with old system
    - Validate latency targets met

---

## 7. GNARLY DEPENDENCIES REQUIRING EXTRA CARE

### 🚨 TOP 5 MIGRATION RISKS

#### 1. **CPU Isolation (Kernel Cmdline)** - Risk: 10/10
**Why Gnarly:**
- Requires GRUB modification
- Needs reboot
- Must match CPU topology of new hardware
- If wrong, entire HFT architecture fails
- No runtime fix - must reboot again

**Mitigation:**
- Detect CPU topology before setting
- Validate in GRUB preview
- Test boot in single-user mode first
- Have rollback GRUB entry

---

#### 2. **Systemd Service Paths** - Risk: 8/10
**Why Gnarly:**
- 8 service files with hardcoded ExecStart paths
- CPU affinity in 6 override directories
- Must update all to new workspace location
- Forget one = service fails silently
- Dependencies between services (market-data.target)

**Mitigation:**
- Use templating for service file generation
- Single source of truth for workspace path
- Deployment script validates all paths exist
- Test with `systemd-analyze verify`

---

#### 3. **Onload Kernel Module** - Risk: 9/10
**Why Gnarly:**
- Proprietary Solarflare software
- Kernel version specific
- DKMS must rebuild on kernel updates
- Different installation between Ubuntu/Debian/RHEL
- Network bypass fails silently if wrong version
- No official Docker/container support

**Mitigation:**
- Bundle exact Onload version
- Document kernel version compatibility
- Test on target OS/kernel before migration
- Have fallback to standard sockets

---

#### 4. **Cold Storage Mount (`/mnt/hdd/questdb/cold`)** - Risk: 7/10
**Why Gnarly:**
- External storage dependency
- Must be mounted before QuestDB starts
- Wrong permissions = lifecycle manager fails
- Different mount point on new system
- Archive data inaccessible if wrong

**Mitigation:**
- Make cold storage optional
- Detect mount before using
- Graceful fallback to single-tier
- Document fstab entry requirements

---

#### 5. **Network Interface Name** - Risk: 6/10
**Why Gnarly:**
- Hardcoded as `enp130s0f0` in multiple scripts
- New hardware has different name
- IRQ affinity script fails silently
- No network optimizations applied
- Affects 5+ scripts

**Mitigation:**
- Auto-detect Solarflare NIC
- Make configurable via environment variable
- Update all 5 scripts in parallel
- Validation script checks interface exists

---

### Additional Gnarly Items

#### 6. **RAPL Permissions** (udev rule with hardcoded workspace path)
- Udev rule contains absolute path to script
- Must update or script never runs
- Silent failure - monitoring just missing data

#### 7. **Python sys.path Hardcoding**
- 6 scripts with hardcoded `/home/youssefbahloul/...`
- Import errors on new system
- Tests fail to run

#### 8. **Performance Baselines in `/opt/redis-hft/`**
- Root-owned directory
- Lose baseline comparisons if not migrated
- Performance gate fails without baseline

---

## 8. RECOMMENDED MIGRATION WORKFLOW

### Step 1: Pre-Migration Audit (1 hour)
```bash
# Run on OLD system
cd /home/youssefbahloul/ai-trading-station

# 1. Backup all systemd service files
./Monitoring/Scripts/backup_systemd_services.sh

# 2. Export kernel parameters
cat /proc/cmdline > Documentation/kernel_cmdline_old.txt
sudo sysctl -a > Documentation/sysctl_old.txt

# 3. Export network configuration
ip addr show > Documentation/network_old.txt
ethtool -g enp130s0f0 > Documentation/nic_settings_old.txt

# 4. Export Grafana dashboards (already done)
# 5. Package /opt/redis-hft/
sudo tar -czf Archive/opt_redis_hft_backup.tar.gz /opt/redis-hft/

# 6. Document installed packages
dpkg --get-selections > Documentation/packages_old.txt
pip freeze > requirements_frozen.txt

# 7. Export cold storage mount
grep questdb /etc/fstab > Documentation/fstab_cold_storage.txt
```

### Step 2: Create Deployment Bundle (2 hours)
```bash
# Create self-contained deployment package
./scripts/create_deployment_bundle.sh

# Should generate:
# - ai-trading-station-deployment-v1.0.tar.gz
#   ├── workspace/ (full git repo)
#   ├── configs/ (all /etc/ files)
#   ├── systemd/ (service files with template vars)
#   ├── installers/ (Onload, Java, etc.)
#   └── deploy.sh (master deployment script)
```

### Step 3: New System Preparation (3 hours)
```bash
# On NEW system

# 1. Update kernel cmdline (from exported file)
sudo nano /etc/default/grub
# Copy GRUB_CMDLINE_LINUX from kernel_cmdline_old.txt
sudo update-grub
sudo reboot

# 2. Install system packages
sudo apt-get update
sudo apt-get install -y $(awk '{print $1}' packages_old.txt)

# 3. Set up network
# Configure static IP for trading NIC
# Detect NIC: lshw -C network | grep -i solarflare

# 4. Apply sysctl configs
sudo cp configs/sysctl.d/* /etc/sysctl.d/
sudo sysctl --system

# 5. Set up storage
# Mount cold storage if available
sudo mkdir -p /mnt/hdd/questdb/cold
# Add to /etc/fstab if needed
```

### Step 4: Deploy Trading Station (2 hours)
```bash
# Extract deployment bundle
tar -xzf ai-trading-station-deployment-v1.0.tar.gz
cd ai-trading-station-deployment/

# Run master deployment script
sudo ./deploy.sh --workspace-path /opt/ai-trading-station \
                 --trading-nic enp1s0 \
                 --enable-cold-storage /mnt/hdd/questdb/cold

# This script will:
# 1. Copy workspace to /opt/ai-trading-station
# 2. Update all hardcoded paths
# 3. Install systemd services
# 4. Apply CPU affinity configs
# 5. Set up Onload/Solarflare
# 6. Configure GPU clocking
# 7. Set up Redis/QuestDB
# 8. Install Grafana
# 9. Run validation checks
```

### Step 5: Validation (1 hour)
```bash
cd /opt/ai-trading-station

# 1. Verify kernel parameters
./Monitoring/Scripts/post_reboot_diagnostic.sh

# 2. Check CPU affinity
./Monitoring/Scripts/verify_cpu_affinity.sh

# 3. Test connectivity
./QuestDB/scripts/datafeed_v2.sh test

# 4. Run performance gate
./Monitoring/Scripts/PerformanceGate/perf-gate.sh --bootstrap

# 5. Start services
./QuestDB/scripts/datafeed_v2.sh start

# 6. Verify data flow
./QuestDB/scripts/datafeed_v2.sh metrics

# 7. Check Grafana
curl http://localhost:3000/api/health
```

### Step 6: Performance Comparison (30 min)
```bash
# Compare new vs old baselines
./QuestDB/scripts/comprehensive_latency_baseline.py

# Check P99 latencies match old system
# Verify capture rate >99%
# Ensure no error spikes
```

---

## 9. QUICK REFERENCE: ALL EXTERNAL PATHS

### System Configuration
| Path | Type | Critical | Backup Location |
|------|------|----------|----------------|
| `/etc/systemd/system/*.service` (9 files) | Services | 🔴 | `Archive/systemd_services_backup_*/` |
| `/etc/systemd/system/*.service.d/` (6 dirs) | Overrides | 🟡 | Generated by `configure_cpu_affinity.sh` |
| `/etc/sysctl.d/99-solarflare-trading.conf` | Kernel params | 🔴 | `Tuning/Network/99-solarflare-trading.conf` |
| `/etc/sysctl.d/99-trading-hugepages.conf` | Huge pages | 🟡 | `Tuning/Onload/99-trading-hugepages.conf` |
| `/etc/sysctl.d/10-*.conf` (9 files) | System hardening | 🟢 | **NOT BACKED UP** - Need to backup |
| `/etc/default/grub` | Kernel cmdline | 🔴 | Document via `cat /proc/cmdline` |
| `/etc/udev/rules.d/90-rapl-permissions.rules` | Power monitoring | 🟢 | `Monitoring/Config/90-rapl-permissions.rules` |
| `/etc/udev/rules.d/99-trading-nic-irq.rules` | NIC IRQ affinity | 🔴 | `Tuning/Network/99-trading-nic-irq.rules` |
| `/etc/modprobe.d/sfc.conf` | Solarflare driver | 🟡 | `Tuning/Solarflare/sfc.conf.backup` |
| `/etc/depmod.d/sfc-depmod.conf` | Module deps | 🟡 | `Tuning/Solarflare/depmod-sfc.conf.backup` |
| `/etc/modprobe.d/onload.conf` | Onload module | 🟡 | `Tuning/Onload/modprobe-onload-hft.conf.backup` |
| `/etc/modprobe.d/blacklist-nouveau.conf` | **GPU blacklist** | 🔴 | **NOT BACKED UP** - Need to backup |
| `/etc/modprobe.d/blacklist-*.conf` (3 more) | Other blacklists | 🟢 | **NOT BACKED UP** - Optional |
| `/etc/dkms/framework.conf` | DKMS config | 🟡 | Need to backup |

### System Binaries (/usr/local/bin/)
| Path | Type | Critical | Status |
|------|------|----------|--------|
| `datafeed` → workspace | Symlink | 🟡 | Primary operator interface |
| `ultra-low-latency-nic.sh` → workspace | Symlink | 🟡 | Used by systemd service |
| `vm-manager.sh` → workspace | Symlink | 🟢 | Convenience |
| `redis-hft-cli` → /opt | Symlink | 🟢 | Convenience |
| `configure-nic-irq-affinity.sh` | Standalone | 🔴 | **Used by service, in workspace Tuning/Network/** |
| `post_boot_core_isolation_verification.sh` | Standalone | 🟡 | CPU validation |
| `toggle_trading_mode_enhanced.sh` | Standalone | 🟡 | Mode switcher |
| `setup_solarflare.sh` | Standalone | 🟢 | One-time setup |
| `prevent_snap_firefox.sh` | Standalone | 🟢 | System optimization |
| `onload_*` (7 binaries) | Onload | 🟡 | Installed by Onload installer |
| `*.backup*` (3 files) | Backups | 🟢 | **Do not migrate** |

### Application Configuration (/opt/redis-hft/)
| Path | Type | Critical | Backup Status |
|------|------|----------|---------------|
| `/opt/redis-hft/config/redis-hft.conf` | Redis config | 🔴 | `Tuning/Redis/redis-hft.conf` |
| `/opt/redis-hft/config/redis-auth.txt` | Auth credentials | 🔴 | **NOT BACKED UP** |
| `/opt/redis-hft/config/redis-pass.txt` | Password | 🔴 | **NOT BACKED UP** |
| `/opt/redis-hft/scripts/redis-hft-monitor.sh` | Monitoring | 🟡 | External script |
| `/opt/redis-hft/scripts/redis-hft-cli` | CLI wrapper | 🟡 | External script |
| `/opt/redis-hft/baselines/fingerprint-baseline.json` | Perf baseline | 🟡 | Should be in deployment bundle |
| `/opt/redis-hft/metrics/*.json` | Live metrics | 🟢 | Regenerated at runtime |
| `/opt/redis-hft/archive/*` | Old configs | � | Historical, optional |
| `/var/lib/redis-hft/` | Redis data dir | 🔴 | Ephemeral (no persistence) |

### Grafana
| Path | Type | Critical | Backup Location |
|------|------|----------|----------------|
| `/etc/grafana/grafana.ini` | Main config | 🟡 | Generated by install script |
| `/etc/grafana/provisioning/datasources/` | Data sources | 🟡 | Generated by install script |
| `/etc/grafana/provisioning/dashboards/` | Dashboard config | 🟡 | Generated by install script |
| `/var/lib/grafana/dashboards/` | Dashboard JSONs | 🟢 | `Monitoring/*.json` |

### Storage
| Path | Type | Critical | Backup Location |
|------|------|----------|----------------|
| `/mnt/hdd/questdb/cold` | Cold storage | 🟡 | External mount point |
| `/var/lib/grafana/` | Grafana data | 🟢 | Regenerated |

### Kernel/Hardware Interfaces (ephemeral)
| Path | Type | Access |
|------|------|--------|
| `/proc/cmdline` | Kernel cmdline | Read (set via GRUB) |
| `/proc/interrupts` | IRQ info | Read |
| `/proc/irq/*/smp_affinity` | IRQ affinity | Write (scripts) |
| `/sys/devices/system/cpu/*/cpufreq/` | CPU frequency | Write (scripts) |
| `/sys/class/net/*/` | Network interface | Write (scripts) |
| `/sys/class/powercap/intel-rapl/` | Power monitoring | Read (via udev rule) |

---

## 10. AUTOMATED MIGRATION CHECKLIST

Create this as `scripts/migration_checklist.sh`:

```bash
#!/bin/bash
# Pre-flight migration checklist

echo "AI Trading Station - Migration Pre-flight Checklist"
echo "==================================================="
echo ""

ISSUES=0

check() {
    local name="$1"
    local command="$2"
    
    printf "%-50s" "$name..."
    if eval "$command" >/dev/null 2>&1; then
        echo "✅ OK"
    else
        echo "❌ MISSING"
        ((ISSUES++))
    fi
}

echo "SYSTEM BACKUPS:"
check "Systemd services backed up" "[ -d Archive/systemd_services_backup_* ]"
check "Kernel cmdline documented" "[ -f Documentation/kernel_cmdline_old.txt ]"
check "Network config exported" "[ -f Documentation/network_old.txt ]"
check "/opt/redis-hft/ backed up" "[ -f Archive/opt_redis_hft_backup.tar.gz ]"

echo ""
echo "WORKSPACE INTEGRITY:"
check "Git repository clean" "git diff-index --quiet HEAD --"
check "All scripts executable" "find . -name '*.sh' -not -perm -u=x | wc -l | grep -q '^0$'"
check "No hardcoded paths (Python)" "! grep -r '/home/youssefbahloul' QuestDB/scripts/*.py"

echo ""
echo "CONFIGURATION FILES PRESENT:"
check "Redis config in workspace" "[ -f Tuning/Redis/redis-hft.conf ]"
check "Sysctl network tuning" "[ -f Tuning/Network/99-solarflare-trading.conf ]"
check "Systemd service templates" "[ -f Archive/systemd_services_backup_*/questdb.service ]"
check "Grafana dashboards" "[ -f Monitoring/01-system-overview.json ]"

echo ""
echo "EXTERNAL DEPENDENCIES DOCUMENTED:"
check "Java requirement noted" "grep -q 'Java 11' Documentation/EXTERNAL_DEPENDENCIES_AUDIT.md"
check "Onload version documented" "onload --version 2>/dev/null | grep -q '[0-9]'"
check "NIC model identified" "lshw -C network 2>/dev/null | grep -q 'Solarflare\|X2522'"

echo ""
echo "============================================"
if [ $ISSUES -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED - Ready for migration"
    exit 0
else
    echo "❌ FOUND $ISSUES ISSUES - Fix before migration"
    exit 1
fi
```

---

## 11. CONCLUSION

### Summary Statistics
- **Total External Dependencies:** 123 (Updated from initial 73)
- **Critical (System Breaks):** 35 items
- **High (Performance Loss):** 54 items
- **Medium (Monitoring/Security/Optional):** 34 items

### Key Takeaways

1. **Current State:** System is NOT portable
   - 123 dependencies outside workspace
   - 35 critical missing = complete failure
   - Hardcoded paths in 40+ locations
   - 20 files in /usr/local/bin/ (mix of symlinks and standalone)
   - 17 files in /opt/redis-hft/ infrastructure
   - 13 kernel/module configs (9 hardening + 4 blacklists)

2. **Main Blockers:**
   - **GPU blacklist (blacklist-nouveau.conf)** - MUST install before NVIDIA drivers
   - Kernel parameters (requires reboot)
   - Systemd services (9 files + 6 override directories, plus questdb conflicts)
   - System binaries (/usr/local/bin/ - 20 files)
   - External storage (`/mnt/hdd/`)
   - Performance baselines (`/opt/redis-hft/`)
   - Network interface hardcoding
   - Onload binaries (7 files from installer)

3. **Migration Complexity:**
   - Simple: 45 items (copy files, includes backups and symlinks)
   - Moderate: 48 items (configuration updates, symlink recreation)
   - Complex: 30 items (system integration, kernel parameters, Onload installation)

### Critical Findings from Update

**Newly Discovered:**
1. 🔴 **blacklist-nouveau.conf** - CRITICAL for GPU, was completely missing from audit
2.  **datafeed symlink** - Primary operator interface not documented
3. 🟡 **17 /opt/redis-hft/ files** - Only directory mentioned, not file-level details
4. 🟡 **questdb.service.d/ conflicts** - Disabled overrides create migration confusion
5. 🟢 **System hardening (9 files)** - Security layer not backed up
6. 🟢 **7 Onload binaries** - Part of Onload install, auto-deployed

**Not Backed Up (Action Required):**
- `/etc/sysctl.d/10-*.conf` (9 system hardening files)
- `/etc/modprobe.d/blacklist-*.conf` (4 blacklist files, including critical nouveau)
- `/opt/redis-hft/config/redis-auth.txt` (authentication credentials)
- `/opt/redis-hft/config/redis-pass.txt` (password file)

### Recommended Next Steps

1. **URGENT (Today):**
   - Backup blacklist-nouveau.conf: `sudo cp /etc/modprobe.d/blacklist-*.conf ~/ai-trading-station/Tuning/Module-Blacklists/`
   - Backup system hardening: `sudo cp /etc/sysctl.d/10-*.conf ~/ai-trading-station/Tuning/System-Hardening/`
   - Backup Redis credentials: `sudo cp /opt/redis-hft/config/redis-{auth,pass}.txt ~/ai-trading-station/Archive/`

2. **Short Term (This Week):**
   - Run existing backup scripts for all external configs
   - Create deployment bundle structure
   - Fix Python hardcoded paths (6 scripts)
   - Document current system state fully
   - Update migration checklist with new 123 dependencies

3. **Medium Term (Next Sprint):**
   - Build deployment automation scripts
   - Create systemd service templates with path variables
   - Make network interface configurable
   - Package Onload installer
   - Create /usr/local/bin/ symlink recreation script

3. **Before Migration:**
   - Test deployment on VM/staging system
   - Validate all scripts find their dependencies
   - Run migration checklist
   - Document rollback procedure

### Risk Mitigation

**Highest Risk Items to Test First:**
1. CPU isolation (kernel cmdline)
2. Systemd service paths
3. Onload installation
4. Network interface detection
5. Cold storage mount

**Success Criteria:**
- All services start automatically
- P99 latency matches old system
- Capture rate >99%
- No hardcoded path errors
- Grafana displays data

---

## APPENDIX A: Quick Command Reference

### Check Current System State
```bash
# Kernel cmdline
cat /proc/cmdline

# Systemd services
systemctl list-units | grep -E 'redis-hft|questdb|binance|batch-writer'

# CPU isolation
grep isolcpus /proc/cmdline

# Network interface
ip link show | grep enp

# Storage mounts
df -h | grep questdb

# External configs
ls -la /etc/systemd/system/*.service /etc/sysctl.d/*.conf /etc/udev/rules.d/*trading*
```

### Backup Commands
```bash
# Systemd services
sudo cp -r /etc/systemd/system/{redis-hft,questdb,binance-*,batch-writer,market-data.target}* Archive/

# Sysctl configs
sudo cp /etc/sysctl.d/99-*.conf Archive/

# Redis HFT
sudo tar -czf Archive/opt_redis_hft.tar.gz /opt/redis-hft/

# Grafana
sudo tar -czf Archive/etc_grafana.tar.gz /etc/grafana/
sudo tar -czf Archive/var_lib_grafana.tar.gz /var/lib/grafana/
```

### Validation Commands
```bash
# After migration
cd /opt/ai-trading-station

./Monitoring/Scripts/post_reboot_diagnostic.sh
./Monitoring/Scripts/verify_cpu_affinity.sh
./QuestDB/scripts/datafeed_v2.sh test
./Monitoring/Scripts/PerformanceGate/perf-gate.sh --bootstrap
```

---

## 12. MONITORING FOLDER AUDIT - Complete Classification

### 📊 SUMMARY STATISTICS
- **Total Files:** 40 files
- **Production:** 12 files (30%) - **CORRECTED from 8**
- **Test/Development:** 0 files (0%) - **PerformanceGate reclassified as Production**
- **Setup/Installation:** 6 files (15%)
- **Legacy/Unused:** 22 files (55%)

---

## ✅ PRODUCTION FILES (Currently Used)

### **1. redis_health_monitor.py**
- **Purpose:** Monitor Redis connections, auto-restart on leaks
- **Status:** Running as systemd service (PID 2539)
- **Dependencies:** prometheus_client, redis-cli
- **Hard-coded paths:** None
- **Service:** redis-health-monitor.service
- **External Dependencies:** `/etc/systemd/system/redis-health-monitor.service`
- **Action:** KEEP

### **2. monitor_dashboard_complete.py**
- **Purpose:** Main monitoring dashboard (monitordash alias)
- **Status:** Running (PID 3613), actively used
- **Dependencies:** monitor_trading_system_optimized.py, curses
- **Hard-coded paths:** None (uses relative import)
- **Alias:** ~/.bashrc monitordash
- **Action:** KEEP

### **3. monitor_trading_system_optimized.py**
- **Purpose:** Core monitoring backend for dashboard
- **Status:** Imported by monitor_dashboard_complete.py
- **Dependencies:** monitor_questdb.py
- **Hard-coded paths:** 
  - Line 22: `sys.path.insert(0, '/home/youssefbahloul/ai-trading-station/QuestDB/scripts')`
  - Line ~100: `config_path="/home/youssefbahloul/ai-trading-station/Monitoring/Config/monitor_config.json"`
- **Action:** KEEP, **REQUIRES PATH REFACTORING**

### **4. monitor_questdb.py**
- **Purpose:** QuestDB metrics collection
- **Status:** Imported by monitor_trading_system_optimized.py
- **Dependencies:** psutil, subprocess
- **Hard-coded paths:** `self.hot_storage = "/home/youssefbahloul/ai-trading-station/QuestDB/data/hot"`
- **External Dependencies:** QuestDB data directories
- **Action:** KEEP, **REQUIRES PATH REFACTORING**

### **5. PerformanceGate/** (8 files) - **PRODUCTION VALIDATION INFRASTRUCTURE**
- **Status:** ✅ **PRODUCTION** (NOT test harness)
- **Purpose:** System validation, performance baseline verification
- **Files:**
  - `perf-gate.sh` - Main validation orchestrator
  - `runtime-fingerprint.sh` - System state fingerprinting
  - `extended_tail_sampler.py` - Tail latency validation
  - `gate_decision_ledger.py` - Decision tracking
  - `network_latency_harness.py` - Network testing
  - `parse_network_results.py` - Results parser
  - `tail_aware_gate.py` - Gate logic
  - `ultra_low_latency_checker.py` - ULL validation
- **External Dependencies:** 
  - `/opt/redis-hft/baselines/` (Section 1.5)
  - `/opt/redis-hft/metrics/` (Section 1.5)
- **Used In:**
  - Migration Phase 3: "`perf-gate.sh --bootstrap`"
  - Validation Phase 5: Deployment validation
  - Pre-migration workflow: Baseline validation
- **Hard-coded paths:** Multiple references to /home/youssefbahloul (needs refactoring)
- **Action:** KEEP - **PRODUCTION CRITICAL**

### **6. verify_cpu_affinity.sh**
- **Purpose:** Verification script for CPU isolation
- **Status:** ✅ **OPERATIONAL VALIDATION**
- **Used In:**
  - Section 1.2: CPU affinity verification
  - Validation Phase 5: System validation
  - Migration workflow: Post-deployment checks
- **External Dependencies:** Checks `/etc/systemd/system/*.service.d/cpu-affinity.conf`
- **Hard-coded paths:** Multiple references to /home/youssefbahloul
- **Action:** KEEP - **OPERATIONAL**

### **7. post_reboot_diagnostic.sh**
- **Purpose:** Post-reboot diagnostics
- **Status:** ✅ **OPERATIONAL DIAGNOSTIC**
- **Used In:**
  - Validation Phase 5: Explicitly listed
  - Section 5.1: Kernel cmdline validation
  - Post-reboot verification workflow
- **External Dependencies:** Reads `/proc/cmdline`, checks service status
- **Hard-coded paths:** Report file in Monitoring/Logs/
- **Action:** KEEP - **OPERATIONAL**

### **8. backup_systemd_services.sh**
- **Purpose:** Backup systemd services
- **Status:** ✅ **DISASTER RECOVERY**
- **Used In:**
  - Section 1.1: Service backup utility (lines 19-25)
  - Pre-migration workflow: "Run backup scripts for all external configs"
  - Disaster recovery procedures
- **External Dependencies:** Backs up `/etc/systemd/system/*.service`
- **Hard-coded paths:** `/home/youssefbahloul/ai-trading-station/Archive/systemd_services_backup_$(date)`
- **Action:** KEEP - **OPERATIONAL**

### **9. configure_cpu_affinity.sh**
- **Purpose:** Initial CPU affinity setup
- **Status:** ✅ **DEPLOYMENT UTILITY**
- **Used In:**
  - Section 1.2: Creates CPU affinity overrides
  - System setup/reconfiguration
- **External Dependencies:** Creates `/etc/systemd/system/*.service.d/cpu-affinity.conf`
- **Hard-coded paths:** Multiple references to /home/youssefbahloul
- **Action:** KEEP - **UTILITY**

### **10. rollback_cpu_affinity.sh**
- **Purpose:** Rollback CPU affinity changes
- **Status:** ✅ **DISASTER RECOVERY**
- **Action:** KEEP - **UTILITY**

### **11. perf-thresholds.env**
- **Purpose:** Performance gate thresholds (MAX_CLIENTS=25)
- **Status:** Actively used by performance gate
- **Dependencies:** None
- **Hard-coded paths:** None
- **Action:** KEEP

### **12. monitor_config.json**
- **Purpose:** Configuration for monitor_trading_system_optimized.py
- **Status:** Active config file
- **Dependencies:** None
- **Action:** KEEP

---

## 📄 DOCUMENTATION (Keep)

### **13. GRAFANA_SETUP_GUIDE.md**
- **Purpose:** Grafana installation documentation
- **Status:** Reference documentation
- **Action:** KEEP

### **14. grafana_redis_health_dashboard.json**
- **Purpose:** Grafana dashboard for Redis health
- **Status:** Imported/active dashboard
- **Dependencies:** Prometheus metrics from redis_health_monitor.py
- **Action:** KEEP

### **15. Config/MIGRATION_REPORT.md**
- **Purpose:** Historical migration documentation
- **Status:** Reference documentation
- **Action:** KEEP

### **16. Logs/BUG_FIX_REPORT.md**
- **Purpose:** Historical bug report
- **Status:** Historical documentation
- **Action:** KEEP (in Logs/)

### **17. Logs/LOGGING_UPGRADE_SUMMARY.md**
- **Purpose:** Historical upgrade doc
- **Status:** Historical documentation
- **Action:** KEEP (in Logs/)

---

## 🔧 SETUP/INSTALLATION (Archive After Initial Setup Complete)

### **18. install_grafana.sh**
- **Purpose:** Grafana installation
- **Status:** One-time install
- **Action:** ARCHIVE AFTER SETUP CONFIRMED

### **19. secure_grafana.sh**
- **Purpose:** Grafana security hardening
- **Status:** One-time setup
- **Action:** ARCHIVE AFTER SETUP CONFIRMED

### **20. setup_dashboards.sh**
- **Purpose:** One-time dashboard setup
- **Status:** Setup script
- **Dependencies:** import_dashboard.sh
- **Action:** ARCHIVE AFTER SETUP CONFIRMED

### **21. import_dashboard.sh**
- **Purpose:** Import Grafana dashboards
- **Status:** Utility script, one-time use
- **Hard-coded paths:** grafana-tiered-cache-dashboard.json
- **Action:** ARCHIVE AFTER SETUP CONFIRMED

### **22. setup_alerts.sh**
- **Purpose:** Configure Grafana alerts
- **Status:** One-time setup
- **Action:** ARCHIVE AFTER SETUP CONFIRMED

### **23. phase3_setup.sh**
- **Purpose:** Phase 3 implementation setup
- **Status:** Historical setup script
- **Hard-coded paths:** `SCRIPT_DIR="/home/youssefbahloul/ai-trading-station/Monitoring/Scripts"`
- **Action:** ARCHIVE AFTER SETUP CONFIRMED

---

## 🗄️ LEGACY/UNUSED FILES (Can Archive Now)

### **24. Redis/** (3 files)
- **health-report.sh** - Redis health report generator (superseded)
- **live-dashboard.sh** - Live Redis dashboard (superseded by monitor_dashboard_complete.py)
- **redis-hft-monitor_to_json.sh** - JSON converter (not used)
- **Status:** All superseded by redis_health_monitor.py
- **Action:** ARCHIVE NOW

### **25. network_latency_monitor.py**
- **Purpose:** Network latency continuous monitoring
- **Status:** Not running, not used
- **Hard-coded paths:** `MONITORING_DIR = Path("/home/youssefbahloul/ai-trading-station/Monitoring")`
- **Action:** ARCHIVE NOW

### **26. Grafana Dashboard Files** (3 files)
- **01-system-overview.json** - Old dashboard format
- **02-performance-metrics.json** - Old dashboard format
- **tiered-cache-clean.json** - Old dashboard version
- **Status:** Superseded by grafana-tiered-cache-dashboard.json
- **Action:** ARCHIVE NOW

### **27. .monitoring_mode**
- **Purpose:** Monitoring mode flag file
- **Status:** Empty flag file, not actively used
- **Action:** ARCHIVE NOW

### **28. set_monitoring_mode.sh**
- **Purpose:** Mode setter
- **Status:** Not used
- **Action:** ARCHIVE NOW

### **29. CPU_AFFINITY_README.md**
- **Purpose:** Documentation
- **Status:** Historical documentation (CPU affinity disabled)
- **Action:** ARCHIVE NOW

---

## 🚨 HARD-CODED PATHS REQUIRING MIGRATION

### **Critical (Production Files):**
1. **monitor_trading_system_optimized.py:**
   - Line 22: `sys.path.insert(0, '/home/youssefbahloul/ai-trading-station/QuestDB/scripts')`
   - Line ~100: `config_path="/home/youssefbahloul/ai-trading-station/Monitoring/Config/monitor_config.json"`
   - **Fix:** Use `Path(__file__).parent.parent` or environment variable `WORKSPACE_ROOT`

2. **monitor_questdb.py:**
   - `self.hot_storage = "/home/youssefbahloul/ai-trading-station/QuestDB/data/hot"`
   - **Fix:** Use environment variable `QUESTDB_DATA_DIR` or workspace-relative path

3. **PerformanceGate/** scripts:
   - Multiple hardcoded paths in Python/shell scripts
   - **Fix:** Use environment variable `WORKSPACE_ROOT` or workspace-relative paths

4. **verify_cpu_affinity.sh:**
   - Multiple workspace path references
   - **Fix:** Use environment variable `WORKSPACE_ROOT`

5. **post_reboot_diagnostic.sh:**
   - Report file path
   - **Fix:** Use workspace-relative path

6. **backup_systemd_services.sh:**
   - Archive path hardcoded
   - **Fix:** Use workspace-relative path

7. **configure_cpu_affinity.sh:**
   - Workspace path references
   - **Fix:** Use environment variable `WORKSPACE_ROOT`

---

## 📋 RELOCATION PLAN

### **Structure After Cleanup:**
```
Monitoring/
├── Config/
│   ├── monitor_config.json (KEEP)
│   ├── perf-thresholds.env (KEEP)
│   └── MIGRATION_REPORT.md (KEEP - docs)
├── Scripts/
│   ├── redis_health_monitor.py (KEEP - Production)
│   ├── monitor_dashboard_complete.py (KEEP - Production)
│   ├── monitor_trading_system_optimized.py (KEEP - Production, needs refactor)
│   ├── monitor_questdb.py (KEEP - Production, needs refactor)
│   ├── PerformanceGate/ (KEEP - 8 files, Production validation)
│   ├── verify_cpu_affinity.sh (KEEP - Operational)
│   ├── post_reboot_diagnostic.sh (KEEP - Operational)
│   ├── backup_systemd_services.sh (KEEP - Operational)
│   ├── configure_cpu_affinity.sh (KEEP - Utility)
│   └── rollback_cpu_affinity.sh (KEEP - Utility)
├── Dashboards/
│   ├── grafana_redis_health_dashboard.json (KEEP)
│   └── grafana-tiered-cache-dashboard.json (KEEP)
├── Logs/ (KEEP - active logs + 2 historical docs)
├── GRAFANA_SETUP_GUIDE.md (KEEP - docs)
└── Archive/
    ├── Setup/ (6 setup scripts - archive after confirmed)
    ├── Redis/ (3 scripts - archive now)
    ├── Dashboards/ (3 old JSON files - archive now)
    ├── network_latency_monitor.py (archive now)
    ├── .monitoring_mode (archive now)
    ├── set_monitoring_mode.sh (archive now)
    └── CPU_AFFINITY_README.md (archive now)
```

---

## ⚠️ EXTERNAL DEPENDENCIES SUMMARY

### **The Monitoring folder references these EXTERNAL dependencies:**

| External Path | Type | Risk | Section | Used By |
|---------------|------|------|---------|---------|
| `/etc/systemd/system/redis-health-monitor.service` | Service | 🔴 CRITICAL | 1.1 | redis_health_monitor.py |
| `/etc/systemd/system/*.service.d/cpu-affinity.conf` | Overrides | 🟡 HIGH | 1.2 | configure_cpu_affinity.sh |
| `/opt/redis-hft/baselines/` | Performance gate | 🟡 HIGH | 1.5 | PerformanceGate/ |
| `/opt/redis-hft/metrics/` | Performance gate | 🟡 HIGH | 1.5 | PerformanceGate/ |
| `/opt/redis-hft/config/redis-hft.conf` | Redis config | 🔴 CRITICAL | 1.5 | PerformanceGate/ |
| `/etc/grafana/` | Grafana config | 🟡 HIGH | 1.3 | install_grafana.sh |
| `/var/lib/grafana/` | Grafana data | 🟢 MEDIUM | 1.3 | Grafana |
| `/proc/cmdline` | Kernel params | 🔴 CRITICAL | 5.1 | post_reboot_diagnostic.sh |
| QuestDB data directories | Storage | 🔴 CRITICAL | 2.1 | monitor_questdb.py |

**Total Monitoring-Related External Dependencies:** ~15 items (out of 123 total system-wide)

---

## 🎯 ACTION PLAN

### **Immediate Actions:**
1. ✅ **KEEP PerformanceGate/** - Production validation infrastructure (8 files)
2. ✅ **KEEP verify_cpu_affinity.sh** - Operational validation
3. ✅ **KEEP post_reboot_diagnostic.sh** - Operational diagnostic
4. ✅ **KEEP backup_systemd_services.sh** - Disaster recovery
5. ✅ **KEEP configure_cpu_affinity.sh & rollback_cpu_affinity.sh** - Utilities
6. 🔧 **REFACTOR 7 files** - Remove hardcoded paths (see list above)
7. 🗄️ **ARCHIVE NOW** - Redis/ scripts (3), old dashboards (3), network_latency_monitor.py, .monitoring_mode, set_monitoring_mode.sh, CPU_AFFINITY_README.md
8. 🟡 **ARCHIVE AFTER SETUP CONFIRMED** - Setup scripts (6 files)

### **Post-Refactor Testing:**
1. Test `monitordash` after path refactoring
2. Verify redis-health-monitor.service still works
3. Run `perf-gate.sh --bootstrap` to validate performance gate
4. Run `verify_cpu_affinity.sh` to validate CPU isolation
5. Run `post_reboot_diagnostic.sh` to validate diagnostics

---

## 🌐 WORKSPACE PORTABILITY STATUS

**Assessment:** ❌ **NOT PORTABLE** - System has 123 external dependencies

**Monitoring Folder Contribution:**
- ~15 external dependencies (12% of total)
- 7 production files with hardcoded workspace paths requiring refactoring
- External dependencies span `/etc/`, `/opt/`, `/proc/`, and QuestDB storage

**Key Insight:** Organizing workspace files (archiving 22 legacy files) improves maintainability but does NOT address the 123 external dependencies that block portability. Both efforts are needed but serve different purposes.

---

## 13. COMPLETE SYSTEM FILE INVENTORY

**Purpose:** Comprehensive file-by-file listing of all workspace and external files with classifications and symlink mappings.

**Scope:** 164 files total (82 workspace + 82 external)

**Classification Flags:**
- 🟢 **Production-Runtime**: Active execution in production
- 🔵 **Production-Config**: Configuration used by running services
- 🟡 **Test**: Testing/validation scripts
- 🟠 **Legacy**: Superseded/archived but kept for reference
- ⚪ **Utility**: Support scripts (setup, backup, diagnostics)

**Symlink Notation:**
- `→` indicates symlink pointing to target
- Source listed first, target after arrow
- All symlinks explicitly documented with both endpoints

---

### 13.1 MONITORING FOLDER - Complete File List

#### Active Production Files (16 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟢 | Runtime | `Scripts/redis_health_monitor.py` | No | SystemD service - Redis health monitoring (PID 2539) |
| 🟢 | Runtime | `Scripts/monitor_dashboard_complete.py` | No | Live monitoring dashboard (monitordash alias) |
| 🟢 | Runtime | `Scripts/monitor_questdb.py` | No | QuestDB metrics collection |
| 🟢 | Runtime | `Scripts/monitor_trading_system_optimized.py` | No | Comprehensive system monitoring |
| 🔵 | Config | `Config/perf-thresholds.env` | No | Performance gate thresholds |
| 🔵 | Config | `Config/monitor_config.json` | No | Monitor runtime configuration |
| 🔵 | Config | `grafana_redis_health_dashboard.json` | No | Active Grafana dashboard |
| 🔵 | Config | `grafana-tiered-cache-dashboard.json` | No | Tiered cache Grafana dashboard |
| 🟢 | Runtime | `Scripts/PerformanceGate/perf-gate.sh` | No | Production validation orchestrator |
| 🟢 | Runtime | `Scripts/PerformanceGate/runtime-fingerprint.sh` | No | System state fingerprinting |
| 🟢 | Runtime | `Scripts/PerformanceGate/network_latency_harness.py` | No | Network latency testing |
| 🟢 | Runtime | `Scripts/PerformanceGate/parse_network_results.py` | No | Parse latency test results |
| 🟢 | Runtime | `Scripts/PerformanceGate/ultra_low_latency_checker.py` | No | ULL validation |
| 🟢 | Runtime | `Scripts/PerformanceGate/gate_decision_ledger.py` | No | Gate decision logging |
| 🟢 | Runtime | `Scripts/PerformanceGate/tail_aware_gate.py` | No | Tail latency analysis |
| 🟢 | Runtime | `Scripts/PerformanceGate/extended_tail_sampler.py` | No | Extended tail sampling |

**Note:** PerformanceGate classified as Production (used in migration validation, deployment checks, baseline verification).

#### Utility Scripts (11 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| ⚪ | Utility | `import_dashboard.sh` | No | Import Grafana dashboards |
| ⚪ | Utility | `Scripts/install_grafana.sh` | No | Grafana installation |
| ⚪ | Utility | `Scripts/setup_dashboards.sh` | No | Dashboard setup automation |
| ⚪ | Utility | `Scripts/setup_alerts.sh` | No | Alert configuration |
| ⚪ | Utility | `Scripts/secure_grafana.sh` | No | Grafana security hardening |
| ⚪ | Utility | `Scripts/backup_systemd_services.sh` | No | Backup systemd configs |
| ⚪ | Utility | `Scripts/post_reboot_diagnostic.sh` | No | Post-boot validation |
| ⚪ | Utility | `Scripts/set_monitoring_mode.sh` | No | Switch monitoring modes |
| ⚪ | Utility | `Scripts/configure_cpu_affinity.sh` | No | Setup CPU affinity |
| ⚪ | Utility | `Scripts/rollback_cpu_affinity.sh` | No | Rollback CPU affinity |
| ⚪ | Utility | `Scripts/verify_cpu_affinity.sh` | No | CPU affinity verification |


#### Legacy/Archived (8 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------||
| 🟠 | Legacy | `Scripts/phase3_setup.sh` | No | Old implementation script |
| 🟠 | Legacy | `Scripts/network_latency_monitor.py` | No | Superseded by PerformanceGate |
| 🔵 | Config | `01-system-overview.json` | No | Active Grafana dashboard |
| 🔵 | Config | `02-performance-metrics.json` | No | Active Grafana dashboard |
| 🟠 | Legacy | `tiered-cache-clean.json` | No | Old dashboard version |
| � | Config | `Scripts/Redis/redis-hft-monitor_to_json.sh` | No | Production JSON wrapper |
| 🟠 | Legacy | `Scripts/Redis/health-report.sh` | No | Old health report |
| 🟠 | Legacy | `Scripts/Redis/live-dashboard.sh` | No | Old live dashboard |

**Monitoring Folder Total: 35 files**

**Monitoring Folder Total: 35 files** (16 production runtime + 11 utility + 5 active config + 3 legacy)

---

### 13.2 QUESTDB FOLDER - Complete File List

#### Active Production Files (7 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟢 | Runtime | `scripts/redis_to_questdb_v2.py` | No | Batch writer service |
| 🟢 | Runtime | `scripts/binance_bookticker_collector.py` | No | Book ticker collector |
| 🟢 | Runtime | `scripts/binance_trades_collector.py` | No | Trades collector |
| 🟢 | Runtime | `scripts/tiered_cache_service.py` | No | Tiered cache service |
| 🔵 | Config | `data/hot/conf/server.conf` | No | QuestDB server config |
| 🔵 | Config | `data/hot/conf/log.conf` | No | QuestDB logging config |
| 🔵 | Config | `questdb-9.1.0-rt-linux-x86-64/conf/server.conf` | No | Default QuestDB config |

#### QuestDB Binary & Support (3 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Binary | `questdb-9.1.0-rt-linux-x86-64/bin/questdb.sh` | No | QuestDB startup script |
| 🔵 | Binary | `questdb-9.1.0-rt-linux-x86-64/bin/env.sh` | No | Environment setup |
| 🔵 | Binary | `questdb-9.1.0-rt-linux-x86-64/bin/print-hello.sh` | No | Version info |

#### Utility Scripts (7 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| ⚪ | Utility | `scripts/datafeed_v2.sh` | ← `/usr/local/bin/datafeed` | Primary operator interface |
| ⚪ | Utility | `scripts/lifecycle-manager.sh` | No | QuestDB lifecycle management |
| ⚪ | Utility | `scripts/retention-cleanup.sh` | No | Data retention cleanup |
| ⚪ | Utility | `scripts/post_reboot_sanity_check.sh` | No | Post-boot validation |
| ⚪ | Utility | `scripts/post_reboot_validation.sh` | No | Post-boot validation |
| ⚪ | Utility | `scripts/final_production_validation.sh` | No | Production readiness check |
| ⚪ | Utility | `backups/backup_existing_data.sh` | No | Data backup script |

#### Test/Development Files (14 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟡 | Test | `scripts/feature_engineering.py` | No | Feature extraction pipeline |
| 🟡 | Test | `scripts/tiered_market_data.py` | No | Tiered data access layer |
| 🟡 | Test | `scripts/query_existing_cache.py` | No | Cache query testing |
| 🟡 | Test | `scripts/exchange_to_redis.py` | No | Exchange→Redis ingestion |
| 🟡 | Test | `scripts/comprehensive_latency_baseline.py` | No | Latency baseline measurements |
| 🟡 | Test | `scripts/benchmark_tiered_cache.py` | No | Cache performance benchmarks |
| 🟡 | Test | `scripts/test_batch_writer_performance.py` | No | Batch writer performance test |
| 🟡 | Test | `scripts/test_multistream_ingestion.py` | No | Multi-stream ingestion test |
| 🟡 | Test | `scripts/test_multistream_performance.sh` | No | Multi-stream performance test |
| 🟡 | Test | `scripts/test_cache_integration.py` | No | Cache integration testing |
| 🟡 | Test | `scripts/test_cache_activity.py` | No | Cache activity monitoring |
| 🟡 | Test | `scripts/test_copy_protocol.py` | No | COPY protocol testing |
| 🟡 | Test | `scripts/test_orderbook_generator.py` | No | Order book generation test |
| 🟡 | Test | `scripts/test_questdb_connection.py` | No | Connection testing |

#### Legacy/Deprecated (7 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟠 | Legacy | `scripts/redis_to_questdb.py` | No | Old batch writer (superseded) |
| 🟠 | Legacy | `scripts/redis_to_questdb_v2_day1_backup.py` | No | Backup of v2 batch writer |
| 🟠 | Legacy | `scripts/redis_health_monitor.py` | No | Moved to Monitoring/ |
| 🟠 | Legacy | `scripts/redis_health_monitor_with_alerts.py` | No | Old monitoring with alerts |
| 🟠 | Legacy | `scripts/create_table_pg.py` | No | PostgreSQL table creation |
| 🟠 | Legacy | `backups/schema_metadata_20251020_203855.json` | No | Old schema backup |
| 🟠 | Legacy | `data.corrupted_20251024/` | No | Corrupted data archive |

**QuestDB Folder Total: 38 files** (Note: ~30 additional internal Java legal symlinks not counted)

---

### 13.3 TRADING FOLDER - Complete File List

#### Active Production Files (1 file)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Runtime | `Wrapper/onload-trading` | No | Onload wrapper for trading apps |

**Trading Folder Total: 1 file**

---

### 13.4 TUNING FOLDER - Complete File List

#### Network Tuning - Production (3 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟢 | Runtime | `Network/ultra-low-latency-nic.sh` | ← `/usr/local/bin/ultra-low-latency-nic.sh` | NIC tuning at boot |
| 🟢 | Runtime | `Network/configure-nic-irq-affinity.sh` | No (service ref) | NIC IRQ affinity at boot |
| 🔵 | Config | `Network/99-solarflare-trading.conf` | No | Solarflare NIC sysctl config |

#### GPU Tuning - Production (5 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟢 | Runtime | `GPU/gpu_clock_lock.sh` | No | Lock GPU clocks at boot |
| 🔵 | Config | `GPU/blackwell_sm12_optimized_config.py` | No | Blackwell GPU optimization |
| 🔵 | Module | `GPU/__init__.py` | No | Python package marker |
| 🔵 | Module | `GPU/trading_inference.py` | No | GPU-accelerated inference |
| 🔵 | Module | `GPU/pytorch_metrics.py` | No | PyTorch performance metrics |

#### GPU Testing (4 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟡 | Test | `GPU/Test/verify_gpu_setup.py` | No | GPU setup verification |
| 🟡 | Test | `GPU/Test/torch_compile_test.py` | No | PyTorch compilation testing |
| 🟡 | Test | `GPU/Test/precision_tests.py` | No | Floating-point precision tests |
| 🟡 | Test | `GPU/Test/variance_test_10k.py` | No | 10K sample variance test |

#### Other Configs (2 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Config | `Onload/99-trading-hugepages.conf` | No | Hugepages sysctl config |
| 🔵 | Config | `Redis/redis-hft.conf` | → `/opt/redis-hft/config/redis-hft.conf` | Redis HFT config template |

**Critical Reverse Symlink:** `Tuning/Redis/redis-hft.conf → /opt/redis-hft/config/redis-hft.conf` (workspace is source, external is target - for version control)

**Tuning Folder Total: 14 files**

---

### 13.5 EXTERNAL DEPENDENCIES - Complete File List

#### /opt/redis-hft/ - Redis HFT Infrastructure (15 files)

**Active Production (10 files)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Config | `/opt/redis-hft/config/redis-hft.conf` | ← `workspace/Tuning/Redis/redis-hft.conf` | Active Redis config |
| 🔵 | Config | `/opt/redis-hft/config/redis.env` | No | Redis environment vars |
| 🔵 | Config | `/opt/redis-hft/config/redis-auth.txt` | No | Redis authentication |
| 🔵 | Config | `/opt/redis-hft/config/redis-pass.txt` | No | Redis password |
| 🟢 | Runtime | `/opt/redis-hft/scripts/redis-hft-monitor.sh` | No | Redis monitoring script |
| 🔵 | Runtime | `/opt/redis-hft/scripts/redis-hft-cli` | ← `/usr/local/bin/redis-hft-cli` | Redis CLI tool |
| 🔵 | Metrics | `/opt/redis-hft/metrics/redis-metrics.json` | No | Current Redis metrics |
| 🔵 | Metrics | `/opt/redis-hft/metrics/current-fingerprint.json` | No | Performance fingerprint |
| 🔵 | Metrics | `/opt/redis-hft/metrics/network-ull-metrics.json` | No | Network ULL metrics |
| 🔵 | Metrics | `/opt/redis-hft/baselines/fingerprint-baseline.json` | No | Performance baseline |

**Legacy/Backup (5 files)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟠 | Legacy | `/opt/redis-hft/config/redis-hft.conf.backup_20251008` | No | Config backup |
| 🟠 | Legacy | `/opt/redis-hft/archive/redis-hft.conf.backup` | No | Old config backup |
| 🟠 | Legacy | `/opt/redis-hft/archive/redis-hft.conf.backup.20250930_125659` | No | Timestamped backup |
| 🟠 | Legacy | `/opt/redis-hft/archive/redis-hft-memory-patch-fixed.conf` | No | Old memory patch config |
| 🟠 | Legacy | `/opt/redis-hft/baselines/fingerprint-baseline.json.backup_kernel_6.8.0-84` | No | Kernel upgrade baseline |

---

#### /usr/local/bin/ - System Binaries (21 files)

**A. Workspace Symlinks (4 files)**

| Status | Type | File | Symlink Target | Purpose |
|--------|------|------|----------------|---------|
| 🔵 | Symlink | `/usr/local/bin/datafeed` | `→ workspace/QuestDB/scripts/datafeed_v2.sh` | Primary operator interface |
| 🔵 | Symlink | `/usr/local/bin/ultra-low-latency-nic.sh` | `→ workspace/Tuning/Network/ultra-low-latency-nic.sh` | NIC tuning |
| 🔵 | Symlink | `/usr/local/bin/vm-manager.sh` | `→ workspace/vm-manager.sh` | VM management utility |
| 🔵 | Symlink | `/usr/local/bin/redis-hft-cli` | `→ /opt/redis-hft/scripts/redis-hft-cli` | Redis CLI wrapper |

**B. Standalone System Scripts (6 files)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟢 | Script | `/usr/local/bin/post_boot_core_isolation_verification.sh` | No | Validate CPU isolation |
| 🟢 | Script | `/usr/local/bin/prevent_snap_firefox.sh` | No | Disable Firefox snap |
| 🟢 | Script | `/usr/local/bin/setup_solarflare.sh` | No | Solarflare NIC setup |
| 🟢 | Script | `/usr/local/bin/toggle_trading_mode_enhanced.sh` | No | Trading mode switcher |
| 🔵 | Binary | `/usr/local/bin/httpx` | No | HTTP client tool |
| 🔵 | Binary | `/usr/local/bin/vm-manager` | No | VM management binary |

**C. Onload Binaries (7 files - installed by Onload installer)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Binary | `/usr/local/bin/onload_cp_client` | No | Onload control plane client |
| 🔵 | Binary | `/usr/local/bin/onload_cp_server` | No | Onload control plane server |
| 🔵 | Binary | `/usr/local/bin/onload_fuser` | No | Show processes using Onload |
| 🔵 | Binary | `/usr/local/bin/onload_mibdump` | No | Dump Onload MIB statistics |
| 🔵 | Binary | `/usr/local/bin/onload_tcpdump.bin` | No | Onload-aware packet capture |
| 🔵 | Binary | `/usr/local/bin/onload_tool` | No | Onload configuration utility |
| 🔵 | Binary | `/usr/local/bin/onload_*` (additional) | No | Additional Onload utilities |

**D. Backup Files (4 files - should be archived)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟠 | Legacy | `/usr/local/bin/configure-nic-irq-affinity.sh.bak-20250901-200453` | No | Old IRQ config backup |
| 🟠 | Legacy | `/usr/local/bin/toggle_trading_mode.sh.backup` | No | Old trading mode script |
| 🟠 | Legacy | `/usr/local/bin/ultra-low-latency-nic.sh.backup_v3_20251002` | No | ULL script backup v3 |
| 🟠 | Legacy | `/usr/local/bin/ultra-low-latency-nic-v2.0.backup` | No | ULL script backup v2 |

---

#### /etc/systemd/system/ - SystemD Services (17 files)

**Active Services (10 files)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟢 | Service | `/etc/systemd/system/redis-hft.service` | No | Redis HFT server (RUNNING) |
| 🟢 | Service | `/etc/systemd/system/questdb.service` | No | QuestDB database (RUNNING) |
| 🟢 | Service | `/etc/systemd/system/redis-health-monitor.service` | No | Redis health monitor (RUNNING) |
| 🟢 | Service | `/etc/systemd/system/ultra-low-latency-nic.service` | No | NIC tuning at boot |
| 🟢 | Service | `/etc/systemd/system/configure-nic-irq-affinity.service` | No | NIC IRQ affinity at boot |
| 🟢 | Service | `/etc/systemd/system/dual-gpu-trading-config.service` | No | GPU config at boot |
| 🔵 | Service | `/etc/systemd/system/batch-writer.service` | No | Batch writer (STOPPED) |
| 🔵 | Service | `/etc/systemd/system/binance-bookticker.service` | No | Book ticker collector (STOPPED) |
| 🔵 | Service | `/etc/systemd/system/binance-trades.service` | No | Trades collector (STOPPED) |
| 🔵 | Service | `/etc/systemd/system/tiered-cache.service` | No | Tiered cache service (DISABLED) |

**CPU Affinity Overrides (6 directories)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Override | `/etc/systemd/system/prometheus.service.d/cpu-affinity.conf` | No | CPU 2 assignment |
| 🔵 | Override | `/etc/systemd/system/redis-hft.service.d/cpu-affinity.conf` | No | CPU 4 assignment |
| 🔵 | Override | `/etc/systemd/system/binance-trades.service.d/cpu-affinity.conf` | No | CPU 3 assignment |
| 🔵 | Override | `/etc/systemd/system/binance-bookticker.service.d/cpu-affinity.conf` | No | CPU 3 assignment |
| 🔵 | Override | `/etc/systemd/system/questdb.service.d/cpu-affinity.conf` | No | CPU 5 assignment |
| 🔵 | Override | `/etc/systemd/system/batch-writer.service.d/cpu-affinity.conf` | No | CPU 6-7 assignment |

**Legacy (1 file)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟠 | Legacy | `/etc/systemd/system/nic-irq-affinity.service` | No | Old/duplicate service |

---

#### /etc/sysctl.d/ - Kernel Parameters (14 files)

**Trading-Specific (2 files)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Config | `/etc/sysctl.d/99-solarflare-trading.conf` | No | Network performance tuning |
| 🔵 | Config | `/etc/sysctl.d/99-trading-hugepages.conf` | No | Hugepages allocation |

**System Hardening (9 files - not backed up)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Config | `/etc/sysctl.d/10-kernel-hardening.conf` | No | Kernel security hardening |
| 🔵 | Config | `/etc/sysctl.d/10-network-security.conf` | No | Network stack security |
| 🔵 | Config | `/etc/sysctl.d/10-ptrace.conf` | No | Restrict ptrace |
| 🔵 | Config | `/etc/sysctl.d/10-bufferbloat.conf` | No | TCP congestion control |
| 🔵 | Config | `/etc/sysctl.d/10-ipv6-privacy.conf` | No | IPv6 privacy extensions |
| 🔵 | Config | `/etc/sysctl.d/10-map-count.conf` | No | vm.max_map_count |
| 🔵 | Config | `/etc/sysctl.d/10-zeropage.conf` | No | Transparent hugepages |
| 🔵 | Config | `/etc/sysctl.d/10-magic-sysrq.conf` | No | Magic SysRq key |
| 🔵 | Config | `/etc/sysctl.d/10-console-messages.conf` | No | Kernel console log level |

**Legacy/Backup (3 files)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🟠 | Legacy | `/etc/sysctl.d/99-trading-hugepages.conf.backup_20251018_104839` | No | Hugepages backup |
| 🟠 | Legacy | `/etc/sysctl.d/99-trading-hugepages.conf.backup_20251018_205833` | No | Hugepages backup |
| 🟠 | Legacy | `/etc/sysctl.d/99-trading-network-performance.conf.bak` | No | Network tuning backup |

---

#### /etc/modprobe.d/ - Kernel Module Configs (7 files)

**Production (7 files)**

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔴 | Config | `/etc/modprobe.d/blacklist-nouveau.conf` | No | **CRITICAL: GPU driver conflict prevention** |
| 🔵 | Config | `/etc/modprobe.d/blacklist-framebuffer.conf` | No | GPU stability |
| 🔵 | Config | `/etc/modprobe.d/onload-hft.conf` | No | Onload physical mode |
| 🔵 | Config | `/etc/modprobe.d/sfc.conf` | No | Solarflare performance profile |
| 🔵 | Config | `/etc/modprobe.d/onload.conf` | No | Generic Onload config |
| 🟢 | Config | `/etc/modprobe.d/blacklist-firewire.conf` | No | Security: disable FireWire |
| 🟢 | Config | `/etc/modprobe.d/blacklist-rare-network.conf` | No | Cleanup: disable unused modules |

**Note:** blacklist-nouveau.conf is CRITICAL but not currently backed up.

---

#### /etc/udev/rules.d/ - Device Rules (2 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Config | `/etc/udev/rules.d/90-rapl-permissions.rules` | No | RAPL power monitoring |
| 🔵 | Config | `/etc/udev/rules.d/99-trading-nic-irq.rules` | No | NIC IRQ affinity trigger |

---

#### /etc/default/grub - Kernel Boot Parameters (1 file)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔴 | Config | `/etc/default/grub` | No | **CRITICAL: CPU isolation, power mgmt** |

**Parameters:** `isolcpus=2,3,4,5,6,7 nohz_full=2,3,4,5,6,7 rcu_nocbs=2,3,4,5,6,7 intel_pstate=disable processor.max_cstate=1 intel_idle.max_cstate=0`

---

#### /etc/grafana/ - Grafana Configuration (3 files)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Config | `/etc/grafana/grafana.ini` | No | Main configuration |
| 🔵 | Config | `/etc/grafana/provisioning/datasources/` | No | Data source configs |
| 🔵 | Config | `/etc/grafana/provisioning/dashboards/` | No | Dashboard provisioning |

---

#### /var/lib/grafana/ - Grafana Data (1 location)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Data | `/var/lib/grafana/` | No | Grafana runtime data |

---

#### /mnt/hdd/questdb/cold - Cold Storage (1 mount)

| Status | Type | File | Symlink | Purpose |
|--------|------|------|---------|---------|
| 🔵 | Mount | `/mnt/hdd/questdb/cold` | No | QuestDB cold storage (archive >30 days) |

---

### 13.6 SUMMARY STATISTICS

#### Workspace Files by Folder

| Folder | Production | Test | Legacy | Utility | Total |
|--------|-----------|------|--------|---------|-------|
| **Monitoring** | 16 | 0 | 8 | 11 | **35** |
| **QuestDB** | 10 | 14 | 7 | 7 | **38** |
| **Trading** | 1 | 0 | 0 | 0 | **1** |
| **Tuning** | 10 | 4 | 0 | 0 | **14** |
| **TOTAL** | **37** | **18** | **15** | **18** | **88** |

#### External Dependencies by Location

| Location | Production | Legacy | Symlinks | Total |
|----------|-----------|--------|----------|-------|
| **/opt/redis-hft/** | 10 | 5 | 1 in | **15** |
| **/usr/local/bin/** | 17 | 4 | 4 out | **21** |
| **/etc/systemd/system/** | 16 | 1 | 0 | **17** |
| **/etc/sysctl.d/** | 11 | 3 | 0 | **14** |
| **/etc/modprobe.d/** | 7 | 0 | 0 | **7** |
| **/etc/udev/rules.d/** | 2 | 0 | 0 | **2** |
| **/etc/default/** | 1 | 0 | 0 | **1** |
| **/etc/grafana/** | 3 | 0 | 0 | **3** |
| **/var/lib/grafana/** | 1 | 0 | 0 | **1** |
| **/mnt/hdd/** | 1 | 0 | 0 | **1** |
| **TOTAL** | **69** | **13** | **5** | **82** |

#### Grand Total: 170 files tracked (88 workspace + 82 external)

#### Symlink Summary (5 total)

**Workspace → External (4 symlinks):**
1. `/usr/local/bin/datafeed` → `workspace/QuestDB/scripts/datafeed_v2.sh`
2. `/usr/local/bin/ultra-low-latency-nic.sh` → `workspace/Tuning/Network/ultra-low-latency-nic.sh`
3. `/usr/local/bin/vm-manager.sh` → `workspace/vm-manager.sh`
4. `/usr/local/bin/redis-hft-cli` → `/opt/redis-hft/scripts/redis-hft-cli` (external→external)

**External ← Workspace (1 symlink - reverse for version control):**
1. `/opt/redis-hft/config/redis-hft.conf` ← `workspace/Tuning/Redis/redis-hft.conf`

---

### 13.7 CRITICAL MISSING BACKUPS

**Action Required - Backup These Files to Workspace:**

1. 🔴 **CRITICAL**: `/etc/modprobe.d/blacklist-nouveau.conf` - GPU will not work without this
2. `/etc/sysctl.d/10-*.conf` (9 files) - System hardening configs
3. `/etc/modprobe.d/blacklist-*.conf` (3 more files) - Other module blacklists
4. `/opt/redis-hft/config/redis-auth.txt` - Authentication credentials
5. `/opt/redis-hft/config/redis-pass.txt` - Password file

**Backup Commands:**
```bash
# Critical GPU blacklist
sudo cp /etc/modprobe.d/blacklist-nouveau.conf ~/ai-trading-station/Tuning/Module-Blacklists/

# System hardening
sudo cp /etc/sysctl.d/10-*.conf ~/ai-trading-station/Tuning/System-Hardening/

# Other blacklists
sudo cp /etc/modprobe.d/blacklist-*.conf ~/ai-trading-station/Tuning/Module-Blacklists/

# Redis credentials (secure storage)
sudo cp /opt/redis-hft/config/redis-{auth,pass}.txt ~/ai-trading-station/Archive/redis-credentials/
```

---

### 13.8 CLEANUP OPPORTUNITIES

**Files That Can Be Archived/Removed:**

1. **Monitoring**: 8 legacy files (23% of folder)
2. **QuestDB**: 7 legacy files (18% of folder)
3. **/usr/local/bin/**: 4 backup files
4. **/opt/redis-hft/archive/**: 4 old backups
5. **/etc/sysctl.d/**: 3 backup files

**Estimated Space Recovery:** ~50MB + reduced clutter

---

### 13.9 SYMLINK MIGRATION CHECKLIST

**Pre-Migration (Old System):**
```bash
# Document all symlinks
find /usr/local/bin/ -type l -ls > symlinks_inventory.txt
readlink -f /usr/local/bin/datafeed
readlink -f /usr/local/bin/ultra-low-latency-nic.sh
readlink -f /usr/local/bin/vm-manager.sh
readlink -f /usr/local/bin/redis-hft-cli
readlink -f /opt/redis-hft/config/redis-hft.conf
```

**Post-Migration (New System):**
```bash
# Recreate workspace → /usr/local/bin/ symlinks
NEW_WORKSPACE="/opt/ai-trading-station"  # Update as needed
sudo ln -sf "$NEW_WORKSPACE/QuestDB/scripts/datafeed_v2.sh" /usr/local/bin/datafeed
sudo ln -sf "$NEW_WORKSPACE/Tuning/Network/ultra-low-latency-nic.sh" /usr/local/bin/ultra-low-latency-nic.sh
sudo ln -sf "$NEW_WORKSPACE/vm-manager.sh" /usr/local/bin/vm-manager.sh
sudo ln -sf /opt/redis-hft/scripts/redis-hft-cli /usr/local/bin/redis-hft-cli

# Handle reverse symlink (redis-hft.conf) - Option A: Copy instead
sudo cp "$NEW_WORKSPACE/Tuning/Redis/redis-hft.conf" /opt/redis-hft/config/redis-hft.conf

# Verify all symlinks
datafeed status
ultra-low-latency-nic.sh --help
vm-manager.sh --help
redis-hft-cli ping
```

---

**End of Complete File Inventory**

---

**End of Audit Report**

*This document should be kept up-to-date as new dependencies are added or existing ones are refactored for portability.*
