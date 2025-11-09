# AI Trading Station - Dependency Categorization Analysis

**Generated:** November 4, 2025  
**Purpose:** Categorize 123 dependencies: Standard Ubuntu vs Custom AI Trading configs

## Executive Summary

Of the 123 external dependencies, here's the breakdown:

- **🔴 CRITICAL CUSTOM (Must Backup):** ~45 dependencies
- **🟡 UBUNTU HARDENING (Optional):** ~15 dependencies  
- **🟢 STANDARD UBUNTU (Skip):** ~25 dependencies
- **🔵 RUNTIME/HARDWARE (Skip):** ~38 dependencies

---

## Category 1: 🔴 CRITICAL CUSTOM - Must Backup (45 items)

### Custom SystemD Services (9 services)
```
batch-writer.service
binance-bookticker.service  
binance-trades.service
configure-nic-irq-affinity.service
dual-gpu-trading-config.service
questdb.service
redis-hft.service
redis-health-monitor.service (if exists)
ultra-low-latency-nic.service (if exists)
```
**Analysis:** These are 100% custom AI trading services you created.

### CPU Affinity Overrides (6 files)
```
batch-writer.service.d/cpu-affinity.conf
binance-bookticker.service.d/cpu-affinity.conf
binance-trades.service.d/cpu-affinity.conf
questdb.service.d/cpu-affinity.conf
redis-hft.service.d/override.conf
prometheus.service.d/cpu-affinity.conf
```
**Analysis:** Custom CPU pinning for HFT performance.

### AI Trading Sysctl Configs (3 files)
```
99-solarflare-trading.conf
99-trading-hugepages.conf  
99-sysctl.conf (if contains custom trading settings)
```
**Analysis:** Custom network and memory tuning for AI trading.

### Onload/Solarflare Configs (5 files) 
```
onload.conf
onload-hft.conf
sfc.conf
sfc-depmod.conf (if exists)
onload-version.txt
```
**Analysis:** Custom HFT network acceleration configs.

### Redis HFT Complete Directory (17 files)
```
/opt/redis-hft/config/redis-hft.conf
/opt/redis-hft/config/redis.env
/opt/redis-hft/config/redis-auth.txt
/opt/redis-hft/config/redis-pass.txt
/opt/redis-hft/scripts/redis-hft-cli
/opt/redis-hft/scripts/redis-hft-monitor.sh
/opt/redis-hft/metrics/redis-metrics.json
/opt/redis-hft/metrics/current-fingerprint.json
/opt/redis-hft/metrics/network-ull-metrics.json
/opt/redis-hft/baselines/fingerprint-baseline.json
/opt/redis-hft/archive/redis-hft.conf.backup
/opt/redis-hft/archive/redis-hft-memory-patch-fixed.conf
/opt/redis-hft/CONFIG_README.md
+ 4 temp/backup files
```
**Analysis:** 100% custom Redis HFT optimization.

### Custom UDEV Rules (1 file)
```
99-trading-nic-irq.rules  
```
**Analysis:** Custom IRQ affinity for trading NIC.

### Custom GPU Config (1+ files)
```
blackwell_maxq_optimizer.sh (in gpu/)
GPU clock locking configs
```
**Analysis:** Custom GPU optimization for AI trading.

---

## Category 2: 🟡 UBUNTU HARDENING - Optional Backup (15 items)

### System Hardening Sysctl (9 files)
```
10-kernel-hardening.conf
10-network-security.conf  
10-ptrace.conf
10-bufferbloat.conf
10-ipv6-privacy.conf
10-map-count.conf
10-zeropage.conf
10-magic-sysrq.conf
10-console-messages.conf
```
**Analysis:** These are system hardening configs. You probably customized them, but similar configs exist in hardened Ubuntu distributions.

### Custom Module Blacklists (4 files)
```
blacklist-nouveau.conf (custom for trading GPUs)
blacklist-framebuffer.conf
blacklist-firewire.conf  
blacklist-rare-network.conf
```
**Analysis:** Customized for trading stability, but could be recreated.

### RAPL Permissions (1 file)
```
90-rapl-permissions.rules
```
**Analysis:** Custom power monitoring access.

### Grafana Config (1 logical unit)
```
/etc/grafana/ (custom dashboards)
/var/lib/grafana/ (custom data)
```
**Analysis:** Custom monitoring dashboards for trading.

---

## Category 3: 🟢 STANDARD UBUNTU - Skip Backup (25 items)

### Standard Module Configs (8 files)
```
blacklist-ath_pci.conf (standard Ubuntu)
amd64-microcode-blacklist.conf (standard Ubuntu)
intel-microcode-blacklist.conf (standard Ubuntu)
blacklist.conf (standard Ubuntu)
iwlwifi.conf (standard Ubuntu)
mdadm.conf (standard Ubuntu)
dkms.conf (standard Ubuntu)
```
**Analysis:** These are standard Ubuntu package configs.

### Standard System Info (11 files)
```
cpu.txt (runtime snapshot)
current-sysctl.txt (runtime snapshot)
distro.txt (runtime snapshot)  
gpu-nvidia.txt (runtime snapshot)
kernel-cmdline.txt (runtime snapshot)
kernel.txt (runtime snapshot)
memory.txt (runtime snapshot)
network-interfaces.txt (runtime snapshot)
network-routes.txt (runtime snapshot)
storage.txt (runtime snapshot)
versions-manifest.json (runtime snapshot)
```
**Analysis:** Runtime system information - not configs to backup.

### Standard Network Info (6 files)
```
current-interfaces.txt (runtime)
current-nic-settings.txt (runtime)
current-gpu-state.txt (runtime)
INSTALLATION_GUIDE.md (documentation)
+ network documentation files
```
**Analysis:** Runtime state and documentation.

---

## Category 4: 🔵 RUNTIME/HARDWARE - Skip Backup (38 items)

### /usr/local/bin/ Scripts (20 files)
**Analysis:** These might be symlinks to workspace scripts (already backed up) or standard binaries.

### Onload Binaries (7 files)
**Analysis:** These are installed by Onload package, not custom configs.

### Python Hardcoded Paths (6 paths)
**Analysis:** Runtime dependencies, not configs.

### Hardware Dependencies (5+ items)
```
GPU hardware detection
NIC hardware (enp130s0f0)  
CPU cores availability
Storage mounts
Kernel modules
```
**Analysis:** Hardware-specific, will be different on new system.

---

## RECOMMENDATION: What to Actually Backup

### 🔴 MUST BACKUP (45 items) - These are YOUR custom AI trading configs:
1. **All 9 custom systemd services**
2. **All 6 CPU affinity overrides** 
3. **All 17 Redis HFT configs**
4. **All 5 Onload/Solarflare custom configs**
5. **3 trading-specific sysctl configs**
6. **1 custom UDEV rule**
7. **Custom GPU optimization scripts**
8. **Custom Grafana dashboards**

### 🟡 CONSIDER BACKING UP (15 items) - Your customized hardening:
1. **9 system hardening sysctl files** (if heavily customized)
2. **4 module blacklists** (if trading-specific)
3. **RAPL permissions**

### 🟢 SKIP BACKUP (63 items) - Standard Ubuntu or runtime:
1. **Standard Ubuntu module configs** (25 items)
2. **Runtime system information** (25 items) 
3. **Onload binaries** (7 items)
4. **Hardware dependencies** (6+ items)

## VERDICT

**Of the 123 dependencies, only ~45-60 are actually YOUR custom AI trading configurations that need Git backup.**

The other ~65-80 are either:
- Standard Ubuntu configs (will exist on new Ubuntu)
- Runtime information (generated fresh on new system)  
- Hardware dependencies (system-specific)
- Package-installed binaries (reinstall packages)

**Your 16 missing configs are definitely in the "MUST BACKUP" category.**