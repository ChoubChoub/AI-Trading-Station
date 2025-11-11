# AI Trading Station System Configuration

**⚠️ CRITICAL: This directory contains COPIES of system configuration files ⚠️**

All files in this directory are **portable backup copies** of configurations deployed to system locations (`/etc/`, `/usr/local/bin/`, `/opt/`). This serves as the **authoritative source** for trading infrastructure configuration that can be version-controlled and deployed to any compatible system.

## 📖 Table of Contents

- [Directory Structure](#-directory-structure)
- [Purpose & Philosophy](#-purpose--philosophy)
- [Quick Start Deployment](#-quick-start-deployment)
- [Deployment Modes](#-deployment-modes-explained)
- [Available Components](#-available-components)
- [Service Mapping](#-complete-service--script--systemconfig-mapping)
- [Common Workflows](#-common-deployment-workflows)
- [File Locations](#-file-locations-reference)
- [Service Dependencies](#-service-dependency-chain)
- [Maintenance](#-maintenance--updates)
- [Troubleshooting](#-troubleshooting)
- [Performance Impact](#-performance-impact)
- [Emergency Recovery](#-emergency-recovery)

---

## 📁 Directory Structure

```
SystemConfig/
├── README.md                          # 📖 This comprehensive guide
├── deploy-system-config-enhanced.sh   # 🚀 Enhanced deployment script (recommended)
├── cron/                              # ⏰ Cron schedules for portability
│   └── crontab.txt                    # Full crontab export (restore: crontab crontab.txt)
├── kernel/                           # ⚙️ Kernel-level configurations
│   ├── grub/                         # GRUB boot parameters (manual integration)
│   ├── modprobe.d/                   # Kernel module settings → /etc/modprobe.d/
│   ├── modules-load.d/               # Modules to load at boot → /etc/modules-load.d/
│   └── README_kvm.md                 # KVM/virtualization configuration guide
├── network/                          # 🌐 Network optimization scripts → /usr/local/bin/
│   ├── configure-nic-irq-affinity.sh
│   └── ultra-low-latency-nic.sh
├── onload/                           # 🏎️ Solarflare Onload configs → /etc/modprobe.d/
│   ├── onload.conf
│   └── onload-hft.conf
├── redis/                            # 💾 Redis HFT config → /opt/redis-hft/config/
│   └── redis-hft.conf
├── sysctl.d/                         # 🔧 System tuning → /etc/sysctl.d/
│   ├── 99-solarflare-trading.conf
│   └── 99-trading-hugepages.conf
├── systemd/                          # 🔄 SystemD management → /etc/systemd/system/
│   ├── overrides/                    # CPU affinity & resource limits
│   │   ├── binance-bookticker.service.d/
│   │   ├── binance-trades.service.d/
│   │   ├── batch-writer.service.d/
│   │   ├── questdb.service.d/
│   │   └── redis-hft.service.d/
│   └── services/                     # Service definitions (ALL trading services)
│       ├── batch-writer.service
│       ├── binance-bookticker.service
│       ├── binance-trades.service
│       ├── configure-nic-irq-affinity.service
│       ├── dual-gpu-trading-config.service
│       ├── questdb.service
│       ├── redis-hft.service
│       └── ultra-low-latency-nic.service
├── sudoers.d/                        # 🔐 Passwordless sudo configs → /etc/sudoers.d/
│   ├── desktop-toggle
│   ├── lspci-monitoring
│   └── redis-monitoring
├── udev/                             # 🔌 Hardware device rules → /etc/udev/rules.d/
│   └── rules.d/
└── shell/                            # 🐚 Shell environment & management scripts
    ├── bashrc.trading                # Trading environment → ~/.bashrc.trading
    ├── datafeed.sh                   # Data feed management → /usr/local/bin/datafeed
    └── vm-manager.sh                 # VM lifecycle management → /usr/local/bin/vm-manager
```

**Legacy Script:** `Services/System/deploy-system-config.sh` (original version, still functional)

---

## 🚀 Quick Start Deployment

**Working Directory:** Always run from `SystemConfig/` directory

```bash
cd /home/youssefbahloul/ai-trading-station/SystemConfig/
```

### Safe Testing (Recommended First Step)

```bash
# Test everything safely - NO system changes
./deploy-system-config-enhanced.sh --test

# Test specific components only
./deploy-system-config-enhanced.sh --test network redis cron
```

### Production Deployment

```bash
# Deploy everything
sudo ./deploy-system-config-enhanced.sh

# Deploy specific components
sudo ./deploy-system-config-enhanced.sh systemd-services network

# Interactive menu selection
sudo ./deploy-system-config-enhanced.sh --select

# Deploy single service
sudo ./deploy-system-config-enhanced.sh --service redis-hft
```

---

## 🎯 Purpose & Philosophy

### What This Directory Contains

**SystemConfig is a COPY/BACKUP repository** containing:
1. **All trading-related systemd service files** — both infrastructure (Redis, network tuning) and application services (Binance collectors, batch writer, QuestDB)
2. **System configuration files** required for ultra-low-latency trading (kernel modules, network scripts, sysctl parameters)
3. **Service overrides** that pin services to specific CPU cores for isolation
4. **Cron schedules** — portable crontab exports for scheduled tasks (QuestDB health monitoring, retention cleanup)

### What This Directory Does NOT Contain

- Application code (Python scripts, trading algorithms) — these remain in workspace folders
- Generic Linux system configurations unrelated to trading
- Security hardening or driver blacklists

### Two-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│          Trading Applications (Workspace)           │
│  Python scripts in Services/{QuestDB,Trading,GPU}/  │
│  • binance_trades_collector.py                      │
│  • binance_bookticker_collector.py                  │
│  • redis_to_questdb_v2.py                           │
│  • QuestDB binary + Java launcher                   │
└─────────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────────┐
│     Trading Infrastructure (SystemConfig copies)    │
│  Service definitions + configs in /etc/, /usr/, etc │
│  • Service files define HOW to run applications     │
│  • Network scripts optimize NIC for low latency     │
│  • CPU affinity pins services to dedicated cores    │
└─────────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────────┐
│              Linux System (Standard OS)             │
│            kernel, drivers, base system             │
└─────────────────────────────────────────────────────┘
```

---

## �️ Complete Service → Script → SystemConfig Mapping

### Infrastructure Services (Network & Database)

| Service File | Deployed Location | Script/Binary | Script Location | SystemConfig Copy |
|--------------|-------------------|---------------|-----------------|-------------------|
| `configure-nic-irq-affinity.service` | `/etc/systemd/system/` | `configure-nic-irq-affinity.sh` | `/usr/local/bin/` (system) | `SystemConfig/network/` |
| `ultra-low-latency-nic.service` | `/etc/systemd/system/` | `ultra-low-latency-nic.sh` | `/usr/local/bin/` (system) | `SystemConfig/network/` |
| `redis-hft.service` | `/etc/systemd/system/` | `redis-server` | `/usr/bin/` (system package) | N/A (standard binary) |
| └─ config | | `redis-hft.conf` | `/opt/redis-hft/config/` (system) | `SystemConfig/redis/` |

### Trading Application Services

| Service File | Deployed Location | Script/Binary | Script Location | SystemConfig Copy |
|--------------|-------------------|---------------|-----------------|-------------------|
| `binance-trades.service` | `/etc/systemd/system/` | `binance_trades_collector.py` | `Services/QuestDB/Runtime/` (workspace) | N/A (app code) |
| `binance-bookticker.service` | `/etc/systemd/system/` | `binance_bookticker_collector.py` | `Services/QuestDB/Runtime/` (workspace) | N/A (app code) |
| `batch-writer.service` | `/etc/systemd/system/` | `redis_to_questdb_v2.py` | `Services/QuestDB/Runtime/` (workspace) | N/A (app code) |
| `questdb.service` | `/etc/systemd/system/` | QuestDB Java launcher | `Services/QuestDB/questdb-*/bin/java` (workspace) | N/A (app binary) |
| `dual-gpu-trading-config.service` | `/etc/systemd/system/` | `blackwell_maxq_optimizer.sh` | `Services/GPU/` (workspace) | N/A (app script) |

### Service Overrides (CPU Affinity)

| Override Directory | Deployed Location | Purpose | SystemConfig Copy |
|--------------------|-------------------|---------|-------------------|
| `binance-trades.service.d/` | `/etc/systemd/system/binance-trades.service.d/` | Pin to CPU core 2 | `SystemConfig/systemd/overrides/` |
| `binance-bookticker.service.d/` | `/etc/systemd/system/binance-bookticker.service.d/` | Pin to CPU core 3 | `SystemConfig/systemd/overrides/` |
| `batch-writer.service.d/` | `/etc/systemd/system/batch-writer.service.d/` | Pin to CPU core 2 | `SystemConfig/systemd/overrides/` |
| `questdb.service.d/` | `/etc/systemd/system/questdb.service.d/` | Pin to housekeeping cores 0,1 | `SystemConfig/systemd/overrides/` |
| `redis-hft.service.d/` | `/etc/systemd/system/redis-hft.service.d/` | Pin to housekeeping cores 0,1 | `SystemConfig/systemd/overrides/` |

**Key Insight:**
- **Service files** (`.service`) → Always copied to `/etc/systemd/system/` → Backed up in `SystemConfig/systemd/services/`
- **Application scripts** referenced by services → Remain in workspace `Services/` folders → NOT in SystemConfig
- **Infrastructure scripts** → Deployed to `/usr/local/bin/` → Backed up in `SystemConfig/network/`
- **Configuration files** → Deployed to `/etc/` or `/opt/` → Backed up in respective SystemConfig folders

---

## � Deployment Modes Explained

### Test Mode (`--test`)
- **No root required**
- **No system files modified**
- Creates temporary directory: `/tmp/systemconfig-test-XXXXXX`
- Shows what would be created/updated
- Shows diffs for changed files
- Perfect for validation before deployment

**Example:**
```bash
./deploy-system-config-enhanced.sh --test
./deploy-system-config-enhanced.sh --test network redis
```

### Production Mode (default)
- **Requires sudo**
- **Modifies system files**
- Creates timestamped backups: `.backup.YYYYMMDD_HHMMSS`
- Deploys files to: `/etc/`, `/usr/local/bin/`, `/opt/`
- Reloads systemd, sysctl, and udev
- Validates deployment after completion

**Example:**
```bash
sudo ./deploy-system-config-enhanced.sh
sudo ./deploy-system-config-enhanced.sh network
```

### Dry-Run Mode (`--dry-run`)
- **Requires sudo** (to check existing files)
- **No system files modified**
- Shows what would be deployed
- No diffs shown (use `--test` for diffs)
- Quick preview mode

**Example:**
```bash
sudo ./deploy-system-config-enhanced.sh --dry-run
```

---

## 🎛️ Available Components

| Number | Component | Description | Files Deployed |
|--------|-----------|-------------|----------------|
| 1 | **systemd-services** | SystemD service definitions | 8 services → `/etc/systemd/system/` |
| 2 | **systemd-overrides** | CPU affinity and resource limits | 5 override dirs → `/etc/systemd/system/*.service.d/` |
| 3 | **kernel** | Kernel module configurations | 3 configs → `/etc/modprobe.d/` |
| 4 | **network** | Network optimization scripts | 2 scripts → `/usr/local/bin/` |
| 5 | **redis** | Redis HFT configuration | 1 config → `/opt/redis-hft/config/` |
| 6 | **onload** | Solarflare Onload configuration | 2 configs → `/etc/modprobe.d/` |
| 7 | **sysctl** | System tuning parameters | 2 configs → `/etc/sysctl.d/` |
| 8 | **udev** | Hardware device rules | 1 rule → `/etc/udev/rules.d/` |
| 9 | **sudoers** | Passwordless sudo configurations | 3 files → `/etc/sudoers.d/` |
  10 | **kvm** | KVM virtualization modules | 1 config → `/etc/modules-load.d/` |
  c | **cron** | Scheduled jobs (crontab) | Appends to user crontab |
  s | **shell** | Shell environment & management scripts | 3 files → `~/.bashrc.trading`, `/usr/local/bin/` |
  a | **all** | Deploy everything | All of the above |

---

## 🔄 Common Deployment Workflows

### Workflow 1: Test First, Then Deploy All
```bash
# 1. Test everything safely
./deploy-system-config-enhanced.sh --test

# 2. Review output and confirm no issues

# 3. Deploy to production
sudo ./deploy-system-config-enhanced.sh
```

### Workflow 2: Selective Component Deployment
```bash
# 1. Test specific components
./deploy-system-config-enhanced.sh --test network redis cron

# 2. If satisfied, deploy those components only
sudo ./deploy-system-config-enhanced.sh network redis cron
```

### Workflow 3: Single Service Update
```bash
# 1. Test the specific service
./deploy-system-config-enhanced.sh --test --service redis-hft

# 2. Deploy that service only
sudo ./deploy-system-config-enhanced.sh --service redis-hft

# 3. Restart the service
sudo systemctl restart redis-hft
```

### Workflow 4: Interactive Selection
```bash
# 1. Test with interactive menu
./deploy-system-config-enhanced.sh --test --select

# 2. Select components interactively
#    Enter numbers: 1 4 5 or 'a' for all

# 3. Review output

# 4. Deploy with same selection
sudo ./deploy-system-config-enhanced.sh --select
```

### Workflow 5: Update Network Script After Editing
```bash
# After editing SystemConfig/network/configure-nic-irq-affinity.sh

# Test the change
./deploy-system-config-enhanced.sh --test --network-script configure-nic-irq-affinity

# Review the diff output

# Deploy the updated script
sudo ./deploy-system-config-enhanced.sh --network-script configure-nic-irq-affinity

# Restart the service to apply changes
sudo systemctl restart configure-nic-irq-affinity
```

---

## �🚀 Deployment

### Automated Deployment (Recommended)

**Enhanced Script Location:** `SystemConfig/deploy-system-config-enhanced.sh`  
**Complete Guide:** See `SystemConfig/DEPLOYMENT-GUIDE.md` for detailed usage examples

```bash
# Navigate to SystemConfig directory
cd /home/youssefbahloul/ai-trading-station/SystemConfig/

# Safe test mode - no system changes, shows diffs
./deploy-system-config-enhanced.sh --test

# Deploy all configurations to system locations
sudo ./deploy-system-config-enhanced.sh

# Deploy specific components only
sudo ./deploy-system-config-enhanced.sh network redis

# Deploy a single service
sudo ./deploy-system-config-enhanced.sh --service redis-hft

# Interactive component selection
sudo ./deploy-system-config-enhanced.sh --select
```

**Original Script (Legacy):** `/home/youssefbahloul/ai-trading-station/Services/System/deploy-system-config.sh`

**What the deployment script does:**
1. **Copies service files** from `SystemConfig/systemd/services/*.service` → `/etc/systemd/system/`
2. **Copies service overrides** from `SystemConfig/systemd/overrides/*` → `/etc/systemd/system/`
3. **Copies network scripts** from `SystemConfig/network/*.sh` → `/usr/local/bin/` (makes executable)
4. **Copies Redis config** from `SystemConfig/redis/redis-hft.conf` → `/opt/redis-hft/config/`
5. **Copies kernel/module configs** from `SystemConfig/kernel/modprobe.d/` and `SystemConfig/onload/` → `/etc/modprobe.d/`
6. **Copies sysctl parameters** from `SystemConfig/sysctl.d/` → `/etc/sysctl.d/`
7. **Copies udev rules** from `SystemConfig/udev/rules.d/` → `/etc/udev/rules.d/`
8. **Creates timestamped backups** of any existing files before overwriting
9. **Reloads system services:** `systemctl daemon-reload`, `sysctl --system`, `udevadm control --reload-rules`

**Enhanced Features:**
- **Test mode** (`--test`): No root needed, no changes, shows diffs
- **Selective deployment**: Deploy specific components or single files
- **Interactive selection**: Menu-driven component picker
- **Unchanged detection**: Skips files that are already up-to-date

**See `SystemConfig/DEPLOYMENT-GUIDE.md` for complete usage examples and workflows.**

### Manual Deployment (Advanced)

```bash
# SystemD services (infrastructure + trading applications)
sudo cp SystemConfig/systemd/services/*.service /etc/systemd/system/
sudo cp -r SystemConfig/systemd/overrides/* /etc/systemd/system/

# Network optimization scripts
sudo cp SystemConfig/network/*.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/*.sh

# Kernel modules & Onload
sudo cp SystemConfig/kernel/modprobe.d/*.conf /etc/modprobe.d/
sudo cp SystemConfig/onload/*.conf /etc/modprobe.d/

# Redis HFT configuration
sudo mkdir -p /opt/redis-hft/config
sudo cp SystemConfig/redis/redis-hft.conf /opt/redis-hft/config/

# System tuning parameters
sudo cp SystemConfig/sysctl.d/*.conf /etc/sysctl.d/
sudo cp SystemConfig/udev/rules.d/*.rules /etc/udev/rules.d/

# Reload all system services
sudo systemctl daemon-reload
sudo sysctl --system
sudo udevadm control --reload-rules
```

---

## � File Locations Reference

| Component Type | SystemConfig (Source) | System Location (Deployed) | Workspace Scripts |
|----------------|----------------------|---------------------------|-------------------|
| **Service Definitions** | `systemd/services/` | `/etc/systemd/system/` | N/A |
| **Service Overrides** | `systemd/overrides/` | `/etc/systemd/system/*.service.d/` | N/A |
| **Network Scripts** | `network/` | `/usr/local/bin/` | N/A |
| **Redis Config** | `redis/` | `/opt/redis-hft/config/` | N/A |
| **Kernel Modules** | `kernel/modprobe.d/` | `/etc/modprobe.d/` | N/A |
| **Onload Config** | `onload/` | `/etc/modprobe.d/` | N/A |
| **Sysctl Parameters** | `sysctl.d/` | `/etc/sysctl.d/` | N/A |
| **Udev Rules** | `udev/rules.d/` | `/etc/udev/rules.d/` | N/A |
| **Cron Schedules** | `cron/crontab.txt` | Per-user crontab (restore: `crontab crontab.txt`) | N/A |
| **Shell Environment** | `shell/bashrc.trading` | `~/.bashrc.trading` | N/A |
| **Datafeed Manager** | `shell/datafeed.sh` | `/usr/local/bin/datafeed` | N/A |
| **VM Manager** | `shell/vm-manager.sh` | `/usr/local/bin/vm-manager` | N/A |
| **KVM Modules** | `kernel/modules-load.d/` | `/etc/modules-load.d/` | N/A |
| **Trading Scripts** | N/A | N/A | `Services/QuestDB/Runtime/` |
| **GPU Scripts** | N/A | N/A | `Services/GPU/` |
| **QuestDB Binary** | N/A | N/A | `Services/QuestDB/questdb-*/` |

---

## � Service Dependency Chain

Services are started in this order, with dependencies enforced by systemd:

```
1. configure-nic-irq-affinity.service    [Assigns NIC IRQs to housekeeping cores 0,1]
   ↓
2. ultra-low-latency-nic.service         [Configures ring buffers, XPS, coalescing]
   ↓
3. redis-hft.service                     [Starts Redis on cores 0,1]
   ↓
4. binance-trades.service (core 2)  ←→  binance-bookticker.service (core 3)
   [Collect market data and write to Redis]
   ↓
5. batch-writer.service (core 2)    →   questdb.service (cores 0,1)
   [Read from Redis and batch-write to QuestDB]
```

**Dependencies configured via:**
- `Requires=` and `After=` directives in service files
- CPU affinity set via override files in `systemd/overrides/`

---

## 🔧 Maintenance & Updates

### When to Update SystemConfig

Update SystemConfig copies whenever you modify system-deployed files:

```bash
# After editing a service file in /etc/systemd/system/
sudo cp /etc/systemd/system/redis-hft.service \
   /home/youssefbahloul/ai-trading-station/SystemConfig/systemd/services/

# After editing a network script in /usr/local/bin/
sudo cp /usr/local/bin/configure-nic-irq-affinity.sh \
   /home/youssefbahloul/ai-trading-station/SystemConfig/network/

# After editing Redis config
sudo cp /opt/redis-hft/config/redis-hft.conf \
   /home/youssefbahloul/ai-trading-station/SystemConfig/redis/
```

### Cron Schedules

SystemConfig includes portable cron schedule backups in `cron/crontab.txt`:

**Current Scheduled Tasks:**
1. **QuestDB Retention Cleanup** — Daily at 2:00 AM
   - Script: `/Services/QuestDB/Utility/retention-cleanup.sh`
   - Log: `/Services/QuestDB/logs/retention-cron.log`
   
2. **QuestDB Health Monitor** — Every 5 minutes
   - Script: `/Services/Monitoring/Scripts/Runtime/check_questdb_health.sh`
   - Log: `/Services/Monitoring/logs/health-check-cron.log`
   - Purpose: Detects WAL suspension, transaction lag, stale data, service failures

**Export/Update Cron Backup:**
```bash
crontab -l > /home/youssefbahloul/ai-trading-station/SystemConfig/cron/crontab.txt
```

**Restore on New System:**
```bash
crontab /home/youssefbahloul/ai-trading-station/SystemConfig/cron/crontab.txt
```

**View Current Cron Jobs:**
```bash
crontab -l
```

### Backup Strategy

- **SystemConfig IS the backup** — it's the authoritative source for infrastructure config
- Deployment script creates `.backup.<timestamp>` files before overwriting
- Original system files are never modified without backup
- Workspace application code is separate and not managed by SystemConfig

### Validation & Troubleshooting

```bash
# Check if services are properly installed
systemctl list-unit-files | grep -E 'redis-hft|binance|batch-writer|questdb|configure-nic'

# Verify service status
systemctl status redis-hft binance-trades binance-bookticker batch-writer

# Check service logs
journalctl -u redis-hft -f
journalctl -u binance-trades -n 50

# Verify CPU affinity assignments
taskset -cp $(pgrep redis-server)
taskset -cp $(pgrep -f binance_trades)
taskset -cp $(pgrep -f binance_bookticker)

# Check NIC IRQ assignments (should be on cores 0,1 only)
grep enp130s0f0 /proc/interrupts

# Verify network optimizations
ethtool -c enp130s0f0  # Check interrupt coalescing
cat /sys/class/net/enp130s0f0/queues/tx-*/xps_cpus  # Check XPS masks
```

### Common Operations

```bash
# Restart all trading services
sudo systemctl restart binance-trades binance-bookticker batch-writer

# Stop trading stack (in reverse order)
sudo systemctl stop batch-writer binance-bookticker binance-trades

# Re-apply network optimizations
sudo systemctl restart configure-nic-irq-affinity ultra-low-latency-nic

# Reload sysctl parameters after editing
sudo sysctl --system

# Reload udev rules after editing
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 🔍 Troubleshooting

### Problem: Test mode shows "Would CREATE" but files already exist in production

**Solution**: This is normal. Test mode creates a clean temp directory, so it shows what would be created there. In production mode, it will detect existing files and update them.

### Problem: Permission denied errors

**Solution**: Production deployment requires `sudo`. Test mode does not.

### Problem: Services not found after deployment

**Solution**: Run `sudo systemctl daemon-reload` to refresh systemd's service list.

### Problem: Want to rollback a deployment

**Solution**: Each deployment creates timestamped backups. To rollback:
```bash
# Find the backup
ls -la /etc/systemd/system/*.backup.*

# Restore it
sudo cp /etc/systemd/system/redis-hft.service.backup.20251105_143022 \
        /etc/systemd/system/redis-hft.service

# Reload systemd
sudo systemctl daemon-reload
```

### Problem: Cron jobs not running

**Solution**: Check cron installation and logs:
```bash
# Verify cron jobs are installed
crontab -l

# Check cron logs
grep CRON /var/log/syslog | tail -20

# Ensure scripts are executable
chmod +x /home/.../Services/QuestDB/Utility/retention-cleanup.sh
chmod +x /home/.../Services/Monitoring/Scripts/Runtime/check_questdb_health.sh
```

### Problem: IRQ violations after deployment

**Solution**: Verify NIC IRQ affinity configuration:
```bash
# Check service status
systemctl status configure-nic-irq-affinity

# Verify IRQ assignments (should be on cores 0,1 only)
grep enp130s0f0 /proc/interrupts

# Restart the service
sudo systemctl restart configure-nic-irq-affinity
```

---

## 📈 Performance Impact

These configurations deliver:

| Metric | Before | After | Configuration |
|--------|--------|-------|---------------|
| **Network Latency** | ~50μs | **<5μs** | Solarflare Onload + kernel bypass |
| **CPU Isolation** | Shared cores | **Dedicated cores 2,3** | SystemD CPU affinity overrides |
| **IRQ Handling** | All cores | **Housekeeping cores 0,1 only** | NIC IRQ affinity script |
| **Memory Access** | Standard | **Physical + DMA** | Huge pages + CAP_IPC_LOCK |
| **Data Path** | Kernel network stack | **User-space with Onload** | Solarflare acceleration |

---

## ✅ Best Practices

1. **Always test first**: Use `--test` mode before production deployment
2. **Deploy selectively**: Use component or single-file deployment for targeted changes
3. **Verify after deployment**: Check service status after deploying
4. **Keep SystemConfig in sync**: After manual edits to production files, copy them back to SystemConfig
5. **Document changes**: Update this README when adding new components
6. **Export cron regularly**: Keep `cron/crontab.txt` updated with `crontab -l > cron/crontab.txt`
7. **Review backups**: Check `.backup.*` files before deleting them
8. **Test after reboot**: Verify services start correctly after system reboot

---

## ⚠️ Critical Notes

1. **SystemConfig contains COPIES only** — Changes here do not affect the running system until deployed
2. **Application code is NOT in SystemConfig** — Python scripts remain in workspace `Services/` folders
3. **Service files reference workspace paths** — e.g., `ExecStart=/home/.../Services/QuestDB/Runtime/script.py`
4. **Network scripts are deployed to system** — They live in `/usr/local/bin/` for early boot access
5. **Deployment script is in SystemConfig** — Located at `SystemConfig/deploy-system-config-enhanced.sh`
6. **Cron jobs are additive** — Deployment appends jobs; won't duplicate existing entries

---

## �️ Management Scripts

SystemConfig includes management scripts that are deployed to `/usr/local/bin/` for system-wide access. These scripts provide command-line tools for managing various aspects of the trading system.

### Available Scripts

| Script | Deployed As | Purpose | Usage |
|--------|-------------|---------|-------|
| `datafeed.sh` | `/usr/local/bin/datafeed` | Manage market data feed services | `datafeed {start\|stop\|status\|logs\|health\|metrics}` |
| `vm-manager.sh` | `/usr/local/bin/vm-manager` | Manage development VM lifecycle | `vm-manager {start\|stop\|status\|console\|ssh}` |

### Datafeed Manager

The `datafeed` command controls all market data collection services:

**Commands:**
```bash
# Start all data feed services
datafeed start

# Stop all data feed services
datafeed stop

# Check status of all services
datafeed status

# View logs (trades, orderbook, batch, or all)
datafeed logs trades
datafeed logs all

# Health check
datafeed health

# View metrics
datafeed metrics

# Test connection to Redis and QuestDB
datafeed test
```

**Managed Services:**
- `binance-trades.service` - Trade data collector
- `binance-bookticker.service` - Order book data collector  
- `batch-writer.service` - Redis to QuestDB batch writer

### VM Manager

The `vm-manager` command controls the development VM:

**Commands:**
```bash
# Start the development VM
vm-manager start

# Stop the VM
vm-manager stop

# Check VM status
vm-manager status

# Open VM console
vm-manager console

# SSH into VM
vm-manager ssh [username]
```

**Features:**
- Automatic KVM/TCG detection (hardware acceleration when available)
- Workspace folder sharing via 9p/virtfs
- Port forwarding for SSH access
- Safe isolated environment for development

### Deployment

All management scripts are deployed as part of the `shell` component:

```bash
# Deploy all shell scripts
sudo ./deploy-system-config-enhanced.sh shell

# Test deployment
./deploy-system-config-enhanced.sh --test shell

# Verify deployment
which datafeed vm-manager
datafeed --help
vm-manager status
```

---

## �🖥️ KVM Virtualization & Development VM

### Overview

The trading station includes KVM (Kernel-based Virtual Machine) support for running a development VM. This provides a **safe, isolated environment for testing changes** without affecting the production trading system.

**Important:** KVM modules and VM management have **zero performance impact on production trading** when the VM is not running.

### Configuration Files

| File | SystemConfig Location | System Location | Purpose |
|------|----------------------|-----------------|---------|
| `kvm.conf` | `kernel/modules-load.d/` | `/etc/modules-load.d/` | Auto-load KVM modules at boot |
| `vm-manager.sh` | `shell/` | `/usr/local/bin/vm-manager` | VM lifecycle management script |

### Deployment

The KVM modules and VM manager script are deployed as part of their respective components:

```bash
# Deploy KVM modules configuration
cd /home/youssefbahloul/ai-trading-station/SystemConfig/
sudo ./deploy-system-config-enhanced.sh kvm

# Deploy VM manager script
sudo ./deploy-system-config-enhanced.sh shell

# Or deploy both together
sudo ./deploy-system-config-enhanced.sh kvm shell

# Verify deployment
ls -l /etc/modules-load.d/kvm.conf
ls -l /usr/local/bin/vm-manager
lsmod | grep kvm
```

**After deployment:**
- KVM modules will load automatically on every boot
- VM manager will be available system-wide as `vm-manager` command
- VM will NOT start automatically (must be started manually)

### Requirements

1. **BIOS Settings** (one-time setup):
   - Enable **Intel VT-x (VMX)** or **AMD-V (SVM)** in BIOS
   - Location: `Advanced` → `CPU Configuration` → `Intel (VMX) Virtualization Technology` → `[Enabled]`
   - Optional: Enable **VT-d** for device passthrough

2. **User Group Membership**:
   ```bash
   sudo usermod -aG kvm youssefbahloul
   # Log out and back in for changes to take effect
   ```

3. **KVM Modules** (loaded automatically at boot after deployment):
   ```bash
   lsmod | grep kvm
   # Should show: kvm_intel, kvm
   ```

### VM Management

**Start VM** (manual, never automatic):
```bash
vm-manager start
```

**Check VM Status**:
```bash
vm-manager status
```

**Stop VM**:
```bash
vm-manager stop
```

**SSH into VM**:
```bash
ssh trading-vm          # From your Mac
ssh -p 2222 username@localhost  # From the host
```

### Performance & Production Safety

**KVM modules are safe for production:**
- ✅ **Zero CPU overhead** when VM is not running
- ✅ **Zero memory overhead** when VM is not running
- ✅ **No network impact** on production NIC
- ✅ **VM never starts automatically** at boot
- ✅ **Trading services unaffected** by KVM presence

**The VM is designed for development only:**
- Test code changes before production deployment
- Safe environment for experimentation
- Workspace folder shared with host via 9p/virtfs
- No impact on ultra-low latency trading when VM is stopped

### Troubleshooting

**KVM modules not loading:**
```bash
# Check if Intel VT-x is enabled
egrep -c '(vmx|svm)' /proc/cpuinfo  # Should return > 0

# Manually load modules
sudo modprobe kvm
sudo modprobe kvm_intel

# Check for errors
dmesg | grep kvm
```

**Permission denied when starting VM:**
```bash
# Check user is in kvm group
groups | grep kvm

# If not, add user and re-login
sudo usermod -aG kvm $USER
# Log out and back in
```

**VM won't start:**
```bash
# Check VM log for errors
cat ~/.ai-trading-vm/ai-trading-dev.log

# Verify KVM device exists
ls -l /dev/kvm
```

---

## 🆘 Emergency Recovery

If trading services fail after system update or reboot:

```bash
# 1. Re-deploy all configurations from SystemConfig
cd /home/youssefbahloul/ai-trading-station/SystemConfig/
sudo ./deploy-system-config-enhanced.sh

# 2. Reload systemd and apply settings
sudo systemctl daemon-reload
sudo sysctl --system

# 3. Restart infrastructure services
sudo systemctl restart configure-nic-irq-affinity
sudo systemctl restart ultra-low-latency-nic
sudo systemctl restart redis-hft

# 4. Restart trading services
sudo systemctl restart binance-trades binance-bookticker batch-writer

# 5. Verify everything is running
systemctl status redis-hft binance-trades binance-bookticker batch-writer questdb
```

---

*Last Updated: November 8, 2025*  
*SystemConfig serves as the portable, version-controlled source of truth for AI Trading Station infrastructure configuration.*