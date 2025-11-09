# VPS Phase 0: Infrastructure Setup & Optimization
**Complete VPS Provisioning Plan for Tokyo Vultr Instance**

**Date:** October 29, 2025  
**Author:** GitHub Copilot  
**Reviewer:** Claude Opus  
**Status:** READY FOR IMPLEMENTATION

---

## Executive Summary

This document covers **Phase 0** - the critical infrastructure setup that must be completed **before** any data collection or VPS-Brain integration begins. Phase 0 ensures the VPS environment mirrors the Brain's optimized configuration while adding VPS-specific tuning for market data ingestion.

### VPS Specifications (from screenshot)
```yaml
Provider: Vultr
Location: Tokyo, JP
Instance: voc-c-2c-4gb-50s
Cores: 2 vCPUs (dedicated)
Memory: 4 GB
Storage: 50 GB NVMe
Network: 10 Gbps
Cost: Base + $8/mo (backups enabled)
```

### CPU Allocation Strategy
```
CPU 0: Trading & Control Plane
- WireGuard VPN
- systemd services
- Monitoring agents
- SSH sessions

CPU 1: Data Ingestion (EXCLUSIVE)
- Redis server
- Binance WebSocket collectors
- Batch writers
- High-frequency data processing
```

---

## Phase 0 Implementation Checklist

### Day 1: Base System Setup (4 hours)

#### 1.1 Initial Provisioning
```bash
# SSH into fresh VPS
ssh root@vps.aistation.trading

# Update system
apt update && apt upgrade -y

# Install essential tools
apt install -y \
    build-essential \
    linux-headers-$(uname -r) \
    git curl wget htop iotop \
    net-tools sysstat numactl \
    cpufrequtils irqbalance
```

#### 1.2 Kernel Optimization (matching Brain)
```bash
# Create kernel parameters file
cat > /etc/sysctl.d/99-trading-optimizations.conf << 'EOF'
# Network Performance (critical for exchange connections)
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.rmem_default = 16777216
net.core.wmem_default = 16777216
net.ipv4.tcp_rmem = 4096 87380 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq

# Increase connection tracking
net.netfilter.nf_conntrack_max = 1048576
net.nf_conntrack_max = 1048576

# TCP optimization for low latency
net.ipv4.tcp_low_latency = 1
net.ipv4.tcp_timestamps = 0
net.ipv4.tcp_sack = 1
net.ipv4.tcp_fastopen = 3

# OPUS ADDITION: Reduce retransmission latency for exchange connections
net.ipv4.tcp_retries2 = 5       # Default: 15 (too many retries)
net.ipv4.tcp_syn_retries = 2     # Default: 6 (connection setup faster)
net.ipv4.tcp_synack_retries = 2  # Default: 5 (handshake faster)

# File descriptors for Redis
fs.file-max = 2097152

# Hugepages for Redis (if using RDB persistence)
vm.nr_hugepages = 128

# Disable swap (critical for latency)
vm.swappiness = 0
EOF

# Apply immediately
sysctl -p /etc/sysctl.d/99-trading-optimizations.conf

# Verify BBR enabled
sysctl net.ipv4.tcp_congestion_control
# Output: net.ipv4.tcp_congestion_control = bbr
```

#### 1.3 CPU Isolation (CRITICAL - Opus Addition)
```bash
# CRITICAL: Isolate CPU 1 from kernel scheduler
# This prevents ANY process from running on CPU 1 unless explicitly pinned

# Edit GRUB configuration
nano /etc/default/grub

# Modify GRUB_CMDLINE_LINUX line to add:
GRUB_CMDLINE_LINUX="isolcpus=1 nohz_full=1 rcu_nocbs=1"

# Explanation:
# - isolcpus=1: Kernel won't schedule processes on CPU 1
# - nohz_full=1: Disable scheduling-clock ticks on CPU 1 (reduces interrupts)
# - rcu_nocbs=1: Move RCU callbacks off CPU 1 (reduces kernel overhead)

# Update GRUB and reboot
update-grub
echo "Rebooting to apply CPU isolation..."
reboot

# After reboot, verify isolation
cat /sys/devices/system/cpu/isolated
# Should show: 1
```

#### 1.4 Disable Transparent Hugepages (CRITICAL - Opus Addition)
```bash
# CRITICAL: THP causes Redis latency spikes (100ms+)
# Must be disabled for consistent sub-millisecond performance

# Disable immediately
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# Verify disabled
cat /sys/kernel/mm/transparent_hugepage/enabled
# Should show: always madvise [never]

# Make permanent - add to rc.local
cat >> /etc/rc.local << 'EOF'
#!/bin/bash
# Disable THP for Redis (prevents latency spikes)
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
exit 0
EOF

chmod +x /etc/rc.local

# Enable rc-local service
systemctl enable rc-local
```

#### 1.5 Disable Swap (CRITICAL)
```bash
# Disable swap permanently
swapoff -a
sed -i '/swap/d' /etc/fstab

# Verify no swap
free -h
# Swap line should show 0B
```

#### 1.6 CPU Frequency Scaling
```bash
# Set CPUs to performance mode
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu
done

# Make permanent
cat > /etc/systemd/system/cpu-performance.service << 'EOF'
[Unit]
Description=Set CPU Governor to Performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $cpu; done'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable cpu-performance.service
systemctl start cpu-performance.service
```

---

### Day 2: Redis Installation & CPU Affinity (4 hours)

#### 2.1 Install Redis 7.0+ from Source
```bash
# Install dependencies
apt install -y libjemalloc-dev tcl

# Download Redis 7.2 (latest stable)
cd /tmp
wget https://download.redis.io/redis-stable.tar.gz
tar xzf redis-stable.tar.gz
cd redis-stable

# Compile with optimizations
make USE_JEMALLOC=yes MALLOC=jemalloc -j2

# Install
make install

# Create Redis user
useradd -r -s /bin/false redis

# Create directories
mkdir -p /var/lib/redis /var/log/redis /etc/redis
chown redis:redis /var/lib/redis /var/log/redis
```

#### 2.2 Redis Configuration (matching Brain)
```bash
cat > /etc/redis/redis.conf << 'EOF'
# Network Configuration
bind 0.0.0.0
port 6379
protected-mode yes
requirepass YOUR_SECURE_PASSWORD_HERE

# Performance Tuning
tcp-backlog 511
timeout 0
tcp-keepalive 300
maxclients 10000

# Memory Management
maxmemory 3gb  # Leave 1GB for system
maxmemory-policy allkeys-lru

# Persistence Strategy (for recovery)
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis

# AOF for durability
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Logging
loglevel notice
logfile /var/log/redis/redis.log

# Slowlog
slowlog-log-slower-than 10000
slowlog-max-len 128

# Latency Monitoring
latency-monitor-threshold 100

# Jemalloc
jemalloc-bg-thread yes
EOF

# Set permissions
chmod 640 /etc/redis/redis.conf
chown redis:redis /etc/redis/redis.conf
```

#### 2.3 CPU Affinity Configuration (CRITICAL)
```bash
# Create systemd service with CPU affinity
cat > /etc/systemd/system/redis-hft.service << 'EOF'
[Unit]
Description=Redis HFT Data Ingestion Server
After=network.target

[Service]
Type=notify
User=redis
Group=redis

# CPU AFFINITY: Pin to CPU 1 (data ingestion core)
CPUAffinity=1
Nice=-10

# Process Limits
LimitNOFILE=65535
LimitNPROC=32000

# Memory
MemoryMax=3.5G
OOMPolicy=stop

# Execution
ExecStart=/usr/local/bin/redis-server /etc/redis/redis.conf
ExecStop=/bin/kill -s TERM $MAINPID

# Restart Strategy
Restart=always
RestartSec=5s

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=yes

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
systemctl daemon-reload
systemctl enable redis-hft.service
systemctl start redis-hft.service

# Verify CPU affinity
taskset -cp $(pgrep redis-server)
# Should show: pid X's current affinity list: 1
```

#### 2.4 Validate Redis Performance
```bash
# Benchmark on CPU 1
redis-benchmark -h localhost -p 6379 -a YOUR_PASSWORD \
    -t set,get,lpush,lpop -n 1000000 -q -c 50

# Expected results:
# SET: >100,000 ops/sec
# GET: >150,000 ops/sec
# LPUSH: >100,000 ops/sec
```

---

### Day 3: Network & WireGuard VPN (3 hours)

#### 3.1 Install WireGuard
```bash
apt install -y wireguard wireguard-tools

# Generate VPS keys
wg genkey | tee /etc/wireguard/vps_private.key | wg pubkey > /etc/wireguard/vps_public.key
chmod 600 /etc/wireguard/vps_private.key
```

#### 3.2 WireGuard Configuration
```bash
cat > /etc/wireguard/wg0.conf << 'EOF'
[Interface]
PrivateKey = PASTE_VPS_PRIVATE_KEY_HERE
Address = 10.0.0.2/24
ListenPort = 51820

# VPS runs on CPU 0 (control plane)
# No CPU affinity needed for WireGuard (kernel module)

[Peer]
# Brain peer
PublicKey = PASTE_BRAIN_PUBLIC_KEY_HERE
AllowedIPs = 10.0.0.1/32
PersistentKeepalive = 25
EOF

chmod 600 /etc/wireguard/wg0.conf

# Enable and start
systemctl enable wg-quick@wg0
systemctl start wg-quick@wg0

# Verify tunnel
wg show
ping 10.0.0.1  # Should reach Brain
```

#### 3.3 Firewall Configuration
```bash
# Install UFW
apt install -y ufw

# Default policies
ufw default deny incoming
ufw default allow outgoing

# SSH (change to your IP)
ufw allow from YOUR_HOME_IP to any port 22

# WireGuard
ufw allow 51820/udp

# Redis (only via WireGuard)
ufw allow from 10.0.0.0/24 to any port 6379

# Enable firewall
ufw enable

# Verify rules
ufw status numbered
```

#### 3.4 Network Latency Testing
```bash
# Test VPS → Brain latency
cat > /root/test_latency.sh << 'EOF'
#!/bin/bash
echo "Testing VPS → Brain latency over WireGuard..."

# ICMP ping
echo -n "ICMP: "
ping -c 10 10.0.0.1 | grep avg | awk '{print $4}' | cut -d'/' -f2

# Redis ping via VPN
echo -n "Redis: "
for i in {1..100}; do
    redis-cli -h 10.0.0.1 -p 6379 -a BRAIN_PASSWORD PING > /dev/null
done
echo "100 pings completed - check Redis latency stats"
EOF

chmod +x /root/test_latency.sh
/root/test_latency.sh

# Target: <5ms average latency
```

---

### Day 4: IRQ Affinity & Final Tuning (3 hours)

#### 4.1 Network Card IRQ Affinity
```bash
# Find network interface
NETIF=$(ip route | grep default | awk '{print $5}')

# Find IRQs for network card
IRQ_LIST=$(grep $NETIF /proc/interrupts | awk -F: '{print $1}')

# Pin network IRQs to CPU 1 (data ingestion)
for IRQ in $IRQ_LIST; do
    echo 2 > /proc/irq/$IRQ/smp_affinity  # Bitmask: 2 = CPU 1
done

# Make permanent
cat > /etc/systemd/system/irq-affinity.service << 'EOF'
[Unit]
Description=Pin Network IRQs to CPU 1
After=network.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'NETIF=$(ip route | grep default | awk "{print \$5}"); for IRQ in $(grep $NETIF /proc/interrupts | awk -F: "{print \$1}"); do echo 2 > /proc/irq/$IRQ/smp_affinity; done'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable irq-affinity.service
systemctl start irq-affinity.service
```

#### 4.2 Disable IRQ Balance (CRITICAL)
```bash
# Stop irqbalance (interferes with manual affinity)
systemctl stop irqbalance
systemctl disable irqbalance
systemctl mask irqbalance

# Verify it's dead
systemctl status irqbalance
```

#### 4.3 Process Priority Configuration
```bash
# Redis already has Nice=-10 in systemd unit

# For data collectors (will be set in systemd units later)
# Binance collectors: CPU 1, Nice=-5
# Batch writers: CPU 1, Nice=-5
# WireGuard: CPU 0, Nice=0 (default)
```

#### 4.4 Monitoring Setup
```bash
# Install monitoring tools
apt install -y prometheus-node-exporter redis-exporter

# Configure node-exporter (CPU 0)
cat > /etc/systemd/system/node-exporter.service.d/override.conf << 'EOF'
[Service]
CPUAffinity=0
EOF

systemctl daemon-reload
systemctl restart prometheus-node-exporter

# Configure redis-exporter (CPU 0)
cat > /etc/systemd/system/redis-exporter.service << 'EOF'
[Unit]
Description=Redis Exporter for Prometheus
After=network.target redis-hft.service

[Service]
Type=simple
User=redis
CPUAffinity=0
ExecStart=/usr/local/bin/redis_exporter \
    --redis.addr=localhost:6379 \
    --redis.password=YOUR_PASSWORD
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl enable redis-exporter
systemctl start redis-exporter
```

---

## Phase 0 Validation Checklist

### System Health Verification
```bash
#!/bin/bash
# Save as /root/phase0_validation.sh

echo "=== Phase 0 Validation ==="
echo

# 1. CPU Affinity
echo "1. Redis CPU Affinity:"
taskset -cp $(pgrep redis-server)
echo

# 2. CPU Governor
echo "2. CPU Frequency Scaling:"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo

# 3. Swap Status
echo "3. Swap (should be 0):"
free -h | grep Swap
echo

# 4. Network Tuning
echo "4. TCP Congestion Control:"
sysctl net.ipv4.tcp_congestion_control
echo

# 5. WireGuard Status
echo "5. WireGuard VPN:"
wg show
echo

# 6. Redis Health
echo "6. Redis Performance:"
redis-cli -a YOUR_PASSWORD INFO stats | grep instantaneous
echo

# 7. IRQ Affinity
echo "7. Network IRQ Affinity (should be CPU 1):"
NETIF=$(ip route | grep default | awk '{print $5}')
grep $NETIF /proc/interrupts | head -1
for IRQ in $(grep $NETIF /proc/interrupts | awk -F: '{print $1}'); do
    echo "IRQ $IRQ: $(cat /proc/irq/$IRQ/smp_affinity)"
done
echo

# 8. Firewall
echo "8. Firewall Status:"
ufw status | grep Status
echo

echo "=== Validation Complete ==="
```

### Expected Results
```
✅ Redis pinned to CPU 1 only
✅ All CPUs in performance mode
✅ Swap disabled (0B used)
✅ BBR congestion control active
✅ WireGuard tunnel up, ping <5ms
✅ Redis throughput >100K ops/sec
✅ Network IRQs on CPU 1
✅ Firewall active, only WireGuard + SSH open
```

---

## CPU Allocation Verification

### Real-Time CPU Usage Monitoring
```bash
# Watch CPU usage by core
watch -n 1 'mpstat -P ALL 1 1'

# Expected pattern:
# CPU 0: 10-30% (control plane: SSH, WireGuard, monitoring)
# CPU 1: 60-90% (data ingestion: Redis, collectors)
```

### Process Affinity Quick Check
```bash
# List all trading-related processes and their CPU affinity
ps -eo pid,comm,psr,ni | grep -E "redis|binance|wireguard|python"

# Example output:
# PID    COMMAND         CPU  NI
# 1234   redis-server     1  -10  ✅ CPU 1, high priority
# 5678   binance-ws       1   -5  ✅ CPU 1 (to be added Phase 1)
# 9012   wg               0    0  ✅ CPU 0, normal priority
```

---

## System Requirements Validation

### Minimum Requirements Met
```yaml
✅ Redis: 
   - Version: 7.2
   - Memory: 3GB allocated
   - Persistence: AOF + RDB
   - Max clients: 10,000

✅ Network:
   - Latency VPS↔Brain: <5ms (WireGuard)
   - Bandwidth: 10 Gbps (Vultr backbone)
   - Packet loss: <0.01% (monitor with mtr)

✅ Disk:
   - Emergency spill path: /tmp/trading_emergency (created Phase 1)
   - Free space: 50GB NVMe (>100GB recommended if using emergency spill)
   - IOPS: >10,000 (NVMe standard)

✅ CPU:
   - Cores: 2 dedicated vCPUs
   - Affinity: CPU 0 = control, CPU 1 = ingestion
   - Governor: Performance mode
   - IRQ: Network on CPU 1
```

---

## Post-Phase 0 Next Steps

Once Phase 0 validation passes:

1. **Phase 1:** Deploy data collectors (binance-ws, binance-bookticker)
2. **Phase 2:** Implement VPS→Brain data transfer (pub/sub + streams)
3. **Phase 3:** Integrate unstructured data sources
4. **Phase 4:** Production cutover with monitoring

---

## Troubleshooting Common Issues

### Issue: Redis slow to start
```bash
# Check if transparent hugepages interfering
echo never > /sys/kernel/mm/transparent_hugepage/enabled

# Add to /etc/rc.local for persistence
```

### Issue: High latency VPS→Brain
```bash
# Test MTU size
ping -M do -s 1472 10.0.0.1  # Should not fragment

# If fails, lower WireGuard MTU
# Edit /etc/wireguard/wg0.conf:
# Add: MTU = 1420
```

### Issue: CPU 1 not reaching high usage
```bash
# Verify Redis is using CPU 1
top -H -p $(pgrep redis-server)
# Press 'f', enable 'P' (last used CPU), should show CPU 1
```

---

## Security Hardening (Optional but Recommended)

```bash
# Disable root SSH login
sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config

# Create trading user
useradd -m -s /bin/bash trading
usermod -aG sudo trading

# Install fail2ban
apt install -y fail2ban
systemctl enable fail2ban
systemctl start fail2ban
```

---

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Day 1: Base system | 4 hours | Fresh VPS |
| Day 2: Redis + CPU affinity | 4 hours | Day 1 complete |
| Day 3: Network + VPN | 3 hours | Brain WireGuard configured |
| Day 4: IRQ + Final tuning | 3 hours | Day 2-3 complete |
| **Total Phase 0** | **14 hours** | - |

---

**Phase 0 Completion Criteria:**
- ✅ All validation checks pass
- ✅ VPS→Brain latency <5ms sustained
- ✅ Redis throughput >100K ops/sec
- ✅ CPU affinity correctly set
- ✅ No swap usage
- ✅ Firewall configured

**Status after Phase 0:** Ready for data collector deployment (Phase 1)
