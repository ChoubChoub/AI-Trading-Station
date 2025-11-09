# VPS-Brain Architecture: Quick Start Card
**Essential Information for Phase 0 Implementation**

---

## 🎯 Mission Statement

Deploy production-grade VPS-Brain distributed market data architecture for HFT trading system. VPS collects Binance market data in Tokyo (low latency), transfers to home Brain for signal generation. **Goal: <5ms sustained VPS→Brain latency, zero data loss.**

---

## 📋 Phase 0 Checklist (Print This!)

### Pre-Flight (Before You Begin)
- [ ] VPS provisioned (Vultr Tokyo, 2 vCPU, 4GB RAM)
- [ ] VPS IP address recorded: ___________________
- [ ] Root password secured
- [ ] SSH access verified from workstation
- [ ] Phase 0 plan open: `VPS-Phase-0-Infrastructure-Setup.md`
- [ ] Implementation log open: `VPS-Phase-0-Implementation-Log.md`
- [ ] Coffee/energy drink acquired ☕
- [ ] Estimated time: 14 hours over 4 days

### Day 1: Base System (4 hours)
- [ ] `apt update && apt upgrade -y` completed
- [ ] Build tools installed (gcc, make, headers)
- [ ] BBR enabled: `sysctl net.ipv4.tcp_congestion_control=bbr`
- [ ] Network tuning applied (tcp_retries2=5, etc.)
- [ ] **CPU isolation added to /etc/default/grub**: `isolcpus=1 nohz_full=1 rcu_nocbs=1`
- [ ] **THP disabled via /etc/rc.local**: `echo never > transparent_hugepage/enabled`
- [ ] System rebooted
- [ ] CPU 1 verified isolated: `cat /sys/devices/system/cpu/isolated` = "1"
- [ ] THP verified disabled: `cat /sys/.../enabled` = "[never]"

### Day 2: Redis (4 hours)
- [ ] Redis 7.0+ compiled with jemalloc
- [ ] Configuration file created: `/etc/redis/redis-hft.conf`
- [ ] maxmemory 3gb set
- [ ] appendonly yes (AOF enabled)
- [ ] requirepass configured (password: ********)
- [ ] systemd service created with **CPUAffinity=1**
- [ ] Redis started: `systemctl start redis-hft`
- [ ] CPU affinity verified: `taskset -cp $(pgrep redis-server)` = "1"
- [ ] Benchmark run: SET ops/sec > 100K ✅

### Day 3: Network (3 hours)
- [ ] WireGuard installed: `apt install wireguard`
- [ ] VPS keys generated: `wg genkey | tee privatekey | wg pubkey > publickey`
- [ ] Brain public key received
- [ ] /etc/wireguard/wg0.conf configured
- [ ] Tunnel started: `systemctl start wg-quick@wg0`
- [ ] Ping test: `ping -c 100 10.0.0.1` → avg < 5ms ✅
- [ ] Redis ping via VPN: latency < 5ms ✅
- [ ] UFW firewall configured (SSH, WireGuard only)

### Day 4: Validation (3 hours)
- [ ] Network IRQs identified: `cat /proc/interrupts | grep eth0`
- [ ] IRQs pinned to CPU 1
- [ ] irqbalance disabled: `systemctl disable irqbalance`
- [ ] Node exporter installed (CPU 0 affinity)
- [ ] Redis exporter installed (CPU 0 affinity)
- [ ] **FULL VALIDATION RUN**: All 10 metrics PASS
- [ ] 24-hour burn-in started
- [ ] Phase 0 COMPLETE ✅

---

## 🚨 Critical Commands (Keep Handy)

### Verify CPU Isolation (Should output "1")
```bash
cat /sys/devices/system/cpu/isolated
```

### Verify THP Disabled (Should output "[never]")
```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
```

### Check Redis CPU Affinity (Should output "1")
```bash
taskset -cp $(pgrep redis-server)
```

### Test VPN Latency (Should be <5ms average)
```bash
ping -c 100 10.0.0.1
```

### Redis Benchmark (Should be >100K ops/sec)
```bash
redis-benchmark -h localhost -p 6379 -a YOUR_PASSWORD -t set,get -q
```

### Check BBR Enabled (Should output "bbr")
```bash
sysctl net.ipv4.tcp_congestion_control
```

### View Network IRQs
```bash
cat /proc/interrupts | grep eth0
```

### Pin IRQ to CPU 1 (replace XX with IRQ number)
```bash
echo 2 > /proc/irq/XX/smp_affinity
```

---

## ⚠️ Critical Don'ts (Will Break System)

1. **DON'T** skip CPU isolation reboot verification
   - If CPU 1 not isolated, Redis will share with systemd (latency spikes)

2. **DON'T** forget THP disable
   - Causes Redis 100ms+ latency spikes randomly

3. **DON'T** set Redis CPU affinity without isolcpus
   - Won't prevent scheduler from stealing CPU 1

4. **DON'T** expose Redis port (6379) to public internet
   - Security risk, only allow via WireGuard VPN

5. **DON'T** proceed to Phase 1 if ANY validation check fails
   - Foundation must be solid

---

## 📊 Success Metrics (All Must PASS)

| Metric | Target | Command | Pass Criteria |
|--------|--------|---------|---------------|
| CPU Isolation | CPU 1 | `cat /sys/devices/system/cpu/isolated` | Output: "1" |
| THP Status | Disabled | `cat /sys/.../transparent_hugepage/enabled` | Output: "[never]" |
| Redis CPU | CPU 1 only | `taskset -cp $(pgrep redis-server)` | Output: "1" |
| Redis Speed | >100K ops/s | `redis-benchmark -t set,get -q` | SET >100K |
| VPN Latency | <5ms avg | `ping -c 100 10.0.0.1` | avg <5ms |
| BBR | Enabled | `sysctl net.ipv4.tcp_congestion_control` | Output: "bbr" |
| Swap | 0 bytes | `free -h | grep Swap` | Used: 0B |
| IRQs | CPU 1 | `cat /proc/interrupts | grep eth0` | All on CPU 1 |
| Kernel Warnings | None | `dmesg | grep -i error` | Empty |

**ALL 9 metrics must PASS before Phase 1!**

---

## 🔧 Common Issues & Fixes

### Issue: CPU 1 not isolated after reboot
**Symptom:** `cat /sys/devices/system/cpu/isolated` outputs empty  
**Fix:** Check GRUB config, ensure `isolcpus=1 nohz_full=1 rcu_nocbs=1` present, run `update-grub`, reboot

### Issue: THP re-enabled after reboot
**Symptom:** `cat /sys/.../enabled` shows "[always] madvise never"  
**Fix:** Verify /etc/rc.local has execute permissions (`chmod +x`), check if systemd service enabled

### Issue: VPN latency >5ms
**Symptom:** `ping 10.0.0.1` shows avg >5ms  
**Fix:** Check BBR enabled, verify WireGuard keepalive=25, test home ISP latency

### Issue: Redis benchmark <100K ops/sec
**Symptom:** SET ops/sec only 50K  
**Fix:** Verify CPU affinity on CPU 1, check if THP disabled, ensure jemalloc compiled

### Issue: Redis not on CPU 1
**Symptom:** `taskset -cp` shows multiple CPUs (0-1)  
**Fix:** Check systemd service has `CPUAffinity=1`, restart service

---

## 📞 Emergency Contacts

**If stuck during Phase 0:**
1. Document issue in `VPS-Phase-0-Implementation-Log.md`
2. Include: symptom, commands tried, output
3. Ask GitHub Copilot for help with context

**VPS Provider Support:**
- Vultr Support: https://my.vultr.com/support/
- Emergency: Create ticket with "HFT setup blocked"

---

## 📚 Document Quick Links

**Primary Guides:**
- Phase 0 Plan: `/Documentation/VPS-Phase-0-Infrastructure-Setup.md` (688 lines)
- Implementation Log: `/Documentation/VPS-Phase-0-Implementation-Log.md` (fill live)
- Opus Fixes: `/Documentation/VPS-Brain-Opus-Critical-Fixes-Implementation.md` (1000+ lines)

**Reference:**
- Main Architecture: `/VPS-Brain Distributed Market Data Architecture.txt` (62KB)
- Decision Doc: `/Documentation/VPS-READY-FOR-IMPLEMENTATION.md`
- Visual Status: `/Documentation/VPS-VISUAL-STATUS.txt`

---

## 🎯 Key Architectural Decisions

1. **Replay Strategy:** SEQUENTIAL (Opus's strong recommendation)
   - 2-6 hour recovery acceptable for determinism

2. **CPU Allocation:**
   - CPU 0: Control plane (systemd, SSH, monitoring)
   - CPU 1: Data ingestion (Redis, collectors) - ISOLATED

3. **Data Flow:** Strictly unidirectional VPS → Brain
   - Brain NEVER sends data to VPS

4. **Backpressure:** Stepped 80% → 50% → 20% → 0%
   - Prevents oscillation

5. **Network Partition:** VPS authoritative
   - Brain resyncs from VPS (never wins conflicts)

---

## 🚀 Phase 1 Preview (After Phase 0 Complete)

**What comes next:**
1. Deploy Binance WebSocket collectors on VPS (CPU 1)
2. Implement VPS→Brain Redis XREAD transfer
3. Integration testing (24 hour stability)
4. Signal generation on Brain using VPS data
5. Monitoring dashboards (Grafana)

**Timeline:** Nov 4-8 (5 days)  
**Target:** Production cutover Nov 11

---

## ✅ Phase 0 Sign-Off Template

```
PHASE 0 COMPLETE

Date: _______________
Duration: _____ hours (target: 14h)

Validation Results:
✅ CPU Isolation: PASS
✅ THP Disabled: PASS  
✅ Redis CPU Affinity: PASS
✅ Redis Throughput: _____ ops/sec (>100K)
✅ VPN Latency: _____ ms (<5ms)
✅ BBR Enabled: PASS
✅ Swap: 0 bytes
✅ IRQs on CPU 1: PASS
✅ No kernel warnings: PASS

Status: READY FOR PHASE 1

Operator: _______________
Signature: _______________
```

---

**Print this card and keep it next to your workstation during Phase 0!**

**Last Updated:** October 29, 2024  
**Version:** 1.0  
**Status:** 🟢 READY TO USE
