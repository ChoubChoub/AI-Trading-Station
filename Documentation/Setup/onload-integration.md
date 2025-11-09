# Redis + Onload Acceleration Integration

This document explains how Redis in this system is integrated with Solarflare Onload acceleration **without embedding Onload into the Redis systemd service itself**, and how operational safety is preserved.

---
## 1. Design Goals
- Keep the Redis daemon (`redis-hft.service`) lean, deterministic, and isolated.
- Apply Onload kernel-bypass acceleration only for client paths that need it (trading + selective diagnostics).
- Avoid IRQ churn or unintended CPU affinity side-effects during admin/monitor operations.
- Preserve clean separation between:
  - IRQ handling cores (0,1)
  - Trading cores (2,3)
  - Dedicated Redis CPU (planned: 4) or housekeeping (1) depending on your chosen layout.

---
## 2. What Is Accelerated?
Redis server itself runs as a normal userspace process (no LD_PRELOAD inside its systemd unit). **Acceleration is applied to the clients** (trading processes, selective tools) that connect to Redis via Onload.

This follows the typical HFT pattern:
- Server process: stable, minimal moving parts.
- Client workflow: accelerated networking stack (userland TCP fast-paths, reduced syscall overhead, lower jitter).

---
## 3. The `onload-trading` Wrapper
Location: `scripts/onload-trading`

Responsibilities:
1. Export tuned Onload environment variables (polling, spin, queue sizes, CTPIO, etc.).
2. Optionally enforce IRQ isolation policy (when **not** in no‑tune mode).
3. Pin the launched process to the configured trading cores (default: `2,3`).
4. Provide a guarded “no side-effects” mode via `ONLOAD_NO_TUNE=1`.

### Key Modes
| Mode | Variable | IRQ Re-Isolation | CPU Pinning | Onload Env | Use Case |
|------|----------|------------------|-------------|------------|----------|
| Default | (unset) | Yes | Yes | Yes | Launch trading / long-lived procs |
| No-Tune | `ONLOAD_NO_TUNE=1` | Skipped | Yes | Yes | Diagnostics / short ops |

---
## 4. Redis Server CPU Isolation
`redis-hft.service` pins Redis to a housekeeping or dedicated core using `CPUAffinity=`. This reduces cross-interference:
- IRQs (0,1) separated from trading (2,3).
- Redis runs on a non-trading core (e.g., 4) → avoids head-of-line contention and preserves cache predictability.

Check current Redis affinity:
```bash
taskset -pc $(systemctl show -p MainPID --value redis-hft.service)
```

---
## 5. Why Not Run Redis Under Onload Directly?
Running Redis itself under Onload adds complexity without direct win for in-host loopback traffic. Since the trading clients are the latency-sensitive TCP endpoints, accelerating their socket operations (connect, write, poll) yields the meaningful benefit, while keeping the server path stable.

Benefits of client-side acceleration only:
- Reduced risk surface (no need to LD_PRELOAD in systemd service).
- Easier troubleshooting (server logs/behavior unaffected by Onload runtime quirks).
- Finer-grained control: per-process enablement.

---
## 6. Usage Patterns
### Trading Application Launch
```bash
./scripts/onload-trading ./bin/trading_gateway --config prod.yaml
```
### Low-Risk Redis Diagnostic (No IRQ Side-Effects)
```bash
ONLOAD_NO_TUNE=1 ./scripts/onload-trading redis-cli -h 127.0.0.1 -p 6379 ping
```
### Native (No Onload) Diagnostic
```bash
redis-cli -h 127.0.0.1 -p 6379 info server
```
### Optional Admin Wrapper (if you create `onload-admin`)
```bash
./scripts/onload-admin redis-cli latency doctor
```

---
## 7. IRQ Policy Interaction
- Boot / NIC script sets intended affinity (balanced 0+1 or consolidated to 0).
- Wrapper *only* re-applies IRQ layout when **not** in no‑tune mode.
- `ONLOAD_NO_TUNE=1` prevents IRQ re-writes → avoids churn during short-lived commands.

### Verifying No Drift
```bash
grep enp130s0f0 /proc/interrupts
```
(Ensure no counts accumulate on trading cores: 2,3.)

---
## 8. Safety Guidelines
| Action | Recommendation |
|--------|---------------|
| Routine monitoring | Use native `redis-cli` (no wrapper) |
| Latency micro-test w/ Onload | Add `ONLOAD_NO_TUNE=1` |
| Launch/Restart trading engines | Wrapper (default mode) |
| Investigate IRQ anomalies | Use wrapper default (to reassert policy) |
| High-volume benchmarking | Use wrapper default (full tuning) |

---
## 9. Minimal Verify Script (Optional)
Drop in `scripts/verify_nic_irq_isolation.sh` to assert cleanliness:
```bash
#!/usr/bin/env bash
set -euo pipefail
TRADING_CORES="2 3"
NIC_PATTERN="enp130s0f0"
bad=0
grep "$NIC_PATTERN" /proc/interrupts | while read -r line; do
  cols=($line)
  # counts start at index 1, device label near end
  cpu_index=0
  for val in "${cols[@]:1:${#cols[@]}-2}"; do
    val=${val//,/}
    for c in $TRADING_CORES; do
      if [[ $cpu_index -eq $c && $val -ne 0 ]]; then
        echo "ALERT: IRQ activity on trading CPU $c (value=$val)"; bad=1
      fi
    done
    cpu_index=$((cpu_index+1))
  done
done
[[ $bad -eq 0 ]] && echo "OK: No NIC IRQ leakage onto trading cores" || exit 2
```

---
## 10. Troubleshooting Quick Table
| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| Tail latency spike after admin cmd | Wrapper ran w/o no-tune | Re-run admin with `ONLOAD_NO_TUNE=1` |
| Redis pinned wrong core | Service override mismatch | Check `redis-hft.service` CPUAffinity + restart |
| IRQs on trading cores | Driver reset or external script | Reapply NIC affinity script; investigate logs |
| No performance gain | Client not launched via wrapper | Confirm command path / env vars |

---
## 11. Key Takeaways
- **Server stable, clients accelerated** is intentional.
- Use **no-tune** selectively to avoid IRQ churn.
- Maintain a clear CPU role map: IRQ (0/1), Trading (2/3), Redis (4), system/GPU feeders (5–7).
- Keep wrapper launches purposeful—do not wrap every trivial command.

---
## 12. Future (Optional Enhancements)
- Add a tiny systemd timer to run the verify script and log anomalies.
- Introduce an environment whitelist (only apply tuning for approved binaries).
- Parameterize Redis CPU in a single config file for easier migration.

---
**End of Document**
