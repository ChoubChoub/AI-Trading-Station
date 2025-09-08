# AI Trading Station

## Overview
This project is a cost-efficient, medium-frequency algorithmic trading platform designed to execute strategies on 1-minute to 1-hour timeframes using tick-level market data. It employs a hybrid competitive-collaborative multi-agent architecture for intelligent decision-making, executing orders in ≤ 50ms. The platform leverages dual RTX 6000 Pro configurations to combine institutional-grade intelligence with retail-level costs.

## Key Features
- **Tick-Level Data Utilization**: Employs tick-level data to implement strategies that operate on 1-minute to 1-hour frequencies, ensuring precise and timely decision-making.
- **Hybrid Architecture**: Implements a competitive-collaborative framework that delivers 85-90% of the benefits of a pure competitive framework with reduced operational risk.
- **AI-Driven Intelligence**: Focuses on competitive performance through advanced AI rather than pure speed.
- **High VRAM Capacity**: Utilizes 192GB total VRAM (2× RTX 6000 Pro with 96GB each) for strategic model selection and phased competitive implementation.
- **Evidence-Based Innovation**: Ensures sustainable trading performance through an evidence-based approach rather than relying solely on architectural sophistication.

## Multi-Agent System Architecture
### Strategic Philosophy
Recent studies show that hybrid systems combining competition and collaboration among trading agents can effectively balance innovation with operational stability. Our platform applies this principle through a phased competitive framework: specialized agents participate in weekly tournaments, where their trading impact is initially capped at 10% of total capital. Agents are tested through adversarial validation, and capital is allocated according to performance. This structure ensures constant competitive pressure to innovate while preserving overall portfolio stability and coordination. Over time, the allocation cap may be raised, provided operational robustness is maintained.

### Competitive Framework Implementation
- **Foundation with Limited Competition**: Weekly tournament with 4 strategy agents.
- **Competitive allocation**: Limited to 10% of total capital.
- **Tournament Mechanics**:
  - Each agent generates predictions for next trading session.
  - Challenge Generator identifies weaknesses in each strategy's reasoning.
  - Agents must defend their positions against challenges.
  - Performance scoring with challenge resilience.
  - Capital allocation based on tournament results (capped at 10% initially)
- **Initial Capital Allocation Rules**:
   - Base allocation (90%) proportional to recent risk-adjusted performance (70% weight).
   - Competitive allocation (10% max) based on tournament results.
   - Innovation bonus for novel approaches showing promise (within competitive allocation).
   - Diversity adjustment to prevent strategy homogeneity (within competitive allocation).

### Agent Communication Architecture
- **Primary Bus**: Redis Streams with UNIX domain sockets for <2μs inter-agent latency
- **Event Types**: MarketEvent, SignalEvent, RiskAlert, OrderEvent, FillEvent, ChallengeEvent,
DefenseEvent, TournamentResultEvent
- **Message Format**: JSON with timestamp, agent_id, confidence_score, competitive_score, and
payload
- **Consumer Groups**: Each agent subscribes to relevant streams with automatic
acknowledgment
- **Persistence**: All events written to QuestDB with SHA-256 hash chain for audit trail
- **Competition-Specific Enhancements**:
   - ChallengeEvent: Contains adversarial questions targeting strategy weaknesses.
   - DefenseEvent : Contains strategy agent's counter-arguments.
   - TournamentResultEvent : Contains final capital allocation decisions

## Hardware Architecture and Configuration
### Hardware Specification

| **Component** | **Specification**                  | **Configuration Details**                                | **Rationale**                                      |
|---------------|------------------------------------|----------------------------------------------------------|----------------------------------------------------|
| **CPU**       | Intel Ultra 9 285K                 | 8 P-cores @ 5.7GHz, E-cores disabled in BIOS             | Deterministic latency, eliminates scheduling jitter |
| **GPU**      | 2 × RTX 6000 Pro Max-Q             | 96GB GDDR7 each, 300W power draw per unit                | Combined 192GB VRAM enables 30-70B parameter models |
| **RAM**       | 160GB DDR5-6000 CL30               | 2×48GB + 2×32GB Corsair Vengeance sticks                 | Matches GPU VRAM capacity for efficient data preprocessing |
| **Motherboard**| ASUS ROG Maximus Extreme Z890     | 2×PCIe 4.0 x16, 4×DIMM DDR5 slots                        | Supports dual GPU and full RAM configuration        |
| **NIC**       | Solarflare X2522 10GbE             | OpenOnload kernel bypass capability                      | Sub-10μs network latency with user-space TCP/UDP    |
| **Storage**   | Primary: 4TB Samsung 990 Pro NVMe | 7,350MB/s sequential read/write                          | Real-time data processing and model checkpoints     |
|               | Archive: 8TB Seagate HDD          | 5,400 RPM for cold storage                               | Historical data and backup storage                  |
| **PSU**       | Corsair HX1600i 80+ Platinum       | 1,200W capacity with 300W headroom                       | Handles dual 300W GPUs plus CPU and peripherals     |
| **Cooling**   | Arctic Liquid Freezer III 360 AIO  | CPU cooling with 3×120mm Arctic P12 Pro fans             | Maintains <80°C CPU temps under sustained load      |
| **Router**    | Ubiquiti Cloud Gateway Fiber       | Enterprise-grade routing                                 | Low-jitter WAN connectivity                         |

### Operating System and Low-Latency Optimization

**Operating System**: 
- Ubuntu Server 24.04.3 LTS with Minimal GUI Setup for Low-Latency, providing just enough desktop functionality for monitoring, management, or launching trading applications, while minimizing background activity and resource usage.
- **Display Manager:** [LightDM](https://github.com/canonical/lightdm) is used as the lightweight login/session manager, providing fast startup and minimal overhead.
- **X Server:** [Xorg](https://www.x.org/wiki/) is selected for its compatibility and configurability with most graphical environments and remote tools.
- **Desktop Environment:** [XFCE](https://www.xfce.org/) is chosen for its lightweight footprint and efficient resource usage, running only the essential panel, window manager, and desktop components.
- **Minimal Overhead:** Only the necessary GUI components are installed and running, reducing CPU usage and background “noise” on the system.
- **Fast Startup:** LightDM and XFCE are optimized for quick login and minimal graphical bloat.
- **Configuration Flexibility:** XFCE allows for easy customization, and components like compositing, notifications, and desktop effects can be disabled for additional performance gains.

**Low Latency Optimizations**: 
- **IRQ Affinity Isolation:** Ensures that network interrupts are strictly confined to dedicated housekeeping CPU cores (0 and 1), preventing unpredictable latency spikes on trading-critical cores.
  - **Automatic NIC Detection:** The `configure-nic-irq-affinity.sh` script auto-detects the system’s primary network interface card (NIC) by parsing the default routing table, ensuring all relevant network paths are covered.
  - **Comprehensive IRQ Mapping:** All IRQs associated with the detected NIC are discovered and logged. These are then distributed in a round-robin fashion between CPU cores 0 and 1.
  - **CPU Affinity Enforcement:** The script applies affinity masks so that only cores 0 and 1 can process NIC interrupts, fully isolating the trading application’s CPU cores (2, 3, …) from network-layer interruptions.
  - **Systemd Integration (Auto-Start at Boot):** The script is managed by a dedicated systemd service unit. It runs automatically at system boot, after all network interfaces are loaded, ensuring reliable and repeatable IRQ configuration on every restart.
  - **Production Safety:** Built-in checks guarantee target cores are online and available before any affinity changes are applied, minimizing operational risk.
  - **Audit Logging:** Every configuration change is logged to `/var/log/nic-irq-affinity.log` for transparency and troubleshooting.

- **Process Isolation:** All GUI processes (LightDM, Xorg, XFCE components) are pinned to specific CPU cores (e.g., 0 and 1) using CPU affinity directive in `/etc/systemd/system.conf`. This guarantees that routine OS services, desktop environments, daemons, and background processes do not run on isolated (trading) cores.

- **OnLoad Trading Wrapper:** Launches trading workloads with guaranteed CPU and network isolation for ultra-low latency execution.
  - **Trading Core Pinning:** Binds trading applications to dedicated, isolated CPU cores (typically 2 and 3), ensuring no background or OS processes interfere with critical computation.
  - **Safety Checks:** Before launch, the script verifies that the configured trading cores exist and are online, preventing misconfiguration or resource contention.
  - **OnLoad Kernel Bypass:** Utilizes Solarflare OnLoad to bypass the Linux kernel network stack, enabling direct user-space networking for sub-microsecond packet processing and minimal jitter.
  - **OnLoad Network Acceleration:** Automatically applies a suite of Solarflare OnLoad environment variables (e.g., `EF_POLL_USEC=0`, queue sizing, spin tuning) to maximize network throughput, minimize latency, and bypass the kernel network stack where possible.
  - **Flexible Fallback Modes:** Supports automatic (`auto`, the default), onload-only, or strict operation if CPU pinning is unavailable, so trading processes always launch safely and optimally.  
    - **Default Mode:** If CPU pinning is not possible, the script defaults to `auto` mode and will run the trading application with OnLoad but without CPU affinity pinning.
  - **User Configurable:** Trading cores, fallback behavior, and debug output can all be set via environment variables, making adaptation to new deployments simple and robust.
  - **How to Launch:** Run your trading application via:```bash scripts/onload-trading <your_trading_binary> [args...]```
  - **Result:** Trading engines are launched in a deterministic, interference-free environment, achieving sub-microsecond network operations and consistently predictable performance thanks to full kernel bypass and CPU isolation.

- **Kernel Optimizations:** Disables or tunes kernel features that can introduce microseconds of latency, while supporting large bursts and minimizing drops. FIle is at `/etc/sysctl.d/99-solarflare-trading.conf`.
  - **Large socket and TCP buffers:** Supports high burst rates and minimizes packet loss.
  - **Busy polling:** Reduces interrupt latency (at the cost of higher CPU, but appropriate for HFT).
  - **Explicit TCP feature selection:** Ensures SACK, window scaling, and FACK are on for optimal TCP robustness.
  - **Reduced swapping:** Maximizes RAM use for trading workloads.
  - **Short timeouts:** Reduces FIN_WAIT state time and allows TIME_WAIT reuse for connection churn.
  - **High listen backlog:** Prevents connection drops under fast connection churn or spikes.
  - **UDP tuning:** Ensures minimal drops for UDP-based market data feeds.


**Monitoring Core Isolation, IRQ Affinity, and Hardware Health**

Ensuring the performance and reliability of a low-latency trading platform requires continuous monitoring of several critical system KPIs. This monitoring framework is designed to provide early detection of configuration drift, resource contention, or hardware anomalies that could jeopardize latency, determinism, or system stability.

- **Core Isolation & IRQ Affinity Monitoring:**

**Objective**: Guarantee that isolated CPU cores (e.g., cores 2 and 3) remain free from OS, user, and device interrupt noise, preserving them exclusively for latency-sensitive trading workloads.
Key KPIs:
CPU Utilization per Core: Cores assigned for trading should exhibit near 100% idle when the application is not running, and show no irq or softirq activity.
IRQ Event Distribution: All network (NIC) and other relevant hardware interrupts must be pinned to non-isolated cores (e.g., cores 0 and 1). Regular checks of /proc/interrupts ensure that no IRQ events are handled by isolated cores.
Process Affinity: Only essential kernel threads should appear on isolated cores outside trading application runtime.

- **Hardware Health Monitoring:**

**Objective**: Prevent performance degradation or hardware failure by tracking vital system temperatures and other health metrics.
Key KPIs:
CPU Temperature: Continuous polling of CPU thermal sensors to detect overheating.
GPU Temperature (if applicable): Monitoring via vendor utilities (e.g., nvidia-smi for NVIDIA GPUs).
Fan Speeds, Power Consumption: Optional, for early detection of cooling or power delivery issues.

- **Implementation Considerations:**

Automated Scripts: Deploy custom or open-source scripts to check isolation status, IRQ distribution, and sensor data, with results logged and/or visualized.
Alerting: Integrate with monitoring platforms (e.g., Prometheus, Zabbix, Nagios) to trigger alerts if KPIs fall outside acceptable ranges (e.g., IRQs on isolated cores, excessive temperatures).
Audit and Validation: Regularly validate that systemd services and boot-time configurations enforce the intended affinity and isolation settings after reboots or upgrades.

- **Operational Benefits:**

Reduced Latency and Jitter: By preventing interrupt and process pollution of trading cores, the system maintains deterministic performance.
Proactive Issue Detection: Early warnings on hardware or configuration drift minimize downtime and performance incidents.
Auditability: Persistent monitoring logs and reports provide compliance and troubleshooting evidence.

## Transferring OnLoad Configuration from PROD to VM

To safely copy your critical OnLoad configuration file from your production machine to your development VM, use the following SCP command. This ensures you can inspect the configuration before activating it on your VM.

```bash
scp -P 2222 /etc/onload.conf youssefbahloul@localhost:~/ai-trading-station/config/onload.conf.prod
```

**Notes:**
- Replace `youssefbahloul` with your VM username if different.
- This command assumes your VM is accessible via SSH on port 2222 (`ssh -p 2222 username@localhost`).
- The config file will be placed in your project directory on the VM (`~/ai-trading-station/config/onload.conf.prod`).
- After transfer, inspect the file to ensure correctness:
  ```bash
  cat ~/ai-trading-station/config/onload.conf.prod
  ```
- To activate the config on your VM, copy it to the system location:
  ```bash
  sudo cp ~/ai-trading-station/config/onload.conf.prod /etc/onload.conf
  ```

This procedure keeps your production environment safe and ensures your VM uses the correct OnLoad configuration for development and testing.
