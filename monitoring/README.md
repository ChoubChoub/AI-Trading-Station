# AI Trading Station - Monitoring System

## Overview and Architecture

The AI Trading Station monitoring system provides comprehensive real-time monitoring of system performance, trading latency, and infrastructure health. The monitoring subsystem is designed to work seamlessly with the high-performance OnLoad trading wrapper while maintaining minimal system overhead.

### Purpose
- Monitor trading system performance and detect anomalies
- Provide real-time dashboard for traders and system administrators
- Track OnLoad wrapper performance and system optimization
- Ensure trading cores remain isolated and optimized
- Alert on performance degradation or configuration drift

### Main Components

The monitoring system consists of two primary scripts that work together:

#### 1. `monitor_dashboard_complete.py` (Dashboard/UI)
- **Role**: Real-time visual dashboard and user interface
- **Purpose**: Provides comprehensive visualization of system metrics, trading performance, and alerts
- **Features**:
  - Real-time system performance display
  - OnLoad wrapper status monitoring
  - Core isolation verification
  - Alert system for performance thresholds
  - Metrics export functionality
  - User-friendly interface for traders

#### 2. `monitor_trading_system_optimized.py` (Optimized Monitoring Core)
- **Role**: High-performance data collection engine
- **Purpose**: Collects system metrics with minimal overhead (<1% CPU usage)
- **Features**:
  - Sub-100ms monitoring intervals
  - Trading process monitoring
  - Latency measurement
  - IRQ and interrupt tracking
  - Per-core CPU usage monitoring
  - Anomaly detection

#### 3. Configuration Management
- **File**: `monitoring_config.json`
- **Role**: Centralized configuration for all monitoring components
- **Contents**: Alert thresholds, monitoring intervals, display settings, and integration options

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Trading Station                      │
│                    Monitoring System                       │
└─────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
    ┌───────────▼───────────┐  │  ┌──────────▼──────────┐
    │  monitor_dashboard_   │  │  │ monitor_trading_    │
    │  complete.py          │  │  │ system_optimized.py │
    │  (Dashboard/UI)       │  │  │ (Core Monitor)      │
    └───────────────────────┘  │  └─────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────┐
    │           monitoring_config.json                    │
    │           (Configuration Management)                │
    └─────────────────────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────┐
    │         ../scripts/onload-trading                   │
    │         (Performance Wrapper - Single Source)      │
    └─────────────────────────────────────────────────────┘
```

**Data Flow:**
1. `monitor_trading_system_optimized.py` collects raw metrics
2. Configuration from `monitoring_config.json` controls behavior
3. `monitor_dashboard_complete.py` displays real-time data
4. Both scripts monitor the OnLoad wrapper status and performance
5. All components reference the single onload-trading wrapper by absolute path

## OnLoad Trading Wrapper (onload-trading)

### Performance Wrapper Overview
The OnLoad trading wrapper is the core performance component that enables the AI Trading Station to achieve its **4.37μs mean trading latency**. This wrapper provides:

- **Kernel Bypass**: OnLoad technology bypasses the Linux kernel networking stack
- **CPU Isolation**: Dedicated CPU cores (2,3) for trading processes
- **Zero-Latency Polling**: Eliminates interrupt-driven I/O overhead
- **Production Safety**: Comprehensive checks and fallback modes

### Single Source of Truth
**CRITICAL**: The OnLoad wrapper has a single authoritative location:

```
ai-trading-station/scripts/onload-trading
```

**Important Guidelines:**
- ❌ **NEVER** duplicate this wrapper in other subfolders
- ✅ **ALWAYS** reference it by absolute path
- ✅ All monitoring scripts use the absolute path to avoid conflicts
- ✅ Maintain a single version to ensure consistency

### Wrapper Integration
The monitoring system integrates with the OnLoad wrapper by:
- Checking wrapper availability and status
- Monitoring processes launched through the wrapper
- Verifying CPU core isolation configured by the wrapper
- Tracking performance metrics achieved by the wrapper

## Launching the Monitoring Dashboard

### Using the monitordash Alias

The fastest way to launch the monitoring dashboard is using the `monitordash` alias:

```bash
# Launch the interactive monitoring dashboard
monitordash
```

**What the alias does:**
- Executes `python3 /path/to/monitoring/monitor_dashboard_complete.py`
- Loads configuration from `monitoring_config.json`
- Starts real-time monitoring with 2-second refresh intervals
- Displays system metrics, OnLoad status, and alerts

### Manual Launch Commands

You can also launch the dashboard explicitly from any directory:

```bash
# From repository root
python3 monitoring/monitor_dashboard_complete.py

# With custom configuration
python3 monitoring/monitor_dashboard_complete.py --config /path/to/custom_config.json

# Export metrics to file and exit
python3 monitoring/monitor_dashboard_complete.py --export /tmp/metrics.json
```

### Shell Wrapper/Alias Approach

The alias approach is specifically designed for traders who need rapid access during trading sessions:

**Setup (one-time):**
```bash
# Make aliases available in current session
source monitoring/setup_monitoring.sh

# Make aliases permanent (add to ~/.bashrc)
echo "source $(pwd)/monitoring/setup_monitoring.sh" >> ~/.bashrc
```

**Available aliases:**
- `monitordash` - Launch interactive dashboard
- `monitorcore` - Launch optimized core monitor
- `monitorstatus` - Quick status export
- `monitor_quick` - Fast system check

## Configuration Management

### Configuration File Location
The monitoring configuration is stored in:
```
monitoring/monitoring_config.json
```

### Configuration Structure
The configuration file contains several sections:

#### Dashboard Settings
```json
{
  "dashboard": {
    "refresh_interval_seconds": 2,
    "dashboard_title": "AI Trading Station Monitor",
    "show_detailed_metrics": true,
    "display_width": 120,
    "max_history_records": 1000
  }
}
```

#### Alert Thresholds
```json
{
  "alert_thresholds": {
    "cpu_usage_percent": 80,
    "memory_usage_percent": 85,
    "latency_threshold_us": 10.0,
    "isolated_core_usage_percent": 5
  }
}
```

#### Trading Infrastructure
```json
{
  "trading_infrastructure": {
    "trading_cores": [2, 3],
    "onload_wrapper_path": "../scripts/onload-trading",
    "expected_performance": {
      "strict_mode_latency_us": 4.37,
      "onload_only_latency_us": 8.5
    }
  }
}
```

### Updating Configuration

1. **Edit the configuration file:**
   ```bash
   nano monitoring/monitoring_config.json
   ```

2. **Common customizations:**
   - Adjust refresh intervals for different monitoring frequencies
   - Modify alert thresholds based on system characteristics
   - Configure trading cores for different CPU configurations
   - Enable/disable specific monitoring features

3. **Validate configuration:**
   ```bash
   # Test configuration with dashboard
   monitordash --config monitoring/monitoring_config.json
   ```

### Configuration Best Practices
- Keep alert thresholds conservative for trading environments
- Use faster refresh intervals (0.1s) for core monitoring during active trading
- Use slower intervals (2-5s) for dashboard display to reduce overhead
- Ensure trading_cores match your actual CPU isolation configuration

## Synchronizing the Monitoring Folder

### Git Operations for Monitoring

To keep your local monitoring system synchronized with the GitHub repository:

#### Pull Latest Changes
```bash
# From repository root
git pull origin main

# Verify monitoring files are updated
ls -la monitoring/
```

#### Push Local Changes
```bash
# Stage monitoring changes
git add monitoring/

# Commit with descriptive message
git commit -m "Update monitoring configuration and scripts"

# Push to repository
git push origin main
```

#### Check Monitoring Status
```bash
# Check which monitoring files have changes
git status monitoring/

# See differences in monitoring files
git diff monitoring/
```

### Synchronization Best Practices

1. **Before trading sessions:**
   ```bash
   # Ensure you have latest monitoring code
   git pull origin main
   source monitoring/setup_monitoring.sh
   ```

2. **After configuration changes:**
   ```bash
   # Test changes locally first
   monitordash --config monitoring/monitoring_config.json
   
   # If working correctly, commit and push
   git add monitoring/monitoring_config.json
   git commit -m "Update monitoring thresholds for production"
   git push origin main
   ```

3. **Regular synchronization:**
   ```bash
   # Weekly sync recommended
   git pull origin main
   git push origin main  # if you have local changes
   ```

### Backup and Recovery
```bash
# Backup current configuration
cp monitoring/monitoring_config.json monitoring/monitoring_config.json.backup

# Restore from backup if needed
cp monitoring/monitoring_config.json.backup monitoring/monitoring_config.json
```

## Usage Notes and Best Practices

### Best Practices for Script Usage

#### Dashboard Usage
1. **During trading hours:**
   - Use `monitordash` for real-time monitoring
   - Keep refresh interval at 2-5 seconds to minimize overhead
   - Monitor the "ALERTS" section for performance issues

2. **For analysis:**
   - Export metrics for post-trading analysis
   - Use longer monitoring periods for trend analysis
   - Archive metrics data for compliance

#### Core Monitor Usage
1. **High-frequency monitoring:**
   ```bash
   # Monitor during active trading (0.1s intervals)
   monitorcore --config monitoring/monitoring_config.json
   ```

2. **Data collection:**
   ```bash
   # Collect 5 minutes of detailed metrics
   monitorcore --export /tmp/trading_metrics.json --duration 300
   ```

### Wrapper Invocation Best Practices

1. **Always use absolute paths:**
   ```bash
   # Correct - absolute path
   /path/to/ai-trading-station/scripts/onload-trading ./trading-app

   # Incorrect - relative path (can cause issues)
   ../scripts/onload-trading ./trading-app
   ```

2. **Check wrapper status before trading:**
   ```bash
   # Verify wrapper is available and OnLoad is working
   monitor_quick
   ```

3. **Monitor wrapper performance:**
   ```bash
   # Track applications launched through wrapper
   monitorcore --daemon
   ```

### System Maintainability

#### File Organization
- Keep monitoring scripts in `monitoring/` directory only
- Use configuration file for all customizable settings
- Document any system-specific modifications

#### Performance Considerations
- Core monitor uses <1% CPU overhead when properly configured
- Dashboard refresh rate should be balanced against system load
- Use export functionality for detailed analysis rather than continuous high-frequency monitoring

#### Integration Guidelines
- All monitoring tools reference the single onload-trading wrapper
- Configuration changes should be tested before production use
- Alert thresholds should be tuned to your specific hardware

### Troubleshooting Common Issues

1. **OnLoad not detected:**
   ```bash
   # Check OnLoad installation
   which onload
   
   # Verify wrapper permissions
   ls -la scripts/onload-trading
   chmod +x scripts/onload-trading
   ```

2. **Permission errors:**
   ```bash
   # Make monitoring scripts executable
   source monitoring/setup_monitoring.sh
   ```

3. **Configuration errors:**
   ```bash
   # Validate JSON configuration
   python3 -m json.tool monitoring/monitoring_config.json
   ```

4. **Missing Python dependencies:**
   ```bash
   # Install required packages
   pip install psutil
   ```

## Architecture Diagram

### Text-Based System Architecture

```
AI Trading Station Monitoring Architecture
==========================================

┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                          │
├─────────────────────────────────────────────────────────────────┤
│  monitordash (alias) -> monitor_dashboard_complete.py          │
│  ├── Real-time dashboard display                               │
│  ├── Alert visualization                                       │
│  ├── OnLoad wrapper status                                     │
│  └── System metrics overview                                   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Configuration
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  monitoring_config.json                                        │
│  ├── Alert thresholds                                          │
│  ├── Monitoring intervals                                      │
│  ├── Trading core configuration                                │
│  └── Integration settings                                      │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Data Collection
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING CORE                             │
├─────────────────────────────────────────────────────────────────┤
│  monitor_trading_system_optimized.py                          │
│  ├── <100ms metric collection                                  │
│  ├── Trading process monitoring                                │
│  ├── Latency measurements                                      │
│  ├── IRQ/interrupt tracking                                    │
│  └── Anomaly detection                                         │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Performance Monitoring
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                 TRADING INFRASTRUCTURE                          │
├─────────────────────────────────────────────────────────────────┤
│  ../scripts/onload-trading (SINGLE SOURCE OF TRUTH)           │
│  ├── OnLoad kernel bypass (4.37μs latency)                    │
│  ├── CPU isolation (cores 2,3)                                │
│  ├── Zero-latency polling                                     │
│  └── Production safety checks                                  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Hardware Monitoring
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  ├── CPU Core Isolation Status                                 │
│  ├── Memory Usage Tracking                                     │
│  ├── Network Interface Monitoring                              │
│  ├── IRQ Affinity Verification                                 │
│  └── OnLoad Driver Status                                      │
└─────────────────────────────────────────────────────────────────┘

Data Flow:
==========
1. System Layer → Monitoring Core (metrics collection)
2. Configuration Layer → Both Dashboard and Core (settings)
3. Monitoring Core → Dashboard (real-time data)
4. Trading Infrastructure ← Monitoring Core (status checks)
5. User Interface ← Dashboard (visualization)

Performance Impact:
==================
├── Dashboard: ~0.1% CPU (2s refresh rate)
├── Core Monitor: <1% CPU (100ms collection)
├── OnLoad Wrapper: 0% overhead (kernel bypass)
└── Total System Impact: <2% CPU overhead
```

### Component Interaction Flow

1. **Startup Sequence:**
   - User executes `monitordash` alias
   - Dashboard loads configuration from `monitoring_config.json`
   - Core monitor starts background data collection
   - OnLoad wrapper status is verified

2. **Runtime Operation:**
   - Core monitor collects metrics every 100ms
   - Dashboard updates display every 2 seconds
   - Alert system checks thresholds continuously
   - OnLoad wrapper performance is tracked

3. **Data Pipeline:**
   - Raw system metrics → Core monitor processing
   - Processed data → Dashboard visualization
   - Alerts → User notification
   - Historical data → Export functionality

### Future Enhancements

This architecture supports future additions:
- **Integration Layer**: Redis, Prometheus, webhook alerts
- **Analytics Layer**: Machine learning anomaly detection
- **Visualization Layer**: Web-based dashboards, Grafana integration
- **API Layer**: REST API for external monitoring systems

---

## Quick Reference

### Essential Commands
```bash
# Setup monitoring (one-time)
source monitoring/setup_monitoring.sh

# Launch dashboard
monitordash

# Quick status check
monitor_quick

# Export metrics
monitorcore --export /tmp/metrics.json --duration 60

# Check OnLoad wrapper
ls -la scripts/onload-trading
```

### Key Files
- `monitoring/monitor_dashboard_complete.py` - Dashboard UI
- `monitoring/monitor_trading_system_optimized.py` - Core monitor
- `monitoring/monitoring_config.json` - Configuration
- `scripts/onload-trading` - Performance wrapper (SINGLE SOURCE)

### Support
For issues or questions about the monitoring system, check:
1. Configuration file syntax: `python3 -m json.tool monitoring/monitoring_config.json`
2. OnLoad wrapper status: `monitor_quick`
3. Log files: `/tmp/trading_monitor.log`
4. Repository documentation: This README.md