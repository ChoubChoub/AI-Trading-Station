# Redis Health Monitor - Grafana Integration Guide

**Date:** October 27, 2025  
**Status:** ✅ Ready to Use

---

## Overview

The Redis Health Monitor now exports Prometheus metrics on **port 9093** and includes a pre-built Grafana dashboard.

### Metrics Available

| Metric | Description | Type |
|--------|-------------|------|
| `redis_connections_total` | Current Redis ESTABLISHED connections | Gauge |
| `redis_latency_milliseconds` | Redis PING latency in ms | Gauge |
| `redis_health_status` | Health status (0=healthy, 1=warning, 2=leak, 3=critical) | Gauge |
| `redis_auto_restarts_total` | Number of automatic batch-writer restarts | Counter |
| `redis_connection_leaks_total` | Number of connection leaks detected | Counter |

---

## Step 1: Verify Metrics Endpoint

```bash
# Check metrics are being exported
curl http://localhost:9093/metrics | grep -E "^redis_"

# Expected output:
# redis_connections_total 24.0
# redis_latency_milliseconds 3.4
# redis_health_status 0.0
# redis_auto_restarts_total 0.0
# redis_connection_leaks_total 0.0
```

---

## Step 2: Configure Prometheus

Add the health monitor to your Prometheus scrape config:

**File:** `/etc/prometheus/prometheus.yml` (or wherever your Prometheus config is)

```yaml
scrape_configs:
  # Existing batch-writer metrics
  - job_name: 'batch-writer'
    static_configs:
      - targets: ['localhost:9092']
  
  # NEW: Redis health monitor metrics
  - job_name: 'redis-health'
    static_configs:
      - targets: ['localhost:9093']
    scrape_interval: 60s  # Matches monitor check interval
```

Then reload Prometheus:

```bash
# If Prometheus is running as systemd service
sudo systemctl reload prometheus

# Or send SIGHUP
sudo pkill -HUP prometheus

# Or restart
sudo systemctl restart prometheus
```

---

## Step 3: Verify Prometheus is Scraping

1. Open Prometheus web UI: `http://localhost:9090`
2. Go to **Status → Targets**
3. You should see `redis-health (1/1 up)` in the targets list
4. Test a query:
   - Go to **Graph**
   - Enter query: `redis_connections_total`
   - Click **Execute**
   - You should see current value (around 24)

---

## Step 4: Import Grafana Dashboard

### Option A: Import via UI (Easiest)

1. Open Grafana: `http://localhost:3000` (or your Grafana URL)
2. Click **+ → Import** in left sidebar
3. Click **Upload JSON file**
4. Select: `/home/youssefbahloul/ai-trading-station/Monitoring/grafana_redis_health_dashboard.json`
5. Select your Prometheus datasource
6. Click **Import**

### Option B: Import via File

1. Copy dashboard to Grafana provisioning folder:

```bash
# If using Grafana provisioning
sudo cp /home/youssefbahloul/ai-trading-station/Monitoring/grafana_redis_health_dashboard.json \
       /etc/grafana/provisioning/dashboards/

# Restart Grafana
sudo systemctl restart grafana-server
```

### Option C: Import via Copy-Paste

1. Open Grafana: `http://localhost:3000`
2. Click **+ → Import**
3. Paste the JSON content from `grafana_redis_health_dashboard.json`
4. Click **Load**
5. Select your Prometheus datasource
6. Click **Import**

---

## Step 5: Dashboard Overview

Once imported, you'll see:

### Top Row - Time Series Graphs
- **Left:** Redis Connection Count over time
- **Right:** Redis Latency (ms) over time

### Middle Row - Current Status Panels
- **Connection Count:** Current value with color coding
  - Green: < 25 (healthy)
  - Yellow: 25-49 (warning)
  - Orange: 50-99 (leak detected)
  - Red: ≥100 (critical)
  
- **Current Latency:** Current Redis response time
  - Green: <2ms (excellent)
  - Yellow: 2-5ms (good)
  - Orange: 5-10ms (slow)
  - Red: >10ms (critical)
  
- **Health Status:** Text status (HEALTHY/WARNING/LEAK/CRITICAL)
- **Auto-Restarts:** Count of automatic restarts
- **Connection Leaks:** Count of detected leaks

### Bottom Row - 24h Trend
- Connection count over last 24 hours
- Shows threshold lines:
  - Yellow dashed: Baseline (25)
  - Orange dashed: Leak threshold (50)
  - Red dashed: Critical threshold (100)

---

## Dashboard Features

### Auto-Refresh
- Dashboard refreshes every 1 minute (configurable)
- Matches health monitor check interval (60 seconds)

### Color Coding
- **Green:** Everything normal
- **Yellow:** Elevated but acceptable
- **Orange:** Problem detected
- **Red:** Critical issue

### Time Range
- Default: Last 6 hours
- Change via time picker (top right)
- Useful ranges:
  - Last 1 hour: Real-time monitoring
  - Last 24 hours: Daily trends
  - Last 7 days: Weekly patterns

---

## Example Queries for Custom Panels

If you want to add custom panels:

```promql
# Connection count
redis_connections_total

# Latency
redis_latency_milliseconds

# Connection growth rate (per hour)
rate(redis_connections_total[1h])

# Average latency (5 min window)
avg_over_time(redis_latency_milliseconds[5m])

# Max connections in last hour
max_over_time(redis_connections_total[1h])

# Check if leak is happening (1=leak, 0=ok)
redis_health_status >= 2

# Total restarts this week
increase(redis_auto_restarts_total[7d])
```

---

## Setting Up Alerts (Future)

To get notified when issues occur, configure Prometheus Alertmanager:

**Example Alert Rules:**

```yaml
groups:
  - name: redis_health
    interval: 60s
    rules:
      - alert: RedisConnectionLeak
        expr: redis_connections_total > 50
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Redis connection leak detected"
          description: "{{ $value }} connections (threshold: 50)"
      
      - alert: RedisConnectionCritical
        expr: redis_connections_total > 100
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis connections CRITICAL"
          description: "{{ $value }} connections - auto-restart triggered"
      
      - alert: RedisHighLatency
        expr: redis_latency_milliseconds > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis latency high"
          description: "Latency: {{ $value }}ms (normal: <2ms)"
```

---

## Troubleshooting

### Metrics Not Showing in Prometheus

```bash
# Check health monitor is running
systemctl status redis-health-monitor

# Check metrics endpoint
curl http://localhost:9093/metrics

# Check Prometheus logs
journalctl -u prometheus -f

# Verify Prometheus config
promtool check config /etc/prometheus/prometheus.yml
```

### Dashboard Shows "No Data"

1. Check Prometheus datasource is configured in Grafana
2. Verify Prometheus is scraping health monitor (Status → Targets)
3. Check time range is recent (default: last 6 hours)
4. Verify queries are correct (click panel → Edit → check query)

### Health Monitor Not Starting

```bash
# Check logs
journalctl -u redis-health-monitor -n 50

# Verify port 9093 is not in use
netstat -tuln | grep 9093

# Check Python dependencies
python3 -c "import prometheus_client"
```

---

## Current Status

✅ **Health monitor running:** `systemctl status redis-health-monitor`  
✅ **Metrics endpoint:** `http://localhost:9093/metrics`  
✅ **Dashboard JSON:** `/home/youssefbahloul/ai-trading-station/Monitoring/grafana_redis_health_dashboard.json`  
✅ **Service location:** `/home/youssefbahloul/ai-trading-station/Monitoring/Scripts/redis_health_monitor.py`

---

## Quick Start

```bash
# 1. Add to Prometheus config
sudo nano /etc/prometheus/prometheus.yml
# Add redis-health job (see Step 2 above)

# 2. Reload Prometheus
sudo systemctl reload prometheus

# 3. Import dashboard to Grafana
# Use Grafana UI: + → Import → Upload grafana_redis_health_dashboard.json

# 4. View dashboard
# Open Grafana → Dashboards → Redis Health Monitor
```

---

**Documentation:** `/home/youssefbahloul/ai-trading-station/Documentation/HEALTH_MONITOR_EXPLAINED.md`  
**Support:** Check systemd logs with `journalctl -u redis-health-monitor -f`
