#!/bin/bash
# Alert Rules Setup for Market Data Monitoring
# Configures Grafana alerts for critical system events

set -e

echo "=========================================="
echo "  Market Data Alert Rules Setup"
echo "=========================================="
echo ""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ALERT_DIR="/etc/grafana/provisioning/alerting"

echo -e "${YELLOW}🚨 Creating Alert Rules...${NC}"
echo ""

# Create alert provisioning directory
sudo mkdir -p "$ALERT_DIR"

# Alert Rule 1: Low Capture Rate
echo "Creating Low Capture Rate alert..."
cat << 'EOF' | sudo tee "$ALERT_DIR/market-data-alerts.yaml"
apiVersion: 1

groups:
  - name: market_data_alerts
    interval: 1m
    folder: Market Data
    rules:
      - uid: low_capture_rate
        title: Low Capture Rate
        condition: A
        data:
          - refId: A
            relativeTimeRange:
              from: 300
              to: 0
            datasourceUid: prometheus
            model:
              expr: (sum(rate(ticks_inserted_total[1m])) / sum(rate(messages_received_total[1m]))) * 100 < 95
              refId: A
        noDataState: NoData
        execErrState: Error
        for: 2m
        annotations:
          description: "Capture rate is {{ $value }}%, below 95% threshold"
          summary: "Market data capture rate critically low"
        labels:
          severity: critical
          
      - uid: no_messages_received
        title: No Messages Received
        condition: A
        data:
          - refId: A
            relativeTimeRange:
              from: 120
              to: 0
            datasourceUid: prometheus
            model:
              expr: sum(rate(messages_received_total[1m])) == 0
              refId: A
        noDataState: Alerting
        execErrState: Error
        for: 1m
        annotations:
          description: "No messages received in the last 60 seconds"
          summary: "Data collection appears to have stopped"
        labels:
          severity: critical
          
      - uid: high_reconnections
        title: High WebSocket Reconnections
        condition: A
        data:
          - refId: A
            relativeTimeRange:
              from: 300
              to: 0
            datasourceUid: prometheus
            model:
              expr: increase(reconnections_total[5m]) > 10
              refId: A
        noDataState: NoData
        execErrState: Error
        for: 1m
        annotations:
          description: "{{ $value }} reconnections in last 5 minutes (threshold: 10)"
          summary: "Excessive WebSocket reconnections detected"
        labels:
          severity: warning
          
      - uid: high_error_rate
        title: High Error Rate
        condition: A
        data:
          - refId: A
            relativeTimeRange:
              from: 300
              to: 0
            datasourceUid: prometheus
            model:
              expr: sum(rate(write_errors_total[5m])) * 300 > 5
              refId: A
        noDataState: NoData
        execErrState: Error
        for: 2m
        annotations:
          description: "{{ $value }} errors in last 5 minutes (threshold: 5)"
          summary: "High error rate in data processing"
        labels:
          severity: warning
          
      - uid: high_memory_usage
        title: High Memory Usage
        condition: A
        data:
          - refId: A
            relativeTimeRange:
              from: 300
              to: 0
            datasourceUid: prometheus
            model:
              expr: process_resident_memory_bytes / 1024 / 1024 > 450
              refId: A
        noDataState: NoData
        execErrState: Error
        for: 5m
        annotations:
          description: "Memory usage: {{ $value }} MB (threshold: 450 MB)"
          summary: "Service approaching memory limit"
        labels:
          severity: warning
          
      - uid: service_down
        title: Service Down
        condition: A
        data:
          - refId: A
            relativeTimeRange:
              from: 60
              to: 0
            datasourceUid: prometheus
            model:
              expr: up{job="market-data"} == 0
              refId: A
        noDataState: Alerting
        execErrState: Error
        for: 30s
        annotations:
          description: "Service {{ $labels.service }} is not responding"
          summary: "Market data service is down"
        labels:
          severity: critical
          
      - uid: high_write_latency
        title: High Write Latency
        condition: A
        data:
          - refId: A
            relativeTimeRange:
              from: 300
              to: 0
            datasourceUid: prometheus
            model:
              expr: histogram_quantile(0.99, rate(write_latency_seconds_bucket[1m])) > 0.010
              refId: A
        noDataState: NoData
        execErrState: Error
        for: 3m
        annotations:
          description: "P99 write latency: {{ $value }}s (threshold: 10ms)"
          summary: "Database write latency degraded"
        labels:
          severity: warning
          
      - uid: low_disk_space
        title: Low Disk Space
        condition: A
        data:
          - refId: A
            relativeTimeRange:
              from: 300
              to: 0
            datasourceUid: prometheus
            model:
              expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) < 0.10
              refId: A
        noDataState: NoData
        execErrState: Error
        for: 5m
        annotations:
          description: "Available disk space: {{ $value | humanizePercentage }} (threshold: 10%)"
          summary: "Critical disk space remaining"
        labels:
          severity: critical
EOF

echo -e "${GREEN}✓ Alert rules created${NC}"
echo ""

# Create notification channel template
echo -e "${YELLOW}📧 Creating Notification Channel Template...${NC}"
echo ""

cat << 'EOF' | sudo tee "$ALERT_DIR/notification-channels.yaml"
apiVersion: 1

# Delete existing notification channels (optional)
deleteContactPoints: []

contactPoints:
  - orgId: 1
    name: "Console Logging"
    receivers:
      - uid: console_logging
        type: webhook
        settings:
          url: http://localhost:8001/alerts
          httpMethod: POST
        disableResolveMessage: false

# Note: Configure additional channels in Grafana UI:
# - Email (SMTP setup required)
# - Slack (webhook URL required)
# - PagerDuty (integration key required)
# - Webhook (custom endpoint)
EOF

echo -e "${GREEN}✓ Notification channel template created${NC}"
echo ""

# Set permissions
sudo chown -R grafana:grafana "$ALERT_DIR"
sudo chmod 644 "$ALERT_DIR"/*.yaml

echo -e "${GREEN}=========================================="
echo "  ✅ Alert Rules Configured!"
echo "==========================================${NC}"
echo ""
echo "🚨 Alert Rules Created:"
echo "   1. Low Capture Rate (<95%)"
echo "   2. No Messages Received (60 seconds)"
echo "   3. High Reconnections (>10 in 5 min)"
echo "   4. High Error Rate (>5 in 5 min)"
echo "   5. High Memory Usage (>450 MB)"
echo "   6. Service Down (30 seconds)"
echo "   7. High Write Latency (p99 > 10ms)"
echo "   8. Low Disk Space (<10%)"
echo "   7. High Write Latency (p99 >10ms)"
echo ""
echo "📧 Notification Channels:"
echo "   Default: Console Logging (localhost:8001/alerts)"
echo ""
echo "🔧 Additional Setup Required:"
echo "   Configure notification channels in Grafana UI:"
echo "   1. Go to Alerting → Contact points"
echo "   2. Add Email, Slack, or other channels"
echo "   3. Test notifications"
echo ""
echo "🔄 Restart Grafana to load alert rules:"
echo "   sudo systemctl restart grafana-server"
echo ""
echo "📍 View Alerts:"
echo "   http://localhost:3000/alerting/list"
echo ""
