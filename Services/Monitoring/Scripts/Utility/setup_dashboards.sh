#!/bin/bash
# Dashboard Setup Script for Market Data Monitoring
# Creates and imports Grafana dashboards with alerting

set -e

echo "=========================================="
echo "  Market Data Dashboard Setup"
echo "=========================================="
echo ""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DASHBOARD_DIR="/var/lib/grafana/dashboards"

echo -e "${YELLOW}📊 Creating Market Data Monitoring Dashboards...${NC}"
echo ""

# Dashboard 1: System Overview
echo "Creating System Overview dashboard..."
cat << 'EOF' | sudo tee "$DASHBOARD_DIR/01-system-overview.json"
{
  "title": "Market Data - System Overview",
    "tags": ["market-data", "overview"],
    "timezone": "browser",
    "schemaVersion": 38,
    "version": 1,
    "refresh": "5s",
    "panels": [
      {
        "id": 1,
        "title": "Service Health Status",
        "type": "stat",
        "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
        "targets": [
          {
            "expr": "up{job=\"market-data\"}",
            "legendFormat": "{{service}}",
            "refId": "A"
          }
        ],
        "options": {
          "colorMode": "background",
          "graphMode": "none",
          "orientation": "auto"
        },
        "fieldConfig": {
          "defaults": {
            "mappings": [
              {"type": "value", "value": "1", "text": "UP"},
              {"type": "value", "value": "0", "text": "DOWN"}
            ],
            "thresholds": {
              "steps": [
                {"value": 0, "color": "red"},
                {"value": 1, "color": "green"}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Total Events Ingested",
        "type": "stat",
        "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
        "targets": [
          {
            "expr": "sum(rate(ticks_inserted_total[1m])) * 60",
            "legendFormat": "Events/min",
            "refId": "A"
          }
        ],
        "options": {
          "colorMode": "value",
          "graphMode": "area",
          "orientation": "auto"
        },
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "decimals": 0,
            "thresholds": {
              "steps": [
                {"value": 0, "color": "red"},
                {"value": 1000, "color": "yellow"},
                {"value": 5000, "color": "green"}
              ]
            }
          }
        }
      },
      {
        "id": 3,
        "title": "Capture Rate",
        "type": "gauge",
        "gridPos": {"x": 12, "y": 0, "w": 6, "h": 4},
        "targets": [
          {
            "expr": "(sum(rate(ticks_inserted_total[1m])) / sum(rate(messages_received_total[1m]))) * 100",
            "legendFormat": "Capture %",
            "refId": "A"
          }
        ],
        "options": {
          "showThresholdLabels": true,
          "showThresholdMarkers": true
        },
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                {"value": 0, "color": "red"},
                {"value": 95, "color": "yellow"},
                {"value": 99, "color": "green"}
              ]
            }
          }
        }
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "stat",
        "gridPos": {"x": 18, "y": 0, "w": 6, "h": 4},
        "targets": [
          {
            "expr": "sum(rate(write_errors_total[5m])) * 300",
            "legendFormat": "Errors/5min",
            "refId": "A"
          }
        ],
        "options": {
          "colorMode": "background",
          "graphMode": "area"
        },
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "decimals": 0,
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 1, "color": "yellow"},
                {"value": 10, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 5,
        "title": "Message Rate by Stream Type",
        "type": "timeseries",
        "gridPos": {"x": 0, "y": 4, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "sum by(stream_type) (rate(ticks_inserted_total[1m])) * 60",
            "legendFormat": "{{stream_type}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ops",
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "smooth",
              "fillOpacity": 10
            }
          }
        }
      },
      {
        "id": 6,
        "title": "Message Rate by Symbol",
        "type": "timeseries",
        "gridPos": {"x": 12, "y": 4, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "sum by(symbol) (rate(ticks_inserted_total[1m])) * 60",
            "legendFormat": "{{symbol}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ops",
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "smooth",
              "fillOpacity": 10
            }
          }
        }
      },
      {
        "id": 7,
        "title": "WebSocket Reconnections",
        "type": "timeseries",
        "gridPos": {"x": 0, "y": 12, "w": 12, "h": 6},
        "targets": [
          {
            "expr": "increase(reconnections_total[5m])",
            "legendFormat": "{{service}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "custom": {
              "drawStyle": "bars",
              "fillOpacity": 100
            },
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 5, "color": "yellow"},
                {"value": 10, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 8,
        "title": "Memory Usage",
        "type": "timeseries",
        "gridPos": {"x": 12, "y": 12, "w": 12, "h": 6},
        "targets": [
          {
            "expr": "process_resident_memory_bytes / 1024 / 1024",
            "legendFormat": "{{service}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "decmbytes",
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "smooth",
              "fillOpacity": 10
            },
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 400, "color": "yellow"},
                {"value": 450, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 9,
        "title": "Active Symbols Health",
        "type": "stat",
        "gridPos": {"x": 0, "y": 18, "w": 24, "h": 4},
        "targets": [
          {
            "expr": "count by(symbol) (rate(ticks_inserted_total[1m]) > 0)",
            "legendFormat": "{{symbol}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "custom": {
              "displayMode": "gradient-gauge"
            },
            "thresholds": {
              "steps": [
                {"value": 0, "color": "red"},
                {"value": 1, "color": "green"}
              ]
            }
          }
        },
        "options": {
          "orientation": "horizontal",
          "textMode": "value_and_name",
          "colorMode": "background"
        }
      }
    ]
}
EOF

echo -e "${GREEN}✓ System Overview dashboard created${NC}"

# Dashboard 2: Performance Metrics
echo "Creating Performance Metrics dashboard..."
cat << 'EOF' | sudo tee "$DASHBOARD_DIR/02-performance-metrics.json"
{
  "title": "Market Data - Performance Metrics",
    "tags": ["market-data", "performance"],
    "timezone": "browser",
    "schemaVersion": 38,
    "version": 1,
    "refresh": "10s",
    "templating": {
      "list": [
        {
          "name": "symbol",
          "label": "Symbol",
          "type": "query",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus-market-data"
          },
          "query": "label_values(ticks_inserted_total, symbol)",
          "multi": true,
          "includeAll": true,
          "allValue": ".*",
          "refresh": 1
        }
      ]
    },
    "panels": [
      {
        "id": 1,
        "title": "Write Latency (p50, p99)",
        "type": "timeseries",
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(write_latency_seconds_bucket[1m]))",
            "legendFormat": "p50",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.99, rate(write_latency_seconds_bucket[1m]))",
            "legendFormat": "p99",
            "refId": "B"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "smooth"
            },
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 0.001, "color": "yellow"},
                {"value": 0.010, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Batch Size Distribution",
        "type": "timeseries",
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(batch_size_bucket[1m]))",
            "legendFormat": "p50",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.95, rate(batch_size_bucket[1m]))",
            "legendFormat": "p95",
            "refId": "B"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "smooth"
            }
          }
        }
      },
      {
        "id": 3,
        "title": "Processing Rate (Events/sec)",
        "type": "timeseries",
        "gridPos": {"x": 0, "y": 8, "w": 24, "h": 8},
        "targets": [
          {
            "expr": "sum(rate(ticks_inserted_total[1m]))",
            "legendFormat": "Total Throughput",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ops",
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "smooth",
              "fillOpacity": 20
            },
            "thresholds": {
              "steps": [
                {"value": 0, "color": "red"},
                {"value": 1000, "color": "yellow"},
                {"value": 5000, "color": "green"}
              ]
            }
          }
        }
      },
      {
        "id": 4,
        "title": "CPU Usage",
        "type": "timeseries",
        "gridPos": {"x": 0, "y": 16, "w": 12, "h": 6},
        "targets": [
          {
            "expr": "rate(process_cpu_seconds_total[1m]) * 100",
            "legendFormat": "{{service}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "smooth"
            },
            "max": 200
          }
        }
      },
      {
        "id": 5,
        "title": "Network I/O",
        "type": "timeseries",
        "gridPos": {"x": 12, "y": 16, "w": 12, "h": 6},
        "targets": [
          {
            "expr": "rate(network_bytes_received_total[1m])",
            "legendFormat": "Received",
            "refId": "A"
          },
          {
            "expr": "rate(network_bytes_sent_total[1m])",
            "legendFormat": "Sent",
            "refId": "B"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "Bps",
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "smooth"
            }
          }
        }
      }
    ]
}
EOF

echo -e "${GREEN}✓ Performance Metrics dashboard created${NC}"

# Set permissions
sudo chown -R grafana:grafana "$DASHBOARD_DIR"
sudo chmod 644 "$DASHBOARD_DIR"/*.json

echo ""
echo -e "${GREEN}=========================================="
echo "  ✅ Dashboards Created Successfully!"
echo "==========================================${NC}"
echo ""
echo "📊 Dashboards Available:"
echo "   1. System Overview - Real-time health monitoring"
echo "   2. Performance Metrics - Latency and throughput"
echo ""
echo "📍 Access:"
echo "   URL: http://localhost:3000"
echo "   Navigate to: Dashboards → Market Data folder"
echo ""
echo "🔄 Restart Grafana to load dashboards:"
echo "   sudo systemctl restart grafana-server"
echo ""
