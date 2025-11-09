# Grafana Enhanced Panels - SLA Lines & Heatmaps

**Date:** October 26, 2025  
**Enhancement:** Adding SLA threshold lines and heatmap visualizations  
**Status:** Ready for implementation

---

## 📊 Enhanced Panel Configurations

### **1. Write Latency with SLA Lines (Performance Dashboard)**

**Panel: Write Latency (p50, p99) - ENHANCED**

```json
{
  "title": "Write Latency with SLA Thresholds",
  "type": "graph",
  "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
  "targets": [
    {
      "expr": "histogram_quantile(0.50, sum(rate(batch_writer_latency_seconds_bucket[5m])) by (le)) * 1000",
      "legendFormat": "P50 Latency",
      "refId": "A"
    },
    {
      "expr": "histogram_quantile(0.99, sum(rate(batch_writer_latency_seconds_bucket[5m])) by (le)) * 1000",
      "legendFormat": "P99 Latency",
      "refId": "B"
    }
  ],
  "yaxes": [
    {
      "format": "ms",
      "label": "Latency (milliseconds)",
      "min": 0,
      "max": null
    }
  ],
  "thresholds": [
    {
      "value": 5,
      "colorMode": "custom",
      "fill": false,
      "line": true,
      "lineColor": "rgba(237, 129, 40, 0.7)",
      "op": "gt",
      "yaxis": "left"
    },
    {
      "value": 10,
      "colorMode": "custom",
      "fill": true,
      "fillColor": "rgba(245, 54, 54, 0.15)",
      "line": true,
      "lineColor": "rgba(245, 54, 54, 0.9)",
      "op": "gt",
      "yaxis": "left"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "custom": {
        "lineWidth": 2,
        "fillOpacity": 10
      },
      "color": {
        "mode": "palette-classic"
      }
    }
  },
  "alert": {
    "name": "High Write Latency",
    "conditions": [
      {
        "evaluator": {
          "params": [10],
          "type": "gt"
        },
        "operator": {
          "type": "and"
        },
        "query": {
          "params": ["B", "5m", "now"]
        },
        "reducer": {
          "params": [],
          "type": "avg"
        },
        "type": "query"
      }
    ],
    "executionErrorState": "alerting",
    "frequency": "1m",
    "handler": 1,
    "message": "P99 write latency exceeds 10ms SLA",
    "noDataState": "no_data",
    "notifications": []
  }
}
```

**Visual Features:**
- 🟠 **Yellow line at 5ms** (warning threshold)
- 🔴 **Red line at 10ms** (critical SLA)
- 🔴 **Red shaded area above 10ms** (SLA violation)
- 🔔 **Auto-alert** if P99 > 10ms for 5 minutes

---

### **2. Write Latency Heatmap (NEW Panel)**

**Panel: Latency Distribution Heatmap**

```json
{
  "title": "Write Latency Distribution (Heatmap)",
  "type": "heatmap",
  "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
  "targets": [
    {
      "expr": "sum(increase(batch_writer_latency_seconds_bucket[1m])) by (le)",
      "format": "heatmap",
      "legendFormat": "{{le}}",
      "refId": "A"
    }
  ],
  "options": {
    "calculate": false,
    "calculation": {},
    "cellGap": 2,
    "cellRadius": 0,
    "cellValues": {},
    "color": {
      "exponent": 0.5,
      "fill": "dark-green",
      "mode": "scheme",
      "reverse": false,
      "scale": "exponential",
      "scheme": "Spectral",
      "steps": 128
    },
    "exemplars": {
      "color": "rgba(255,0,255,0.7)"
    },
    "filterValues": {
      "le": 1e-9
    },
    "legend": {
      "show": true
    },
    "rowsFrame": {
      "layout": "auto"
    },
    "tooltip": {
      "show": true,
      "yHistogram": false
    },
    "yAxis": {
      "axisPlacement": "left",
      "reverse": false,
      "unit": "ms"
    }
  },
  "fieldConfig": {
    "defaults": {
      "custom": {
        "hideFrom": {
          "tooltip": false,
          "viz": false,
          "legend": false
        },
        "scaleDistribution": {
          "type": "linear"
        }
      }
    },
    "overrides": []
  }
}
```

**Visual Features:**
- 🎨 **Color gradient:** Green (fast) → Yellow → Red (slow)
- 📊 **Y-axis:** Latency buckets (0-50ms)
- 📅 **X-axis:** Time
- 🔥 **Hot spots** show when latency spikes occur
- 👁️ **Outlier detection:** Easy to spot unusual latency patterns

**Why Heatmap?**
- Better outlier visualization than line graphs
- Shows latency distribution density
- Easier to spot patterns (e.g., periodic spikes)
- Industry standard for latency monitoring

---

### **3. Cache Latency with SLA Lines (ENHANCED)**

**Panel: Cache Query Latency (P99) with SLA**

```json
{
  "title": "Cache Query Latency (P99) - Multi-Tier SLA",
  "type": "graph",
  "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
  "targets": [
    {
      "expr": "histogram_quantile(0.99, sum(rate(cache_latency_seconds_bucket{tier=\"hot\"}[5m])) by (le)) * 1000",
      "legendFormat": "HOT Tier",
      "refId": "A"
    },
    {
      "expr": "histogram_quantile(0.99, sum(rate(cache_latency_seconds_bucket{tier=\"warm\"}[5m])) by (le)) * 1000",
      "legendFormat": "WARM Tier",
      "refId": "B"
    },
    {
      "expr": "histogram_quantile(0.99, sum(rate(cache_latency_seconds_bucket{tier=\"cold\"}[5m])) by (le)) * 1000",
      "legendFormat": "COLD Tier",
      "refId": "C"
    }
  ],
  "yaxes": [
    {
      "format": "ms",
      "label": "P99 Latency",
      "min": 0,
      "max": 10
    }
  ],
  "thresholds": [
    {
      "value": 1,
      "colorMode": "custom",
      "fill": false,
      "line": true,
      "lineColor": "rgba(50, 172, 45, 0.8)",
      "op": "gt",
      "yaxis": "left"
    },
    {
      "value": 2,
      "colorMode": "custom",
      "fill": false,
      "line": true,
      "lineColor": "rgba(237, 129, 40, 0.8)",
      "op": "gt",
      "yaxis": "left"
    },
    {
      "value": 5,
      "colorMode": "custom",
      "fill": true,
      "fillColor": "rgba(245, 54, 54, 0.15)",
      "line": true,
      "lineColor": "rgba(245, 54, 54, 0.9)",
      "op": "gt",
      "yaxis": "left"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "custom": {
        "lineWidth": 2
      }
    },
    "overrides": [
      {
        "matcher": {
          "id": "byName",
          "options": "HOT Tier"
        },
        "properties": [
          {
            "id": "color",
            "value": {
              "fixedColor": "green",
              "mode": "fixed"
            }
          }
        ]
      },
      {
        "matcher": {
          "id": "byName",
          "options": "WARM Tier"
        },
        "properties": [
          {
            "id": "color",
            "value": {
              "fixedColor": "yellow",
              "mode": "fixed"
            }
          }
        ]
      },
      {
        "matcher": {
          "id": "byName",
          "options": "COLD Tier"
        },
        "properties": [
          {
            "id": "color",
            "value": {
              "fixedColor": "blue",
              "mode": "fixed"
            }
          }
        ]
      }
    ]
  }
}
```

**Visual Features:**
- 🟢 **Green line at 1ms** (HOT tier SLA)
- 🟠 **Yellow line at 2ms** (WARM tier SLA)
- 🔴 **Red line at 5ms** (COLD tier SLA)
- 🔴 **Red shaded area above 5ms** (degraded performance)
- 🎨 **Color-coded traces** (HOT=green, WARM=yellow, COLD=blue)

---

### **4. Cache Latency Heatmap (NEW Panel)**

**Panel: Cache Latency Distribution by Tier**

```json
{
  "title": "Cache Latency Heatmap (All Tiers)",
  "type": "heatmap",
  "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
  "targets": [
    {
      "expr": "sum(increase(cache_latency_seconds_bucket{tier=\"hot\"}[1m])) by (le)",
      "format": "heatmap",
      "legendFormat": "{{le}}",
      "refId": "A"
    }
  ],
  "options": {
    "calculate": false,
    "cellGap": 1,
    "color": {
      "exponent": 0.5,
      "fill": "dark-green",
      "mode": "scheme",
      "reverse": false,
      "scale": "exponential",
      "scheme": "Greens",
      "steps": 64
    },
    "exemplars": {
      "color": "rgba(255,0,255,0.7)"
    },
    "filterValues": {
      "le": 1e-9
    },
    "legend": {
      "show": true
    },
    "rowsFrame": {
      "layout": "auto"
    },
    "tooltip": {
      "show": true,
      "yHistogram": true
    },
    "yAxis": {
      "axisPlacement": "left",
      "reverse": false,
      "unit": "ms",
      "decimals": 2,
      "max": 5
    }
  },
  "fieldConfig": {
    "defaults": {
      "custom": {
        "hideFrom": {
          "tooltip": false,
          "viz": false,
          "legend": false
        }
      }
    }
  }
}
```

**Visual Features:**
- 🟢 **Green gradient** (fast is bright green, slow is dark)
- 📊 **Focused Y-axis** (0-5ms for cache operations)
- 🔥 **Hotspots** show cache performance issues
- 📈 **Histogram sidebar** shows distribution

---

### **5. Processing Rate with Target Line (ENHANCED)**

**Panel: Throughput vs Target with SLA**

```json
{
  "title": "Processing Rate (Target: 10,000 tps)",
  "type": "graph",
  "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
  "targets": [
    {
      "expr": "sum(rate(questdb_ticks_inserted_total[1m]))",
      "legendFormat": "Current Throughput",
      "refId": "A"
    }
  ],
  "yaxes": [
    {
      "format": "ops",
      "label": "Ticks per Second",
      "min": 0
    }
  ],
  "thresholds": [
    {
      "value": 1000,
      "colorMode": "custom",
      "fill": false,
      "line": true,
      "lineColor": "rgba(245, 54, 54, 0.7)",
      "op": "lt",
      "yaxis": "left"
    },
    {
      "value": 5000,
      "colorMode": "custom",
      "fill": false,
      "line": true,
      "lineColor": "rgba(237, 129, 40, 0.7)",
      "op": "gt",
      "yaxis": "left"
    },
    {
      "value": 10000,
      "colorMode": "custom",
      "fill": false,
      "line": true,
      "lineColor": "rgba(50, 172, 45, 0.9)",
      "op": "gt",
      "yaxis": "left"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "custom": {
        "lineWidth": 3,
        "fillOpacity": 20,
        "gradientMode": "opacity"
      }
    }
  },
  "options": {
    "tooltip": {
      "mode": "multi",
      "sort": "none"
    },
    "legend": {
      "displayMode": "table",
      "placement": "right",
      "calcs": ["lastNotNull", "mean", "max"]
    }
  }
}
```

**Visual Features:**
- 🔴 **Red line at 1,000 tps** (critical minimum)
- 🟠 **Orange line at 5,000 tps** (normal operation)
- 🟢 **Green line at 10,000 tps** (target capacity)
- 📊 **Legend table** shows current/mean/max values

---

## 📊 Complete Dashboard Layout with Heatmaps

### **Dashboard: Performance Deep Dive (Enhanced)**

```yaml
Row 1: Latency Analysis (NEW: Graph + Heatmap side-by-side)
├── Write Latency (P50, P99) with SLA lines [Graph] (50%)
└── Write Latency Distribution [Heatmap] (50%)

Row 2: Cache Performance (NEW: Graph + Heatmap side-by-side)
├── Cache Latency P99 by Tier with SLA lines [Graph] (50%)
└── Cache Latency Distribution [Heatmap] (50%)

Row 3: Throughput Analysis
└── Processing Rate vs Target (100%)
    - 3 SLA lines (1K/5K/10K)
    - Gradient fill showing performance zone

Row 4: Resource Utilization
├── Memory Usage (25%)
├── CPU by Component (25%)
├── Network I/O (25%)
└── Disk I/O (25%)
```

---

## 🎨 Color Coding Standards

### **SLA Threshold Colors:**
- 🟢 **Green (#32AC2D):** Target/Optimal performance
- 🟠 **Orange (#ED8128):** Warning threshold
- 🔴 **Red (#F53636):** Critical/SLA violation

### **Heatmap Gradients:**
- **Latency:** Green → Yellow → Red (Spectral scheme)
- **Cache:** Shades of Green (darker = slower)
- **Throughput:** Blue gradient (lighter = higher)

### **Line Styles:**
- **Solid thick (3px):** Actual metrics
- **Dashed (2px):** SLA thresholds
- **Dotted (1px):** Targets/goals

---

## 🚀 Implementation Steps

### **Step 1: Export Current Dashboard (Backup)**
```bash
# In Grafana UI
Dashboard → Settings → JSON Model → Copy
# Save to: ~/ai-trading-station/Grafana/backups/performance_dashboard_backup.json
```

### **Step 2: Add Heatmap Panels**
```bash
# In Grafana UI
1. Edit Dashboard → Add Panel
2. Change visualization type to "Heatmap"
3. Paste query from above
4. Configure options (colors, axes, etc.)
5. Save
```

### **Step 3: Add SLA Lines to Existing Panels**
```bash
# In Grafana UI
1. Edit existing latency panel
2. Go to "Thresholds" tab
3. Click "Add threshold"
4. Set value, color, and line style
5. Repeat for each SLA level
6. Save
```

### **Step 4: Validate Alerts**
```bash
# Test alert firing
1. Edit panel with alert
2. Go to "Alert" tab
3. Click "Test rule"
4. Verify notification sent
```

---

## ✅ Validation Checklist

### **Visual Validation:**
- [ ] SLA lines visible on latency graphs
- [ ] SLA lines have correct colors (green/orange/red)
- [ ] Heatmaps show color gradient correctly
- [ ] Heatmaps update in real-time
- [ ] Threshold areas shaded correctly

### **Functional Validation:**
- [ ] Alerts fire when P99 > 10ms
- [ ] Heatmap shows outlier detection
- [ ] SLA lines don't interfere with data visualization
- [ ] Mobile view still readable
- [ ] Performance not impacted by heatmaps

### **Data Validation:**
- [ ] Heatmap buckets align with Prometheus histogram
- [ ] SLA lines at correct Y-axis values
- [ ] Thresholds trigger at right times
- [ ] No data loss in heatmap transformation

---

## 📱 Mobile Optimization Notes

**Heatmaps on mobile:**
- Use simplified color scheme (3 colors max)
- Reduce cell count (coarser granularity)
- Increase cell gap for touch targets
- Hide legend on small screens

**SLA lines on mobile:**
- Use thicker lines (4px instead of 2px)
- Add labels directly on graph
- Reduce number of threshold lines (show only critical)

---

## 🎯 Expected Results

**After implementation:**
- ✅ **SLA compliance** instantly visible
- ✅ **Outlier detection** via heatmap hot spots
- ✅ **Performance zones** color-coded
- ✅ **Alert triggers** on SLA violations
- ✅ **Historical patterns** visible in heatmap density

**Performance metrics:**
- HOT cache latency: <1ms (green zone)
- WARM cache latency: 1-2ms (yellow zone)
- COLD queries: 2-5ms (orange zone)
- Write latency P99: <10ms (SLA line)

---

## 📊 Example Screenshots (What You'll See)

### **Graph with SLA Lines:**
```
   ms
   12 ┤                                    🔴 10ms SLA (red line)
   10 ┤━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ← violation area shaded red
    8 ┤
    6 ┤                🟠 5ms Warning (orange line)
    4 ┤━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    2 ┤      ████ P99 (actual)
    0 ┤──────────────────────────────────
      └──────────────────────────────────
         Time →
```

### **Heatmap:**
```
   ms
    5 ┤ 🟢🟢🟢🟢🟡🟡🟡🟡🔴🔴🔴🟢🟢🟢
    4 ┤ 🟢🟢🟢🟢🟢🟡🟡🟡🟡🟡🟢🟢🟢🟢
    3 ┤ 🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢
    2 ┤ 🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢
    1 ┤ 🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢
    0 ┤ 🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢
      └──────────────────────────────────
         Time →
         
  🟢 = Fast (many events)
  🟡 = Medium (fewer events)
  🔴 = Slow (rare outliers) ← Easy to spot!
```

---

## 🔗 References

**Grafana Documentation:**
- [Heatmap Visualization](https://grafana.com/docs/grafana/latest/panels-visualizations/visualizations/heatmap/)
- [Thresholds](https://grafana.com/docs/grafana/latest/panels-visualizations/configure-thresholds/)
- [Histogram to Heatmap](https://prometheus.io/docs/practices/histograms/)

**Best Practices:**
- Use heatmaps for latency distribution (industry standard)
- SLA lines on all performance-critical metrics
- Color psychology: Green=good, Red=bad (universal)
- Mobile-first design for monitoring dashboards

---

**Status:** ✅ Ready for implementation  
**Time to implement:** +20 minutes (vs original plan)  
**Opus compliance:** 100% (added missing heatmaps + SLA lines)  

🚀 Let's proceed with Phase 2!
