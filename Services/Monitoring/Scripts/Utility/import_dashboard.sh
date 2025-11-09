#!/bin/bash
# Import Tiered Cache Dashboard into Grafana
# Date: October 26, 2025

set -e

GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="admin"
GRAFANA_PASS="admin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_FILE="${SCRIPT_DIR}/Config/grafana-tiered-cache-dashboard.json"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║           📊 IMPORTING GRAFANA DASHBOARD - TIERED CACHE                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Grafana is running
echo "🔍 Checking if Grafana is running..."
if ! curl -s -o /dev/null -w "%{http_code}" "${GRAFANA_URL}/api/health" | grep -q "200"; then
    echo "❌ Grafana is not running or not accessible at ${GRAFANA_URL}"
    echo "   Please start Grafana: sudo systemctl start grafana-server"
    exit 1
fi
echo "✅ Grafana is running"
echo ""

# Check if dashboard file exists
echo "🔍 Checking dashboard file..."
if [ ! -f "${DASHBOARD_FILE}" ]; then
    echo "❌ Dashboard file not found: ${DASHBOARD_FILE}"
    exit 1
fi
echo "✅ Dashboard file found"
echo ""

# Import dashboard
echo "📤 Importing dashboard to Grafana..."
RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -u "${GRAFANA_USER}:${GRAFANA_PASS}" \
    -d @"${DASHBOARD_FILE}" \
    "${GRAFANA_URL}/api/dashboards/db")

# Check response
if echo "${RESPONSE}" | jq -e '.status == "success"' > /dev/null 2>&1; then
    DASHBOARD_UID=$(echo "${RESPONSE}" | jq -r '.uid')
    DASHBOARD_URL=$(echo "${RESPONSE}" | jq -r '.url')
    
    echo "✅ Dashboard imported successfully!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Dashboard Details:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "   Title: Tiered Cache Performance"
    echo "   UID:   ${DASHBOARD_UID}"
    echo "   URL:   ${GRAFANA_URL}${DASHBOARD_URL}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Dashboard Panels (8 total):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "   1. ✅ Cache Hit Rate by Tier (%) - WITH SLA LINES"
    echo "         • HOT: >60% target (green line)"
    echo "         • Overall: >95% target (bright green line)"
    echo ""
    echo "   2. ⚡ Cache Latency P99 (ms) - WITH 3 SLA THRESHOLDS"
    echo "         • HOT: 1ms SLA (green line)"
    echo "         • WARM: 2ms SLA (yellow line)"
    echo "         • COLD: 5ms SLA (orange line with fill)"
    echo ""
    echo "   3. 📦 Cache Size (Entries per Tier) - Stat Panel"
    echo "         • HOT Cache (Memory)"
    echo "         • WARM Cache (Redis)"
    echo "         • COLD Storage (QuestDB)"
    echo ""
    echo "   4. 🔥 Write Latency Heatmap - Spectral Colors"
    echo "         • Green = fast, Yellow = moderate, Red = slow"
    echo "         • Outlier detection enabled"
    echo ""
    echo "   5. 🌡️  Cache Latency Heatmap - Greens Colors"
    echo "         • Bright = high density, Dark = low density"
    echo "         • Log scale for better visibility"
    echo ""
    echo "   6. 📱 Cache Status (Mobile) - Traffic Light"
    echo "         • 🟢 Green: >95% hit rate"
    echo "         • 🟡 Yellow: 90-95% hit rate"
    echo "         • 🔴 Red: <90% hit rate"
    echo ""
    echo "   7. 📊 Cache Requests Rate (per second)"
    echo "         • Total requests/sec"
    echo "         • HOT/WARM/COLD hits/sec breakdown"
    echo ""
    echo "   8. ⏱️  Average Cache Latency by Tier"
    echo "         • HOT avg latency"
    echo "         • WARM avg latency"
    echo "         • COLD avg latency"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯 Dashboard Features:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "   • ⏱️  Auto-refresh: 5 seconds"
    echo "   • 🕐 Time range: Last 1 hour (adjustable)"
    echo "   • 🎨 SLA threshold lines on latency panels"
    echo "   • 🔥 Heatmap visualizations for outlier detection"
    echo "   • 📱 Mobile-optimized status panel"
    echo "   • 🏷️  Template variables: symbol, data_type"
    echo "   • 🔔 Annotation support for alerts"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 Access Your Dashboard:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "   🌐 URL: ${GRAFANA_URL}${DASHBOARD_URL}"
    echo ""
    echo "   Or navigate in Grafana UI:"
    echo "   Home → Dashboards → Tiered Cache Performance"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "✅ Phase 2 - Grafana Dashboard Implementation: 100% COMPLETE!"
    echo ""
    
elif echo "${RESPONSE}" | jq -e '.message' > /dev/null 2>&1; then
    ERROR_MSG=$(echo "${RESPONSE}" | jq -r '.message')
    echo "⚠️  Dashboard import status: ${ERROR_MSG}"
    echo ""
    echo "Response: ${RESPONSE}"
    echo ""
    
    # Check if it's an authentication error
    if echo "${ERROR_MSG}" | grep -qi "invalid username or password"; then
        echo "💡 Tip: Default Grafana credentials are admin/admin"
        echo "   If you changed them, update GRAFANA_USER and GRAFANA_PASS in this script"
    fi
    
    exit 1
else
    echo "❌ Dashboard import failed"
    echo "Response: ${RESPONSE}"
    exit 1
fi
