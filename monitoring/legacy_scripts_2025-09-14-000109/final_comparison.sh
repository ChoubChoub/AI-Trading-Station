#!/bin/bash

echo "═══════════════════════════════════════════════════════════"
echo "    FINAL LATENCY COMPARISON - PERSISTENT vs NEW CONNECTIONS"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "1️⃣  YOUR BASELINE (Persistent Connection)"
echo "───────────────────────────────────────"
cd ~/ai-trading-station
./ai-trading-station.sh test 2>&1 | grep -A 5 "Test 3:" | grep -E "Mean:|P99:"
echo ""

echo "2️⃣  MONITOR WITH PERSISTENT CONNECTION (Should Match!)"
echo "──────────────────────────────────────────────────"
cd ~/ai-trading-station/monitoring/scripts
onload-trading python3 monitor_persistent_connection.py 2>&1 | grep -E "Mean:|P99:|Running"
echo ""

echo "3️⃣  MONITOR WITH NEW CONNECTIONS (Higher Latency)"
echo "────────────────────────────────────────────────"
onload-trading python3 monitor_exact_match.py 2>&1 | grep -E "Mean:|P99:" | head -2
echo ""

echo "📊 ANALYSIS:"
echo "───────────"
echo "• Persistent connection: ~4-6μs (like real trading)"
echo "• New connections: ~15μs (connection overhead)"
echo "• For monitoring persistent connections, use monitor_persistent_connection.py"
echo ""
echo "═══════════════════════════════════════════════════════════"
