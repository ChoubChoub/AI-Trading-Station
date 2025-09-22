#!/bin/bash

echo "═══════════════════════════════════════════════════════"
echo "         EXACT COMPARISON TEST"
echo "═══════════════════════════════════════════════════════"
echo ""

echo "1️⃣  YOUR ORIGINAL (ai-trading-station.sh test)"
echo "────────────────────────────────────────────────"
cd ~/ai-trading-station
timeout 10 ./ai-trading-station.sh test 2>&1 | grep -A 5 "Test 3: Standard TCP" | grep -E "Mean:|P99:"
echo ""

echo "2️⃣  EXACT MATCH MONITOR"
echo "───────────────────────"
cd ~/ai-trading-station/monitoring/scripts
onload-trading python3 monitor_exact_match.py 2>&1 | grep -E "Mean:|P99:|Running"
echo ""

echo "3️⃣  DIRECT PYTHON TEST"
echo "──────────────────────"
cd ~/ai-trading-station/tests
onload-trading python3 comprehensive_onload_test.py 2>&1 | grep -A 5 "Test 3: Standard TCP" | grep -E "Mean:|P99:"
echo ""

echo "═══════════════════════════════════════════════════════"
