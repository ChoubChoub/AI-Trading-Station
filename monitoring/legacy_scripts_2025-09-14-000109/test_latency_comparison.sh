#!/bin/bash

echo "═══════════════════════════════════════════════════════"
echo "    AI TRADING STATION - LATENCY TEST COMPARISON"
echo "═══════════════════════════════════════════════════════"
echo ""

# Test 1: Your original test
echo "1️⃣  ORIGINAL TEST (ai-trading-station.sh)"
echo "──────────────────────────────────────"
cd ~/ai-trading-station
./ai-trading-station.sh test 2>&1 | grep -E "Mean:|P99:|Min:|Max:" | head -5
echo ""

# Test 2: Python monitoring with onload-trading
echo "2️⃣  MONITORING WITH onload-trading"
echo "───────────────────────────────────"
cd ~/ai-trading-station/monitoring/scripts
~/ai-trading-station/scripts/onload-trading python3 monitor_trading_latency_wrapped.py --samples 1000 2>&1 | grep -E "Mean:|P99:|Min:|Max:|Running with" | head -6
echo ""

# Test 3: Check what's different
echo "3️⃣  ENVIRONMENT CHECK"
echo "────────────────────"
~/ai-trading-station/scripts/onload-trading python3 -c "
import os
import socket

print('Python environment:')
for key in sorted(os.environ.keys()):
    if 'EF_' in key or 'ONLOAD' in key:
        print(f'  {key}={os.environ[key]}')

# Test socket creation
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
print(f'\nSocket FD: {s.fileno()}')
s.close()
" 2>&1 | head -20
echo ""

echo "═══════════════════════════════════════════════════════"
