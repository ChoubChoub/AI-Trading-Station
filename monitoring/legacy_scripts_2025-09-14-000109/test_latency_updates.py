#!/usr/bin/env python3
"""Test if latency measurements are actually changing"""

import time
from monitor_trading_system_v2 import TradingSystemMonitor

monitor = TradingSystemMonitor()

print("Testing 3 consecutive latency measurements (should be slightly different):")
print("="*60)

for i in range(3):
    print(f"\nMeasurement {i+1}:")
    result = monitor.measure_trading_latency()
    if result:
        print(f"  Mean: {result['mean']:.2f}μs")
        print(f"  Min:  {result['min']:.2f}μs")
        print(f"  Max:  {result['max']:.2f}μs")
        print(f"  P99:  {result['p99']:.2f}μs")
    else:
        print("  ERROR: Measurement failed!")
    
    if i < 2:
        print("  Waiting 2 seconds before next measurement...")
        time.sleep(2)

print("\n" + "="*60)
print("If all three measurements are IDENTICAL, there's a caching issue.")
print("If they're slightly different, the dashboard has an update problem.")
