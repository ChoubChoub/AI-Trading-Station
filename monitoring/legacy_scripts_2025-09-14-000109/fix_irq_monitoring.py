#!/usr/bin/env python3
"""Fix the IRQ monitoring to check for active violations only"""

import re

# Read the current monitor script
with open('monitor_trading_system.py', 'r') as f:
    lines = f.readlines()

# Find the check_irq_affinity method and replace it
in_method = False
method_indent = None
new_lines = []
method_start = -1

for i, line in enumerate(lines):
    if 'def check_irq_affinity(self):' in line:
        in_method = True
        method_start = i
        method_indent = len(line) - len(line.lstrip())
        # Add the new method
        new_lines.append(line)
        new_lines.append(' ' * (method_indent + 4) + '"""Check for ACTIVE IRQ violations on isolated cores (not historical)"""\n')
        new_lines.append(' ' * (method_indent + 4) + 'import time\n')
        new_lines.append(' ' * (method_indent + 4) + 'alerts = []\n')
        new_lines.append(' ' * (method_indent + 4) + '\n')
        new_lines.append(' ' * (method_indent + 4) + '# Take initial snapshot\n')
        new_lines.append(' ' * (method_indent + 4) + 'initial_irqs = {}\n')
        new_lines.append(' ' * (method_indent + 4) + 'with open("/proc/interrupts", "r") as f:\n')
        new_lines.append(' ' * (method_indent + 8) + 'for line in f:\n')
        new_lines.append(' ' * (method_indent + 12) + 'parts = line.strip().split()\n')
        new_lines.append(' ' * (method_indent + 12) + 'if parts and parts[0][:-1].isdigit():\n')
        new_lines.append(' ' * (method_indent + 16) + 'irq_num = parts[0][:-1]\n')
        new_lines.append(' ' * (method_indent + 16) + '# Get counts for isolated cores (cores 2,3)\n')
        new_lines.append(' ' * (method_indent + 16) + 'if len(parts) > 4:\n')
        new_lines.append(' ' * (method_indent + 20) + 'try:\n')
        new_lines.append(' ' * (method_indent + 24) + 'count_core2 = int(parts[3])\n')
        new_lines.append(' ' * (method_indent + 24) + 'count_core3 = int(parts[4])\n')
        new_lines.append(' ' * (method_indent + 24) + 'initial_irqs[irq_num] = (count_core2, count_core3)\n')
        new_lines.append(' ' * (method_indent + 20) + 'except (ValueError, IndexError):\n')
        new_lines.append(' ' * (method_indent + 24) + 'pass\n')
        new_lines.append(' ' * (method_indent + 4) + '\n')
        new_lines.append(' ' * (method_indent + 4) + '# Wait to detect active interrupts\n')
        new_lines.append(' ' * (method_indent + 4) + 'time.sleep(2)\n')
        new_lines.append(' ' * (method_indent + 4) + '\n')
        new_lines.append(' ' * (method_indent + 4) + '# Take second snapshot and compare\n')
        new_lines.append(' ' * (method_indent + 4) + 'with open("/proc/interrupts", "r") as f:\n')
        new_lines.append(' ' * (method_indent + 8) + 'for line in f:\n')
        new_lines.append(' ' * (method_indent + 12) + 'parts = line.strip().split()\n')
        new_lines.append(' ' * (method_indent + 12) + 'if parts and parts[0][:-1].isdigit():\n')
        new_lines.append(' ' * (method_indent + 16) + 'irq_num = parts[0][:-1]\n')
        new_lines.append(' ' * (method_indent + 16) + 'if len(parts) > 4 and irq_num in initial_irqs:\n')
        new_lines.append(' ' * (method_indent + 20) + 'try:\n')
        new_lines.append(' ' * (method_indent + 24) + 'count_core2 = int(parts[3])\n')
        new_lines.append(' ' * (method_indent + 24) + 'count_core3 = int(parts[4])\n')
        new_lines.append(' ' * (method_indent + 24) + '\n')
        new_lines.append(' ' * (method_indent + 24) + '# Check if counts increased (active violations)\n')
        new_lines.append(' ' * (method_indent + 24) + 'delta_core2 = count_core2 - initial_irqs[irq_num][0]\n')
        new_lines.append(' ' * (method_indent + 24) + 'delta_core3 = count_core3 - initial_irqs[irq_num][1]\n')
        new_lines.append(' ' * (method_indent + 24) + '\n')
        new_lines.append(' ' * (method_indent + 24) + 'if delta_core2 > 0 or delta_core3 > 0:\n')
        new_lines.append(' ' * (method_indent + 28) + 'irq_desc = " ".join(parts[-2:]) if len(parts) > 6 else parts[-1]\n')
        new_lines.append(' ' * (method_indent + 28) + 'if delta_core2 > 0:\n')
        new_lines.append(' ' * (method_indent + 32) + 'alerts.append(f"ACTIVE IRQ {irq_num} ({irq_desc}) on core 2: +{delta_core2} interrupts/2s")\n')
        new_lines.append(' ' * (method_indent + 28) + 'if delta_core3 > 0:\n')
        new_lines.append(' ' * (method_indent + 32) + 'alerts.append(f"ACTIVE IRQ {irq_num} ({irq_desc}) on core 3: +{delta_core3} interrupts/2s")\n')
        new_lines.append(' ' * (method_indent + 20) + 'except (ValueError, IndexError):\n')
        new_lines.append(' ' * (method_indent + 24) + 'pass\n')
        new_lines.append(' ' * (method_indent + 4) + '\n')
        new_lines.append(' ' * (method_indent + 4) + 'return alerts\n')
        in_method = False
    elif in_method:
        # Skip old method lines until we find the next method or dedent
        if line.strip() and not line.startswith(' ' * (method_indent + 4)):
            in_method = False
            new_lines.append(line)
        elif line.strip().startswith('def '):
            in_method = False
            new_lines.append(line)
    else:
        new_lines.append(line)

# Write the fixed version
with open('monitor_trading_system.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Fixed IRQ monitoring to check for ACTIVE violations only (2-second sampling)")
print("   - Will only alert on IRQs actively happening on isolated cores")
print("   - Ignores historical counts from boot time")
